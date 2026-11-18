"""GPT-2 small with ccml kernels swapped in (ROADMAP Phase 14; reference.md
sections 3, 5 "end-to-end model latency", 9 "bindings usable in a model";
enhancements C4 model parity, G1 CUDA Graphs, P1 torch.compile, P2 drop-ins).

    python scripts/model_bench.py --out-dir results/sm_75/pytorch_extension
        [--model gpt2] [--batches 1,8] [--seqs 128,512] [--eval-tokens 4096]

Parity (C4), written to model_parity.json:
  * logits max |error| against an FP32 copy of the same model, for stock FP16
    and ccml FP16, on the first evaluation window;
  * perplexity of FP32, stock FP16 and ccml FP16 on WikiText-2 (test split,
    non-overlapping 512-token windows of the first --eval-tokens tokens).
Latency, written to model_latency.csv (scripts/benchlib.py schema and method:
CUDA events, warmup 20, >= 100 iterations, interleaved, median and spread):
  {eager, compile, cudagraph} x {stock (transformers' default attention,
  recorded: SDPA, a fused attention kernel), stock-eager-attn (transformers'
  unfused eager attention: the same matmul, softmax, matmul structure as
  ccml's), +ccml (every swap), +ccml-nolinear (norms and attention softmax
  only; cuBLAS linears)}
  for every (batch, seq): one forward pass, logits out, no KV cache. The
  no-linear variant attributes the end-to-end time between ccml's matmul and
  its row-wise kernels.
Dispatch cost, written to model_dispatch.csv: each ccml op and its ATen
counterpart on decode-sized inputs, 50 calls per timed iteration between CUDA
events (per call = median / 50); at these sizes the time is launch and
dispatch bound, so it measures the binding, not the kernel.
Every candidate is gated before it is timed: its logits must be within
max(2 x stock FP16's own error, 1e-2 x logit scale) of the FP32 model on the
same input (reference.md section 11: never time what fails its check).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import sys
import time
import traceback

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
try:  # the installed extension (setup.py); else the source tree (JIT build)
    import ccml_ext  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO / "python" / "ccml_ext"))


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def eval_ids(tokenizer, n_tokens: int) -> tuple[torch.Tensor, str]:
    """WikiText-2 test split (needs internet once); a committed fallback text
    otherwise. Returns (token ids, source description)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
        source = "wikitext-2-raw-v1 test"
    except Exception as e:  # noqa: BLE001
        log(f"wikitext unavailable ({type(e).__name__}: {e}); using scripts/data/eval_text.txt")
        text = (REPO / "scripts" / "data" / "eval_text.txt").read_text(encoding="utf-8")
        source = "scripts/data/eval_text.txt"
    ids = tokenizer(text, return_tensors="pt").input_ids[0][:n_tokens]
    return ids, source


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, window: int = 512) -> float:
    nll, count = 0.0, 0
    for start in range(0, ids.numel() - 1, window):
        chunk = ids[start:start + window + 1].to("cuda")
        if chunk.numel() < 2:
            break
        logits = model(chunk[None, :-1], use_cache=False).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0], chunk[1:], reduction="sum")
        nll += loss.item()
        count += chunk.numel() - 1
    return math.exp(nll / count)


def graph_forward(model, ids: torch.Tensor):
    """Capture one forward pass as a CUDA Graph; returns a replay function."""
    static = ids.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):  # warm up allocations outside the capture
            model(static, use_cache=False)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        out = model(static, use_cache=False).logits

    def replay():
        # The graph reads `static`'s memory: the closure must keep it alive
        # (first T4 run: freed and reused, replays gathered garbage token ids).
        graph.replay()
        return out if static is not None else None
    return replay


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--seqs", default="128,512")
    ap.add_argument("--eval-tokens", type=int, default=4096)
    ap.add_argument("--peaks")
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers

    import ccml_ext

    torch.backends.cuda.matmul.allow_tf32 = False
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    ref32 = transformers.GPT2LMHeadModel.from_pretrained(a.model).to("cuda").eval()
    stock = copy.deepcopy(ref32).half().eval()
    ccml_model = copy.deepcopy(stock)
    report = ccml_ext.patch_model(ccml_model)
    nolin = copy.deepcopy(stock)
    report_nolin = ccml_ext.patch_model(nolin, linear=False)
    stock_eager_attn = copy.deepcopy(stock)
    if hasattr(stock_eager_attn, "set_attn_implementation"):
        stock_eager_attn.set_attn_implementation("eager")
    else:
        stock_eager_attn.config._attn_implementation = "eager"
    attn_impls = {name: getattr(m.config, "_attn_implementation", "unknown")
                  for name, m in (("stock", stock), ("stock_eager_attn", stock_eager_attn),
                                  ("ccml", ccml_model), ("ccml_nolinear", nolin))}
    log(f"attention implementations: {attn_impls}")
    log(f"patched: {report}; no-linear: {report_nolin}")

    # ------------------------------------------------------------ parity --
    ids, source = eval_ids(tok, a.eval_tokens)
    with torch.no_grad():
        first = ids[:512].to("cuda")[None]
        l32 = ref32(first, use_cache=False).logits.float()
        ccml_ext.modules.ATTENTION_CALLS = 0
        ccml_ext.modules.UNFUSED_MASK_CALLS = 0
        l_ccml = ccml_model(first, use_cache=False).logits.float()
        attention_calls = ccml_ext.modules.ATTENTION_CALLS
        unfused_mask_calls = ccml_ext.modules.UNFUSED_MASK_CALLS
        l_nolin = nolin(first, use_cache=False).logits.float()
        l_eager_attn = stock_eager_attn(first, use_cache=False).logits.float()
        l_stock = stock(first, use_cache=False).logits.float()
    parity = {
        "model": a.model,
        "eval_source": source,
        "eval_tokens": int(ids.numel()),
        "patch_report": report,
        "patch_report_nolinear": report_nolin,
        "attention_implementations": attn_impls,
        "ccml_attention_calls_in_one_forward": attention_calls,
        "of_which_mask_applied_in_pytorch": unfused_mask_calls,
        "logits_max_abs_err_stock_fp16_vs_fp32": (l_stock - l32).abs().max().item(),
        "logits_max_abs_err_ccml_fp16_vs_fp32": (l_ccml - l32).abs().max().item(),
        "logits_max_abs_diff_ccml_vs_stock_fp16": (l_ccml - l_stock).abs().max().item(),
        "logits_max_abs_err_ccml_nolinear_fp16_vs_fp32": (l_nolin - l32).abs().max().item(),
        # Eager attention rounds the scores to FP16 before the softmax, as
        # ccml's does; SDPA (the default) does not.
        "logits_max_abs_err_stock_eager_attn_fp16_vs_fp32":
            (l_eager_attn - l32).abs().max().item(),
        "logits_scale_max_abs_fp32": l32.abs().max().item(),
        "top1_agreement_ccml_vs_fp32": (l_ccml.argmax(-1) == l32.argmax(-1)).float().mean().item(),
        "top1_agreement_stock_vs_fp32": (l_stock.argmax(-1) == l32.argmax(-1)).float().mean().item(),
        "perplexity_fp32": perplexity(ref32, ids),
        "perplexity_stock_fp16": perplexity(stock, ids),
        "perplexity_ccml_fp16": perplexity(ccml_model, ids),
        "perplexity_ccml_nolinear_fp16": perplexity(nolin, ids),
        "perplexity_stock_eager_attn_fp16": perplexity(stock_eager_attn, ids),
    }
    parity["perplexity_delta_ccml_vs_stock_pct"] = 100.0 * (
        parity["perplexity_ccml_fp16"] / parity["perplexity_stock_fp16"] - 1.0)
    parity["gpu"] = torch.cuda.get_device_name()
    parity["commit"] = os.environ.get("CCML_COMMIT", "unknown")
    parity["torch"] = torch.__version__
    parity["transformers"] = transformers.__version__
    (out_dir / "model_parity.json").write_text(json.dumps(parity, indent=2) + "\n",
                                                encoding="utf-8")
    log(json.dumps(parity, indent=2))

    # ----------------------------------------------------------- latency --
    tel = bl.Telemetry()
    p = torch.cuda.get_device_properties(0)
    peaks = bl.load_peaks(a.peaks)
    ctx = {"gpu": p.name, "cc": f"{p.major}.{p.minor}", "driver": tel.driver(),
           "cuda": torch.version.cuda or "unknown", "commit": parity["commit"],
           "fp32_tflops_theory": peaks.get("fp32_tflops_theory", 0.0),
           "fp32_tflops_measured": peaks.get("fp32_tflops_measured", 0.0),
           "bandwidth_gbps_theory": peaks.get("bandwidth_gbps_theory", 0.0),
           "bandwidth_gbps_measured": peaks.get("bandwidth_gbps_measured", 0.0)}
    path = out_dir / "model_latency.csv"
    with open(path, "w", encoding="utf-8") as out:
        out.write(f"# gpu: {ctx['gpu']} (cc {ctx['cc']})  driver: {ctx['driver']}  "
                  f"torch: {torch.__version__}  transformers: {transformers.__version__}  "
                  f"commit: {ctx['commit']}\n")
        out.write(f"# model: {a.model} fp16, one forward pass (logits), no KV cache; "
                  f"patch: {report}; nolinear patch: {report_nolin}\n")
        out.write(bl.CSV_HEADER + "\n")
        for b in (int(v) for v in a.batches.split(",")):
            for s in (int(v) for v in a.seqs.split(",")):
                t0 = time.time()
                g = torch.Generator(device="cuda").manual_seed(b * 1000 + s)
                x = torch.randint(0, tok.vocab_size, (b, s), device="cuda", generator=g)
                with torch.no_grad():
                    ref = ref32(x, use_cache=False).logits.float()
                    own_err = (stock(x, use_cache=False).logits.float() - ref).abs().max().item()
                bound = max(2.0 * own_err, 1e-2 * ref.abs().max().item())
                torch._dynamo.reset()
                cands, notes = [], {}

                def nograd(f):
                    def run():
                        with torch.no_grad():
                            return f()
                    return run

                models = (("", stock), ("[eager-attn]", stock_eager_attn), ("+ccml", ccml_model),
                          ("+ccml-nolinear", nolin))
                stocks = (stock, stock_eager_attn)
                for suffix, m in models:
                    cands.append(bl.Candidate("pytorch_eager" if m in stocks else "ccml",
                                              "eager" + suffix,
                                              nograd(lambda m=m: m(x, use_cache=False).logits)))
                for suffix, m in models:
                    name = "compile" + suffix
                    try:
                        cm = torch.compile(m)
                        cands.append(bl.Candidate("torch_compile" if m in stocks else "ccml", name,
                                                  nograd(lambda cm=cm: cm(x, use_cache=False).logits)))
                    except Exception as e:  # noqa: BLE001
                        notes[name] = f"{type(e).__name__}: {e}"
                for suffix, m in models:
                    name = "cudagraph" + suffix
                    try:
                        cands.append(bl.Candidate("cuda_graph" if m in stocks else "ccml", name,
                                                  graph_forward(m, x)))
                    except Exception as e:  # noqa: BLE001
                        notes[name] = f"{type(e).__name__}: {e}"
                        log(f"  {name} capture failed b{b} s{s}: {notes[name]}")
                gated = []
                for c in cands:
                    try:
                        err = (c.fn().float() - ref).abs().max().item()
                        torch.cuda.synchronize()
                        gate = "PASS" if err <= bound else "FAIL"
                        gated.append((c, gate, err))
                    except Exception:  # noqa: BLE001
                        log(f"  {c.algo} not run b{b} s{s}:\n{traceback.format_exc()}")
                        gated.append((c, "NOT_RUN", None))
                passing = [c for c, gate, _ in gated if gate == "PASS"]
                measured = bl.time_interleaved(passing, False, None, tel=tel) if passing else []
                k = 0
                shape = {"tag": f"{a.model}.b{b}.s{s}", "rows": b, "cols": s}
                for c, gate, err in gated:
                    m = None
                    if gate == "PASS":
                        m, k = measured[k], k + 1
                    out.write(bl.csv_row(op="gpt2_forward", impl=c.impl, algo=c.algo,
                                         dtype="f16", shape=shape, cold=False, gate=gate,
                                         max_abs=err, max_rel=None, flops=0.0, nbytes=0.0,
                                         measured=m, tel=(m.telemetry if m and m.telemetry
                                                          else tel.sample()), ctx=ctx) + "\n")
                for name, why in notes.items():
                    out.write(f"# b{b} s{s} {name}: NOT_RUN ({why})\n")
                out.flush()
                log(f"done b{b} s{s} ({time.time() - t0:.1f}s)")
    log(f"wrote {path}")
    dispatch_cost(out_dir, tel, ctx, ccml_ext)
    return 0


def dispatch_cost(out_dir: pathlib.Path, tel, ctx, ccml_ext) -> None:
    """Per-call cost of each ccml op and its ATen counterpart at decode sizes
    (GPT-2 small, one token), where launch and dispatch dominate."""
    import torch.nn.functional as F
    from ccml_ext import ops

    calls = 50
    h = torch.float16
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(1, 768, device="cuda", generator=g).to(h)
    w = torch.randn(768, device="cuda", generator=g).to(h)
    b = torch.randn(768, device="cuda", generator=g).to(h)
    wk = (torch.randn(768, 3072, device="cuda", generator=g) * 0.03).to(h)
    bk = torch.randn(3072, device="cuda", generator=g).to(h)
    wt = wk.t().contiguous()
    s = torch.randn(12, 1, 128, device="cuda", generator=g).to(h)

    def loop(f):
        def run():
            with torch.no_grad():
                for _ in range(calls):
                    f()
        return run

    pairs = {
        "layernorm": (lambda: ops.layernorm(x, w, b, 1e-5),
                      lambda: F.layer_norm(x, (768,), w, b, 1e-5)),
        "softmax_masked": (lambda: ops.softmax_masked(s, None, 0.125, 1),
                           lambda: torch.softmax(s * 0.125, dim=-1)),
        "linear_gelu": (lambda: ops.linear(x, wk, bk, None, ops.ACT_GELU_TANH),
                        lambda: F.gelu(F.linear(x, wt, bk), approximate="tanh")),
    }
    # fp64 references at the tolerances of python/ccml_ext/tests/test_ext.py:
    # every row is gated before it is timed. (causal_q = 1 with 128 keys masks
    # nothing: bottom-right alignment lets the one query see every key.)
    prod = x.double() @ wk.double()
    abs_t, rel_t = bl.tolerance_dot(768, prod.pow(2).mean().sqrt().item())
    refs = {
        "layernorm": (F.layer_norm(x.double(), (768,), w.double(), b.double(), 1e-5),
                      bl.widen_for_storage(bl.tolerance_reduction(768), "f16")),
        "softmax_masked": (torch.softmax(s.double() * 0.125, dim=-1),
                           bl.widen_for_storage(bl.tolerance_softmax(128), "f16")),
        "linear_gelu": (F.gelu(prod + bk.double(), approximate="tanh"),
                        bl.widen_for_storage((2 * abs_t, rel_t), "f16")),
    }
    path = out_dir / "model_dispatch.csv"
    with open(path, "w", encoding="utf-8") as out:
        out.write(f"# per-call cost = median_us / {calls}; decode-sized inputs (1 x 768 "
                  f"fp16, 12 x 1 x 128 scores, 1 x 768 x 3072 linear); commit "
                  f"{ctx['commit']}\n")
        out.write(bl.CSV_HEADER + "\n")
        for name, (ours, theirs) in pairs.items():
            ref, tol = refs[name]
            gated = []
            for impl, f in (("ccml", ours), ("pytorch_eager", theirs)):
                with torch.no_grad():
                    ok, mabs, mrel = bl.compare(f(), ref, *tol)
                gated.append((bl.Candidate(impl, f"{name} x{calls}", loop(f)),
                              "PASS" if ok else "FAIL", mabs, mrel))
            passing = [c for c, gate, _, _ in gated if gate == "PASS"]
            measured = iter(bl.time_interleaved(passing, False, None, tel=tel) if passing else [])
            for c, gate, mabs, mrel in gated:
                m = next(measured) if gate == "PASS" else None
                out.write(bl.csv_row(op="dispatch", impl=c.impl, algo=c.algo, dtype="f16",
                                     shape={"tag": name, "rows": 1, "cols": 768}, cold=False,
                                     gate=gate, max_abs=mabs, max_rel=mrel, flops=0.0,
                                     nbytes=0.0, measured=m,
                                     tel=(m.telemetry if m and m.telemetry else tel.sample()),
                                     ctx=ctx) + "\n")
                if m is not None:
                    log(f"dispatch {name} {c.impl}: {m.summary.median / calls:.2f} us/call")
    log(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
