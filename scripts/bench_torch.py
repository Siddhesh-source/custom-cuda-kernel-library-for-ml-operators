"""Python-side baselines: PyTorch eager, torch.compile (Inductor) and Triton
(reference.md §4; enhancement B1), measured with the same method, gate and CSV
schema as the C++ benches (scripts/benchlib.py mirrors bench/support/).

    python scripts/bench_torch.py --out-dir results/sm_75 --peaks results/sm_75/peaks.json
        [--ops matmul,layernorm,softmax] [--impls eager,compile,triton]
        [--filter TEXT] [--cache warm|cold|both] [--compile-mode max-autotune]

PyTorch eager matmul is cuBLAS and eager softmax is PyTorch's own CUDA kernel;
eager layer_norm is the layernorm vendor baseline (cuDNN has no legacy-API
layernorm; see bench/baselines/cudnn_softmax.h).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
EPS = 1e-5


def context(peaks_path: str | None, tel: bl.Telemetry) -> dict:
    p = torch.cuda.get_device_properties(0)
    peaks = bl.load_peaks(peaks_path)
    return {
        "gpu": p.name, "cc": f"{p.major}.{p.minor}", "driver": tel.driver(),
        "cuda": torch.version.cuda or "unknown",
        "commit": os.environ.get("CCML_COMMIT", "unknown"),
        "fp32_tflops_theory": peaks.get("fp32_tflops_theory", 0.0),
        "fp32_tflops_measured": peaks.get("fp32_tflops_measured", 0.0),
        "bandwidth_gbps_theory": peaks.get("bandwidth_gbps_theory", 0.0),
        "bandwidth_gbps_measured": peaks.get("bandwidth_gbps_measured", 0.0),
        "l2": getattr(p, "L2_cache_size", 4 << 20),
    }


def compiled(fn, mode: str):
    torch._dynamo.reset()  # one compile per shape; no recompile-limit surprises
    return torch.compile(fn, mode=mode, dynamic=False, fullgraph=True)


def setup(op: str, shape: dict, impls: list[str], mode: str):
    """Returns (inputs, fp64 reference, (abs_tol, rel_tol), flops, bytes, cands)."""
    g = torch.Generator(device="cuda").manual_seed(1234)
    cands: list[bl.Candidate] = []
    triton_mod = None
    if "triton" in impls:
        try:
            import triton_kernels as triton_mod  # noqa: F811
        except Exception as e:  # noqa: BLE001
            print(f"  triton unavailable: {e}", file=sys.stderr)

    if op == "matmul":
        M, N, K = shape["M"], shape["N"], shape["K"]
        a = torch.randn(M, K, device="cuda", generator=g)
        b = torch.randn(K, N, device="cuda", generator=g)
        ref = a.double() @ b.double()
        tol = bl.tolerance_dot(K, ref.pow(2).mean().sqrt().item())
        fn = lambda: torch.mm(a, b)  # noqa: E731
        if "eager" in impls:
            cands.append(bl.Candidate("pytorch_eager", "torch.mm", fn))
        if "compile" in impls:
            cf = compiled(lambda x, y: torch.mm(x, y), mode)
            cands.append(bl.Candidate("torch_compile", mode, lambda: cf(a, b)))
        if triton_mod is not None:
            cands.append(bl.Candidate("triton", "tutorial_autotuned",
                                      lambda: triton_mod.matmul(a, b)))
        return ref, tol, bl.matmul_flops(M, N, K), bl.matmul_bytes(M, N, K, 4), cands

    rows, cols = shape["rows"], shape["cols"]
    x = torch.randn(rows, cols, device="cuda", generator=g)
    if op == "layernorm":
        w = torch.rand(cols, device="cuda", generator=g) + 0.5
        bias = torch.rand(cols, device="cuda", generator=g) * 2 - 1
        ref = F.layer_norm(x.double(), (cols,), w.double(), bias.double(), EPS)
        tol = bl.tolerance_reduction(cols)
        if "eager" in impls:
            cands.append(bl.Candidate("pytorch_eager", "F.layer_norm",
                                      lambda: F.layer_norm(x, (cols,), w, bias, EPS)))
        if "compile" in impls:
            cf = compiled(lambda t, ww, bb: F.layer_norm(t, (cols,), ww, bb, EPS), mode)
            cands.append(bl.Candidate("torch_compile", mode, lambda: cf(x, w, bias)))
        if triton_mod is not None:
            cands.append(bl.Candidate("triton", "tutorial_looped",
                                      lambda: triton_mod.layernorm(x, w, bias, EPS)))
        return ref, tol, 0.0, bl.layernorm_bytes(rows, cols, 4), cands

    ref = torch.softmax(x.double(), dim=-1)
    tol = bl.tolerance_reduction(cols)
    if "eager" in impls:
        cands.append(bl.Candidate("pytorch_eager", "torch.softmax",
                                  lambda: torch.softmax(x, dim=-1)))
    if "compile" in impls:
        cf = compiled(lambda t: torch.softmax(t, dim=-1), mode)
        cands.append(bl.Candidate("torch_compile", mode, lambda: cf(x)))
    if triton_mod is not None:
        cands.append(bl.Candidate("triton", "online_looped", lambda: triton_mod.softmax(x)))
    return ref, tol, 0.0, bl.softmax_bytes(rows, cols, 4), cands


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default=str(REPO / "bench" / "shapes.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--peaks")
    ap.add_argument("--ops", default="matmul,layernorm,softmax")
    ap.add_argument("--impls", default="eager,compile,triton")
    ap.add_argument("--filter", default="")
    ap.add_argument("--cache", default="both", choices=["warm", "cold", "both"])
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument("--min-iters", type=int, default=100)
    ap.add_argument("--max-iters", type=int, default=200)
    args = ap.parse_args()
    if args.min_iters < 100:
        ap.error("--min-iters below 100 violates the measurement method (reference.md)")
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1

    # FP32 means FP32: no TF32 in cuBLAS or cuDNN (T4 has none anyway; Ada does).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tel = bl.Telemetry()
    ctx = context(args.peaks, tel)
    flush = torch.empty(2 * ctx["l2"] // 4, device="cuda", dtype=torch.float32)
    shapes = json.loads(pathlib.Path(args.shapes).read_text(encoding="utf-8"))
    impls = args.impls.split(",")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for op in args.ops.split(","):
        path = out_dir / f"torch_{op}.csv"
        with open(path, "w", encoding="utf-8") as out:
            out.write(f"# gpu: {ctx['gpu']} (cc {ctx['cc']})  driver: {ctx['driver']}  "
                      f"cuda: {ctx['cuda']}  torch: {torch.__version__}  commit: {ctx['commit']}\n")
            out.write(f"# compile mode: {args.compile_mode}; telemetry: "
                      f"{'NVML' if tel.ok else 'unavailable'}; other compute processes: "
                      f"{tel.other_processes()}\n")
            out.write(bl.CSV_HEADER + "\n")
            for shape in shapes[op]:
                if args.filter and args.filter not in shape["tag"]:
                    continue
                t0 = time.time()
                try:
                    ref, tol, flops, nbytes, cands = setup(op, shape, impls, args.compile_mode)
                except Exception:  # noqa: BLE001
                    print(f"setup failed {op} {shape['tag']}:\n{traceback.format_exc()}",
                          file=sys.stderr)
                    continue
                gated = []
                for c in cands:
                    try:
                        ok, mabs, mrel = bl.compare(c.fn(), ref, *tol)
                        torch.cuda.synchronize()
                        gated.append((c, "PASS" if ok else "FAIL", mabs, mrel))
                    except Exception as e:  # noqa: BLE001
                        print(f"  {c.impl} not run on {shape['tag']}: {e}", file=sys.stderr)
                        gated.append((c, "NOT_RUN", None, None))
                del ref
                passing = [c for c, gate, *_ in gated if gate == "PASS"]
                for cold in (False, True):
                    if (cold and args.cache == "warm") or (not cold and args.cache == "cold"):
                        continue
                    if cold and args.cache == "both" and nbytes > 4 * ctx["l2"]:
                        continue  # same rule as the C++ side: cold == warm here
                    measured = bl.time_interleaved(passing, cold, flush,
                                                   min_iters=args.min_iters,
                                                   max_iters=args.max_iters,
                                                   tel=tel) if passing else []
                    t = tel.sample()
                    k = 0
                    for c, gate, mabs, mrel in gated:
                        m = None
                        if gate == "PASS":
                            m = measured[k]
                            k += 1
                            if not m.valid:
                                print(f"  timing invalid {c.impl} {shape['tag']}: {m.error}",
                                      file=sys.stderr)
                        out.write(bl.csv_row(op=op, impl=c.impl, algo=c.algo, dtype="f32",
                                             shape=shape, cold=cold, gate=gate,
                                             max_abs=mabs, max_rel=mrel, flops=flops,
                                             nbytes=nbytes, measured=m,
                                             tel=(m.telemetry if (m is not None and m.telemetry)
                                                  else t),
                                             ctx=ctx) + "\n")
                    out.flush()
                print(f"done {op} {shape['tag']} ({time.time() - t0:.1f}s)", file=sys.stderr)
                torch.cuda.empty_cache()
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
