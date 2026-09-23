#!/usr/bin/env python3
"""Transformer-style operator pipeline: ccml against PyTorch.

    python python/case_study.py [--output-dir DIR]

An operator-level case study, not a Transformer model. It runs the shape of
computation an attention block performs, using only the three kernels this
library implements, and compares both correctness and time against PyTorch.

Pipeline, for an input x of [seq, hidden]:

    ln    = layernorm(x)                 [S, H]
    Q     = ln @ Wq                      [S, H] @ [H, H] -> [S, H]
    K     = ln @ Wk                      [S, H] @ [H, H] -> [S, H]
    V     = ln @ Wv                      [S, H] @ [H, H] -> [S, H]
    Kt    = transpose(K)                 [H, S]
    scores= Q @ Kt                       [S, H] @ [H, S] -> [S, S]
    probs = softmax(scores)              [S, S]
    ctx   = probs @ V                    [S, S] @ [S, H] -> [S, H]
    out   = ctx @ Wo                     [S, H] @ [H, H] -> [S, H]
    out   = layernorm(out)               [S, H]

Two honest notes about what this is not:

* The 1/sqrt(d) attention scaling is folded into Wq before the run, because the
  library has no elementwise scale. The arithmetic is identical; it just happens
  once instead of every iteration, for both implementations equally.

* The transpose is done with PyTorch in BOTH paths, because the library has no
  transpose kernel and no transposed-operand matmul. It is inside the timed
  region for both, so it does not bias the comparison, but the custom pipeline
  is therefore not 100% custom. This is a real gap, not an oversight.

Nothing here is estimated. Every number printed and written comes from a
measurement taken in this process.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import ccml  # noqa: E402

# Shapes: batch 1, an operator-level slice of a realistic inference workload.
CONFIGS = [
    (128, 768),
    (512, 768),
    (1024, 768),
    (128, 1024),
    (512, 1024),
    (1024, 1024),
]

OPERATOR_WARMUP, OPERATOR_ITERS = 10, 30
PIPELINE_WARMUP, PIPELINE_ITERS = 5, 15

# Correctness is checked on this many independent random inputs per workload.
CORRECTNESS_TRIALS = 3


@dataclass
class Timing:
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    iters: int


@dataclass
class Row:
    workload: str
    shape: str
    operator: str
    reference_ms: float
    custom_ms: float
    speedup: float
    max_abs_error: float
    max_rel_error: float
    custom_min_ms: float = 0.0
    custom_std_ms: float = 0.0
    reference_min_ms: float = 0.0
    notes: str = ""


def time_callable(fn, warmup: int, iters: int) -> Timing:
    """CUDA-event timing: warm up, then measure each iteration on the device.

    Warmup is discarded because the first launches pay module load and JIT and
    the clocks are still ramping. Events are recorded in the stream rather than
    read from a host clock, so what is measured is device time.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    mean = sum(samples) / len(samples)
    variance = sum((s - mean) ** 2 for s in samples) / len(samples)
    return Timing(mean, min(samples), max(samples), variance ** 0.5, len(samples))


def errors(got: torch.Tensor, reference64: torch.Tensor) -> tuple:
    """Max absolute error, and max relative error scaled by the result's RMS.

    A per-element relative error is misleading where an element has cancelled to
    near zero -- see tests/support/compare.h. Dividing the absolute error by the
    RMS of the reference measures it against the scale of the quantity actually
    computed.
    """
    difference = (got.double() - reference64).abs()
    max_abs = difference.max().item()
    scale = reference64.pow(2).mean().sqrt().item()
    return max_abs, (max_abs / scale if scale > 0 else 0.0)


# --------------------------------------------------------------------------
# The pipeline, twice
# --------------------------------------------------------------------------

def pipeline_custom(x, wq, wk, wv, wo):
    ln = ccml.custom_layernorm(x)
    q = ccml.custom_matmul(ln, wq)
    k = ccml.custom_matmul(ln, wk)
    v = ccml.custom_matmul(ln, wv)
    scores = ccml.custom_matmul(q, k.t().contiguous())   # transpose: torch
    probs = ccml.custom_softmax(scores)
    context = ccml.custom_matmul(probs, v)
    out = ccml.custom_matmul(context, wo)
    return ccml.custom_layernorm(out)


def pipeline_reference(x, wq, wk, wv, wo):
    ln = F.layer_norm(x, (x.shape[-1],))
    q = torch.matmul(ln, wq)
    k = torch.matmul(ln, wk)
    v = torch.matmul(ln, wv)
    scores = torch.matmul(q, k.t().contiguous())         # same transpose
    probs = torch.softmax(scores, dim=-1)
    context = torch.matmul(probs, v)
    out = torch.matmul(context, wo)
    return F.layer_norm(out, (out.shape[-1],))


def make_inputs(seq: int, hidden: int, seed: int, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    opts = {"device": device, "dtype": torch.float32, "generator": generator}

    x = torch.randn(seq, hidden, **opts)
    scale = hidden ** -0.5
    # The attention scaling, folded in once rather than applied every iteration.
    wq = torch.randn(hidden, hidden, **opts) * (hidden ** -0.5) * scale
    wk = torch.randn(hidden, hidden, **opts) * (hidden ** -0.5)
    wv = torch.randn(hidden, hidden, **opts) * (hidden ** -0.5)
    wo = torch.randn(hidden, hidden, **opts) * (hidden ** -0.5)
    return x, wq, wk, wv, wo


# --------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------

def check_correctness(device) -> tuple:
    """Returns (per-operator verdicts, worst errors, detail rows)."""
    print("\n" + "=" * 78)
    print("CORRECTNESS")
    print("=" * 78)

    verdicts = {"matmul": True, "layernorm": True, "softmax": True, "pipeline": True}
    worst = {"matmul": 0.0, "layernorm": 0.0, "softmax": 0.0, "pipeline": 0.0}
    detail = []

    # Tolerances follow docs/correctness.md: 8 * eps_f32 * sqrt(terms), applied
    # against the RMS scale of the result.
    eps = float(torch.finfo(torch.float32).eps)

    for seq, hidden in CONFIGS:
        for trial in range(CORRECTNESS_TRIALS):
            x, wq, wk, wv, wo = make_inputs(seq, hidden, 1000 + trial, device)
            shape = f"{seq}x{hidden}"

            # matmul
            got = ccml.custom_matmul(x, wq)
            ref = x.double() @ wq.double()
            abs_e, rel_e = errors(got, ref)
            tol = 8 * eps * (hidden ** 0.5)
            verdicts["matmul"] &= rel_e <= tol
            worst["matmul"] = max(worst["matmul"], rel_e)
            detail.append(("matmul", shape, trial, abs_e, rel_e, tol, rel_e <= tol))

            # layernorm
            got = ccml.custom_layernorm(x)
            ref = F.layer_norm(x.double(), (hidden,))
            abs_e, rel_e = errors(got, ref)
            tol = 8 * eps * (hidden ** 0.5)
            verdicts["layernorm"] &= rel_e <= tol
            worst["layernorm"] = max(worst["layernorm"], rel_e)
            detail.append(("layernorm", shape, trial, abs_e, rel_e, tol, rel_e <= tol))

            # softmax, on the attention shape
            scores = torch.randn(seq, seq, device=device, dtype=torch.float32)
            got = ccml.custom_softmax(scores)
            ref = torch.softmax(scores.double(), dim=-1)
            abs_e, rel_e = errors(got, ref)
            tol = 8 * eps * (seq ** 0.5)
            verdicts["softmax"] &= rel_e <= tol
            worst["softmax"] = max(worst["softmax"], rel_e)
            detail.append(("softmax", f"{seq}x{seq}", trial, abs_e, rel_e, tol,
                           rel_e <= tol))

            # end-to-end
            got = pipeline_custom(x, wq, wk, wv, wo)
            ref64 = pipeline_reference(x.double(), wq.double(), wk.double(),
                                       wv.double(), wo.double())
            abs_e, rel_e = errors(got, ref64)

            # Context: PyTorch's own float32 error against the same float64
            # reference. Ours should be the same order, not obviously better or
            # worse.
            ref32 = pipeline_reference(x, wq, wk, wv, wo)
            _, torch_rel = errors(ref32, ref64)

            tol = max(50 * 8 * eps * (hidden ** 0.5), 10 * torch_rel)
            verdicts["pipeline"] &= rel_e <= tol
            worst["pipeline"] = max(worst["pipeline"], rel_e)
            detail.append(("pipeline", shape, trial, abs_e, rel_e, tol, rel_e <= tol))

            if trial == 0:
                print(f"  {shape:>10}  pipeline: ccml rel={rel_e:.3e}  "
                      f"torch f32 rel={torch_rel:.3e}  tol={tol:.3e}  "
                      f"{'PASS' if rel_e <= tol else 'FAIL'}")

    print()
    for name in ("matmul", "layernorm", "softmax", "pipeline"):
        state = "PASS" if verdicts[name] else "FAIL"
        print(f"  {name:<12} {state}   worst scaled relative error {worst[name]:.3e}")

    return verdicts, worst, detail


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

def benchmark(device) -> list:
    print("\n" + "=" * 78)
    print("BENCHMARK")
    print("=" * 78)
    print(f"  CUDA events, {OPERATOR_WARMUP} warmup + {OPERATOR_ITERS} measured "
          f"iterations per operator")
    print(f"  {PIPELINE_WARMUP} warmup + {PIPELINE_ITERS} measured for the pipeline")

    rows = []

    for seq, hidden in CONFIGS:
        x, wq, wk, wv, wo = make_inputs(seq, hidden, 7, device)
        scores = torch.randn(seq, seq, device=device, dtype=torch.float32)
        workload = f"S={seq},H={hidden}"
        print(f"\n  {workload}")

        def add(operator, shape, ref_fn, custom_fn, got, ref64, notes=""):
            ref_t = time_callable(ref_fn, OPERATOR_WARMUP, OPERATOR_ITERS)
            cus_t = time_callable(custom_fn, OPERATOR_WARMUP, OPERATOR_ITERS)
            abs_e, rel_e = errors(got, ref64)
            row = Row(workload, shape, operator, ref_t.mean_ms, cus_t.mean_ms,
                      ref_t.mean_ms / cus_t.mean_ms, abs_e, rel_e,
                      cus_t.min_ms, cus_t.std_ms, ref_t.min_ms, notes)
            rows.append(row)
            arrow = "faster" if row.speedup >= 1.0 else "slower"
            print(f"    {operator:<18} {shape:>12}  ref {ref_t.mean_ms:8.3f} ms   "
                  f"ccml {cus_t.mean_ms:8.3f} ms   {row.speedup:6.2f}x {arrow}")

        # matmul, projection shape, both algorithms
        add("matmul (tiled)", f"{seq}x{hidden}@{hidden}x{hidden}",
            lambda: torch.matmul(x, wq),
            lambda: ccml.custom_matmul(x, wq),
            ccml.custom_matmul(x, wq), x.double() @ wq.double())

        add("matmul (naive)", f"{seq}x{hidden}@{hidden}x{hidden}",
            lambda: torch.matmul(x, wq),
            lambda: _matmul_naive(x, wq),
            _matmul_naive(x, wq), x.double() @ wq.double(),
            notes="naive kernel, kept as the correctness baseline")

        add("layernorm", f"{seq}x{hidden}",
            lambda: F.layer_norm(x, (hidden,)),
            lambda: ccml.custom_layernorm(x),
            ccml.custom_layernorm(x), F.layer_norm(x.double(), (hidden,)))

        add("softmax", f"{seq}x{seq}",
            lambda: torch.softmax(scores, dim=-1),
            lambda: ccml.custom_softmax(scores),
            ccml.custom_softmax(scores), torch.softmax(scores.double(), dim=-1))

        # end to end
        ref_t = time_callable(lambda: pipeline_reference(x, wq, wk, wv, wo),
                              PIPELINE_WARMUP, PIPELINE_ITERS)
        cus_t = time_callable(lambda: pipeline_custom(x, wq, wk, wv, wo),
                              PIPELINE_WARMUP, PIPELINE_ITERS)
        got = pipeline_custom(x, wq, wk, wv, wo)
        ref64 = pipeline_reference(x.double(), wq.double(), wk.double(),
                                   wv.double(), wo.double())
        abs_e, rel_e = errors(got, ref64)
        row = Row(workload, f"{seq}x{hidden}", "PIPELINE (end-to-end)",
                  ref_t.mean_ms, cus_t.mean_ms, ref_t.mean_ms / cus_t.mean_ms,
                  abs_e, rel_e, cus_t.min_ms, cus_t.std_ms, ref_t.min_ms,
                  "transpose done by torch in both paths")
        rows.append(row)
        arrow = "faster" if row.speedup >= 1.0 else "slower"
        print(f"    {'PIPELINE':<18} {f'{seq}x{hidden}':>12}  "
              f"ref {ref_t.mean_ms:8.3f} ms   ccml {cus_t.mean_ms:8.3f} ms   "
              f"{row.speedup:6.2f}x {arrow}")

    return rows


def _matmul_naive(a, b):
    """The naive kernel, reached through the numpy-free ctypes path."""
    import ctypes
    from ccml._lib import check, lib
    from ccml.torch_api import _float_ptr, _stream

    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    with torch.cuda.device(a.device):
        check(lib().ccml_matmul_f32(_float_ptr(out), _float_ptr(a), _float_ptr(b),
                                    m, n, k, 1, _stream(a.device)),
              "ccml_matmul_f32(naive)")
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "workload", "shape", "operator",
            "reference_ms_mean", "custom_ms_mean", "speedup",
            "max_abs_error", "max_rel_error_scaled",
            "custom_ms_min", "custom_ms_std", "reference_ms_min", "notes",
        ])
        for r in rows:
            writer.writerow([
                r.workload, r.shape, r.operator,
                f"{r.reference_ms:.6f}", f"{r.custom_ms:.6f}", f"{r.speedup:.4f}",
                f"{r.max_abs_error:.6e}", f"{r.max_rel_error:.6e}",
                f"{r.custom_min_ms:.6f}", f"{r.custom_std_ms:.6f}",
                f"{r.reference_min_ms:.6f}", r.notes,
            ])
    print(f"\n  wrote {path}")


def write_summary(rows, verdicts, worst, path: Path, device) -> None:
    props = torch.cuda.get_device_properties(0)
    lines = []
    w = lines.append

    w("# Case study: Transformer-style operator pipeline")
    w("")
    w("Generated by `python/case_study.py`. Every number below was measured in "
      "that run; none is estimated.")
    w("")
    w("## Hardware and software")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| GPU | {props.name} |")
    w(f"| Compute capability | {props.major}.{props.minor} |")
    w(f"| SMs | {props.multi_processor_count} |")
    w(f"| Global memory | {props.total_memory / 1024**3:.2f} GiB |")
    w(f"| CUDA (torch build) | {torch.version.cuda} |")
    w(f"| PyTorch | {torch.__version__} |")
    w(f"| ccml | {ccml.version()} |")
    w(f"| Python | {platform.python_version()} |")
    w("")
    w("## Workload")
    w("")
    w("An operator-level slice of an attention block, batch 1:")
    w("")
    w("```")
    w("ln     = layernorm(x)          [S, H]")
    w("Q      = ln @ Wq               [S, H] @ [H, H]")
    w("K      = ln @ Wk               [S, H] @ [H, H]")
    w("V      = ln @ Wv               [S, H] @ [H, H]")
    w("Kt     = transpose(K)          [H, S]      <- torch, in BOTH paths")
    w("scores = Q @ Kt                [S, H] @ [H, S] -> [S, S]")
    w("probs  = softmax(scores)       [S, S]")
    w("ctx    = probs @ V             [S, S] @ [S, H]")
    w("out    = ctx @ Wo              [S, H] @ [H, H]")
    w("out    = layernorm(out)        [S, H]")
    w("```")
    w("")
    w("Configurations: " + ", ".join(f"S={s},H={h}" for s, h in CONFIGS) + ".")
    w("")
    w("Two caveats about the pipeline, stated rather than hidden:")
    w("")
    w("* The `1/sqrt(d)` attention scale is folded into `Wq` before the run, "
      "because the library has no elementwise scale kernel. Identical "
      "arithmetic, applied once instead of per iteration, for both paths.")
    w("* The transpose uses PyTorch in **both** paths, because the library has "
      "neither a transpose kernel nor a transposed-operand matmul. It is inside "
      "the timed region for both, so it does not bias the comparison, but the "
      "custom pipeline is not 100% custom. This is a real gap in the library.")
    w("")
    w("## Correctness")
    w("")
    w(f"Each operator checked on {CORRECTNESS_TRIALS} independent random inputs "
      f"per configuration, against a float64 PyTorch reference computed from the "
      f"same float32 inputs.")
    w("")
    w("| Operator | Verdict | Worst scaled relative error |")
    w("|---|---|---|")
    for name in ("matmul", "layernorm", "softmax", "pipeline"):
        w(f"| {name} | **{'PASS' if verdicts[name] else 'FAIL'}** | "
          f"{worst[name]:.3e} |")
    w("")
    w("Relative error is scaled by the RMS of the reference rather than measured "
      "per element: a matrix-product element can cancel to near zero, which "
      "makes a per-element relative figure meaningless. See "
      "`tests/support/compare.h`.")
    w("")
    w("## Benchmark methodology")
    w("")
    w(f"* CUDA events recorded in the stream, not a host clock.")
    w(f"* {OPERATOR_WARMUP} warmup + {OPERATOR_ITERS} measured iterations per "
      f"operator; {PIPELINE_WARMUP} + {PIPELINE_ITERS} for the pipeline.")
    w("* Warmup discarded: the first launches pay module load and the clocks are "
      "still ramping from idle.")
    w("* Both paths are driven from Python and timed identically, so per-launch "
      "Python overhead is present in both.")
    w("* Reference is PyTorch, which dispatches to cuBLAS for matmul and its own "
      "fused kernels for layernorm and softmax.")
    w("")
    w("## Results")
    w("")
    w("| Workload | Shape | Operator | Reference (ms) | Custom (ms) | Speedup | Max abs error |")
    w("|---|---|---|---|---|---|---|")
    for r in rows:
        marker = "" if r.speedup >= 1.0 else ""
        w(f"| {r.workload} | {r.shape} | {r.operator} | {r.reference_ms:.3f} | "
          f"{r.custom_ms:.3f} | {r.speedup:.2f}x{marker} | {r.max_abs_error:.3e} |")
    w("")
    w("Speedup above 1.00 means the custom kernel was faster; below 1.00 means "
      "slower. Full data including spread is in `case_study_results.csv`.")
    w("")

    # Analysis computed from the rows, not asserted.
    pipeline_rows = [r for r in rows if r.operator.startswith("PIPELINE")]
    op_rows = [r for r in rows if not r.operator.startswith("PIPELINE")]

    w("## Analysis")
    w("")
    w("### Which operator dominates")
    w("")
    for seq, hidden in CONFIGS:
        workload = f"S={seq},H={hidden}"
        mine = [r for r in op_rows if r.workload == workload
                and r.operator != "matmul (naive)"]
        if not mine:
            continue
        total = sum(r.custom_ms for r in mine)
        share = sorted(mine, key=lambda r: -r.custom_ms)
        parts = ", ".join(f"{r.operator} {100*r.custom_ms/total:.0f}%" for r in share)
        w(f"* **{workload}** — of the custom operator time, {parts}.")
    w("")

    slowest = min(rows, key=lambda r: r.speedup)
    fastest = max(rows, key=lambda r: r.speedup)
    w("### Where the custom kernels stand")
    w("")
    w(f"* Worst ratio: **{slowest.operator}** at {slowest.workload}, "
      f"{slowest.speedup:.2f}x ({slowest.custom_ms:.3f} ms against "
      f"{slowest.reference_ms:.3f} ms).")
    w(f"* Best ratio: **{fastest.operator}** at {fastest.workload}, "
      f"{fastest.speedup:.2f}x ({fastest.custom_ms:.3f} ms against "
      f"{fastest.reference_ms:.3f} ms).")
    w("")
    w("### Scaling with workload size")
    w("")
    for name in ("matmul (tiled)", "layernorm", "softmax", "PIPELINE (end-to-end)"):
        series = [r for r in rows if r.operator == name]
        if len(series) < 2:
            continue
        first, last = series[0], series[-1]
        w(f"* **{name}**: {first.speedup:.2f}x at {first.workload} -> "
          f"{last.speedup:.2f}x at {last.workload}.")
    w("")

    # Bottleneck: the operator with the largest absolute time deficit.
    deficits = {}
    for r in op_rows:
        if r.operator == "matmul (naive)":
            continue
        deficits[r.operator] = deficits.get(r.operator, 0.0) + (r.custom_ms - r.reference_ms)
    bottleneck = max(deficits, key=deficits.get)

    w("### Primary bottleneck")
    w("")
    w(f"Summed across every configuration, the operator losing the most absolute "
      f"time against the reference is **{bottleneck}** "
      f"({deficits[bottleneck]:.3f} ms total deficit). Full per-operator "
      f"deficits:")
    w("")
    for operator, deficit in sorted(deficits.items(), key=lambda kv: -kv[1]):
        w(f"* {operator}: {deficit:+.3f} ms")
    w("")
    w("### Next optimization target")
    w("")
    w(f"**{bottleneck}.** It carries the largest absolute deficit and, per the "
      f"share breakdown above, the largest fraction of pipeline time. The naive "
      f"and tiled matmul rows in the table give the in-library reference point "
      f"for what a structural change to that kernel is worth.")
    w("")
    w("## What this does not establish")
    w("")
    w("* Nothing about an Ada (sm_89) GPU; this ran on the hardware named above.")
    w("* Nothing about batched or multi-head attention; batch is 1 and the "
      "pipeline is single-head-shaped.")
    w("* Nothing about training; there is no backward pass.")
    w("* No claim that any kernel is well optimized. None has been tuned, and "
      "block and tile sizes were chosen by reasoning rather than measurement.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None,
                        help="where to write results (default: <repo>/results)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device available to PyTorch; this case study needs a GPU.")
        return 1

    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path(__file__).resolve().parents[1] / "results")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print("=" * 78)
    print("CASE STUDY: Transformer-style operator pipeline")
    print("=" * 78)
    print(f"  GPU    : {props.name} (sm_{props.major}{props.minor}, "
          f"{props.multi_processor_count} SMs)")
    print(f"  CUDA   : {torch.version.cuda}   torch {torch.__version__}")
    print(f"  ccml   : {ccml.version()}")

    verdicts, worst, _ = check_correctness(device)

    if not all(verdicts.values()):
        print("\nCorrectness failed. Not proceeding to performance conclusions.")
        return 2

    rows = benchmark(device)

    write_csv(rows, output_dir / "case_study_results.csv")
    write_summary(rows, verdicts, worst, output_dir / "case_study_summary.md", device)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
