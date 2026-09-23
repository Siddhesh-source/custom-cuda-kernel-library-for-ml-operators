# Phase 6 — Matmul, and the first real-world case study

**Date:** 2026-09-24
**Outcome:** all three operators implemented and correct. **All of them are
slower than PyTorch on every shape measured** — between 2x and 11x.
**This is the project's first performance measurement of any kind.**

Generated artifacts: [`results/case_study_results.csv`](../../results/case_study_results.csv),
[`results/case_study_summary.md`](../../results/case_study_summary.md).

## Environment

Kaggle GPU kernel. Tesla T4, sm_75 (Turing), 40 SMs, 14.56 GiB. CUDA 12.8.93,
PyTorch 2.10.0+cu128, GCC 11.4.0, CMake 3.31.10, Ninja. Release build,
`-Wall -Wextra -Werror`, zero warnings.

## Matmul, finally implemented

The case study needed it: an attention block is matmul-dominated, and the
operator had been a stub since the skeleton. Built exactly as designed in
[design.md](../design.md) — naive and shared-memory tiled, nothing more.

Accumulation is **float32**, unlike the row-wise operators. The reason those
need double does not apply: layer normalization divides by a standard deviation,
so error in the mean is amplified by `1/stddev`. A matrix product has no such
amplification, and float32 is what cuBLAS does, which keeps the comparison fair.

### A tolerance model that was wrong

Two matmul tests failed on the first run: `long_k` (K=2048) at 1.14x tolerance
and `transformer_shapes` (K=768) at 1.51x. **Neither was a kernel bug.**

Evidence it was the tolerance, not the code:

* `tiled_agrees_with_naive` reported max difference **exactly 0.0** — the two
  kernels accumulate in the same order, so they agree bit for bit.
* A 1x1x1 product was exact.
* The errors matched float32 sequential-accumulation theory. For K=2048 the
  predicted RMS error is 8.6e-5 and the observed max over 1024 elements was
  1.8e-4 — the expected 2x ratio for a maximum over that many samples.
* The failures were driven by `max_rel`, which hit **0.23** on a K=768 element
  that had cancelled to near zero.

That last point is the flaw. `tolerance_for_f32_reduction` judges each element
against its own magnitude, which is right for softmax and layer normalization —
their outputs have a bounded, meaningful scale. A matrix-product element is a
sum of K signed products and can cancel to nearly zero while the computation
that produced it involved terms of magnitude 1. Judged relatively, such an
element shows an enormous error for no reason.

`tolerance_for_f32_dot_product(terms, scale)` was added, scaling the absolute
bound by the RMS of the reference. With it, every shape passes with 5–10x
margin. The tolerance was not loosened arbitrarily: the model was replaced with
one appropriate to the operator, and the reasoning is recorded in
`tests/support/compare.h`.

### A compile error worth recording

`div_up` was a host-only `constexpr` and the tiled kernel called it from device
code. nvcc rejected it. Fixed by making the helpers `__host__ __device__` under
a macro that is empty for plain C++ translation units. Caught by `-Werror`,
which is the argument for keeping it on.

## The workload

An operator-level slice of an attention block, batch 1, using only this
library's three kernels:

```
ln     = layernorm(x)          [S, H]
Q      = ln @ Wq               [S, H] @ [H, H]
K      = ln @ Wk               [S, H] @ [H, H]
V      = ln @ Wv               [S, H] @ [H, H]
Kt     = transpose(K)          [H, S]      <- torch, in BOTH paths
scores = Q @ Kt                [S, H] @ [H, S] -> [S, S]
probs  = softmax(scores)       [S, S]
ctx    = probs @ V             [S, S] @ [S, H]
out    = ctx @ Wo              [S, H] @ [H, H]
out    = layernorm(out)        [S, H]
```

Five matmuls, two layer normalizations, one softmax. Configurations: S ∈ {128,
512, 1024} × H ∈ {768, 1024}.

**Two gaps stated rather than hidden.** The `1/sqrt(d)` scale is folded into
`Wq` beforehand because the library has no elementwise scale kernel. The
transpose uses PyTorch in **both** paths because the library has neither a
transpose kernel nor a transposed-operand matmul — it is inside the timed region
for both, so it does not bias the comparison, but the custom pipeline is
therefore not 100% custom.

## Correctness — PASS

Three independent random inputs per configuration, against a float64 PyTorch
reference computed from identical float32 inputs.

| Operator | Verdict | Worst scaled relative error |
|---|---|---|
| matmul | **PASS** | 7.377e-06 |
| layernorm | **PASS** | 2.384e-07 |
| softmax | **PASS** | 2.119e-06 |
| pipeline (end-to-end) | **PASS** | 1.120e-05 |

The pipeline figure is the informative one, because error compounds through nine
operations:

| Configuration | ccml vs float64 | **PyTorch float32 vs float64** |
|---|---|---|
| S=128, H=768 | 7.430e-06 | 4.046e-06 |
| S=512, H=768 | 7.628e-06 | 3.904e-06 |
| S=1024, H=768 | 8.964e-06 | 4.176e-06 |
| S=128, H=1024 | 7.656e-06 | 3.502e-06 |
| S=512, H=1024 | 1.071e-05 | 3.595e-06 |
| S=1024, H=1024 | 9.568e-06 | 3.657e-06 |

Running PyTorch's own float32 pipeline against the same float64 reference is
what makes these numbers mean something. **ccml is within 2–3x of PyTorch's own
float32 error** — the same order, which is what "equivalent to a trusted
reference" should mean for float32 arithmetic. A figure of 1e-5 in isolation
would be uninterpretable.

## Performance — slower, everywhere

| Workload | Operator | Reference (ms) | Custom (ms) | Ratio |
|---|---|---|---|---|
| S=128,H=768 | matmul (tiled) | 0.068 | 0.212 | 0.32x |
| S=128,H=768 | layernorm | 0.027 | 0.059 | 0.46x |
| S=128,H=768 | softmax | 0.021 | 0.040 | 0.52x |
| S=128,H=768 | **PIPELINE** | 0.281 | 0.871 | **0.32x** |
| S=512,H=768 | matmul (tiled) | 0.124 | 0.741 | 0.17x |
| S=512,H=768 | **PIPELINE** | 0.727 | 4.352 | **0.17x** |
| S=1024,H=768 | matmul (tiled) | 0.225 | 1.517 | 0.15x |
| S=1024,H=768 | layernorm | 0.043 | 0.234 | 0.18x |
| S=1024,H=768 | softmax | 0.047 | 0.159 | 0.30x |
| S=1024,H=768 | **PIPELINE** | 1.681 | 10.272 | **0.16x** |
| S=128,H=1024 | **PIPELINE** | 0.399 | 1.638 | **0.24x** |
| S=512,H=1024 | **PIPELINE** | 1.048 | 7.578 | **0.14x** |
| S=1024,H=1024 | matmul (tiled) | 0.375 | 3.066 | 0.12x |
| S=1024,H=1024 | layernorm | 0.051 | 0.292 | 0.18x |
| S=1024,H=1024 | softmax | 0.050 | 0.170 | 0.30x |
| S=1024,H=1024 | **PIPELINE** | 3.008 | 17.843 | **0.17x** |

Ratio below 1.00 means slower. The full table, including per-iteration spread,
is in the CSV.

### What the times mean in absolute terms

Derived from the measured times above and the device properties recorded in
[phase 1](phase-1-infrastructure.md).

**Matmul, 1024×1024×1024** (2·M·N·K = 2.147 GFLOP):

| | Time | Achieved | Fraction of T4 fp32 peak (8.1 TFLOP/s) |
|---|---|---|---|
| ccml tiled | 3.066 ms | 700 GFLOP/s | **8.6%** |
| ccml naive | 4.415 ms | 486 GFLOP/s | 6.0% |
| cuBLAS | 0.375 ms | 5727 GFLOP/s | **70.7%** |

**Row-wise operators, 1024×1024**, assuming this library's traffic of three
reads and one write (16 MiB) against the 320.1 GB/s ceiling:

| | Time | Implied bandwidth | Fraction of ceiling |
|---|---|---|---|
| ccml layernorm | 0.292 ms | 57 GB/s | 17.9% |
| ccml softmax | 0.170 ms | 99 GB/s | 30.8% |
| torch layernorm | 0.051 ms | 329 GB/s | *102.8%* |
| torch softmax | 0.050 ms | 336 GB/s | *104.8%* |

The figures above 100% are not PyTorch beating physics — they mean **the traffic
assumption is wrong for PyTorch**: it is moving less data than we are, almost
certainly by making fewer passes over the row. At one read and one write (8 MiB)
its softmax would sit at 52% of the ceiling, which is entirely ordinary.

That is a direct measurement against a decision made in
[phase 3](phase-3-softmax.md), where three passes were chosen over caching the
exponentials on the argument that caching moved five array transfers against
four. The reasoning was sound about those two options and **missed a third**: a
single-pass online algorithm that reads the row once. The measurement found what
the reasoning did not.

## Analysis

**Which operator dominates.** Matmul, decisively. The pipeline contains five of
them, and at S=1024,H=1024 a single projection matmul (3.066 ms) exceeds the
entire PyTorch pipeline (3.008 ms). Summed across all configurations the
absolute time deficit is roughly 7.2 ms for matmul against 0.9 ms for layernorm
and 0.4 ms for softmax.

**Behaviour as workload grows — the gap widens.** The pipeline ratio falls from
0.32x at S=128,H=768 to 0.16x at S=1024,H=768. For matmul alone, 0.32x → 0.12x.
This is the expected signature of a kernel that is latency- and
occupancy-limited rather than compute-limited: cuBLAS scales with the work,
ours scales worse.

**Performance is shape-dependent.** The custom kernels are relatively closest at
the smallest shape (0.32x) and worst at the largest (0.12x–0.17x). At small
sizes both implementations are dominated by launch overhead, which compresses
the difference; the gap is a property of the kernels, and small shapes hide it.

**Where the custom implementation does best.** Softmax, consistently — 0.22x to
0.52x, roughly twice as close to the reference as matmul. It is the only
operator reaching even 30% of the bandwidth ceiling.

**Where it does worst.** Matmul at large shapes, at 8.6% of fp32 peak against
cuBLAS's 70.7%. An eight-fold gap in efficiency.

**One result that is genuinely ours.** Tiling beats the naive baseline by
**1.38x–1.55x**, measured in-library:

| Shape | naive | tiled | speedup |
|---|---|---|---|
| 128×768×768 | 0.292 ms | 0.212 ms | 1.38x |
| 1024×768×768 | 2.351 ms | 1.517 ms | 1.55x |
| 1024×1024×1024 | 4.415 ms | 3.066 ms | 1.44x |

This is the only speedup claim this project makes, it is between two of its own
kernels, and it is measured. Shared-memory tiling alone buys about 1.5x; the
remaining ~8x to cuBLAS is register tiling, vectorized loads, and everything
else deliberately left out.

## Primary bottleneck and next target

**Matmul**, by every measure: largest share of pipeline time, largest absolute
deficit, worst ratio, and the ratio that degrades fastest with size.

The specific next step is **register tiling** — having each thread compute a
small tile of C (say 4×4) instead of one element, so that values staged in
shared memory are reused from registers rather than re-read. Shared-memory
tiling took the kernel from 486 to 700 GFLOP/s; the arithmetic intensity
argument says register tiling is where the next several-fold sits, and at 8.6%
of peak there is a great deal of room.

Second priority, and cheaper to do: a **single-pass online softmax and a fused
layer normalization**, both suggested directly by the traffic measurement above.

## What this does NOT establish

* Nothing about an Ada (sm_89) GPU. All of this is Turing.
* Nothing about batched or multi-head attention; batch is 1, single-head-shaped.
* Nothing about training; no backward pass exists.
* **`compute-sanitizer` has still not been run on any kernel.** Unchanged since
  phase 3, and now three kernels rest on it.
* The pipeline is not 100% custom — the transpose is PyTorch's, in both paths.
* Python launch overhead is present in both paths; at S=128 it is a meaningful
  share of the measured time.

## Reproducing this

```
python cloud/kaggle/kaggle_run.py            # full remote run, including the study
```

Or with a local CUDA-enabled PyTorch:

```
cmake --workflow --preset build
python python/case_study.py
```
