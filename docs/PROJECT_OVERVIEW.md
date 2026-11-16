# Project overview

**ccml**: a custom CUDA kernel library for common ML operators. It contains
hand-tuned matmul, layernorm and softmax kernels for production model shapes,
benchmarked against cuBLAS, cuDNN, PyTorch and torch.compile.

The specification is [`reference.md`](reference.md) (Project 1, CUDA & GPU
Research Track). It is binding. The plan is in [ROADMAP.md](ROADMAP.md), and the
additions beyond the specification are in [enhancements.md](enhancements.md).

## Research question

> On which target shapes and dtypes can hand-written kernels beat cuBLAS /
> PyTorch, and which optimizations produce the gain?

**Expected outcome:** a kernel library that beats vendor baselines on target
shapes by a measurable margin, or a documented explanation of why it does not.

## Pipeline [R§2]

```
target-shape list → reference impls (PyTorch fp64) → naive kernels
→ optimized kernels (one optimization at a time, NCU evidence each)
→ correctness tests (FP32/FP16/BF16, compute-sanitizer)
→ benchmark harness (CUDA events, median of 100+, TFLOPS/GB/s/%peak, CSV)
→ Nsight Compute analysis → torch.utils.cpp_extension
→ model integration → shape-by-shape comparison table + roofline
```

## Operators and variant ladders

| Operator | Required variants [R§3] | Enhancements |
|---|---|---|
| matmul | naive → smem tiled → register-blocked → float4 → double-buffered → Tensor Core (WMMA, mma.sync) | swizzle, split-K/stream-K, fused epilogue |
| layernorm | block/warp-per-row, two-pass + Welford, warp shuffle, vectorized, residual+bias fusion | persistent kernel |
| softmax | stable, online, warp-per-row (short) / block-per-row (long), fused scale/mask | persistent kernel |

Each variant is a value of the operator's `algo` enum, and `AUTO` picks per
shape from a measured tuning table. Dtypes are FP32, FP16 and BF16 (BF16 needs
sm_80+), all accumulating in FP32.

## Baselines [R§4]

Naive CUDA · PyTorch eager · cuBLAS / cuDNN · torch.compile (Inductor) · our
best kernel per shape. Triton is added as an extra baseline.

## Metrics [R§6]

Latency (µs) · TFLOPS and % of peak (matmul) · GB/s and % of peak bandwidth
(row-wise ops) · speedup vs cuBLAS/PyTorch · max abs and rel error · NCU
achieved occupancy, memory throughput and warp stall reasons.

## Hardware

| | Arch | Role |
|---|---|---|
| **Kaggle T4** | Turing sm_75, 15 GB | **All GPU work**: build, tests, sanitizer, benchmarks, FP16 Tensor Cores, extension, model. The benchmark target |
| Local GPU | sm_80+ | Only the items in [local.md](local.md) |

Results go in `results/sm_75/`. Numbers from different architectures are never mixed.

## Repository layout [R§7]

| Path | Contents |
|---|---|
| `include/ccml/` | Public C ABI: `ccml_<op>_<f32\|f16\|bf16>(out, in..., dims..., params..., algo, stream)` |
| `src/matmul/ src/layernorm/ src/softmax/` | `<op>.cu` host wrapper and dispatch; `<op>_kernels.cuh` device code only |
| `src/common/` | `CUDA_CHECK`, launch validation, reductions |
| `python/` | `torch.utils.cpp_extension` extension and bindings; ctypes C-ABI path |
| `bench/` | Harness, `shapes.json`, vendor baselines, CSV writer |
| `tests/` | Correctness against references; no timing |
| `profiles/` | Small `.ncu-rep` / `.nsys-rep` files, committed |
| `results/sm_75/` | Raw CSVs, peaks, sanitizer logs, plots |
| `scripts/` | Environment setup, sanitizer, profiling, sweeps, roofline, remote runs |
| `docs/` | Specification, roadmap, design, per-phase results, optimization log, report |
| `cloud/` | Kaggle runner: every GPU job |

## Non-negotiables [R§11, R Method]

1. Correctness first. A kernel that fails its tests gets no speed number.
2. Every CUDA call goes through `CUDA_CHECK`, and every launch is followed by `cudaGetLastError`.
3. Timing uses CUDA events with warmup, and reports the median and spread of 100 or more iterations.
4. Same GPU, same clocks, same dtype and same shapes for every comparison.
5. Report % of hardware peak, not only speedup.
6. Report losses honestly. Beating cuBLAS on large square matmul is not expected.

## Current state (2026-11-16)

FP32 only. The repository has: naive and tiled matmul; two-pass block-per-row
layernorm; three-pass block-per-row softmax; a ctypes Python binding; and a
case study on a T4 in which every operator is 2–11× slower than PyTorch, with
matmul at 8.6% of FP32 peak. The full gap list is in
[ROADMAP.md §1](ROADMAP.md#1-compliance-audit--where-the-repository-stands-against-the-spec).

## Deliverables [R§10]

Kernel library and build system · benchmark harness and raw CSVs · profiling
reports and roofline plots · technical report (`docs/report.md`) · demo
notebook (`notebooks/demo.ipynb`).
