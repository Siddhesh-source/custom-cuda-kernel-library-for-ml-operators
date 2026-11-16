# 40% milestone status

> **Superseded in part (2026-11-16):** `docs/reference.md` is the binding specification. Any scope exclusion below that conflicts with it (Tensor Cores/WMMA, double buffering, vectorized loads, fusion, FP16/BF16, `torch.utils.cpp_extension`) no longer applies. See [ROADMAP.md](ROADMAP.md).

Last updated: 2026-09-24.

## Honest summary

The repository has a clean, reusable library structure with a stable public API,
a test harness, a benchmark harness, a Python binding and documentation. The
common CUDA infrastructure and **all three operators are implemented and
validated on real GPU hardware**, and a Transformer-style case study has
produced the project's first performance measurements.

**Every operator is slower than PyTorch**, by 2x to 11x. That is the honest
experimental finding; see [results/phase-6-case-study.md](results/phase-6-case-study.md).

Everything below was measured on a Kaggle **Tesla T4** (sm_75, Turing), Release
build, CUDA 12.8, GCC 11.4, `-Wall -Wextra -Werror` with zero warnings. Full
per-phase records are in [results/](results/README.md).

| Suite | Passed | Failed | Skipped |
|---|---|---|---|
| `ccml_test_infrastructure` | 16 | 0 | 0 |
| `ccml_test_harness` | 19 | 0 | 0 |
| `ccml_test_layernorm` | **14** | 0 | 0 |
| `ccml_test_matmul` | **11** | 0 | 0 |
| `ccml_test_softmax` | **13** | 0 | 0 |
| `pytest` (Python binding) | **35** | 0 | 0 |

No skips remain in the C++ suites.

## Done

**Structure and build**

- [x] Clean, reusable library structure; kernels separated from host wrappers
- [x] CMake build with install and export rules (`find_package(ccml)`), verified
- [x] One command builds everything; one command runs every test
- [x] No duplicated CUDA utility code; shared helpers in `src/common/`
- [x] Compiled public-API usage example (`examples/usage.cpp`)
- [x] Public C ABI for all three operators, with per-operator algorithm selection
- [x] Compiles clean with warnings-as-errors (GCC/Linux, sm_75 and sm_89)
- [x] Remote build pipelines: Kaggle GPU (`cloud/kaggle/`) and Modal (`cloud/modal_app.py`)

**Infrastructure** — [phase 1](results/phase-1-infrastructure.md)

- [x] Error handling that preserves CUDA's own message through a flat C ABI
- [x] Device query; properties measured and recorded for the T4
- [x] Stream-ordered copies, memset, per-stream synchronisation
- [x] Launch configuration validation against real device limits
- [x] RAII wrappers for device buffers, streams and events
- [x] CUDA-event timing with warmup and spread reporting

**Testing**

- [x] Per-operator test executables, independently runnable
- [x] Harness distinguishing PASS / FAIL / SKIP, where a skip can never read as a pass
- [x] Deterministic, cross-platform-reproducible input generation — tested
- [x] Element-wise tolerance machinery — tested on both sides of its boundary
- [x] Python binding via ctypes, requiring no compiler
- [x] **PyTorch interface** (`ccml.custom_*`): operates on CUDA tensors with no
      copy, on PyTorch's current stream. Validated against torch in float64;
      worst absolute error 2.90e-08 (softmax), 2.38e-07 (layernorm)

**Operators** — [phase 2](results/phase-2-layernorm.md), [phase 3](results/phase-3-softmax.md), [phase 6](results/phase-6-case-study.md)

- [x] **LayerNorm kernel**: two-pass, block-per-row, warp-shuffle reduction.
      Validated over 26 shapes (cols 1–16384, rows 1–20000) against two
      independent references — a double-precision CPU oracle and PyTorch.
      Worst relative error **5.96e-08**, half of float32 machine epsilon.
- [x] **Softmax kernel**: block-per-row, three-pass, maximum subtracted for
      numerical stability. Validated over 28 shapes (cols 1-16384, rows
      1-20000) against two independent references. Worst absolute error
      **4.40e-08**; verified finite, in [0, 1], and rows summing to 1 on every
      case.
- [x] **Matmul kernels**: naive baseline and shared-memory tiled. Validated over
      11 shapes (K 1-2048) against a double-precision CPU oracle and PyTorch.
      Tiling beats naive by a measured **1.38x-1.55x**.
- [x] Double-precision CPU references for all three operators
- [x] Shared block reduction (`src/common/reduce.cuh`) used by both row-wise operators
- [x] Measured error magnitudes recorded in `correctness.md`

**Case study** — [phase 6](results/phase-6-case-study.md)

- [x] Transformer-style operator pipeline (attention block shape), batch 1,
      S x H over {128,512,1024} x {768,1024}
- [x] Correctness against PyTorch float64: matmul, layernorm, softmax and the
      end-to-end pipeline all **PASS**, within 2-3x of PyTorch's own float32 error
- [x] **First performance measurements**, CUDA events with warmup and spread,
      written to `results/case_study_results.csv` and `results/case_study_summary.md`

## Not done

- [ ] **Register tiling for matmul** — the measured bottleneck, at 8.6% of fp32
      peak against cuBLAS's 70.7%
- [ ] Single-pass online softmax and fused layernorm — the traffic measurement
      shows PyTorch making fewer passes than we do
- [ ] `compute-sanitizer` memcheck and racecheck, on any kernel
- [ ] Tolerance policy validated across all three operators
- [ ] Device properties measured on an **Ada (sm_89)** device — the actual target
- [ ] Windows SDK installed; local build verified under MSVC
- [ ] Sponsor target shapes obtained; block and tile sizes revisited
- [ ] First commit pushed

## Explicitly out of scope at this milestone

Autotuning; Tensor Core / WMMA; register and warp-level matmul tiling; double
buffering; vectorized loads; fused operators; mixed precision; multi-GPU;
backward passes; a `torch.utils.cpp_extension` binding.

(A cuBLAS/cuDNN comparison *has* now happened, via PyTorch, in the case study.)

## Claims this project is NOT making

- That any kernel is fast. Measurement says the opposite: every operator is
  2x-11x slower than PyTorch.
- That any kernel is faster than cuBLAS, cuDNN, PyTorch, or anything else.
- That the block or tile sizes in `src/` are tuned. None was chosen by
  measurement.
- That any operator is correct on Ada. All are validated on Turing; correctness
  should carry over, but none has been run there.
- That any kernel is free of races or out-of-bounds access. `compute-sanitizer`
  has not been run.
- That the code builds on Windows. Only the Linux/GCC build is verified.

Each becomes claimable only with a committed, rerunnable script and its output
in the repository.
