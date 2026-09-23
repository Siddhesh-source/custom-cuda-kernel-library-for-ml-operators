# Phase 2 — Layer normalization

**Date:** 2026-09-23
**Outcome:** 14 of 14 C++ tests passed across 26 distinct shapes, plus 4 PyTorch
comparisons. 0 failed, 0 skipped.
**Worst relative error observed anywhere: 5.96e-08.**

## Environment

Kaggle GPU kernel (`isiddhesh3011/ccml-build`, version 7).

| | |
|---|---|
| GPU | NVIDIA Tesla T4, compute capability **7.5 (Turing)** — *not* the Ada 8.9 target |
| Driver / toolkit | 580.159.04 / CUDA 12.8.93 |
| Host compiler | GCC 11.4.0 |
| CMake / generator | 3.31.10 / Ninja |
| Build | Release, `-DCCML_CUDA_ARCHITECTURES=75`, `-Wall -Wextra -Werror` |
| Warnings | 0 |

## What was built

| Component | File |
|---|---|
| Kernel and block reduction | `src/layernorm/layernorm_kernels.cuh` |
| Host wrapper, validation, launch config | `src/layernorm/layernorm.cu` |
| Double-precision CPU oracle | `tests/support/reference.cpp` |
| Tests | `tests/test_layernorm.cpp` (14 tests, 26 shapes) |
| PyTorch comparison | `python/tests/test_ops.py` (existing, now active) |

One block per row; three passes over the row — mean, variance about that mean,
then the normalized write. Reduction is warp shuffles followed by one warp
reducing the per-warp partials.

## The one design decision worth recording

**Reductions accumulate in double, not float32.** This departs from the
project-wide float32 convention in `types.h`, deliberately.

The output divides by the standard deviation, so error in the mean is magnified
by `1/stddev`. A float32 mean carries about half an ulp of error — at magnitude
10000 that is roughly 5e-4, which against a standard deviation of 0.01 is a
**5% error in every output element**. No summation order fixes it; even a
perfectly rounded float32 mean is not precise enough, so the mean must be held
in higher precision and carried in that precision through the subtraction.

This was chosen before running anything, from the arithmetic. The
`large_offset_small_variance` test exists to check it and reports
`max_rel=5.909e-08` — the same order as every other shape, so the offset costs
nothing.

**The cost is unmeasured.** fp64 runs at a fraction of fp32 rate on consumer
GPUs. Layer normalization should be memory-bound, so a few fp64 operations per
element may be hidden entirely — but that is an expectation, not a measurement,
and no timing was taken.

## Results

### CTest

```
1/5 Test #1: infrastructure ...................   Passed    0.61 sec
2/5 Test #2: support ..........................   Passed    0.53 sec
3/5 Test #3: matmul ...........................   Passed    0.01 sec
4/5 Test #4: layernorm ........................   Passed    1.03 sec
5/5 Test #5: softmax ..........................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 5
```

### `ccml_test_layernorm` — 14 passed, 0 failed, 0 skipped

```
[PASS] layernorm.small_dimensions
       8 shapes; worst max_abs=5.799e-08 max_rel=5.357e-08 normalized=0.021
       worst at: rows=8 cols=2 gamma=y beta=y
[PASS] layernorm.single_column_is_finite
       n=128 max_abs=0.000e+00 max_rel=0.000e+00 normalized=0.000 worst_index=0 nonfinite=0
[PASS] layernorm.large_dimensions
       5 shapes; worst max_abs=2.325e-07 max_rel=5.927e-08 normalized=0.002
       worst at: rows=64 cols=1024 gamma=y beta=y
[PASS] layernorm.awkward_dimensions
       6 shapes; worst max_abs=2.322e-07 max_rel=5.930e-08 normalized=0.002
       worst at: rows=33 cols=517 gamma=y beta=y
[PASS] layernorm.row_counts
       6 shapes; worst max_abs=2.384e-07 max_rel=5.957e-08 normalized=0.002
       worst at: rows=4099 cols=768 gamma=y beta=y
[PASS] layernorm.many_rows
       n=5120000 max_abs=2.384e-07 max_rel=5.958e-08 normalized=0.003 worst_index=946076 nonfinite=0
[PASS] layernorm.affine_variants
[PASS] layernorm.large_offset_small_variance
       mean=10000 stddev=0.01: n=16384 max_abs=2.329e-07 max_rel=5.909e-08 normalized=0.002 worst_index=5591 nonfinite=0
[PASS] layernorm.moderate_offset
       mean=1000 stddev=1: n=65536 max_abs=2.378e-07 max_rel=5.934e-08 normalized=0.001 worst_index=9289 nonfinite=0
[PASS] layernorm.constant_row_is_finite
[PASS] layernorm.epsilon_is_configurable
       4096 of 4096 magnitudes shrank when eps went from 1e-5 to 10
[PASS] layernorm.rejects_nonpositive_eps
[PASS] layernorm.rejects_null_and_nonpositive_dimensions
[PASS] layernorm.welford_is_reserved_not_silently_substituted

14 passed, 0 failed, 0 skipped
```

### PyTorch comparison — `torch.nn.functional.layer_norm`

```
test_layernorm_matches_torch[64-768] PASSED
test_layernorm_matches_torch[33-517] PASSED
test_layernorm_matches_torch[1-1024] PASSED
test_layernorm_large_offset_small_variance PASSED
test_layernorm_rejects_wrong_gamma_shape PASSED
```

PyTorch runs on CPU in double precision here; it is an oracle, never the thing
under test. Agreement with both it and the independent C++ oracle is what makes
the result meaningful — one reference agreeing proves only shared assumptions.

## Numerical error

| Group | Shapes | max_abs | max_rel | Fraction of tolerance used |
|---|---|---|---|---|
| Small dimensions (cols 1–33) | 8 | 5.799e-08 | 5.357e-08 | 2.1% |
| Large dimensions (cols 1024–16384) | 5 | 2.325e-07 | 5.927e-08 | 0.2% |
| Awkward dimensions | 6 | 2.322e-07 | 5.930e-08 | 0.2% |
| Row counts (1–4099) | 6 | 2.384e-07 | 5.957e-08 | 0.2% |
| 20000 × 256 (5.12M elements) | 1 | 2.384e-07 | 5.958e-08 | 0.3% |
| mean=10000, stddev=0.01 | 1 | 2.329e-07 | 5.909e-08 | 0.2% |
| mean=1000, stddev=1 | 1 | 2.378e-07 | 5.934e-08 | 0.1% |

**Worst relative error anywhere: 5.958e-08**, which is half of float32 machine
epsilon (1.19e-07) — the error of the final rounding to float32 and essentially
nothing else.

Two things this table shows that are worth naming:

- **Error does not grow with column count.** 16384 columns is no worse than 2.
  That is the double accumulation: the reduction contributes nothing measurable.
- **Error does not grow with the offset.** The mean=10000 case matches the
  zero-mean cases to three significant figures.

`normalized` is error as a fraction of the allowed tolerance; every group used
under 3% of its budget, and the small-dimension group's 2.1% is the largest
because the tolerance is tightest there (`sqrt(cols)` is smallest).

## Supported shapes

| | |
|---|---|
| Rows | any `rows >= 1`; tested 1 to 20000 |
| Columns | any `cols >= 1`; tested 1 to 16384 |
| Alignment | none required. No column count is a special case; rows wider than a block are strided |
| Normalized dimension | the last one. Collapse leading dimensions into `rows`: `[batch, seq, hidden]` normalized over hidden is `rows = batch*seq, cols = hidden` |
| gamma / beta | independently optional; either may be null |
| epsilon | any `eps > 0`; tested 1e-5 and 10.0 |
| dtype | float32 only |

Rejected with `CCML_ERROR_INVALID_ARGUMENT`: null `x` or `y`, `rows <= 0`,
`cols <= 0`, `eps <= 0`. A row count above the device grid limit is caught by
launch validation rather than silently computing part of the tensor.

## What this does NOT establish

- **Nothing about performance.** No timing was taken. The fp64 accumulation cost
  is unmeasured, and no comparison against cuDNN, PyTorch CUDA or any other
  implementation exists.
- **Nothing about the Ada target.** Turing (sm_75). Correctness should carry
  over; block sizes and occupancy do not.
- **Nothing about matmul or softmax.** Both still return
  `CCML_ERROR_NOT_IMPLEMENTED`; their suites still skip.
- **`compute-sanitizer` has not been run** on this kernel. Release build, and
  `CCML_SYNC_LAUNCH_CHECKS` was off. Both memcheck and racecheck are still
  outstanding — racecheck particularly, since the block reduction reuses one
  shared scratch buffer across two reductions and depends on the barrier between
  them.
- **Welford is not implemented.** `CCML_LAYERNORM_WELFORD` returns
  `NOT_IMPLEMENTED`; a test asserts it is not silently substituted.
- **No backward pass.**

## Reproducing this

```
python cloud/kaggle/kaggle_run.py
```
