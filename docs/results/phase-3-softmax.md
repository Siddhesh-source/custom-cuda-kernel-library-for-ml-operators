# Phase 3 — Softmax

**Date:** 2026-09-24
**Outcome:** 13 of 13 C++ tests passed across 28 distinct shapes, plus 5 PyTorch
comparisons. 0 failed, 0 skipped.
**Worst absolute error: 4.398e-08. Worst relative error on normal inputs:
5.631e-07.**

## Environment

Kaggle GPU kernel (`isiddhesh3011/ccml-build`, version 8).

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
| Shared block reduction | `src/common/reduce.cuh` (new) |
| Kernel | `src/softmax/softmax_kernels.cuh` |
| Host wrapper, validation, launch config | `src/softmax/softmax.cu` |
| Double-precision CPU oracle | `tests/support/reference.cpp` |
| Tests | `tests/test_softmax.cpp` (13 tests, 28 shapes) |
| PyTorch comparison | `python/tests/test_ops.py` (existing, now active) |

Softmax needs a maximum reduction as well as a sum, and layer normalization
already had block-reduction machinery. Operators must never include each other,
so the reduction was lifted into `src/common/reduce.cuh` and parameterised by
the combining operation. Layer normalization was moved onto it; its results are
unchanged.

One block per row, three passes: row maximum, sum of shifted exponentials, then
the normalized write.

## Design decisions worth recording

**The maximum subtraction is a correctness requirement, not a refinement.**
`expf` overflows above about x = 88.7, so a row containing 1000 would produce
inf for every large element and inf/inf = NaN for every output. Subtracting the
row maximum makes the largest shifted value exactly 0, so the largest
exponential is exactly 1 and nothing can overflow.

It also removes the division-by-zero case entirely: the maximum element always
contributes exp(0) = 1, so the denominator is at least 1 however extreme the
input. There is no guard to write, because there is nothing to guard against.

**Three passes over x, recomputing the exponential, rather than caching it.**
Caching the exponentials in the output during pass two would evaluate `expf`
once per element instead of twice, but it moves *more* bytes, not fewer:

| | Reads of x | Writes of y | Reads of y | `expf` per element |
|---|---|---|---|---|
| Recompute (chosen) | 3 | 1 | 0 | 2 |
| Cache exponentials | 2 | 2 | 1 | 1 |

Five transfers against four. For a kernel expected to be memory-bound, trading
bytes for arithmetic is the wrong direction. **This is reasoning, not
measurement** — no timing was taken.

**The sum accumulates in double; the exponential is evaluated in float.** Error
in the sum compounds across the whole row, so it gets double. `expf` is accurate
to a couple of ulp, its error is per-element rather than accumulating, and the
output is float32 anyway — a double exponential would cost considerably more for
a difference the output cannot represent. This differs from layer normalization,
where double was needed throughout for a specific reason recorded in
[phase 2](phase-2-layernorm.md).

## Results

### CTest

```
1/5 Test #1: infrastructure ...................   Passed    0.63 sec
2/5 Test #2: support ..........................   Passed    0.52 sec
3/5 Test #3: matmul ...........................   Passed    0.01 sec
4/5 Test #4: layernorm ........................   Passed    0.98 sec
5/5 Test #5: softmax ..........................   Passed    0.99 sec

100% tests passed, 0 tests failed out of 5
```

### `ccml_test_softmax` — 13 passed, 0 failed, 0 skipped

```
[PASS] softmax.small_rows
       8 shapes; worst max_abs=4.398e-08 max_rel=9.925e-08 normalized=0.018
       worst at: rows=8 cols=2
[PASS] softmax.single_column_is_exactly_one
[PASS] softmax.large_rows
       5 shapes; worst max_abs=1.657e-09 max_rel=3.826e-07 normalized=0.000
       worst at: rows=64 cols=1024
[PASS] softmax.non_power_of_two_row_sizes
       8 shapes; worst max_abs=4.441e-09 max_rel=3.129e-07 normalized=0.000
       worst at: rows=11 cols=100
[PASS] softmax.multiple_rows
       6 shapes; worst max_abs=4.477e-09 max_rel=5.013e-07 normalized=0.000
       worst at: rows=4099 cols=512
[PASS] softmax.many_rows
       n=5120000 max_abs=1.305e-08 max_rel=5.631e-07 normalized=0.001 worst_index=1390397 nonfinite=0
[PASS] softmax.large_positive_inputs
       mean=5000: n=4096 max_abs=3.178e-08 max_rel=1.549e-07 normalized=0.001 worst_index=3785 nonfinite=0
[PASS] softmax.large_negative_inputs
       mean=-5000: n=4096 max_abs=2.694e-08 max_rel=1.454e-07 normalized=0.001 worst_index=2242 nonfinite=0
[PASS] softmax.extreme_dynamic_range
       range +/-10000: n=4096 max_abs=2.731e-08 max_rel=1.000e+00 normalized=0.001 worst_index=1267 nonfinite=0
[PASS] softmax.dominant_element_gives_one_hot
[PASS] softmax.constant_row_is_uniform
[PASS] softmax.rejects_null_and_nonpositive_dimensions
[PASS] softmax.warp_per_row_is_reserved_not_silently_substituted

13 passed, 0 failed, 0 skipped
```

Alongside the oracle comparison, every case is checked against three properties
derived from nothing but the definition: every output finite, every output in
[0, 1], and every row summing to 1 within 1e-5. A kernel can match a reference
elementwise and still be wrong if the reference is wrong, so these are kept
independent of it.

### PyTorch comparison — `torch.softmax`

```
test_softmax_matches_torch[64-1024] PASSED
test_softmax_matches_torch[37-519]  PASSED
test_softmax_matches_torch[128-1]   PASSED
test_softmax_rows_sum_to_one        PASSED
test_softmax_survives_large_inputs  PASSED
```

Only matmul now skips: 11 passed, 3 skipped.

## Numerical error

| Group | Shapes | max_abs | max_rel | Tolerance used |
|---|---|---|---|---|
| Small rows (cols 1–33) | 8 | 4.398e-08 | 9.925e-08 | 1.8% |
| Large rows (cols 1024–16384) | 5 | 1.657e-09 | 3.826e-07 | 0.0% |
| Non-power-of-two (100, 517, 999, 1023, 1025, 4095, 4097, 6143) | 8 | 4.441e-09 | 3.129e-07 | 0.0% |
| Row counts (1–4099) | 6 | 4.477e-09 | 5.013e-07 | 0.0% |
| 20000 × 256 (5.12M elements) | 1 | 1.305e-08 | 5.631e-07 | 0.1% |
| mean = +5000 | 1 | 3.178e-08 | 1.549e-07 | 0.1% |
| mean = −5000 | 1 | 2.694e-08 | 1.454e-07 | 0.1% |
| Uniform ±10000 | 1 | 2.731e-08 | **1.000e+00** | 0.1% |

**Worst absolute error anywhere: 4.398e-08**, below half of float32 machine
epsilon. Every group used under 2% of its error budget.

Absolute error *falls* as rows get longer, which is the opposite of the usual
pattern and is expected here: softmax outputs sum to 1, so a longer row means
smaller individual values and smaller absolute deviations. Relative error rises
slightly instead, which is why both are reported.

### The relative error of 1.000e+00, explained

That figure is real and not a defect. With inputs spread uniformly over ±10000,
`x - rowmax` reaches about −20000. Two different underflow thresholds then
apply:

| | `exp` underflows to zero below |
|---|---|
| float32 (the kernel) | x − max ≈ **−103.3** |
| float64 (the oracle) | x − max ≈ **−744.4** |

For the band between them, the kernel returns exactly 0 while the double oracle
still returns a tiny non-zero value. Relative error is then exactly 1.0 by
definition, while the absolute error is around 1e-45.

The kernel is right. A value of 1e-45 relative to the row maximum is not
representable in a float32 output, and any float32 softmax flushes it to zero.

This is precisely the case the combined tolerance was designed for. A pure
relative bound would have failed this test on a correct kernel; a pure absolute
bound would be meaningless elsewhere. The element-wise
`abs_tol + rel_tol * |ref|` form passes it on the absolute term, at 0.1% of
budget — a design decision from phase 1 that this phase put under load.

## Supported shapes

| | |
|---|---|
| Rows | any `rows >= 1`; tested 1 to 20000 |
| Columns | any `cols >= 1`; tested 1 to 16384 |
| Alignment | none required. No column count is a special case; rows wider than a block are strided |
| Softmax dimension | the last one. Collapse leading dimensions into `rows` |
| Input range | any finite float32. Tested ±5000 means and a ±10000 uniform spread |
| dtype | float32 only |

Guaranteed properties, verified on every test case: outputs are finite, lie in
[0, 1], and each row sums to 1 within 1e-5. A single-column row returns exactly
1.0, a constant row returns exactly uniform, and a dominant element takes the
entire mass.

Rejected with `CCML_ERROR_INVALID_ARGUMENT`: null `x` or `y`, `rows <= 0`,
`cols <= 0`.

## What this does NOT establish

- **No performance claim of any kind.** No timing was taken. The three-pass
  choice was argued from byte counts, not measured.
- **Nothing about the Ada target.** Turing (sm_75).
- **Matmul is still not implemented**; its suite still skips.
- **`compute-sanitizer` has not been run** on any kernel. This is now the
  largest outstanding correctness gap: both operators reuse a single shared
  scratch buffer across two reductions and depend on the barrier between them,
  which is exactly what racecheck exists to verify and exactly the kind of bug
  that produces plausible-looking results.
- **Inputs containing inf or NaN are untested and undefined.** Every test used
  finite inputs.
- **`CCML_SOFTMAX_WARP_PER_ROW` is not implemented** and returns
  `NOT_IMPLEMENTED`; a test asserts it is not silently substituted.
- **No backward pass.**

## Reproducing this

```
python cloud/kaggle/kaggle_run.py
```
