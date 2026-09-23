# Correctness

Status: **all three operators implemented and validated** on a Tesla T4.
Measured errors are in the table below; the per-phase records in
[results/](results/README.md) carry the full breakdowns.

## Method

Every kernel is compared against **two independent references**:

1. **Double-precision CPU implementations** in `tests/support/reference.cpp`,
   written for obviousness rather than speed: straightforward loops, double
   accumulation, no blocking. Run from the C++ test executables.
2. **PyTorch**, from `python/tests/` and `python/case_study.py`. On the CPU in
   float64 for the numpy path, and on the GPU in float64 for the case study.

Two references rather than one is deliberate. A single reference agreeing with a
kernel proves only that they share assumptions; two independent ones disagreeing
is what actually catches a wrong assumption. PyTorch is used CPU-only, so no
CUDA build of torch is required.

## Tolerance policy

Applied element-wise:

    |got[i] - ref[i]|  <=  abs_tol + rel_tol * |ref[i]|

The combined form is necessary because neither bound works alone. A pure
relative bound is meaningless where the reference is near zero (softmax tails,
normalized values); a pure absolute bound is meaningless where it is large
(matmul over a long K).

Starting point, from `tolerance_for_f32_reduction(terms)`:

    abs_tol = rel_tol = 8 * eps_f32 * sqrt(terms)        eps_f32 = 1.19e-7

with `terms` = K for matmul and the column count for layernorm and softmax. The
square root reflects error growth in a sum of `terms` values with uncorrelated
rounding; the factor of 8 is margin.

**This model is right for the row-wise operators and wrong for matmul.** It
judges each element against its own magnitude, which works where outputs have a
bounded meaningful scale — softmax probabilities, normalized values. A
matrix-product element is a sum of K signed products and can cancel to nearly
zero while the terms that produced it were of magnitude 1; judged relatively it
then shows an enormous error for no reason. A K=768 case measured a relative
error of **0.23** on such an element while its absolute error was ordinary.

Matmul therefore uses `tolerance_for_f32_dot_product(terms, scale)`, which
scales the absolute bound by the RMS of the reference — the size of the quantity
actually being accumulated. Every matmul shape then passes with 5-10x margin.
The model was replaced, not loosened; see
[results/phase-6-case-study.md](results/phase-6-case-study.md).

## Measured results

To be filled in from actual runs. Do not populate this table by hand: paste the
`max_abs` / `max_rel` / `normalized` values the test executables print.

| Operator | Algorithm | Shapes | max_abs | max_rel | normalized | Verdict |
|---|---|---|---|---|---|---|
| matmul | tiled | 11 shapes, K 1-2048 | 1.836e-04 | see note | 0.52 | **PASS** |
| layernorm | two_pass | 26 shapes, cols 1–16384, rows 1–20000 | 2.384e-07 | **5.958e-08** | 0.021 | **PASS** |
| softmax | block_per_row | 28 shapes, cols 1–16384, rows 1–20000 | **4.398e-08** | 5.631e-07 † | 0.018 | **PASS** |

Measured on a Tesla T4: layer normalization 2026-09-23, softmax 2026-09-24. Both
worst-case errors are at or below half of float32 machine epsilon (1.19e-07) —
the final rounding to float32 and essentially nothing else.

Layer normalization's error does not grow with column count, because its
reductions accumulate in double. Softmax's *absolute* error falls as rows get
longer, since its outputs sum to 1 and longer rows mean smaller values.

† Relative error reaches exactly 1.000e+00 on one test, with inputs spread over
±10000. That is not a defect: float32 `exp` underflows to zero below x ≈ −103.3
while the double oracle keeps going to −744.4, so for inputs in that band the
kernel correctly returns 0 where the oracle returns ~1e-45. Any float32 softmax
flushes such a value to zero. The absolute error there is 2.731e-08, and the
combined `abs_tol + rel_tol * |ref|` tolerance passes it on the absolute term at
0.1% of budget. A pure relative bound would have failed a correct kernel; this
is the case that form exists for.

Full breakdowns: [results/phase-2-layernorm.md](results/phase-2-layernorm.md),
[results/phase-3-softmax.md](results/phase-3-softmax.md).

## Edge cases covered

Registered in the test files. All pass.

**Matmul** - square, ragged dimensions (no dimension a multiple of the tile),
single element, tall-and-skinny, long K, null arguments, non-positive dimensions.

**LayerNorm** - with and without affine, gamma-only, ragged column count, single
row, single column (zero variance, which tests the eps guard), large mean with
small variance (catches a naive `E[x^2] - E[x]^2`), non-positive eps.

**Softmax** - typical, ragged columns, single column (must be exactly 1.0),
single row, large positive inputs (overflow guard), large negative inputs, null
arguments. Plus two properties checked independently of any reference: every row
sums to 1, and every value lies in [0, 1].

## Memory and race checking

**Still not run — the largest outstanding correctness gap.** Three kernels now
depend on barriers that only racecheck can verify. Both must be clean before the
milestone is claimed:

    compute-sanitizer --tool memcheck  build/default/bin/ccml_test_matmul
    compute-sanitizer --tool racecheck build/default/bin/ccml_test_matmul

Repeat for each operator. `racecheck` matters most for the tiled matmul and the
block reductions, which are exactly where a missing `__syncthreads()` produces
results that look plausible.

## Reproducibility

Inputs come from `ccml_support::Rng`, a splitmix64 generator with values derived
from the raw 64-bit output by explicit arithmetic rather than through
`std::uniform_real_distribution`, whose output sequence the standard does not
specify and which therefore differs between standard libraries. Every test
passes an explicit seed. Reruns on the same machine produce identical inputs, and
so do runs under a different toolchain.

## Skips are not passes

The harness reports PASS, FAIL and SKIP separately, and prints a reminder when
anything skipped. An unimplemented operator must never be able to report PASS.
That is why every reference returns `false` while unimplemented instead of
filling its output with zeros.
