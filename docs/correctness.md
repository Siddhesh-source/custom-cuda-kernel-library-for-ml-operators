# Correctness

Status: all three operators are validated in **FP32, FP16 and BF16**
(reference.md section 3). The measured errors are in the tables below; the
per-phase records in [results/](results/README.md) carry the full breakdowns.

## Method

Every kernel is compared against **two independent references**:

1. **Double-precision CPU implementations** in `support/reference.cpp`,
   written for obviousness rather than speed: straightforward loops, double
   accumulation, no blocking. Used by the C++ test executables and by the
   benchmark correctness gate.
2. **PyTorch float64**, the reference the spec names: `python/tests/`
   (`test_torch_api.py` for FP32, `test_torch_dtypes.py` for FP16 and BF16), on
   the GPU in float64. These need a CUDA PyTorch, so they run on Kaggle (the
   `sanitize` job).

Two references rather than one is deliberate. A single reference agreeing with a
kernel proves only that they share assumptions; two independent ones disagreeing
is what actually catches a wrong assumption.

For FP16 and BF16, inputs are rounded to the storage type FIRST, and both
references are computed from the rounded values. The comparison then measures
only what the kernel adds: its float32 arithmetic and one final rounding on
store.

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

**Softmax is relative only** (`tolerance_for_softmax_f32(cols)`, since Phase 8).
Its outputs are probabilities whose size is set by the row: in a 32000-wide row
a typical output is about 3e-5. The earlier absolute bound of
`8 * eps * sqrt(cols)` (about 1.7e-4 at 32000 columns) was larger than the
outputs themselves, so a kernel that wrote zeros would have passed the
element-wise check. Every error source in softmax (`expf`, the double sum, the
division) is relative to each output, so the bound is `rel_tol` alone, with an
absolute floor of FLT_MIN where `expf` underflows. Every existing FP32 softmax
case still passes under the stricter bound.

Matmul therefore uses `tolerance_for_f32_dot_product(terms, scale)`, which
scales the absolute bound by the RMS of the reference — the size of the quantity
actually being accumulated. Every matmul shape then passes with 5-10x margin.
The model was replaced, not loosened; see
[results/phase-6-case-study.md](results/phase-6-case-study.md).

### Per-dtype tolerances

The kernels compute in float32 or wider for every storage type (widen on load,
round once on store; `src/common/dtype.cuh`). So a 16-bit result differs from
its reference only by the f32 arithmetic error plus that final rounding.
`widen_for_storage` adds exactly that:

    rel_tol(dtype) = rel_tol(f32) + 2 * u(dtype)
    abs_tol(dtype) = abs_tol(f32) + floor(dtype)

| dtype | u (unit roundoff) | floor | Why the floor |
|---|---|---|---|
| FP32 | 2^-24 = 5.96e-8 | 0 (softmax: FLT_MIN) | n/a |
| FP16 | 2^-11 = 4.88e-4 | 2^-24 = 5.96e-8 | Spacing of FP16 subnormals; small outputs round in absolute steps |
| BF16 | 2^-8 = 3.91e-3 | FLT_MIN = 1.18e-38 | Same exponent range as FP32 |

The factor 2 on `u`: half a unit from the final rounding, plus the f32 result
itself being up to its own tolerance away from the reference before rounding.
The PyTorch tests (`python/tests/test_torch_dtypes.py`) use the same formulas.

### Unfused 16-bit pipelines (Phase 11)

A pipeline that rounds an intermediate to a 16-bit type before a second kernel
(a matmul followed by a separate bias/GELU/residual kernel, as PyTorch's
`Linear` then `GELU` does) carries one extra rounding the fused contract does
not: up to `u * |intermediate_i|`, element-specific. The benchmark gate admits
exactly that for such candidates, through a per-element allowance
(`ccml_test::compare(got, ref, tol, extra_abs)`, unit-tested in
`harness.compare_per_element_allowance`):

    bound_i = abs_tol + rel_tol * |ref_i| + 1.2 * u(dtype) * |A B|_i

(1.2 covers the GELU slope, at most 1.13.) It applies to "+separate"
candidates and to cuBLASLt's epilogue, whose error measured equal to theirs.
**ccml's fused kernels are held to the strict bound.** See
[optimization-log.md](optimization-log.md) entry 8.

### Layer norm in FP32: the pivot shift (Phase 12)

`TWO_PASS` reduces in FP64 because a FP32 mean accumulated at magnitude 10000
carries ~5e-4 of error, 5% of a 0.01 standard deviation. `WELFORD`,
`WARP_PER_ROW`, `VECTORIZED` and the fused residual kernels reduce in FP32 but
first subtract a pivot (the row's first element): d = x - p is exact when x
and p are within a factor of 2 (Sterbenz) and otherwise rounds relative to |d|,
so the statistics are of small numbers and x - mean is formed as d - mean(d).
They are held to **the same FP32 tolerance** as TWO_PASS, including mean 10000
/ stddev 0.01 and a pivot 1000 standard deviations from its row
([optimization-log.md](optimization-log.md) entry 9).

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

### FP16 and BF16 (Phase 8)

Worst case over the shapes in `tests/test_dtypes.cpp`, copied from
`ccml_test_dtypes` output: local RTX 3050 A Laptop (sm_89), 2026-11-17. **The
Tesla T4 (sm_75) produced identical figures**
([results/phase-8-multi-dtype.md](results/phase-8-multi-dtype.md)).

| Operator | dtype | Shapes | worst max_rel (typical outputs) | Bound (rel_tol) | Verdict |
|---|---|---|---|---|---|
| matmul naive + tiled | f16 | 6, K 1-2048 | 2.37e-03 (a) | ~1.0e-03 + abs | **PASS** |
| matmul naive + tiled | bf16 | 6, K 1-2048 | 4.11e-03 | ~7.9e-03 | **PASS** |
| layernorm | f16 | 7, cols 1-32000, incl. mean 100 | 3.49e-03 (a) | ~1.0e-03 + abs | **PASS** |
| layernorm | bf16 | 7 | 3.89e-03 | ~7.9e-03 | **PASS** |
| softmax | f16 | 7, cols 1-32000, incl. scale 30 | up to 6.9e-03 (b) | ~1.0e-03 + floor 2^-24 | **PASS** |
| softmax | bf16 | 7 | 3.89e-03 | ~7.9e-03 | **PASS** |

BF16 errors sit at about one unit roundoff (2^-8 = 3.9e-3), which is the final
rounding and nothing else. The same holds for FP16 at 2^-11.

(a) The larger relative figures are on elements near zero (matmul
cancellation, normalized values near the mean), where the absolute term of the
bound governs.

(b) Larger relative errors (up to 1.0) occur only on outputs in FP16's
subnormal range or below it. Those are rounded in absolute steps of 2^-24 and
covered by the floor; the absolute error there is at most 2.4e-04 on
scale-30 inputs.

### Tensor Core matmul and the layer norm variants (Phases 11-12, Tesla T4)

Worst normalized error (|error| / bound; <= 1 passes) printed by the test
executables under compute-sanitizer memcheck on Kaggle Tesla T4 sessions:
Phase 11 at commit `5edb43d` (`results/sm_75/tensor_cores/sanitizer/`), Phase
12 at `aab20ff` (`results/sm_75/layernorm_variants/sanitizer/`). BF16 Tensor
Core cases SKIP on the T4 (sm_80+); on the local sm_89 they pass with the same
margins (log entries 4-8).

| Operator | Algorithm | Cases | Worst normalized | Verdict |
|---|---|---|---|---|
| matmul f16 | wmma, mma_sync, split_k, split_k_atomic, mma_sync_swizzled | every TC config x 17 awkward shapes, k-steps 1-1000, misaligned views | 0.499 | **PASS** |
| matmul f16 | fused epilogue (mma_sync, split_k) | 7 epilogue combinations x 8 shapes | 0.488 | **PASS** |
| layernorm f32 | two_pass | 16 cases incl. 4 offsets | 0.011 | **PASS** |
| layernorm f32 | welford, warp_per_row, vectorized | same 16 cases (cols <= 1024 for the warp kernels) | 0.062 | **PASS** |
| layernorm f32 | welford / warp_per_row / vectorized | pivot 1000 stddevs from its row | 0.090 / 0.115 / 0.151 | **PASS** |
| layernorm f32 | warp_per_row, every per-lane bucket | 23 widths, odd widths offset +300 | 0.048 | **PASS** |
| layernorm f32 | vectorized, vector and fallback paths | 20 widths + misaligned views | 0.072 | **PASS** |
| layernorm_residual f32 | vectorized, welford, auto | 10 shapes x 4 combinations + offset + misaligned | 0.040 | **PASS** |

The FP32 layer norm variants carry about 6x the error of FP64 two-pass (0.062
against 0.011 of the bound) and stay 16x inside it. The 16-bit results sit at
about half the bound, which is the final rounding.

**Determinism (C3), T4:** over 20 runs of a 16 x 768 x 3072 matmul, split_k
differed bitwise 0 times and split_k_atomic 19 of 19 times.
`ccml_layernorm_residual`'s sum_out is checked bitwise against the host's
(x + r) + b rounded once, in all three dtypes.

### NaN, infinity and masking policy

The same for every dtype, and identical to `torch.softmax` and
`torch.nn.functional.layer_norm`:

- Softmax over a row with some `-inf` entries gives **exactly 0** at those
  entries; the rest of the row sums to 1. This is additive masking.
- Softmax over a row that is **entirely** `-inf` gives NaN everywhere (0/0), as
  PyTorch does.
- A NaN anywhere in a row makes that whole row NaN (softmax and layernorm).
- FP16 results above 65504 round to `+inf` on store (IEEE). BF16 has float32's
  exponent range, so the same results stay finite.

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

**FP16 / BF16** (`tests/test_dtypes.cpp`, `python/tests/test_torch_dtypes.py`) -
all three operators over representative shapes up to a 32000-wide row, both
matmul variants. Also: FP16 overflow to inf against BF16 staying finite;
+-65504 rows staying finite in softmax and layernorm; partial and full `-inf`
masking; NaN propagation; constant rows giving exactly beta; error messages
naming the dtype entry point; reserved variants reporting NOT_IMPLEMENTED for
every dtype. The output buffer is poisoned with NaN before every run, so an
element the kernel fails to write cannot pass.

## Memory and race checking

`scripts/sanitize.sh` runs memcheck, racecheck, initcheck and synccheck over
every suite, including the FP16/BF16 one. It runs in the Kaggle `sanitize` job
and writes `results/sm_75/sanitizer/summary.txt`; the outcome is recorded in
the Phase 7 and Phase 8 results documents.

    bash scripts/sanitize.sh build/bin results/sm_75/sanitizer `racecheck` matters most for the tiled matmul and the
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
