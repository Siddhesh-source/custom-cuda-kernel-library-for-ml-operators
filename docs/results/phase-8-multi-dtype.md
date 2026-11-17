# Phase 8: FP16 and BF16 for every operator

**Dates:** 2026-11-17
**Branch:** `multi-dtype-kernels`
**Outcome:** every operator has `_f16` and `_bf16` entry points, correct
against both references with explicit per-dtype tolerances, on a Tesla T4
(and on the local sm_89 GPU). ctest 7/7, pytest 68/68 with no skips, and
compute-sanitizer clean on 24/24 suite/tool pairs (one deliberate,
recognized API error).

## Environment

Kaggle `sanitize` session at commit `24e7360` (`results/sm_75/dtypes/`): Tesla
T4 (sm_75), driver 580.178.04, CUDA 13.0, torch 2.11.0+cu130. The build and the
toolchain are as in Phase 7. Locally the same suites passed on the RTX 3050 A
Laptop (sm_89, CUDA 13.2, MSVC 14.44).

## What was added

| Requirement (reference.md) | Implementation |
|---|---|
| FP16 and BF16 (§3, §9: "all kernels pass correctness tests for each dtype") | `ccml_<op>_f16` / `_bf16` for matmul, layernorm and softmax; public 16-bit storage types `ccml_f16`, `ccml_bf16` (C ABI, laid out like `__half` / `__nv_bfloat16`) |
| Same algorithms in every dtype | Kernels templated on the storage type: widen on load, compute in float32 (or double where the operator already did), round once on store (`src/common/dtype.cuh`) |
| PyTorch float64 reference, explicit tolerances per dtype (§3) | `python/tests/test_torch_dtypes.py`; tolerance policy in `docs/correctness.md` |
| Edge behaviour | Overflow, masking, NaN, constant rows, range limits (below) |

### Tests written in this phase

- `tests/test_dtypes.cpp` (13 tests): all three operators over representative
  shapes up to a 32000-wide row, with the fp64 reference computed from the
  16-bit-rounded inputs. Plus:
  - FP16 overflow to `+inf` (300 × 300) while BF16 stays finite;
  - ±65504 rows staying finite in softmax and layernorm;
  - partial and full `-inf` masking, and NaN propagation (both dtypes);
  - constant rows giving exactly beta;
  - error messages naming the dtype entry point;
  - reserved variants returning NOT_IMPLEMENTED.
- `python/tests/test_torch_dtypes.py` (33 tests): the same operators against
  `torch` float64 on the GPU, the masking and NaN policy against
  `torch.softmax`, and mixed-dtype rejection.

## Results (verbatim)

`results/sm_75/dtypes/ctest.txt`:

```
1/7 Test #1: infrastructure ...................   Passed    0.59 sec
2/7 Test #2: harness ..........................   Passed    0.51 sec
3/7 Test #3: matmul ...........................   Passed    1.21 sec
4/7 Test #4: layernorm ........................   Passed    1.00 sec
5/7 Test #5: softmax ..........................   Passed    0.99 sec
6/7 Test #6: dtypes ...........................   Passed    1.19 sec
7/7 Test #7: bench_stats ......................   Passed    0.03 sec
100% tests passed, 0 tests failed out of 7
```

`pytest_binding.txt`: `68 passed in 2.92s` (35 FP32 binding tests plus 33
FP16/BF16 tests against torch float64; **0 skipped**: the GPU tests ran).
`pytest_benchlib.txt`: `8 passed`.

compute-sanitizer (`results/sm_75/dtypes/sanitizer/summary.txt`): all 24
pairs exit 0, including the new suite:

```
dtypes           memcheck   0        ========= ERROR SUMMARY: 0 errors
dtypes           racecheck  0        ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
dtypes           initcheck  0        ========= ERROR SUMMARY: 0 errors
dtypes           synccheck  0        ========= ERROR SUMMARY: 0 errors
infrastructure   memcheck   0        ========= ERROR SUMMARY: 1 error (expected: deliberate OOM in cuda_failures_carry_the_runtime_message)
```

### Measured errors on the T4

Worst case per operator and dtype over the suite's shapes, extracted from
`sanitizer/dtypes_memcheck.log` (the suite's own output):

| dtype | Operator | max abs | max rel | Note |
|---|---|---|---|---|
| f16 | matmul (naive, tiled) | 6.18e-02 | 3.96e-02 | rel is on elements near zero; the abs term governs there |
| f16 | layernorm | 1.95e-03 | 3.49e-03 | |
| f16 | softmax | 2.40e-04 | 1.0 | the rel 1.0 is on outputs below FP16's subnormal range (absolute steps of 2^-24) |
| bf16 | matmul (naive, tiled) | 4.90e-01 | 4.11e-03 | about one BF16 unit roundoff (3.9e-3) |
| bf16 | layernorm | 1.56e-02 | 3.89e-03 | |
| bf16 | softmax | 1.86e-03 | 1.0 | scale-30 inputs; same subnormal effect |

**These are identical, digit for digit, to the local sm_89 run**
(docs/correctness.md). That is expected: the same inputs, float32 arithmetic in
the same order, and one rounding at the end. It also shows the kernels are
deterministic across two GPU architectures.

## Issues found and how they were resolved

| Issue | Resolution |
|---|---|
| The pre-existing softmax tolerance had an absolute part of `8·eps·sqrt(cols)` (1.7e-4 at 32000 columns), larger than the outputs themselves (~3e-5), so a kernel writing zeros would have passed | Softmax tolerance is now relative, with an absolute floor at FLT_MIN. All FP32 softmax cases still pass. The fix is applied to the tests, the benchmark gate and the Python baselines alike |
| Rounding layernorm's double result to 16 bits through float32 rounds twice | Direct `__double2half` / `__double2bfloat16` (`from_double`) |
| The output buffer could hide an unwritten element | Poisoned with NaN (0xFF bytes) before every 16-bit run |

## What this proves, and what it does not

**Proves:** every operator is correct in FP16 and BF16 against two
independent float64 references, with tolerances derived from the format, not
tuned to pass. It is free of memory errors and races on the T4. The masking
and NaN behaviour matches PyTorch.

**Does not prove:**

- Anything about 16-bit *speed*. The kernels compute in float32, so the only
  gain is memory traffic. The dtype comparison is a reference.md section 5
  experiment, run in the full sweep (Phase 15).
- Anything about BF16 Tensor Core kernels, which need sm_80+ (Phase 11,
  docs/local.md L1).

## How to reproduce

    python cloud/kaggle/kaggle_run.py --any-gpu --jobs sanitize
    build/local/bin/ccml_test_dtypes          # local, any CUDA GPU
