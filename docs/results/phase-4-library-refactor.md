# Phase 4 — Library consolidation

**Date:** 2026-09-24
**Outcome:** 64 of 64 tests passed, plus 11 PyTorch comparisons. 0 failed.
12 skips remain, all of them matmul.
**Nothing about the operators changed**; layer normalization and softmax report
the same errors as phases 2 and 3.

## Environment

Kaggle GPU kernel (`isiddhesh3011/ccml-build`, version 13). Tesla T4, sm_75,
CUDA 12.8.93, GCC 11.4.0, CMake 3.31.10, Ninja. Release,
`-DCCML_CUDA_ARCHITECTURES=75`, `-Wall -Wextra -Werror`. **Zero warnings.**

## What changed

### Duplication removed

`threads_for_cols` existed **verbatim in two files** — `src/layernorm/layernorm.cu`
and `src/softmax/softmax.cu` — differing only in which block-size constant it
clamped against. It is now `rowwise_launch_config` in `src/common/rowwise.h`,
which both operators call.

This was the only genuine duplication left. The block reduction had already been
lifted to `src/common/reduce.cuh` in phase 3.

### Test targets made uniform

Every suite now follows one rule, applied by a `ccml_add_test` function:

    tests/test_<name>.cpp|cu  ->  target ccml_test_<name>  ->  ctest name <name>

`tests/test_support.cpp` was the exception — it built `ccml_test_support_selftest`,
which also collided conceptually with the `ccml_test_support` harness library. It
is now `tests/test_harness.cpp` → `ccml_test_harness` → `ctest -R harness`, which
also describes it better: it tests the harness.

### One command to build, one to test

`CMakePresets.json` gained workflow presets, so the two commands the brief asked
for actually exist:

    cmake --workflow --preset build     # configure + build everything
    cmake --workflow --preset default   # configure + build + run every test

This required raising `cmake_minimum_required` from 3.24 to 3.25, the version
that introduced workflow presets. `docs/build.md` and the Kaggle runner's
version check were updated to match.

### A compiled usage example

`examples/usage.cpp` runs all three operators through the public API and
**includes nothing from `src/`**. It is the code the README's usage section
shows, so the documented usage is compiled and executed rather than asserted.
If the public headers are ever incomplete, this stops building.

### The pipeline now survives its own failures

The first attempt at this phase failed to compile, and **Kaggle returned no
output at all** — it discards `/kaggle/working` when a kernel exits non-zero,
which is precisely when the log is needed. The runner now records the failure,
writes the log, and finishes cleanly; `kaggle_run.py` scans for a
`CCML_RUN_FAILED` marker and exits non-zero itself.

Root cause of that build failure: `examples/usage.cpp` used `std::size_t`
without including `<cstddef>`.

## What deliberately did not change

**The `ccml_` prefix stays.** The brief named the operators `custom_matmul`,
`custom_layernorm`, `custom_softmax`. C has no namespaces, so a library's
symbols must carry a prefix that will not collide with anything else a caller
links, and `custom_*` is close to the worst available choice — it is exactly
what another project would also pick. The operator names within the scheme are
already `matmul`, `layernorm`, `softmax`.

**Matmul's signature stays `(C, A, B, M, N, K)`** rather than being forced into
the `(y, x, rows, cols)` shape of the row-wise operators. They are different
problems; M/N/K is the universal convention for a matrix product and renaming it
would make the API less recognisable, not more consistent. What *is* consistent
across all three is the argument order: output, inputs, dimensions, parameters,
`algo`, `stream`.

## Results

### Build

```
[42/42] Linking CXX executable bin/ccml_bench_matmul
```

42 targets, zero warnings. `cmake --install` produced headers, the shared
library and working `find_package(ccml)` export files.

### CTest — one command, every suite

```
1/5 Test #1: infrastructure ...................   Passed    0.61 sec
2/5 Test #2: harness ..........................   Passed    0.52 sec
3/5 Test #3: matmul ...........................   Passed    0.01 sec
4/5 Test #4: layernorm ........................   Passed    1.03 sec
5/5 Test #5: softmax ..........................   Passed    1.02 sec

100% tests passed, 0 tests failed out of 5
Total Test time (real) =   3.20 sec
```

| Suite | Passed | Failed | Skipped |
|---|---|---|---|
| `ccml_test_infrastructure` | 16 | 0 | 0 |
| `ccml_test_harness` | 19 | 0 | 0 |
| `ccml_test_layernorm` | 14 | 0 | 0 |
| `ccml_test_softmax` | 13 | 0 | 0 |
| `ccml_test_matmul` | 2 | 0 | **6** |
| `pytest` (Python binding) | 11 | 0 | **3** |

All 9 remaining skips are matmul.

### Usage example — public API only

```
ccml 0.1.0

  input        0.0000   1.0000   2.0000   3.0000   4.0000   5.0000   6.0000   7.0000
  softmax      0.0006   0.0016   0.0043   0.0116   0.0315   0.0856   0.2326   0.6323
             row sums to 1.000000
  layernorm   -1.5275  -1.0911  -0.6547  -0.2182   0.2182   0.6547   1.0911   1.5275

  matmul:
  ccml_matmul_f32              ccml_matmul_f32: kernels are not implemented in this build
```

Both outputs are independently checkable. For input 0..7: the mean is 3.5 and
the population standard deviation is 2.2913, so the first normalized value is
−3.5 / 2.2913 = **−1.5275**. For softmax, e⁷ / Σe⁰⁻⁷ = 1096.6 / 1734.6 =
**0.6323**. Both match.

The matmul line is included on purpose: it is what a caller sees when an
operator is declared but unimplemented — a clear status and message, not a crash
and not a silently wrong answer.

## What this does NOT establish

- **No performance claim.** Nothing was timed. This phase changed no kernel.
- **Matmul is still not implemented.**
- **`compute-sanitizer` has still not been run** on any kernel. Unchanged from
  phase 3, and still the largest outstanding correctness gap.
- **Turing (sm_75), not the Ada target.**
- **Windows is still unverified**; the development machine has no Windows SDK.

## Reproducing this

```
python cloud/kaggle/kaggle_run.py
```

Or, with a local GPU:

```
cmake --workflow --preset default
```
