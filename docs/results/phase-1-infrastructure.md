# Phase 1 — Common CUDA infrastructure

**Date:** 2026-09-23
**Outcome:** 35 of 35 tests passed. 0 failed, 0 skipped.
**Verdict:** the infrastructure builds clean and executes correctly on a GPU.
Safe to proceed to operators.

## Environment

Run on a Kaggle GPU kernel (`isiddhesh3011/ccml-build`, version 5). Nothing in
this phase was built or run on the development machine, which still cannot
compile CUDA — see [../build.md](../build.md).

| | |
|---|---|
| GPU | NVIDIA Tesla T4 (2 attached, device 0 used) |
| Compute capability | **7.5 (Turing)** — *not* the Ada 8.9 target |
| SMs | 40 |
| Driver | 580.159.04 |
| CUDA toolkit | 12.8.93 |
| Host compiler | GCC 11.4.0 |
| CMake / generator | 3.31.10 / Ninja |
| Build type | Release, `-DCCML_CUDA_ARCHITECTURES=75` |
| Warnings | `-Wall -Wextra -Werror` enabled |

## What was added

| Component | File |
|---|---|
| Launch configuration validation | `src/common/launch.h` (new) |
| Error helpers, now header-only | `src/common/check.h` |
| Stream-ordered copies, memset, per-stream sync | `include/ccml/memory.h`, `src/common/memory.cpp` |
| RAII stream | `support/cuda_stream.h` (new) |
| RAII event and timer | `support/cuda_event.h` (new) |
| Benchmark timer, rewritten onto RAII events | `bench/support/timer.cpp` |
| Interface conventions | `include/ccml/types.h` |
| `CCML_SYNC_LAUNCH_CHECKS` build option | `CMakeLists.txt`, `src/CMakeLists.txt` |

### Tests written in this phase

- **`tests/test_infrastructure.cu`** — 16 tests. The only suite that launches
  kernels of its own. White-box: includes `src/common/launch.h` directly,
  because launch validation cannot be reached through the public API.
- **`tests/test_support.cpp`** — 19 tests. Checks the harness that every other
  suite depends on. If `compare()` were wrong, every operator suite would be
  wrong in the same direction and all of them would still report PASS.

Both are registered with CTest and run per-suite (`ctest -R infrastructure`).

## Results

### Build

```
[40/40] Linking CXX executable bin/ccml_bench_matmul
```

40 of 40 targets, **zero warnings** with `-Werror` on. `cmake --install`
produced the headers, shared library and `find_package(ccml)` export files.

### CTest

```
1/5 Test #1: infrastructure ...................   Passed    0.62 sec
2/5 Test #2: support ..........................   Passed    0.50 sec
3/5 Test #3: matmul ...........................   Passed    0.01 sec
4/5 Test #4: layernorm ........................   Passed    0.01 sec
5/5 Test #5: softmax ..........................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 5
Total Test time (real) =   1.16 sec
```

### `ccml_test_infrastructure` — 16 passed, 0 failed, 0 skipped

```
[PASS] infrastructure.device_query_is_sane
       Tesla T4 sm_75, 40 SMs
[PASS] infrastructure.memory_round_trip
[PASS] infrastructure.memset_zeroes_device_memory
[PASS] infrastructure.kernel_launch_executes
       launched and verified 10000 elements
[PASS] infrastructure.two_kernels_chain_correctly
[PASS] infrastructure.work_runs_on_an_explicit_stream
[PASS] infrastructure.default_constructed_stream_is_the_default_stream
[PASS] infrastructure.launch_validation_accepts_a_reasonable_config
[PASS] infrastructure.launch_validation_rejects_oversized_block
       message: oversized_block: block of 1025x1x1 is 1025 threads, above the device limit of 1024 per block
[PASS] infrastructure.launch_validation_rejects_empty_block
[PASS] infrastructure.launch_validation_rejects_zero_grid
       message: zero_grid: grid.x is zero, so no blocks would run
[PASS] infrastructure.launch_validation_rejects_excess_shared_memory
       message: excess_shared: requested 50176 bytes of shared memory, above the device limit of 49152 per block
[PASS] infrastructure.rejected_launch_leaves_the_device_usable
[PASS] infrastructure.event_timer_measures_a_kernel
       measured 0.629920 ms
[PASS] infrastructure.failures_report_detail_not_just_a_code
       detail: ccml_device_malloc: ptr is null
[PASS] infrastructure.cuda_failures_carry_the_runtime_message
       out of memory -> cudaMalloc failed: out of memory (cudaErrorMemoryAllocation)
```

The three validation messages are the point of that component. Without it, an
oversized block surfaces as a bare `invalid configuration argument` from
whichever call synchronises next, naming neither the kernel nor the dimension.

### `ccml_test_support_selftest` — 19 passed, 0 failed, 0 skipped

```
[PASS] support.compare_identical_inputs_report_zero_error
[PASS] support.compare_reports_the_largest_deviation
       n=3 max_abs=5.000e-01 max_rel=1.667e-01 normalized=0.500 worst_index=2 nonfinite=0
[PASS] support.compare_normalized_error_decides_the_verdict
       inside:  n=1 max_abs=5.000e-02 max_rel=5.000e-02 normalized=0.500 worst_index=0 nonfinite=0
       outside: n=1 max_abs=2.000e-01 max_rel=2.000e-01 normalized=2.000 worst_index=0 nonfinite=0
[PASS] support.compare_uses_absolute_tolerance_near_zero
[PASS] support.compare_flags_nonfinite_results
       n=3 max_abs=0.000e+00 max_rel=0.000e+00 normalized=0.000 worst_index=1 nonfinite=2
[PASS] support.compare_flags_a_size_mismatch
[PASS] support.tolerance_grows_with_reduction_length
       terms=1:     abs_tol=9.537e-07 rel_tol=9.537e-07
       terms=10000: abs_tol=9.537e-05 rel_tol=9.537e-05
[PASS] support.rng_is_reproducible_for_a_given_seed
[PASS] support.rng_differs_between_seeds
[PASS] support.rng_uniform_respects_its_bounds
       range covered: [-1.999846, 4.997363]
[PASS] support.rng_normal_has_the_requested_moments
       mean=4.989133 stddev=2.002836
[PASS] support.div_up_rounds_up
[PASS] support.round_up_reaches_the_next_multiple
[PASS] support.grid_helpers_cover_every_element
[PASS] support.every_status_has_a_distinct_message
[PASS] support.version_string_is_populated
       version: 0.1.0
[PASS] support.freeing_a_null_pointer_is_a_no_op
[PASS] support.device_buffer_move_transfers_ownership
[PASS] support.cuda_stream_move_transfers_ownership
```

Requested `normal(mean=5, stddev=2)` over 100000 samples measured
`mean=4.989133 stddev=2.002836`. The RNG reproducibility test is what backs the
reproducibility claim in [../correctness.md](../correctness.md); it was an
assertion until now.

### Operator suites — unchanged, still stubs

| Suite | Passed | Failed | Skipped |
|---|---|---|---|
| `ccml_test_matmul` | 2 | 0 | 6 |
| `ccml_test_layernorm` | 1 | 0 | 7 |
| `ccml_test_softmax` | 1 | 0 | 6 |
| `pytest` (Python binding) | 2 | 0 | 12 |

Every pass is argument validation. Every skip is an operator with no kernel
behind it.

## What this proves

- A kernel launches on a real GPU and produces correct values, verified
  element by element across 10000 elements with a count that is **not** a
  multiple of the block size.
- Two kernels chained through device memory produce the right result.
- Stream-ordered copy → launch → copy, waited on with `ccml_stream_synchronize`
  rather than a device-wide barrier, is correct.
- Launch validation rejects oversized blocks, empty blocks, zero grids and
  excess shared memory, against **limits read from the device at runtime**, and
  reports which dimension and which limit.
- A rejected launch leaves the device usable.
- CUDA event timing returns a plausible non-zero duration for real GPU work.
- A CUDA failure reaches the caller with its runtime message intact through a
  flat C ABI.
- RAII move semantics empty the moved-from object, so no double free or double
  destroy.

## What this does NOT prove

- **No operator works.** All three still return `CCML_ERROR_NOT_IMPLEMENTED`.
  Nothing here is evidence about matmul, layernorm or softmax.
- **Nothing about performance.** The one timing figure (0.629920 ms) is a
  deliberately wasteful busy-loop kernel written to give the timer something
  longer than its own resolution to measure. It is not a benchmark of anything.
- **Nothing about the Ada target.** This ran on Turing (sm_75). Launch
  validation reads real device limits at runtime, so it is architecture-correct
  wherever it runs, but the recorded values — 1024 threads per block, 49152
  bytes of shared memory — are the T4's.
- **Nothing about Windows or MSVC.** GCC and MSVC `/W4` warn about different
  things; a clean build here does not imply a clean build there.
- **Execution faults are not covered.** The Release build checks only that a
  launch was *accepted*. Faults raised during execution surface at the next
  synchronisation. `CCML_SYNC_LAUNCH_CHECKS` exists for that and is on in Debug,
  but this run was Release, so it was not exercised.

## Issues found and resolved

**1. Hidden visibility blocked the white-box test.** The library is built with
`CXX_VISIBILITY_PRESET hidden`, so `ccml::detail::set_last_error` was not
exported and `tests/test_infrastructure.cu` could not link against it.

Rather than export internals, `src/common/check.h` was made header-only using
C++17 inline variables, which also removed code from `status.cpp`. Consequence,
documented in both files: the error detail buffer is per-binary. Library code
writes to the library's buffer, read by the exported `ccml_last_error_string()`;
a white-box test that calls the helpers directly gets its own, read via
`ccml::detail::last_error()`. Only white-box tests ever see both.

**2. The first Kaggle run could not deliver its own log.** The build tree was
being written inside `/kaggle/working`, everything under which becomes kernel
output. A CMake build tree is enough files that `kaggle kernels output` timed
out before reaching the log. The build moved to `/tmp/ccml-work`, leaving only
`ccml_build_log.txt` as output.

**3. The source payload was growing on every push.** `cloud/kaggle/kaggle_run.py`
was packing its own `_output/` directory from the previous run, taking the
payload from 68 KB to 143 KB. `_build` and `_output` were added to the exclusion
set.

No issue was found in the infrastructure code itself: it compiled clean and
passed on the first GPU run.

## Reproducing this

```
python cloud/kaggle/kaggle_run.py
```

Pushes, waits, and prints the log. See [../../cloud/kaggle/README.md](../../cloud/kaggle/README.md).

For the Ada target instead, once Modal billing is set up:

```
CCML_GPU=L4 modal run cloud/modal_app.py
```
