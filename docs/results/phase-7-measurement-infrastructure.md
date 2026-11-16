# Phase 7: measurement infrastructure

**Dates:** 2026-11-16 to 2026-11-17
**Branch:** `measurement-infrastructure`
**Outcome:** the spec-compliant measurement instrument is built, tested and
validated on a Tesla T4, and has produced the baseline every later phase is
compared against. Correctness tests 6/6, Python suites 42/42, and
compute-sanitizer clean on all 20 suite/tool pairs (one deliberate,
recognized API error). The first full benchmark run exposed **three
measurement defects**; all three were fixed and the benchmarks re-run.

## Environment

Kaggle GPU kernels, one session per job (`sanitize`, `matmul`, `layernorm`,
`softmax`), from `results/sm_75/env_*.txt`:

| | |
|---|---|
| GPU | Tesla T4, compute capability **7.5**, 15360 MiB, 70 W power limit |
| Max clocks | SM 1590 MHz, memory 5001 MHz |
| Driver | 580.178.04 |
| CUDA toolkit | 13.0 (V13.0.88) |
| Tools | ncu 2025.3.1, nsys 2025.3.1, compute-sanitizer 2025.3.1, CMake 3.31.10, Ninja |
| Python | `/usr/bin/python3`: torch 2.11.0+cu130, triton 3.6.0, numpy 2.1.3, pandas 2.3.3, matplotlib 3.10.0, pynvml |
| Build | Release, sm_75, `-DCCML_WARNINGS_AS_ERRORS=ON`, cuDNN from the PyTorch wheel |
| Commits measured | `d8c676b` (tests, sanitizer; first benchmark run), `835f070` (benchmark re-run) |

The library code (`src/`, `include/`) is identical at both commits. Only the
measurement harness changed between them.

## What was added

| Requirement (reference.md) | Implementation |
|---|---|
| Setup: tool checks, venv, packages | `scripts/setup_env.sh` |
| `scripts/`, `profiles/` | Created; small profiler reports allowed in git |
| Target-shape list (§3) | `bench/shapes.json` from `scripts/make_shapes.py`: 76 matmul, 38 layernorm, 32 softmax shapes, odd shapes included |
| CUDA events, warmup, median + spread of 100+ (Method) | `bench/support/timer.cpp`, `stats.cpp`: per-iteration events, warmup 20, 100-200 iterations, median, p10, p90, IQR, min, bootstrap 95% CI; batched-mean cross-check |
| Same GPU/clocks, watch throttling (Method, §11) | NVML telemetry per row; one session per operator, so every comparison is on one GPU |
| TFLOPS, GB/s, % of peak (§3, §6) | `bench/support/metrics.cpp`; theoretical and measured peaks (`ccml_bench_peaks`) |
| Correctness before speed (§11) | In-run fp64 gate: a candidate that fails is never timed |
| Vendor baselines (§4) | cuBLAS, cuBLASLt (`bench/baselines/cublas.cpp`), cuDNN softmax (`cudnn_softmax.cpp`); PyTorch eager, torch.compile, Triton (`scripts/bench_torch.py`) |
| compute-sanitizer memcheck + racecheck (Method) | `scripts/sanitize.sh`, plus initcheck and synccheck |
| CSV output (§3) | `results/sm_75/<op>.csv` and `torch_<op>.csv`, one schema |

### Tests written in this phase

- `tests/test_bench_stats.cpp`: percentiles (checked against numpy's
  definition), median, summary fields, bootstrap determinism, the iteration
  floor of 100, the shape reader (including malformed input), FLOP and byte
  counts, the T4's published peaks, throttle classification, and the rule that
  a failed gate carries no timing.
- `scripts/tests/test_benchlib.py`: the same checks for the Python mirror, so
  the two languages cannot drift apart.

## Correctness and sanitizer results (verbatim)

`ctest` (`results/sm_75/ctest.txt`):

```
1/6 Test #1: infrastructure ...................   Passed    0.57 sec
2/6 Test #2: harness ..........................   Passed    0.46 sec
3/6 Test #3: matmul ...........................   Passed    1.21 sec
4/6 Test #4: layernorm ........................   Passed    0.99 sec
5/6 Test #5: softmax ..........................   Passed    1.00 sec
6/6 Test #6: bench_stats ......................   Passed    0.03 sec

100% tests passed, 0 tests failed out of 6
```

Python (`pytest_benchlib.txt`, `pytest_binding.txt`): `7 passed`, and `35
passed`. The binding suite includes the FP32 comparison against PyTorch in
float64 on the T4.

compute-sanitizer (`results/sm_75/sanitizer/summary.txt`, logs alongside):

```
suite            tool       exit     result
infrastructure   memcheck   99       ========= ERROR SUMMARY: 1 error
infrastructure   racecheck  0        ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
infrastructure   initcheck  0        ========= ERROR SUMMARY: 0 errors
infrastructure   synccheck  0        ========= ERROR SUMMARY: 0 errors
harness          memcheck   0        ========= ERROR SUMMARY: 0 errors
harness          racecheck  0        ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
harness          initcheck  0        ========= ERROR SUMMARY: 0 errors
harness          synccheck  0        ========= ERROR SUMMARY: 0 errors
matmul           memcheck   0        ========= ERROR SUMMARY: 0 errors
matmul           racecheck  0        ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
matmul           initcheck  0        ========= ERROR SUMMARY: 0 errors
matmul           synccheck  0        ========= ERROR SUMMARY: 0 errors
layernorm        memcheck   0        ========= ERROR SUMMARY: 0 errors
layernorm        racecheck  0        ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
layernorm        initcheck  0        ========= ERROR SUMMARY: 0 errors
layernorm        synccheck  0        ========= ERROR SUMMARY: 0 errors
softmax          memcheck   0        ========= ERROR SUMMARY: 0 errors
softmax          racecheck  0        ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
softmax          initcheck  0        ========= ERROR SUMMARY: 0 errors
softmax          synccheck  0        ========= ERROR SUMMARY: 0 errors
```

**The one memcheck error is deliberate.** `infrastructure_memcheck.log`
shows it is `cudaErrorMemoryAllocation (error 2) due to "out of memory" on
CUDA API call to cudaMalloc`, raised inside the test
`cuda_failures_carry_the_runtime_message`. That test requests an impossible
allocation on purpose, to check that CUDA's own message reaches the caller,
and the test passes. memcheck counts every failed API call as an error. This
is not a memory-safety defect.
`scripts/sanitize.sh` now accepts exactly this error (sole error, that test,
that API) and still fails on anything else.

**This is the first time `compute-sanitizer` has run on this library.** Every
barrier in the tiled matmul and in the shared block reduction used by
layernorm and softmax is now machine-checked: racecheck reports 0 hazards.

## Benchmark results

Re-run at commit `835f070`: one Kaggle session per operator, so every
comparison below is on one physical T4. FP32, warm L2, median of 100-200
guarded per-iteration CUDA-event timings. Raw data: `results/sm_75/<op>.csv`
(C++: ccml, cuBLAS, cuBLASLt, cuDNN) and `torch_<op>.csv` (PyTorch eager,
torch.compile, Triton). Measured peaks for each session are in
`peaks_<op>.json`: 7.8-8.1 TFLOPS FP32 and 227 GB/s.

**Correctness gate:** every baseline row PASS. Our NOT_RUN rows are the
reserved variants (Welford, warp-per-row), and naive matmul above 5 GFLOP per
call; neither is timed. The 9 Triton failures of the first run are gone.

This is the **baseline before any optimization phase**: matmul here is only
`naive` and `tiled`. Register blocking and vectorized loads (Phases 9-10) are
measured on their own branch.

### Median % of measured peak (warm L2)

| Operator | ours | cuBLAS / cuDNN | PyTorch eager | torch.compile | Triton |
|---|---|---|---|---|---|
| matmul (TFLOPS) | tiled **6.9%**, naive 4.9% | cuBLAS 41.2%, cuBLASLt 42.2% | 42.6% | 42.7% (default mode = cuBLAS) | 35.8% |
| layernorm (GB/s) | two_pass **13.6%** | (no legacy cuDNN layernorm) | 32.8% | 25.4% | 45.4% |
| softmax (GB/s) | block_per_row **22.3%** | cuDNN 96.8% | 79.9% | 43.3% | 69.1% |

### Our best kernel against the best baseline, shape by shape

| Operator | Shapes | Median speed of ours relative to the best baseline | Shapes where ours is faster |
|---|---|---|---|
| matmul | 76 | **0.17x** | 2 (odd 1x1x1: 1.08x, odd 33x33x33: 1.46x; both tiny) |
| layernorm | 38 | **0.31x** | 0 |
| softmax | 32 | **0.23x** | 0 |

That is the honest starting point the optimization phases work from. The
reference.md section 9 criterion ("at least one operator beats the vendor
baseline on a target shape") is not met yet: the two matmul wins are odd
shapes, not target shapes.

### What the launch-gap fix changed

In the biased first run, our layernorm appeared faster than the best
baseline on **12 of 38** shapes, all of them small. With the guard, it is
faster on **0 of 38**. Those "wins" were the Python baselines' submission time
being counted as kernel time. This is why issue 6 below was fixed before any
number was reported.

### The T4 runs at its power cap under sustained load

The telemetry, sampled mid-run, shows the SM clock falling with kernel
duration. The NVML reason is `0x4` (software power cap, 70 W), not thermal or
hardware throttling:

| Kernel duration | Rows below 1300 MHz | Median SM clock |
|---|---|---|
| < 30 us | 10% | 1590 MHz |
| 30-100 us | 8% | 1590 MHz |
| 0.1-1 ms | 42% | 1380 MHz |
| > 1 ms | 89% | **945 MHz** |

So a long, heavy kernel on a T4 runs at about 60% of the boost clock that the
theoretical peak assumes. Percent-of-peak figures for long kernels include
this power-cap loss for **every** implementation alike. The `sm_clock_mhz`
column allows a clock-normalized view where that matters. The temperature
range was 57-81 °C and no row was thermally throttled.

### Timing-method agreement

The median ratio of batched mean to guarded per-iteration median is 0.98-1.07
in every operator, for C++ and Python alike. Where the batched mean is much
larger (Python layernorm, 75th percentile 2.9x), that is expected: the
batched pattern has no guard, so it measures launch throughput, which for tiny
kernels is bounded by the CPU. The guarded median is the GPU time, and that is
what the tables report.

## Issues found and how they were resolved

Each of these was found by looking at the first run's own output, not assumed.

| # | Symptom in the data | Cause | Fix |
|---|---|---|---|
| 1 | "Measured" T4 peak 3.2 TFLOPS, while cuBLAS reached 6.1 in the same session | The peak microbenchmark warmed up for 3 launches (~20 ms); an idle T4 sits at 300 MHz and ramps over hundreds of ms | 1.5 s GPU-time warmup; SM clock recorded during the measurement. Now 7.81-8.11 TFLOPS (96-99.6% of theoretical) |
| 2 | Telemetry columns empty | Kaggle's image ships only the driver's `libnvidia-ml.so.1`, not the toolkit stub `CUDA::nvml` expects | CMake finds the versioned driver library |
| 3 | Python baselines missing | `python3 -m venv` on Kaggle cannot bootstrap pip, so the venv lacked torch | Use the interpreter that has CUDA PyTorch |
| 4 | Laptop "measured" peak above "theoretical" | CUDA's clock attribute (2205 MHz) is below the GPU's real boost (2550-3105 MHz) | Theoretical peak uses NVML's max boost clock when higher |
| 5 | One job ran code from another branch | The launcher packed the working tree at push time, after edits had started elsewhere | Payloads are snapshotted once, at launch |
| 6 | **Per-iteration medians up to 2.1x the batched mean** for Python kernels under 30 us (C++ under 10 us: 1.3x) | Launch-gap bias: the GPU executed the start event, then idled inside the timed window while the CPU submitted the kernel | A spin kernel before each timed iteration keeps the GPU busy during submission (`bench/support/spin.cu`; `torch.cuda._sleep` in Python) |
| 7 | 521 rows with SM clock 585-1200 MHz, none flagged as throttled | Clocks were sampled after the timing loop, on an already-idle GPU | Sampled per candidate while its last round executes |
| 8 | 9 Triton softmax rows FAILED the correctness gate (non-finite) | Bug in our Triton baseline: lanes past the row end computed `exp(-inf - -inf)` = NaN | Masked lanes contribute exactly 0. **The gate did its job:** none of the 9 rows was timed |
| 9 | Sanitizer summary reported 1 memcheck error | The deliberate OOM test above | Recognized precisely, as described |

Issue 6 matters most. Without the fix, small-shape comparisons were biased
against every Python baseline, which would have made several of our kernels
look faster than they are on small inputs. The affected run is kept, clearly
marked, in `results/sm_75/superseded_run1/`.

## What this phase proves, and what it does not

**Proves:**

- Every timed number in `results/sm_75/` comes from a kernel that passed an
  fp64 correctness check in the same process.
- The kernels are free of memory errors and data races on the tested shapes
  (sanitizer, T4).
- The measured peaks are credible: within 0.4-4% of theoretical FP32, and 71%
  of theoretical bandwidth (the typical achievable fraction for GDDR6).

**Does not prove:**

- That the clocks were identical across sessions. They cannot be locked on
  Kaggle. Each row records its own clock, temperature and throttle state, and
  comparisons are only made within one session.
- Anything about Ada (sm_89). This is Turing data only.
- Anything about end-to-end model latency. That is Phase 14.

## How to reproduce

    python cloud/kaggle/kaggle_run.py --any-gpu      # four sessions; results collected into results/sm_75/
    bash scripts/sanitize.sh build/bin results/sm_75/sanitizer
