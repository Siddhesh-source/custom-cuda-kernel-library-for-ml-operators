# Benchmarking

This is the measurement method. It implements the "Benchmarking method" section
and §3, §4, §6 and §11 of [`reference.md`](reference.md). The extra rigor
measures (M1–M7) are described in [enhancements.md](enhancements.md).

**State on 2026-11-16:** the harness in `bench/` predates this method. It
reports the mean over 50 samples and has no throughput columns. Bringing it to
this standard is Phase 7 ([ROADMAP.md](ROADMAP.md)). Until then, no number from
`bench/` may be reported.

## Where it runs

On a **Kaggle Tesla T4 (sm_75)**, through `cloud/kaggle/kaggle_run.py`, with
the machine shape pinned to T4. Every row records the GPU, compute capability,
driver, CUDA version, SM and memory clocks, temperature and throttle reasons.
Clocks cannot be locked on Kaggle (`nvidia-smi -lgc` is denied, as measured by
the probe on 2026-11-16). The harness therefore records clocks per row and
flags throttled samples, rather than assuming the clocks are fixed.

## Rules [reference.md, Method]

1. **Correctness first.** The harness runs the correctness check for a kernel
   and shape in the same process before timing it, and refuses to time a kernel
   that fails (M7).
2. **CUDA events.** Warmup is at least 20 launches, then at least 100 timed
   iterations (default 200), each bracketed by events. Wall-clock time without
   synchronization is never used.
3. **Statistics.** Report the median, p10, p90, IQR and min, plus a bootstrap
   95% CI on the median (M1).
4. **Identical conditions.** Same GPU, dtype, shapes and session for everything
   in one comparison. Variants run interleaved A/B/A/B to cancel drift (M3).
   The harness checks that no other process is on the GPU (M4).
5. **Cold and warm L2.** Both are reported. Cold flushes L2 before each
   iteration by writing a buffer larger than L2 (M2).
6. **Percent of peak.** TFLOPS for matmul, GB/s for row-wise ops. Each is given
   as a percentage of the theoretical peak and of the measured peak
   (`results/sm_75/peaks.json`, M5).
7. **Every CUDA call is checked**, and `cudaGetLastError` follows every launch.

## Event-timing pattern (from the spec)

```cpp
cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
for (int i = 0; i < WARMUP; ++i) kernel<<<g, blk>>>(args);
cudaEventRecord(a);
for (int i = 0; i < ITERS; ++i) kernel<<<g, blk>>>(args);
cudaEventRecord(b); cudaEventSynchronize(b);
float ms; cudaEventElapsedTime(&ms, a, b);   // per-iter = ms / ITERS
```

This batched form gives the mean per iteration. To get the median and spread,
the harness also records one event pair per iteration and computes the
distribution from those. Both forms are kept so they can be cross-checked.

## Baselines [reference.md §4]

| Baseline | Where |
|---|---|
| Naive CUDA kernel | `algo = NAIVE` in the same harness |
| PyTorch eager | `scripts/bench_torch.py` |
| cuBLAS / cuDNN | `bench/` C++, linked only into benchmarks, never into `src/` |
| torch.compile (Inductor) | `scripts/bench_torch.py`, `mode="max-autotune"` |
| Triton (extra, B1) | `scripts/bench_torch.py` |
| Our best kernel per shape | AUTO dispatch from the tuning table |

## Metrics [reference.md §6]

Latency (µs) · TFLOPS and %peak (matmul) · GB/s and %peak bandwidth
(layernorm, softmax) · speedup against each baseline · max abs and rel error ·
Nsight Compute: achieved occupancy, memory throughput, warp stall reasons.

**FLOP and byte counts** (the minimum required traffic, so that %peak is never
inflated):
- matmul: `2·M·N·K` FLOPs; bytes = `(M·K + K·N + M·N)·sizeof(T)`
- layernorm: bytes = `(2·rows·cols + 2·cols)·sizeof(T)`, plus a residual read when fused
- softmax: bytes = `2·rows·cols·sizeof(T)`, plus a mask read when fused

## CSV schema

`results/sm_75/<op>.csv`, one row per (op, algo, dtype, shape, baseline, cache state):

```
op,impl,algo,dtype,M,N,K,rows,cols,cache,iters,median_us,p10_us,p90_us,iqr_us,min_us,ci95_lo_us,ci95_hi_us,tflops,gbps,pct_peak_theory,pct_peak_measured,max_abs_err,max_rel_err,sm_clock_mhz,mem_clock_mhz,temp_c,throttled,gpu,cc,driver,cuda,commit
```

An empty timing cell means the work did not run. It is never written as zero.

## Profiling [reference.md, Method and §9]

Nsight Systems is used for timelines and Nsight Compute for kernel metrics.
Only small reports are committed, under `profiles/`. Nsight Systems runs on
Kaggle. Nsight Compute counters are blocked there (`ERR_NVGPUCTRPERM`), so NCU
runs locally as described in [local.md](local.md) L3.
