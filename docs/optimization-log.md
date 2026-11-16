# Optimization log

One entry per optimization step (reference.md §3: "one optimization at a
time... record speedup after each step"; §9: Nsight Compute evidence for each
major optimization). Each entry is written in this order, and the hypothesis is
committed **before** the code:

1. **Hypothesis**: the mechanism, and a falsifiable predicted effect.
2. **Change**: what was built, and where.
3. **Evidence**: timings (Kaggle T4, `results/sm_75/`) and Nsight Compute
   metrics (local sm_89, `profiles/`; see docs/local.md L3). The GPU is named
   for every number.
4. **Verdict**: kept, reverted or revised. A prediction that failed stays in
   the log with its explanation.

---

## 1. Matmul: register blocking (`CCML_MATMUL_REGISTER_TILED`)

**Status:** implemented and correct (all dtypes, all tile configurations).
Local sm_89 evidence below supports predictions 1-3. The official T4 timings
are pending (Kaggle, `results/sm_75/matmul.csv`).

### Hypothesis

The shared-memory tiled kernel (16×16 tile, one output per thread) is limited
by **memory traffic per FLOP at two levels**:

- **Global → shared:** a block loads `(16 + 16) × 16` floats (2 KiB) to do
  `2 × 16³ = 8192` FLOPs, so its arithmetic intensity is **4 FLOP/byte**. The
  T4's ridge point is about 8.1 TFLOPS / 320 GB/s ≈ **25 FLOP/byte**, so the
  kernel is memory-bound by design and capped near 4/25 ≈ 16% of the FP32 peak
  (docs/foundations.md §4.2). It was measured at **8.6%** of peak in the case
  study (results/phase-6-case-study.md).
- **Shared → registers:** every FMA reads two operands from shared memory, a
  ratio of 2 shared-memory loads per FMA.

Register blocking gives each thread a **TM × TN = 8 × 8** patch of C, inside a
**BM × BN × BK = 128 × 128 × 8** block tile with 256 threads:

- **Global:** a block loads `(128 + 128) × 8` floats (8 KiB) for
  `2 × 128 × 128 × 8 = 262,144` FLOPs, which is **32 FLOP/byte**, above the
  T4's ridge point. The kernel moves from memory-bound to compute-bound in
  principle.
- **Shared:** per k-step a thread loads `TM + TN = 16` values and performs
  `TM × TN = 64` FMAs: **0.25 loads per FMA**, 8× fewer than now.

**Predictions** (falsifiable, judged on the Kaggle T4, FP32, warm L2):

1. On large shapes (M ≥ 512, N and K ≥ 768), REGISTER_TILED is **3–6× faster
   than TILED**, reaching **30–50% of the theoretical FP32 peak**. It stays
   below cuBLAS (no vectorized loads, no double buffering yet; those are the
   Phase 10 steps).
2. On decode shapes (M ≤ 16), a 128-row block tile wastes ≥ 87% of its rows,
   so the 128×128 configuration is **no faster, possibly slower** than TILED.
   The tile sweep should show a smaller BM winning there. That would be
   evidence for shape-dependent dispatch (enhancement S4), not a failure.
3. Nsight Compute (local sm_89) should show **shared-memory load instructions
   per FMA dropping roughly 8×**, and the top warp-stall reason moving from
   memory dependency (`long_scoreboard` / `mio_throttle`) towards execution or
   barrier stalls.

**What would falsify it:** large-shape speedup under 2×, or NCU showing that
the kernel is still bound by shared-memory throughput. Either would point at
bank conflicts in the transposed A tile, or at register spilling (each thread
holds 64 accumulators plus 16 operands, about 100 registers, which caps
occupancy at roughly 2 blocks/SM).

### Change

- `src/matmul/matmul_kernels.cuh`: `matmul_register_tiled<T, BM, BN, BK, TM, TN>`,
  templated on storage type and tile shape. A is staged transposed and B
  row-major; there are no vectorized loads and no double buffering, so this step
  changes one thing.
- Six tile configurations (`kMatmulTileConfigs`). The algo enum
  `CCML_MATMUL_REGISTER_TILED` uses configuration 0 (128x128x8, 8x8). `AUTO`
  still resolves to `TILED`: it switches only on the T4 measurement.
- `include/ccml/tuning.h`: per-configuration entry points for the tile sweep
  (reference.md section 5); `ccml_bench_matmul --tile-sweep`.
- Tests: every configuration on 11 awkward shapes (M, N or K = 1; nothing a
  multiple of any tile), FP32, FP16 and BF16. All pass. Worst normalized error
  is 0.09-0.13 of the tolerance.

### Evidence

**Local RTX 3050 A Laptop, sm_89 (indicative only; the target is the T4).**
FP32, warm L2, median of 200; measured peak 8.71 TFLOPS
(`results/sm_89/register_tiling_*.csv`, `peaks_register_tiling.json`):

| Shape | tiled | register_tiled (cfg 0) | best tile config | cuBLAS |
|---|---|---|---|---|
| gpt2m ffn_up 2048x4096x1024 | 23,536 us | **4,956 us (4.75x), 39.8% of peak** | cfg 0 | 3,595 us (54.8%) |
| gpt2s ffn_up 512x3072x768 | 3,270 us | **763 us (4.29x), 36.3%** | cfg 0 | 433 us (64.0%) |
| gpt2s qkv 128x2304x768 | 624 us | 194 us (3.2x) | 128x64 cfg 1: 183 us | 101 us |
| gpt2s qkv 16x2304x768 | 90 us | 172 us (**0.52x**) | 32x64x16 cfg 4: 70 us | 40 us |
| gpt2s qkv 1x2304x768 | 89 us | 171 us (**0.52x**) | 32x64x16 cfg 4: 63 us (naive: 35 us) | 16 us |

**Nsight Compute, local sm_89** (`profiles/sm_89/matmul/*.ncu-rep`, metrics in
the `_metrics.csv` files next to them), gpt2s ffn_up 512x3072x768:

| Metric | tiled | register_tiled | |
|---|---|---|---|
| Duration (under the profiler) | 3.78 ms | 0.87 ms | 4.3x |
| Shared-load wavefronts | 56.6 M | 11.8 M | 4.8x fewer |
| Shared-memory bank conflicts (load / store) | 0 / 0 | 4.7 M / 2.1 M | new |
| Stall: MIO throttle (per issue) | 18.2 | 0.33 | the bottleneck is gone |
| Stall: barrier (per issue) | 5.8 | 0.73 | |
| Registers per thread | 37 | 126 | |
| Occupancy limit from registers | 6 blocks/SM | 2 blocks/SM | |
| Warps active | 99% | 31% | the new limiter |

### Verdict

**Kept (provisionally, pending the T4 run).**

- Prediction 1 (3-6x on large shapes, 30-50% of peak): **met** on sm_89, with
  4.3-4.75x and 36-40%.
- Prediction 2 (no gain at M <= 16, and a smaller BM wins): **met**. The
  128x128 tile is 0.52x of tiled; the 32x64x16 configuration is the best ccml
  kernel at M = 16, while at M = 1 the naive kernel still beats every tiled
  one. Decode shapes need their own path: per-shape dispatch (S4), and later
  split-K (S2).
- Prediction 3 (shared loads per FMA down ~8x; the top stall moves off the
  memory pipe): **partly met.** MIO throttle stalls fell 55x, but shared-load
  wavefronts fell only 4.8x. NCU names the reason: 4-way bank conflicts on the
  B-operand reads (16 threads at a stride of TN = 8 floats). The register
  budget (126 per thread) also limits occupancy to 31% of warps.

**Next steps, chosen from this evidence:** Phase 10's vectorized `float4`
shared-memory loads remove the strided B reads (the conflicts), and a
register-lighter variant raises occupancy. `AUTO` moves to per-shape dispatch
once the T4 tile sweep lands.
