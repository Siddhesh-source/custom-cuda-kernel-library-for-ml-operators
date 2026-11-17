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

---

## 2. Matmul: vectorized loads and conflict-free shared memory (`CCML_MATMUL_VECTORIZED`)

**Status:** implemented and correct (all dtypes, every compatible tile
configuration, aligned and misaligned pointers). Local sm_89 evidence supports
all four predictions **after one correction** (below). The official T4 timings
are pending.

### Hypothesis

Entry 1's profile (`profiles/sm_89/matmul/`, gpt2s ffn_up 512x3072x768) names
two inefficiencies in the shared-memory traffic of REGISTER_TILED:

- **4.7 M load bank conflicts.** For the B operand, a warp's 16 thread
  columns read `Bs[k][thread_col * 8 + j]`: a stride of 8 floats. That lands
  them on only 4 of the 32 banks, so the reads serialize 4 ways.
- **2.1 M store bank conflicts.** When A is staged transposed into
  `As[k][m]`, rows that are BM = 128 floats apart (a multiple of 32) map to the
  same bank.

It also issues every access 32 bits at a time: 16 scalar shared loads per
k-step, and scalar global loads.

**Change:** four things, all serving one goal (fewer, wider, conflict-free
memory instructions):

1. **Global loads 128 bits wide** (`float4`, or 4 × 16-bit for FP16/BF16) for
   A and B, when K and N are multiples of 4 and the pointers are aligned.
   Otherwise the same kernel uses scalar global loads.
2. **Shared reads 128 bits wide:** a thread's TM values of A and TN values of
   B come in TM/4 and TN/4 vector loads instead of TM + TN scalar ones.
3. **B columns split into 4-wide chunks** spaced BN/2 apart, instead of 8
   contiguous columns per thread. Consecutive threads then read consecutive
   16-byte chunks, so a warp's B reads are conflict-free.
4. **A tile padded by 4 floats per row** (`As[BK][BM + 4]`). The transposed
   stores then spread over all 32 banks, and row starts stay 16-byte aligned
   for the vector reads.

**Predictions** (FP32, warm L2; T4 official, sm_89 indicative):

1. Shared-memory bank conflicts (load and store) fall **to near zero**
   (from 4.7 M and 2.1 M) on the profiled shape.
2. Shared-load wavefronts fall **from 11.8 M to about 6 M or fewer**.
3. Large shapes get **1.2–1.6x faster than REGISTER_TILED**, reaching
   **45–60% of peak**. Occupancy is unchanged (registers are similar, about
   2 blocks/SM), so the gain comes only from fewer memory instructions.
4. Decode shapes (M <= 16) **barely change**: they are limited by idle block
   rows, not by load width.

**What would falsify it:** conflicts remaining, which would mean the chunked
mapping is wrong, or no speedup despite fewer conflicts, which would mean the
kernel is now limited by occupancy (31% of warps active) rather than memory
instructions. Either result decides whether the next step attacks occupancy or
latency (double buffering).

### Change

- `matmul_vectorized<T, BM, BN, BK, TM, TN, kVec>` in
  `src/matmul/matmul_kernels.cuh`; `CCML_MATMUL_VECTORIZED`, configuration 0.
  The host picks `kVec` (128-bit global access) when K and N are multiples of
  4 and A, B and C are aligned; otherwise the same kernel runs with scalar
  global access. Configuration 5 (TM = 2) is rejected for this variant.
- The tuning API became `ccml_matmul_tuned_<dtype>(..., algo, config, ...)` for
  the whole tiled family; the bench sweep names rows `rt_*` and `vec_*`.
- Tests: every compatible configuration on 15 shapes (4 of them with N and K
  multiples of 4, so the vector path is exercised); misaligned views
  (pointers offset by one element); FP16 and BF16. All pass.

**Correction during the step: an occupancy floor.** The first version was
*slower* than REGISTER_TILED on the largest shape. NCU showed why: the vector
temporaries raised register use from 126 to **155 per thread**, so only **1
block per SM** fit and active warps halved (33% to 16.7%). Adding
`__launch_bounds__(threads, 2)` (at least 2 resident blocks) brought it to
**127 registers with no spills** (`cuobjdump --dump-resource-usage`: `LOCAL:0`).
This is the falsification case the hypothesis named ("no speedup despite fewer
conflicts means occupancy is the limiter"), confirmed and then fixed. Both
measurements are kept below.

### Evidence

**Nsight Compute, local sm_89, before the occupancy floor**
(`profiles/sm_89/matmul/register_tiled_vs_vectorized_*.ncu-rep` and the
`_metrics.csv` files next to them):

| Metric | register_tiled | vectorized (155 regs) | Prediction |
|---|---|---|---|
| Shared-load bank conflicts, 2048x4096x1024 | 33.5 M | **0** | near zero ✅ |
| Shared-store bank conflicts, same shape | 14.7 M | 0.56 M | near zero ✅ |
| Shared-load wavefronts, same shape | 84.1 M | 50.3 M | ~half ✅ (-40%) |
| Global load requests, same shape | 4.19 M | 1.05 M | 4x fewer ✅ |
| Shared-load bank conflicts, 512x3072x768 | 4.72 M | **0** | ✅ |
| MIO-throttle stall (per issue) | 0.28 | 0.016 | |
| Registers per thread | 126 | **155** | not anticipated |
| Blocks per SM (register limit) | 2 | **1** | |
| Warps active | 33% | **16.7%** | |

**Timings, local RTX 3050 A Laptop sm_89 (indicative), FP32, warm L2, median
of 200, candidates interleaved in one run**
(`results/sm_89/vectorized_*.csv`; the pre-fix run is in
`vectorized_before_bound_*.csv`):

| Shape | register_tiled | vectorized, 155 regs | **vectorized, 127 regs** | cuBLAS |
|---|---|---|---|---|
| 2048x4096x1024 | 4,990-5,330 us * | 5,363 us (0.93x) | **4,355 us (1.22x), 45.3% of peak** | 3,911 us (50.4%) |
| 512x3072x768 | 762-764 us | 686 us (1.11x) | **570 us (1.34x), 48.6%** | 458 us (60.5%) |
| 16x2304x768 | 172-199 us | 170 us | 159 us; best config 32x64x16: **49 us** | 40 us |

\* The same kernel varied about 7% between separate runs on the laptop
(thermal and power state). This is why each comparison is made within one
interleaved run, and why the official numbers come from the T4.

### Verdict

**Kept**, with the occupancy floor.

1. Bank conflicts to near zero: **met** (load 0, store down 96%).
2. Shared-load wavefronts about halved: **met** (-40%).
3. 1.2-1.6x over REGISTER_TILED, 45-60% of peak, on large shapes: **met
   only after the occupancy fix** (1.22-1.34x, 45-49%). Without it, 0.93-1.11x.
   The lesson: a memory optimization that costs registers can lose more
   occupancy than it saves, and NCU's register and warps-active counters are
   the check.
4. Decode shapes barely change: **met** for configuration 0. The small
   32x64x16 tile remains the right choice there, which per-shape dispatch (S4)
   will use.

**Gap to cuBLAS** on the largest shape is now 1.11x (local). The next step is
double buffering (entry 3): overlap the next tile's global loads with the
current tile's math, since `long_scoreboard` (waiting on global memory) is now
the largest remaining stall (1.50 per issue).

---

## 3. Matmul: double buffering (`CCML_MATMUL_DOUBLE_BUFFERED`)

**Status:** implemented and correct; local sm_89 evidence meets predictions
1-3. The official T4 timings are pending.

### Hypothesis

After entry 2, the largest remaining stall of VECTORIZED is
**`long_scoreboard`: waiting for global memory** (1.50 per issue on the
2048x4096x1024 profile). The cause is structural. Each k-step runs

    load tile k from global -> store to shared -> barrier -> compute tile k -> barrier

so while a block waits for its loads, it computes nothing. With only 2
resident blocks per SM (the register budget), there is too little other work
to hide that wait.

**Change (register prefetch, works on sm_75):** two shared-memory buffers.
While computing tile k from buffer `k % 2`, each thread has already *issued*
the global loads for tile k+1 into registers. Those loads are in flight during
the math. After the math, it stores them into buffer `(k + 1) % 2`. One barrier
per k-step instead of two.

(The `cp.async` form, a direct global-to-shared copy without registers, needs
sm_80+ and is item L2 in docs/local.md.)

**Predictions** (FP32, warm L2; T4 official, sm_89 indicative):

1. `long_scoreboard` stalls per issue **fall by at least a third**, and barrier
   stalls fall (one barrier per step instead of two).
2. Large shapes get **1.05-1.20x faster than VECTORIZED**. The gain is capped
   because the prefetch registers (2 float4 per thread in the default config)
   push against the 128-register occupancy floor.
3. Decode shapes (M <= 16) do not change measurably.

**What would falsify it:** no stall reduction, which would mean the loads are
not overlapping (the compiler sank them below the math), or register spills
(`LOCAL > 0` in `cuobjdump --dump-resource-usage`) erasing the gain. Either
result is recorded as is.

### Change

- `matmul_double_buffered<T, BM, BN, BK, TM, TN, kVec>`: VECTORIZED's layout,
  `As[2][BK][BM+4]` and `Bs[2][BK][BN]`, fetch / compute / stash, and one
  barrier per k-step. `CCML_MATMUL_DOUBLE_BUFFERED`, configuration 0; sweep
  rows `db_*`.
- 127 registers, **no spills** (`LOCAL:0`), 16.6 KB of shared memory, so
  occupancy is unchanged.
- Tests: every compatible configuration on 15 shapes; **every k-step count**
  (K = 1, 3, 8, 9, 16, 17, 24, 40, 63, 64, 65, 1000: one step, two, odd counts,
  ragged last step), which is where a single-barrier pipeline would break;
  misaligned views; FP16 and BF16. All pass.

### Evidence

**Nsight Compute, local sm_89**
(`profiles/sm_89/matmul/vectorized_vs_double_buffered_*`):

| Metric (2048x4096x1024 / 512x3072x768) | vectorized | double_buffered |
|---|---|---|
| Duration under profiler | 4.32 ms / 678 us | **3.54 ms / 511 us** |
| Stall: long scoreboard (per issue) | 1.46 / 1.20 | **0.022 / 0.026** (-98%) |
| Stall: barrier | 0.40 / 0.38 | 0.33 / 0.31 |
| Stall: MIO throttle | 0.09 / 0.07 | 0.33 / 0.28 (shared traffic is denser in time) |
| Top stall | — | **not_selected** 2.25 / 2.10 (warps ready, waiting for the scheduler) |
| FMA pipe active | 51% / 52% | **63% / 62%** |
| Registers, warps active | 128, 33% | 127, 33% |

**Timings, local sm_89 (indicative), FP32, warm L2, one interleaved run**
(`results/sm_89/double_buffered_*.csv`; the clock is recorded per row):

| Shape | vectorized | double_buffered | best db config | cuBLAS |
|---|---|---|---|---|
| 512x3072x768 (all 2550 MHz) | 572 us | **444 us (1.29x), 62.4% of peak** | cfg 0 | 437 us (63.4%) |
| 2048x4096x1024 | 3,928 us @ 2535 MHz | **3,349 us (1.17x)** @ 2535 MHz | cfg 0 | 3,517 us **@ 2220 MHz** * |
| 16x2304x768 (all 2565 MHz) | 138 us | 121 us | **32x64x16: 36.9 us** | 39.9 us |

\* The laptop's clock fell during the run, so the cuBLAS row ran 12% slower
than ours. Clock-normalized, cuBLAS is still ahead (about 3,080 us at 2535
MHz). **This is not a win and is not claimed as one.**

### Verdict

**Kept.**

1. long_scoreboard falls by at least a third, and barrier stalls fall:
   **met**, by far: -98% and -18%.
2. 1.05-1.20x over VECTORIZED: **met or exceeded** (1.17x and 1.29x on the
   large shapes).
3. Decode shapes do not change measurably: **not quite**. Configuration 0 went
   from 138 to 121 us. The decode-sized tile is the real story there: at
   **36.9 us it is 1.08x faster than cuBLAS on this GPU at identical clocks**,
   the first shape-level win of the project. It is indicative only until the
   T4 confirms it, and that is the official test.

**Where the kernel stands (sm_89, indicative):** at parity with cuBLAS on
512x3072x768 (444 vs 437 us), and about 8% behind on the largest shape after
clock correction. FMA-pipe utilization (63%) and occupancy (33%) are now the
limits. The remaining reference.md ladder step is Tensor Cores (Phase 11),
which changes the compute ceiling itself.
