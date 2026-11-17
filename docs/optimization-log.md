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

### Official T4 result (commit 12cb762, docs/results/phase-9-10-matmul-ladder.md)

tiled -> register_tiled, median per class: **5.02x large, 3.28x mid, 0.38x
decode**; 31.2% of peak on large shapes (prediction: 30-50%). All three
predictions hold on the target GPU.

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

### Official T4 result (commit 12cb762)

register_tiled -> vectorized, median per class: **1.43x large, 1.37x mid, 1.16x
decode**; 43.6% of peak on large shapes (prediction: 1.2-1.6x, 45-60%). The
speedup is met; the peak fraction lands just under the band on the T4, where
the 70 W power cap holds long kernels to ~900 MHz.

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

### Official T4 result (commit 12cb762)

vectorized -> double_buffered, median per class: **1.05x large, 1.22x mid,
1.35x decode** (prediction: 1.05-1.20x). On large shapes it reaches **44.6% of
peak, equal to cuBLAS's median**, and it beats cuBLAS on two target shapes
with the vendor kernel at a *higher* clock: llama1b attn_out 2048^3 (4,641 vs
5,345 us, 1.15x) and gpt2s ffn_up 2048x3072x768 (2,639 vs 2,826 us, 1.07x).
The sm_89 decode win did **not** transfer: on the T4, cuBLAS is 1.14x faster
at M = 16.

---

## 4. Matmul: Tensor Cores through WMMA (`CCML_MATMUL_WMMA`)

**Status:** implemented and correct (FP16 everywhere; BF16 on sm_80+, SKIP on
the T4; all 3 Tensor Core configurations). Local sm_89 evidence below; the
official T4 result is at the end of this entry.

### Hypothesis

Every rung so far computes FP16 and BF16 matmuls on the **FP32 CUDA cores**:
operands are widened to float on load and each thread issues one FFMA per
multiply-add. On the T4 that caps a 16-bit matmul at the FP32 peak, 8.1
TFLOPS, which is **12.5% of the 65 TFLOPS FP16 Tensor Core peak** (40 SMs x
1024 FLOP/clk x 1.59 GHz, FP32 accumulate; `ccml_bench_peaks` now measures it).
cuBLAS `GemmEx` FP16 uses the Tensor Cores; the local sm_89 harness run already
shows it **3.6x faster** than DOUBLE_BUFFERED f16 on gpt2s.qkv.m128 (36.9 vs
134 us), so the gap is the compute unit, not the tiling.

**Change:** a new kernel on the double-buffered tiling of entry 3, with the
inner product moved to Tensor Cores:

- `wmma::mma_sync` on 16x16x16 fragments, FP16 (or BF16) in, FP32 accumulate.
  One warp instruction does 4096 multiply-adds instead of 32.
- Operands stay 16-bit in shared memory (no widening): half the shared bytes
  per tile, and 128-bit global loads move 8 elements instead of 4.
- Block tile 128x128x32, 8 warps, warp tile 64x32 (4x2 accumulator
  fragments); register-prefetch double buffering as in entry 3; rows padded by
  8 elements so `load_matrix_sync` rows start on different banks.
- Epilogue through a per-warp 16x16 FP32 scratch tile (fragment layouts are
  opaque), rounded once to the storage type, bounds-checked, so M, N, K stay
  arbitrary.
- BF16 fragments exist only on sm_80+: the BF16 instantiation is compiled under
  an `__CUDA_ARCH__ >= 800` guard and returns `CCML_ERROR_NOT_IMPLEMENTED` on the
  T4 (tests report SKIP, never PASS). FP32 input returns NOT_IMPLEMENTED too:
  FP32 Tensor Cores (TF32) do not exist on sm_75 and rounding FP32 inputs to
  FP16 would break the f32 contract.

**Predictions** (FP16, warm L2; official on the T4, indicative on sm_89):

1. Large shapes (M >= 512, N and K >= 768): WMMA is **at least 3x faster than
   DOUBLE_BUFFERED f16** and reaches **40-70% of cuBLAS GemmEx FP16**, i.e.
   roughly **15-30% of the theoretical Tensor Core peak**. It stays below
   cuBLAS: `load_matrix_sync` from a padded row-major tile still conflicts in
   some banks, and there is no swizzle or `ldmatrix` control yet (that is the
   MMA_SYNC step).
2. Decode shapes (M <= 16): **no meaningful gain over the best CUDA-core
   configuration**. A 128-row tile wastes 87-99% of its rows, and at M = 1 the
   operation is memory-bound (each weight is used once), so the compute unit
   barely matters. Split-K (S2) is the step aimed there.
3. Nsight Compute (sm_89): the **tensor pipe becomes active and the FMA pipe
   falls to near idle**; executed instructions per FLOP fall **by at least
   5x**. The top stall moves to shared-memory/MIO or to waiting on global
   loads, which names the next step.

**What would falsify it:** a large-shape speedup under 2x over DOUBLE_BUFFERED
f16, which would mean the kernel is limited by something other than math
throughput (the global loads, the epilogue, or shared-memory conflicts on the
fragment loads); NCU decides which.

### Change

- `matmul_wmma<T, BM, BN, BK, WM, WN, kVec>` in `src/matmul/matmul_kernels.cuh`;
  `CCML_MATMUL_WMMA` uses Tensor Core configuration 0 (128x128x32, warp
  64x32). Three configurations (`ccml_matmul_tc_config_*` in
  `include/ccml/tuning.h`; bench rows `wmma_*`): 128x128x32 / 8 warps,
  64x64x32 / 4 warps, 16x128x32 / 4 warps.
- The host reads the architecture the loaded binary was compiled for
  (`cudaFuncAttributes::binaryVersion`) and returns NOT_IMPLEMENTED below
  sm_70 (FP16) or sm_80 (BF16), and always for FP32.
- sm_75 compile check (`-DCCML_CUDA_ARCHITECTURES=75`): FP16 configuration 0
  is 130 registers, 37.9 KB shared, `LOCAL:0`; the BF16 bodies are empty there.
- Tests: `tests/test_matmul_tc.cpp`, new suite `matmul_tc` (also added to
  `scripts/sanitize.sh`): every configuration on 17 awkward shapes, every
  k-step count (K = 1 ... 1000, vector and scalar paths), misaligned views
  (offsets 1, 2, 4 elements), target shapes through the enum, NaN-poisoned
  outputs, f32 refused, config API bounds. sm_89: 10/10 pass, FP16 and BF16;
  worst normalized error 0.50 (the final rounding to 16 bits dominates).

**Correction during the step: the occupancy floor again.** The first build gave
the BF16 instantiation of configuration 0 **132 registers** on sm_89 (FP16:
123), one block per SM instead of two, and BF16 ran **1.75x slower than
FP16** on the same code (2007 vs 1142 us, gpt2m.ffn_up.m2048;
`results/sm_89/wmma_before_bound_gpt2m.ffn_up.m2048.csv`). Entry 2's fix,
`__launch_bounds__(threads, 2)`, brings it to 128 registers with `LOCAL:0`
and 2007 -> 1444 us. A BF16 gap to FP16 remains (1444 vs 1138 us); it is not
explained yet, and the MMA_SYNC step, which controls fragment loads
explicitly, is where it is examined.

### Evidence

**Timings, local RTX 3050 A Laptop sm_89 (indicative), FP16, warm L2, median
of 200, candidates interleaved in one run per shape**
(`results/sm_89/wmma_*.csv`, `peaks_wmma.json`; table by
`python scripts/summarize_matmul.py --prev double_buffered --new wmma --dtype f16 "results/sm_89/wmma_gpt2*.csv"`).
The sm_89 Tensor Core ceiling is 512 FLOP/clk/SM (GeForce, FP32 accumulate):
`ccml_bench_peaks` measured **18.2 TFLOPS at 2550 MHz**, exactly that rate.

| Shape | double_buffered | **wmma** | best ccml row | cuBLAS GemmEx | wmma %peak (theory / measured) |
|---|---|---|---|---|---|
| 2048x4096x1024 | 3,324 us | **1,138 us (2.92x)** | wmma | 983 us * | 67.8% / 82.9% |
| 512x3072x768 | 525 us | **174 us (3.02x)** | wmma | 145 us | 62.4% / 76.2% |
| 128x2304x768 | 135 us | **48.1 us (2.81x)** | wmma | 36.9 us | 42.3% / 51.7% |
| 16x2304x768 | 124 us | 55.3 us | **wmma_64x64x32: 25.6 us** | 16.4 us | 4.6% / 5.6% |

\* cuBLAS ran at 2355 MHz against WMMA's 2505 MHz in that run; clock-normalized
it is about 924 us, so WMMA is at about **0.81x of cuBLAS** there, not 0.86x.

BF16 (sm_89 only; the T4 has no BF16 Tensor Cores): 2.20-2.32x over
double_buffered, 0.62-0.68x of cuBLAS.

**Nsight Compute, local sm_89, FP16**
(`profiles/sm_89/matmul/double_buffered_vs_wmma_*.ncu-rep` and `_metrics.csv`,
from `scripts/ncu_matmul.ps1 -Name double_buffered_vs_wmma -Kernel "matmul_[dw][om]" -Count 2 -Dtype f16 -Shapes gpt2s.ffn_up.m512,gpt2s.qkv.m16`,
run elevated; profiled before the occupancy floor, when FP16 used 123
registers, 127-128 after, with timings unchanged within 1%):

| Metric (512x3072x768 / 16x2304x768) | double_buffered | wmma |
|---|---|---|
| Duration under the profiler | 619 / 156 us | **215 / 56 us** |
| Instructions executed | 45.6 M / 8.35 M | **4.34 M / 0.80 M (10.5x fewer)** |
| FMA pipe active | 58.1% / 50.5% | 4.0% / 3.1% |
| Tensor pipe active | 0 | **36.6% / 29.5%** |
| Top stalls (per issue) | not_selected 1.91, long_scoreboard 1.03 | **math_pipe_throttle 7.38, long_scoreboard 6.84**, barrier 2.49, wait 2.43 |
| Shared-store bank conflicts | 0 | **427,599** / 76,933 |
| Shared-load bank conflicts | 0 | 60,557 / 1,190 |
| Registers, warps active | 124, 32.6% | 123, 30.9% |

### Verdict

**Kept** (FP16 everywhere, BF16 on sm_80+), the T4 result is at the end of this entry, and it is the
real test of prediction 1.

1. At least 3x over double_buffered and 40-70% of cuBLAS on large shapes:
   **speedup borderline, cuBLAS fraction exceeded on sm_89**: 2.92-3.02x, and
   0.81-0.84x of cuBLAS (clock-normalized). sm_89 understates the speedup by
   construction: its GeForce Tensor Cores are only 2x its FP32 rate (512 vs 256
   FLOP/clk/SM), while the T4's are 8x (1024 vs 128). The T4 decides.
2. No meaningful gain at decode sizes: **falsified on sm_89.** The 64x64x32
   configuration runs 16x2304x768 in 25.6 us, 1.9x faster than the best
   CUDA-core row (48.1 us, db_32x64x16). The operands fit in the 32 MB L2
   (warm), so this shape is not DRAM-bound here, and the CUDA-core kernel was
   instruction-bound. Configuration 0 (128 rows) gains nothing (55.3 us), so
   decode shapes still need their own configuration, and cuBLAS is still 1.6x
   ahead: split-K (S2) remains the step aimed there.
3. Tensor pipe active, FMA idle, instructions down at least 5x: **met**
   (10.5x fewer instructions; FMA 58% -> 4%; tensor 37%). The top stalls are
   now **math_pipe_throttle** (the GeForce tensor pipe is saturated by the
   warps that issue) and **long_scoreboard**: one register-prefetch stage does
   not cover global latency once the math per stage shrinks 8x. NCU also
   shows **427 K shared-store bank conflicts**: 16-byte stores to rows 80
   bytes apart put lanes 0 and 7 of each 8-lane phase in the same bank.

**Next step, chosen from this evidence (entry 5, MMA_SYNC):** `mma.sync`
m16n8k8 with `ldmatrix` fragment loads from an XOR-swizzled shared layout
(removes both conflict kinds), and a deeper register-prefetch pipeline to
attack long_scoreboard.

### Official T4 result (commit 5edb43d, docs/results/phase-11-tensor-cores.md)

Kaggle Tesla T4, FP16, warm L2, median per shape class
(`python scripts/step_summary.py results/sm_75/tensor_cores/matmul.csv --pairs double_buffered:wmma`):
WMMA is **2.83x faster than DOUBLE_BUFFERED on large shapes, 3.04x mid, 3.66x
decode**, and runs at **0.40x of cuBLAS** (large-shape median). Prediction 1
(at least 3x, 40-70% of cuBLAS): **met at the bottom edge**: 2.83x on large
shapes, 0.40x of cuBLAS. The T4 runs Tensor Core kernels at its 70 W power cap
(SM clocks 375-1440 MHz, NVML reason 0x4, no thermal throttling), so every
Tensor Core row, ours and cuBLAS's, is far below the 65 TFLOPS theoretical
peak; the WMMA microbenchmark measured 48.4 TFLOPS. Prediction 2 (no decode
gain): **falsified on the T4 too** (3.66x at decode sizes).

---

## 5. Matmul: `mma.sync` m16n8k8 + `ldmatrix` on a swizzled layout (`CCML_MATMUL_MMA_SYNC`)

**Status:** implemented and correct (FP16 from sm_75, BF16 from sm_80, SKIP on
the T4; all 3 Tensor Core configurations). Local sm_89 evidence below, after
**two corrections during the step**; the official T4 result is at the end.

### Hypothesis

Entry 4's profile names two costs that WMMA's API leaves out of reach:

- **Shared-memory bank conflicts: 427,599 on stores, 60,557 on loads**
  (512x3072x768, sm_89). Rows padded to 80 bytes put two lanes of each 8-lane
  128-bit store phase in one bank, and `load_matrix_sync` reads from the same
  padded rows. The fragment layout is opaque, so the shared layout cannot be
  arranged around it.
- **Shared memory caps occupancy on the T4.** The padded tiles take 37.9 KB
  per block. The T4 has 64 KB per SM, so one block (8 warps) is resident, and
  `long_scoreboard` (6.8 per issue) shows those warps waiting on global loads
  with nothing else to run.

**Change:** the same tiling and register-prefetch double buffering, with the
inner product written in PTX:

1. `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` (BF16: `.bf16`, sm_80+).
   The register layout of every fragment is documented, so the epilogue writes
   accumulators straight to global memory (no scratch tile).
2. Fragments loaded with `ldmatrix.sync.aligned.m8n8.x4` (A) and `.x4.trans`
   (B, stored row-major [K][N]): one instruction fills the A operands of two
   k8 steps, or the B operands of two n8 tiles over two k8 steps.
3. **No padding; an XOR swizzle instead.** The 16-byte chunk `c` of row `r` is
   stored at chunk `c ^ f(r)`, where f(r) spreads the eight rows one
   `ldmatrix` phase reads over all 32 banks. Stores of 8 consecutive threads
   still fill 128 contiguous bytes. Shared memory per block falls from 37.9 KB
   to **32 KB**, so **two blocks fit per T4 SM** if registers stay at or below
   128.

**Predictions** (FP16, warm L2; official on the T4, indicative on sm_89):

1. Shared-memory bank conflicts, load and store, fall **to near zero** on the
   profiled shape (from 60.6 K and 427.6 K).
2. Large shapes (M >= 512): **1.15-1.5x faster than WMMA**, and at least
   **0.7x of cuBLAS GemmEx** on the T4 (sm_89: at least 0.9x). On the T4 part
   of the gain comes from the second resident block; on sm_89, where shared
   memory allowed two blocks already, only from the conflicts and the leaner
   epilogue, so the sm_89 gain is the smaller one.
3. Decode shapes (M <= 16) **do not change measurably**: they are limited by
   rows a 16-row MMA wastes and by the grid being smaller than the GPU.

**What would falsify it:** conflicts remaining (the swizzle is wrong), or no
speedup despite conflict-free shared memory, which would mean global-load
latency with only one prefetch stage is the limit; the remedy would then be a
deeper pipeline (cp.async multistage on sm_80+, docs/local.md L2).

### Change

- `matmul_mma_sync<T, BM, BN, BK, WM, WN, kVec>` in `src/matmul/matmul_kernels.cuh`:
  `ldmatrix.x4` / `.x4.trans`, `mma.sync.m16n8k8`, XOR-swizzled unpadded
  tiles (`swizzle_chunk`), register-prefetch double buffering, epilogue from
  the documented accumulator layout. Same 3 configurations as WMMA (bench rows
  `mma_*`); the enum runs **configuration 1 (64x64x32, 4 warps)**, see below.
- Host: needs sm_75 (FP16) / sm_80 (BF16) in the loaded binary, else
  NOT_IMPLEMENTED; the compiled-architecture query is cached per kernel.
- Tests: `matmul_tc` covers it with every WMMA case (the suite iterates over
  the Tensor Core algos). sm_89: 10/10 pass, FP16 and BF16. sm_75 compile
  check: FP16 bodies built, BF16 bodies empty.

**Correction 1: the first version was slower than WMMA (0.88-0.92x)**
(`results/sm_89/mma_sync_spilling_*.csv`,
`profiles/sm_89/matmul/wmma_vs_mma_sync_spilling_*`). The SASS showed why:
loading the A fragments of all four 16-row tiles before the MMAs (16
registers) on top of 64 accumulators and 16 prefetch registers exceeded the
128-register occupancy floor, so 7 values spilled to local memory inside the
K loop, and the shared global loader's `unsigned short[8]` zero-fill was
compiled into 8 byte-permutes (`PRMT`) per 16-byte chunk. Changes: A
fragments are loaded one 16-row tile at a time; the `mma` asm is no longer
`volatile` (it only reads and writes registers, so the compiler may schedule
fragment loads around it); and MMA_SYNC got its own loader (whole `uint4`
chunks, per-thread source pointers fixed for the K loop). `PRMT` is gone and
every shared store is `STS.128`.

**Correction 2: the default configuration.** Configuration 0 (128x128, 8
warps, warp tile 64x32) still spills at the 128-register floor (`STACK:48`).
Configuration 1 (64x64, 4 warps, warp tile 32x32) uses 114 registers, no
stack, 16 KB of shared memory (4 resident blocks on sm_89, and on the T4
where shared memory allows 4 as well), and was fastest in the sm_89 sweep. The
enum uses it; WMMA keeps configuration 0, as measured in entry 4. The T4 sweep
re-checks both choices.

### Evidence

**Timings, local sm_89 (indicative), warm L2, median of 200, interleaved per
shape** (`results/sm_89/mma_sync_*.csv`; `python scripts/summarize_matmul.py
--prev wmma --new mma_sync --dtype f16 "results/sm_89/mma_sync_gpt2*.csv"`).
Event timestamps on this laptop are quantized to about 1 us (every median is a
multiple of 1.024 us), which matters only for the decode rows.

| Shape (FP16) | wmma | **mma_sync** | best ccml row | cuBLAS GemmEx | cuBLASLt top-1 | mma_sync %peak (theory / measured) |
|---|---|---|---|---|---|---|
| 2048x4096x1024 | 1,135 us | **1,027 us (1.10x)** | mma_sync | 1,080 us * | **983 us** | 75.2% / 91.8% |
| 512x3072x768 | 178 us | **158 us (1.13x)** | mma_64x64x32 | 146 us | 145 us | 68.8% / 84.1% |
| 128x2304x768 | 48.1 us | **37.9 us (1.27x)** | mma_sync | 36.9 us | 36.9 us | 53.7% / 65.6% |
| 16x2304x768 | 55.3 us | 21.5 us | **mma_16x128x32: 13.3 us** | 16.4 us | 16.4 us | 11.8% / 14.4% |

\* GemmEx's spread is wide in that run (p10 988, p90 1190 us); cuBLASLt's
heuristic kernel is tight at 983 us, and it is the vendor number to beat: at
1,027 us MMA_SYNC is **0.96x of it, a loss**. All rows of this table ran at
2535 MHz.

BF16 (sm_89 only): MMA_SYNC is **as fast as FP16** (1,027 / 158 / 37.9 / 21.5
us), 1.40-1.62x over WMMA's BF16. The BF16 gap in entry 4 was therefore a
property of the WMMA path, not of BF16 Tensor Cores.

**Decode shape, 16x2304x768, at identical clocks (2535 MHz):** the 16x128x32
configuration takes **13.3 us against cuBLAS's 16.4 us (FP16, 1.23x) and 19.5
us (BF16, 1.46x)**, with p10-p90 ranges that do not overlap. This is the first
Tensor Core shape-level win, **indicative only**: the T4 run is the official
test, and it is not claimed until then.

**Nsight Compute, local sm_89, FP16, enum configurations (WMMA 0, MMA_SYNC 1)**
(`profiles/sm_89/matmul/wmma_vs_mma_sync_*.ncu-rep`, `_metrics.csv`; table by
`python scripts/ncu_table.py "profiles/sm_89/matmul/wmma_vs_mma_sync_gpt2*_metrics.csv"`):

| Metric (512x3072x768 / 16x2304x768) | wmma | mma_sync |
|---|---|---|
| Duration under the profiler | 215 / 56.7 us | **188 / 31.3 us** |
| Shared-load bank conflicts | 60,744 / 1,190 | **6,608 / 0** |
| Shared-store bank conflicts | 425,010 / 77,130 | 130,965 / **0** |
| Registers; block limit (regs, smem) | 128; 2, 2 | 114; **4, 5** |
| Stall long_scoreboard (per issue) | 6.87 / 6.54 | **2.28** / 10.8 |
| Stall math_pipe_throttle | 6.68 / 5.62 | 6.90 / 3.49 |
| Tensor pipe active | 37.0% / 29.0% | **41.4%** / 18.7% |
| DRAM throughput | 20.1% / 49.4% | 23.0% / **89.6%** |

### Verdict

**Kept**, with configuration 1 as its default; T4 result below.

1. Conflicts near zero: **loads met** (60.7 K -> 6.6 K; 0 at decode size).
   **Stores: met at decode size (0), not explained on the larger grid (131 K).**
   The kernel and the store addresses are identical in both profiles, so an
   address-pattern conflict cannot account for the difference; my working
   explanation is contention between concurrent shared-memory requests of the
   4-5 resident blocks, which this counter may include. It is unverified:
   `scripts/ncu_matmul.ps1` now also collects shared wavefronts per
   instruction, which settles it in the next profile.
2. 1.15-1.5x over WMMA and at least 0.9x of cuBLAS on large shapes (sm_89):
   **met at the low end after the corrections**: 1.10-1.27x, and 0.93-0.97x of
   the best vendor kernel. Without the corrections it was falsified (0.88x).
   The gain came from occupancy (4 resident blocks; long_scoreboard -67%) more
   than from the conflicts.
3. Decode shapes unchanged: **falsified, favourably.** At M = 16 the enum
   configuration is 2.6x faster than WMMA, now DRAM-bound (89.6% of DRAM
   peak), and the 16-row configuration beats cuBLAS on sm_89. Split-K (S2,
   entry 6) is still worth testing there: at M = 16 the 16x128 grid has 18
   blocks for 14 SMs on sm_89 and 40 SMs on the T4, so the T4 will be
   under-filled.

### Official T4 result (commit 5edb43d, docs/results/phase-11-tensor-cores.md)

T4, FP16, warm: MMA_SYNC is **1.53x faster than WMMA on large shapes, 2.01x
mid, 2.47x decode**; it reaches **0.58x of cuBLAS** on large shapes (median),
0.77x mid, 0.97x decode. Prediction 2 (1.15-1.5x over WMMA, at least 0.7x of
cuBLAS): **speedup met (above the range), cuBLAS fraction missed** (0.58x).
One shape went the wrong way: llama1b ffn_up 2048x8192x2048, 8,948 us against
WMMA's 6,328 us. Its operands (72 MB) are far beyond the T4's 4 MB L2 and
MMA_SYNC's 64x64 tile gives 4x the CTAs of WMMA's 128x128, so raster order
re-reads B more: the problem entry 7 (swizzle) fixes (5,847 us).

---

## 6. Matmul: split-K for decode shapes (enhancement S2; `CCML_MATMUL_SPLIT_K`, `CCML_MATMUL_SPLIT_K_ATOMIC`)

**Status:** implemented and correct (FP16; BF16 on sm_80+). Local sm_89
evidence below; sm_89 has 14 SMs, so the T4 (40 SMs) is the real test.

### Hypothesis

At decode sizes (M <= 16) a matmul is a weight stream: every element of B is
read once and used for at most 16 rows, so the operation is bound by how fast
the GPU can pull B from memory, and that needs **every SM** issuing loads.
MMA_SYNC's 16-row configuration gives one CTA per 128 output columns and walks
the whole K range inside it: **18 CTAs** for N = 2304 (QKV) and **6** for
N = 768 (out-proj, FFN-down), against **40 SMs on the T4**. Most SMs sit idle
and the busy ones each stream a long K serially.

**Change:** split K into S slices; CTA (tile, s) runs the same MMA_SYNC main
loop over its slice only and writes an FP32 partial tile. S is chosen per call
so that tiles x S >= 2 x SMs, with at least 4 k-steps (BK = 32) per slice.
Two reductions, both behind their own algo value:

- `SPLIT_K` (deterministic): partials to an S x M x N FP32 workspace, then a
  second kernel sums them in fixed slice order and rounds once to the storage
  type. Same input, same bits, every run.
- `SPLIT_K_ATOMIC` (**non-deterministic**, enhancement C3): partials
  `atomicAdd`ed into one zeroed M x N FP32 workspace, then rounded. The order
  of the additions varies between runs, so the low bits can too.

The workspace comes from the stream-ordered allocator (`cudaMallocAsync` on
the caller's stream), so no global state is shared between streams; its cost
is inside every timed call.

**Predictions** (FP16, T4 official; sm_89 has only 14 SMs, so it shows less):

1. Decode shapes whose MMA_SYNC grid is under the SM count (M <= 16 and
   N/128 < 40: QKV 2304, out-proj 768, FFN-down 768 with K = 3072): SPLIT_K is
   **1.3-2.5x faster than the best MMA_SYNC configuration**, the gain largest
   where N is smallest and K largest (FFN-down: 6 CTAs, K = 3072).
2. On the T4, SPLIT_K **beats cuBLAS GemmEx on at least one decode target
   shape**, with non-overlapping confidence intervals (reference.md §9, §11).
3. Shapes whose grid already fills the GPU (M >= 128, or N >= 5120) gain
   nothing; the heuristic then picks S = 1 and the cost is one extra kernel
   launch. Not the target, measured anyway.
4. SPLIT_K_ATOMIC is **at most 10% faster** than SPLIT_K: its saving is only
   the S x M x N partial traffic, which is small at M <= 16; determinism costs
   little here.

**What would falsify it:** no gain on the 6-CTA shapes, which would mean the
kernel is not bandwidth-starved by idle SMs but limited per CTA (e.g. by
global-load latency with one prefetch stage), or the workspace allocation and
second launch costing more than the parallelism wins.

### Change

- The MMA_SYNC kernel body became `mma_sync_block_tile` (a K range and an
  epilogue functor); MMA_SYNC calls it with the whole K and the same store as
  before, and split-K with one slice and an FP32 partial store.
  `matmul_mma_sync_splitk<..., kAtomic>` and `splitk_reduce<T>` in
  `src/matmul/matmul_kernels.cuh`; host `launch_split_k` in `matmul.cu`.
- Slices: enough CTAs for 2 per SM, at least 4 k-steps per slice; S = 1 runs
  MMA_SYNC directly. The enum uses configuration 2 (16x128) for M <= 16 and 1
  otherwise; bench sweep rows `splitk_*`.
- Workspace: `cudaMallocAsync` / `cudaFreeAsync` on the caller's stream,
  freed on every path.
- Tests (`matmul_tc`, now 11 cases): split-K inherits every Tensor Core case
  (awkward shapes with several slices, every k-step count, misaligned views,
  poisoned outputs), plus **determinism (C3)**: over 20 runs of
  16x768x3072, **SPLIT_K differed bitwise 0 times; SPLIT_K_ATOMIC differed 19
  of 19 times** (still within tolerance).

### Evidence

**Timings, local sm_89 (indicative), FP16, warm L2, median of 200,
interleaved per shape, all rows at 2550-2565 MHz** (`results/sm_89/split_k_*.csv`).
"best mma" is the fastest MMA_SYNC tile configuration on that shape.

| Shape (M x N x K) | MMA_SYNC CTAs | best mma | **split_k** | split_k_atomic | cuBLAS (GemmEx / Lt) |
|---|---|---|---|---|---|
| gpt2s ffn_down 1x768x3072 | 6 | 32.8 us | **14.3 us (2.29x)** | 16.4 | 32.8 / 32.8 |
| gpt2s ffn_down 16x768x3072 | 6 | 33.8 | **15.4 (2.20x)** | 17.4 | **FAIL** (gate) |
| gpt2s attn_out 1x768x768 | 6 | 11.3 | **9.2 (1.22x)** | 10.2 | 8.1 / 8.2 |
| gpt2s attn_out 16x768x768 | 6 | 11.3 | **9.2 (1.22x)** | 12.3 | 8.2 / 8.2 |
| llama1b attn_out 1x2048x2048 | 16 | 24.6 | **19.5 (1.26x)** | 20.5 | 68.6 / 68.6 |
| llama1b attn_out 16x2048x2048 | 16 | 25.6 | **19.5 (1.32x)** | 22.5 | 31.7 / 30.7 |
| gpt2s qkv 16x2304x768 | 18 | 13.3 | 13.3 (1.00x) | 16.4 | 16.4 / 16.4 |
| gpt2s ffn_up 16x3072x768 | 24 | 13.3 | 14.3 (0.93x) | 17.4 | 17.5 / 17.4 |
| llama1b ffn_down 1x2048x8192 | 16 | 272 | 276 (0.99x) | 278 | 300 / 299 |
| llama1b ffn_down 16x2048x8192 | 16 | 274 | 282 (0.97x) | 284 | **FAIL** (gate) |

**cuBLAS FP16 fails the fp64 gate on both FFN-down shapes with M = 16**
(max abs error 0.079 at K = 3072, 0.168 at K = 8192; GemmEx and the Lt
heuristic alike), so those rows carry no timing (reference.md §11). Every ccml
kernel passes on the same inputs. Error that large at FP32 accumulation is
consistent with partial sums being rounded to FP16 inside a split-K
reduction; that cause is an inference, not verified.

**Nsight Compute, local sm_89, FP16** (`profiles/sm_89/matmul/mma_sync_vs_split_k_*`;
the enum MMA_SYNC, configuration 1, against split-K deterministic and atomic):

| Metric (gpt2s ffn_down 16x768x3072 / llama1b attn_out 16x2048x2048) | mma_sync (enum) | split_k main kernel |
|---|---|---|
| Duration under the profiler (cold) | 50.5 / 70.7 us | **41.3** / 69.9 us |
| DRAM throughput, % of peak | 74.8% / 94.1% | **91.6% / 94.8%** |
| Warps active | 8.3% / 19.0% | **17.4%** / 19.0% |
| Tensor instructions | 147 K / 262 K | **36.9 K / 65.5 K** (16-row tile: 4x less waste) |
| Shared store wavefronts per STS.128 | 4.0 / 4.0 | 4.0 / 4.0 (the ideal: conflict-free) |

### Verdict

**Kept** (both variants; SPLIT_K is the default choice, being deterministic and
faster). T4 result below.

1. 1.3-2.5x over the best MMA_SYNC configuration where the grid is under the
   SM count: **met where the grid is far below it**: 2.2-2.3x on GPT-2
   FFN-down (6 CTAs, K = 3072), 1.22-1.32x on attn_out. On sm_89 the 18- and
   24-CTA shapes already cover its 14 SMs, so they show nothing (1.00x,
   0.93x); on the T4's 40 SMs they are also under-filled, which the T4 run
   tests.
2. Beats cuBLAS on at least one decode target shape (T4: met, see below). **On sm_89,
   indicative:** 2.29x on gpt2s ffn_down 1x768x3072, 1.58x on llama1b attn_out
   16x2048x2048 (3.5x at M = 1, where cuBLAS picks a much slower kernel), at
   identical clocks. **Loss:** gpt2s attn_out (768x768), 9.2 vs 8.2 us
   (0.89x): at 1.2 MB of weights the second kernel and the workspace
   allocation are a visible share of the time.
3. Shapes whose grid already fills the GPU gain nothing: **met** (0.93-1.00x;
   up to 7% lost on ffn_up, where the heuristic still picked S = 2).
4. Atomic at most 10% faster than deterministic: **falsified, the other way:
   SPLIT_K_ATOMIC is 5-26% slower** on every shape. The zeroing `memset` and
   the atomics cost more than the S x M x N partial traffic they save, which
   at M <= 16 is small. Determinism here is free; the atomic variant is kept
   only as the measured counter-example.

**Under the profiler (cold caches) llama1b attn_out shows no gain** (69.9 vs
70.7 us; both at 94% of DRAM peak): there the warm-L2 gain of 1.32x comes from
L2 bandwidth (its 8 MB of weights fit the 32 MB L2), not DRAM. The T4's 4 MB
L2 will not hold it; the cold-cache rows of the T4 run decide.

### Official T4 result (commit 5edb43d, docs/results/phase-11-tensor-cores.md)

T4, FP16, warm: SPLIT_K vs MMA_SYNC, median **1.18x at decode sizes**, 1.00x
mid, 0.95x large; **vs the best cuBLAS kernel, 1.10x at decode (median)**.
gpt2s ffn_down 1x768x3072: 59.4 -> 26.0 us (2.28x). Prediction 1 (1.3-2.5x
where the grid is under 40 SMs): **met on the smallest grids** (ffn_down,
attn_out), **missed on the 18-24 CTA shapes**: gpt2s qkv 16x2304x768 went
15.5 -> 18.1 us, the heuristic's S = 5 (18 tiles x 5 = 90 CTAs) costs more than it gains there.
Prediction 2 (beat cuBLAS on a decode target shape): **met** on 19 target
decode shapes counting all ccml variants (phase-11 doc). Prediction 3:
**falsified for M >= 128**: the heuristic still splits some 128-row grids,
e.g. gpt2s qkv 128x2304x768 26.0 -> 40.2 us, and large shapes lose 5% at the
median; AUTO must not route M > 16 to SPLIT_K. Prediction 4: atomic within
+-2% of deterministic on the T4 (0.98-1.00x median), so "at most 10% faster"
holds only as "not faster".

---

## 7. Matmul: threadblock swizzle (enhancement S1; `CCML_MATMUL_MMA_SYNC_SWIZZLED`)

**Status:** implemented and correct (FP16; BF16 on sm_80+). Local sm_89
evidence below; T4 result at the end of the entry.

### Hypothesis

CTAs are dispatched in `blockIdx` order, x fastest, so the CTAs resident at
one moment cover a strip of C one block-row high and many block-columns wide:
they share one row panel of A but each reads a different column panel of B.
When the B panels a strip needs exceed L2, the next strip re-reads them all
from DRAM. On the T4 (4 MB L2), 2048x4096x1024 FP16 has 8 MB of B and 4 MB
of A, and MMA_SYNC's 64x64 tile gives a 64x32 grid of 2048 CTAs, about 160
resident at a time (40 SMs x 4): each wave touches ~2.5 block-rows of A but 64
column panels of B.

**Change:** CUTLASS-style group-M rasterization. The grid becomes 1-D; CTA
`pid` is mapped so that consecutive CTAs walk down a group of G = 8
block-rows before moving one block-column right. A resident wave then covers
about an 8 x 20 patch of C instead of a 2.5 x 64 strip: the same A panels plus
far fewer B panels. Everything else is MMA_SYNC (same tile, configuration 1).

**Predictions** (FP16, warm and cold L2):

1. T4, shapes whose operands exceed its 4 MB L2 and whose grid spans several
   waves (2048x4096x1024, 2048x3072x768, 2048x8192x2048, the LM heads with M
   >= 512): **DRAM traffic falls by at least 20%** and time by **3-10%**.
2. Grids of one wave or less (every decode shape, most M = 128 shapes)
   **do not change** (+-2%): there is no second wave to reuse anything.
3. sm_89 (32 MB L2, which holds every target shape's operands): **no
   measurable change** on any shape. The local run can only check that the
   mapping is correct and free.

**What would falsify it:** no DRAM reduction on the T4's large shapes, which
would mean the dispatch order is not raster or the hardware already reorders;
or a slowdown from the index arithmetic.

### Change

- `swizzled_block<kGroup>` and `matmul_mma_sync_swizzled` in
  `src/matmul/matmul_kernels.cuh`: 1-D grid, group of 8 block-rows, then
  `mma_sync_block_tile` (which now takes its block coordinates as arguments;
  MMA_SYNC and split-K pass `blockIdx`, timing unchanged: 1022 / 154 / 37.9 us
  against 1027 / 158 / 37.9 before). Enum `CCML_MATMUL_MMA_SYNC_SWIZZLED`,
  configuration 1 like MMA_SYNC.
- Tests: it joins every `matmul_tc` case; the NaN-poisoned outputs prove the
  mapping covers every tile exactly once, including partial last groups
  (M = 2047 with 16-row tiles). 11/11 pass on sm_89.

### Evidence

**Timings, local sm_89 (indicative), FP16, median of 200, interleaved per
shape** (`results/sm_89/swizzle_*.csv`):

| Shape | operands | mma_sync warm / cold | **swizzled warm / cold** | cuBLAS warm |
|---|---|---|---|---|
| llama1b ffn_up 2048x8192x2048 | 72 MB | 9,825 us (2460 MHz, power cap) / - | **4,393 us (2.24x)** (2535 MHz) / - | 4,158 us |
| gpt2m ffn_up 2048x4096x1024 | 28 MB | 1,021 / 1,091 | 1,034 (0.99x) / 1,048 (1.04x) | 983 |
| gpt2s ffn_up 2048x3072x768 | 19 MB | 586 / 623 | 590 (0.99x) / 613 (1.02x) | 565 |
| gpt2s qkv 128x2304x768 | 4 MB | 37.9 / 61.4 | 37.0 (1.02x) / **70.7 (0.87x)** | 36.9 |
| gpt2s qkv 16x2304x768 | 3.6 MB | 21.5 / 46.1 | 21.5 (1.00x) / 46.1 (1.00x) | 16.4 |

(cold rows exist only where the working set is at most 4x L2; the llama1b
shape is warm only.) The llama1b clock difference (2460 vs 2535 MHz) accounts
for 3%, not 2.24x; the counters below show the cause.

**Nsight Compute, local sm_89** (`profiles/sm_89/matmul/mma_sync_vs_swizzled_*`;
4 launches profiled because split-K runs plain MMA_SYNC when it picks one
slice; columns 1-3 are the same kernel and agree to 1%):

| Metric | llama1b 2048x8192x2048: mma_sync | **swizzled** | gpt2m 2048x4096x1024: mma_sync | swizzled |
|---|---|---|---|---|
| Duration under the profiler | 9.14 ms | **4.64 ms** | 1.206 ms | 1.172 ms |
| DRAM bytes read | **1,091 MB** (27x the 40 MB of A and B) | **151 MB** (-86%) | 12.6 MB | 13.7 MB |
| L2 sector hit rate | 49.3% | **94.4%** | 97.8% | 97.6% |
| DRAM throughput, % of peak | 96.2% | 30.7% | 15.6% | 17.0% |
| Stall long_scoreboard (per issue) | 20.3 | **0.30** | 0.86 | 0.35 |
| Tensor pipe active | 23.8% | **47.0%** | 45.3% | 46.6% |

### Verdict

**Kept** as its own algo (MMA_SYNC's enum is unchanged; which one AUTO picks
per shape is enhancement S4's measured table).

1. DRAM -20% and time -3-10% on shapes larger than L2: **exceeded, on the one
   sm_89 shape larger than its L2**: DRAM reads -86% and 2.24x (warm),
   1.97x under the profiler. Raster order made MMA_SYNC re-read B 27 times
   over and run at 96% of DRAM peak with the tensor pipe at 24%. The T4's 4 MB
   L2 is smaller than every M = 2048 target shape, so the T4 run tests the
   prediction where it was aimed.
2. Grids of one wave do not change: **met warm** (1.00-1.02x); **one
   unexplained cold-L2 loss**: gpt2s qkv 128x2304x768 cold, 70.7 vs 61.4 us
   (0.87x). The grid is 72 CTAs (one wave), so the order should not matter;
   it is recorded as a loss and re-checked in the T4 cold rows.
3. No change on sm_89: **met for the shapes that fit its 32 MB L2**
   (0.99-1.04x, DRAM bytes equal within 9%), **wrong as stated**: I assumed
   every target shape fits 32 MB, and llama1b's FFN at M = 2048 (72 MB) does
   not.

### Official T4 result (commit 5edb43d, docs/results/phase-11-tensor-cores.md)

T4, FP16, warm: swizzled vs MMA_SYNC median **1.00x large, 1.01x mid, 1.01x
decode**; **llama1b ffn_up 2048x8192x2048: 8,948 -> 5,847 us (1.53x)**, the
only target shape whose operands (72 MB) dwarf L2 *and* whose grid spans many
waves. Prediction 1 (DRAM -20% and 3-10% on the M = 2048 shapes larger than
4 MB): **met only on that shape; elsewhere no change**: gpt2m ffn_up
2048x4096x1024 1.02x (1,401 -> 1,376 us). Power-capped clocks differ by up to
2x between rows of the same large shape (450-1035 MHz), so the large-shape
medians are coarse. Prediction 2 (one-wave grids unchanged): **met**
(1.01x). The sm_89 cold-L2 loss on gpt2s qkv 128x2304x768 (0.87x) did not
reproduce on the T4: cold, 32.7 vs 32.3 us (0.99x, within noise of cuBLAS's
32.3 us); it is not explained, only not seen on the target GPU.

---

## 8. Matmul: fused bias / GELU / residual epilogue (enhancements F1, B2; `ccml_matmul_epilogue_<dtype>`)

**Status:** implemented and correct (FP16; BF16 on sm_80+). Local sm_89
evidence below; T4 result at the end of the entry.

### Hypothesis

A transformer linear layer is `y = act(x W + b)` (FFN-up, GELU) or
`y = x W + b + residual` (attention out-projection, FFN-down). With a plain
matmul, the product C goes to DRAM and an elementwise kernel reads it back,
reads b (and the residual), and writes y: **C crosses DRAM three times instead
of once** (write, read, write), plus a second launch. MMA_SYNC's accumulators
are in registers with a documented layout, so the epilogue can be applied
there before the single store.

**Change:** `ccml_matmul_epilogue_f16/bf16(C, A, B, bias, residual, M, N, K,
act, algo, stream)`, `C = act(A B + bias) + residual` (bias [N] or NULL,
residual [M, N] or NULL, act none / ReLU / GELU-tanh), computed in FP32 on the
accumulators and rounded once. `algo` = MMA_SYNC (fused in the store) or
SPLIT_K (fused into the reduction kernel, which already reads every element
once). Baselines: our MMA_SYNC + a separate epilogue kernel; cuBLAS GemmEx + the
same separate kernel; **cuBLASLt with its own fused epilogue** (`GELU_BIAS`,
or `BIAS` with beta = 1 and C = residual) (B2).

**Predictions** (FP16; T4 official):

1. Fused vs our own unfused MMA_SYNC + separate kernel: the saving is the
   separate kernel's traffic, 2 x M x N (+ M x N residual) 16-bit elements
   plus one launch. It is largest relative to the matmul when K is small
   compared with M and N: **1.10-1.30x on M >= 512 FFN-up / attn_out shapes**
   (K = 768-1024), **1.2-1.6x at decode sizes**, where the extra launch is a
   large share of 10-30 us.
2. Fused ccml vs cuBLAS GemmEx + separate kernel: **wins wherever the plain
   ccml kernel is within ~10% of cuBLAS**, because the vendor path pays the
   same extra round-trip.
3. Fused ccml vs cuBLASLt's fused epilogue: **no better than the plain-matmul
   ratio** (Lt fuses too, so fusion gives no relative advantage against it);
   wins only where the plain ccml kernel already wins (decode, via split-K).

**What would falsify it:** fused no faster than unfused, which would mean the
epilogue's extra registers (bias, residual loads in the store path) cost
occupancy, or that the separate kernel's traffic is hidden in L2 at these
sizes; NCU (DRAM bytes, registers) decides.

### Change

- `EpilogueArgs`, `matmul_mma_sync_epilogue` (epilogue in MMA_SYNC's store
  functor) and `splitk_reduce_epilogue` in `src/matmul/matmul_kernels.cuh`;
  `ccml_matmul_epilogue_f16/bf16` and `ccml_activation` in
  `include/ccml/matmul.h`. GELU is the tanh form (GPT-2's `gelu_new`).
- Bench: `ccml_bench_matmul_epilogue` (`bench/bench_matmul_epilogue.cpp`),
  shared dtype traits moved to `bench/support/typed.h`; the separate kernel is
  `bench/baselines/epilogue.cu`; cuBLASLt's `GELU_BIAS` / `BIAS` (+ beta = 1
  residual) in `bench/baselines/cublas.cpp`.
- Tests: new suite `matmul_epilogue` (7 epilogue combinations x 8 shapes x
  {MMA_SYNC, SPLIT_K} x {FP16, BF16}, misaligned views, argument checks;
  NaN-poisoned outputs), worst normalized error 0.49 (FP16), 0.50 (BF16).

**Correction during the step: the gate for unfused pipelines.** The first run
failed every "+separate" candidate, ours included. Not a bug: an unfused 16-bit
pipeline rounds A B to FP16 before the epilogue kernel, one extra rounding of
up to unit x |A B|, which the fused tolerance does not admit where the residual
cancels the product. That rounding is inherent to any unfused 16-bit linear
layer, so unfused candidates are now gated with that per-element allowance
(`ccml_test::compare(..., extra_abs)`, unit-tested in `harness`); ccml's fused
kernels keep the strict bound. **cuBLASLt's fused epilogue showed exactly the
unfused error** (0.0936 vs the fused kernels' 0.0625 on gpt2s ffn_up
512x3072x768): it evidently rounds before its epilogue, so it gets the same
allowance. cuBLAS GemmEx + separate still fails on 8 of 20 shapes with errors
beyond even that (0.106-0.171), the same accuracy loss as entry 6.

### Evidence

**Timings, local sm_89 (indicative), FP16, warm L2, median of 200,
interleaved per shape, 2535-2565 MHz unless noted** (`results/sm_89/epilogue_*.csv`):

| Shape, epilogue | **ccml fused** (best of MMA_SYNC / SPLIT_K) | ccml + separate | cuBLAS + separate | cuBLASLt epilogue |
|---|---|---|---|---|
| gpt2s ffn_up 2048x3072x768, bias+GELU | **606.7 us** | 755.2 (1.24x) | 749.6 (1.24x) | FAIL |
| gpt2s ffn_up 512x3072x768 | **159.9** | 192.5 (1.20x) | 184.9 (1.16x) | 160.8 (1.01x) |
| gpt2s ffn_up 16x3072x768 (2205-2565 MHz) | **16.4** | 17.4 (1.06x) | 21.5 (1.31x) | 26.6 (1.62x) |
| gpt2s ffn_up 1x3072x768 (2205 MHz) | **15.4** | 16.4 (1.06x) | 29.7 (1.93x) | 36.9 (2.40x) |
| gpt2s attn_out 2048x768x768, bias+residual | 162.8 | 190.5 (1.17x) | 182.3 (1.12x) | **146.4 (0.90x)** |
| gpt2s attn_out 128x768x768 | 18.6 | 21.5 (1.16x) | 19.5 (1.05x) | **15.4 (0.83x)** |
| gpt2s attn_out 16x768x768 | **9.2** | 11.3 (1.23x) | 10.2 (1.11x) | FAIL |
| gpt2s attn_out 1x768x768 | 9.2 | 10.2 (1.11x) | 10.2 (1.11x) | **8.2 (0.89x)** |
| gpt2s ffn_down 2048x768x3072 | 596.0 | 645.1 (1.08x) | 605.2 (1.02x) | **554.0 (0.93x)** |
| gpt2s ffn_down 128x768x3072 | 50.2 | 57.3 (1.14x) | **48.3 (0.96x)** | **45.1 (0.90x)** |
| gpt2s ffn_down 16x768x3072 | **15.4** | 16.4 (1.06x) | FAIL | FAIL |
| gpt2s ffn_down 1x768x3072 | **14.3** | 16.4 (1.15x) | 34.8 (2.43x) | FAIL |
| llama1b attn_out 2048x2048x2048 | 1038 | 1249 (1.20x) | 1157 (1.11x) | **999 (0.96x)** |
| llama1b attn_out 16x2048x2048 | **20.5** | 22.5 (1.10x) | 33.8 (1.65x) | FAIL |
| llama1b attn_out 1x2048x2048 | **19.5** | 20.6 (1.06x) | 71.7 (3.68x) | FAIL |

(ratios are the competitor's time over ccml fused: above 1 is a ccml win.
All 20 shapes are in the CSVs; M = 128 and 512 rows not shown follow the same
pattern.)

**Nsight Compute, local sm_89** (`profiles/sm_89/matmul/fused_vs_unfused_*`,
the fused MMA_SYNC kernel against the plain one it replaces):

| Metric (gpt2s ffn_up 2048x3072x768 / attn_out 2048x768x768) | plain matmul kernel | fused kernel |
|---|---|---|
| Duration under the profiler | 685 / 174 us | 715 / 190 us (+4% / +9%) |
| Registers per thread | 114 | 114 (no occupancy cost) |
| Stall long_scoreboard (per issue) | 0.87 / 0.49 | 1.36 / 4.66 (bias and residual loads in the store) |

### Verdict

**Kept.**

1. Fused vs our unfused: **met for M >= 512** with K = 768-2048: 1.17-1.24x
   (K = 3072 FFN-down: 1.08x, as the 1/K argument predicts). **Decode:
   1.06-1.23x, below the predicted 1.2-1.6x**: the separate kernel on 16 x N
   elements costs only ~1-2 us here, less than I assumed.
2. Fused ccml vs cuBLAS + separate: **met**: ccml wins on every shape where
   both pass the gate except gpt2s ffn_down 128x768x3072 (0.96x), by 1.02-1.24x
   at M >= 512 and up to 3.7x at decode sizes (split-K's advantage plus the
   fusion).
3. Against cuBLASLt's fused epilogue: **met**: Lt is faster at M = 128 and
   M = 2048 on the residual shapes (0.83-0.96x for ccml), tied on ffn_up
   512, and ccml wins only at decode sizes (1.6-2.4x on ffn_up). Lt fails the
   gate on 11 of 20 shapes even with the unfused allowance.

The fused kernel itself is 4-9% slower than the plain one (the epilogue's
loads stall the store path), so fusion wins by removing a whole kernel, not by
being free.

### Official T4 result (commit 5edb43d, docs/results/phase-11-tensor-cores.md)

T4, FP16, warm (`python scripts/epilogue_summary.py results/sm_75/tensor_cores/matmul_epilogue.csv`),
median ratio of the competitor's time to ccml fused:

| Class | ours + separate | cuBLAS + separate | cuBLASLt epilogue |
|---|---|---|---|
| large (M >= 512) | 1.13x (n = 24) | **0.77x** (n = 19) | **0.53x** (n = 19) |
| mid (M = 128) | 1.09x | 1.05x | 0.88x |
| decode (M <= 16) | 1.03x | **1.23x** | **1.20x** |

1. Fused vs ours unfused: **met for large shapes** (1.13x median; up to 1.32x
   on ffn_up 512x3072x768), **missed at decode** (1.03x, as on sm_89).
2. Fused ccml vs cuBLAS + separate: **met where the plain kernel is close to
   cuBLAS** (decode, 1.23x), **lost on large shapes** (0.77x), where the
   plain MMA_SYNC is only 0.58x of cuBLAS and fusion cannot make that up.
3. vs cuBLASLt's epilogue: **met**: ccml wins only at decode (1.20x) and
   loses large (0.53x).

---

## 9. Layernorm: single-pass Welford in FP32 on pivot-shifted data (`CCML_LAYERNORM_WELFORD`)

**Status:** implemented and correct (FP32, FP16, BF16). Local sm_89 evidence
below; T4 result at the end of the entry when it lands.

### Hypothesis

`TWO_PASS` (Phase 2) reads each row three times and does every reduction in
**FP64**: two conversions, a subtraction and an FMA per element per pass, and
two 32-bit shuffles per double in each block reduction. FP64 runs at 1/32 of
the FP32 rate on the T4 (0.25 TFLOPS) and 1/64 on GeForce Ada. On the T4 it
reaches only **30-55 GB/s** (Phase 7, `results/sm_75/layernorm.csv`), 13-24% of
the 227.6 GB/s measured peak, while PyTorch eager and Triton reach 150-240 GB/s
on the same rows. Layer normalization should be memory-bound, so the kernel is
bound by something else: the FP64 arithmetic and the extra passes.

FP64 is there for a reason (`layernorm_kernels.cuh`): a FP32 mean at magnitude
10000 carries ~5e-4 of error, which against a standard deviation of 0.01 is a
5% output error. That argument assumes the values are accumulated as they come.
**Subtracting a pivot first removes it:** with p = the row's first element,
d = x - p is exact whenever x and p are within a factor of 2 of each other
(Sterbenz), and otherwise rounds relative to |d| itself. The statistics of d
are small numbers again, FP32 holds them to ~1e-7 of the spread, and
x - mean = d - mean(d) never forms the large intermediate.

**Change:** `CCML_LAYERNORM_WELFORD`, block per row as before, but
- one pass for the statistics: each thread runs Welford's update on its d
  values in FP32; threads merge with Chan's formula through warp shuffles and
  one shared-memory step;
- a second pass writes (d - mean_d) x rstd x gamma + beta, in FP32;
- no FP64 anywhere.

**Predictions** (FP32, warm L2; T4 official, sm_89 NCU):

1. Large shapes (rows >= 2048): **3-6x faster than TWO_PASS**, reaching **at
   least 50% of measured DRAM bandwidth** (>= 115 GB/s on the T4).
2. Correctness at the FP32 reduction tolerance **including mean = 10000,
   stddev = 0.01** (`large_offset_small_variance`), the case the FP64 design
   exists for. If the pivot is the reason, then that test passes and nothing
   else in the suite needs a wider tolerance.
3. Nsight Compute: TWO_PASS shows the **FP64 pipe** busy (or its instructions
   dominating issue) and memory throughput low; WELFORD shows no FP64
   instructions and memory throughput as the top limiter.
4. Small row counts (rows <= 16) barely change: 16 blocks on a 40-SM GPU are
   latency-bound either way. Warp-per-row (entry 10) is the step for those.

**What would falsify it:** a speedup under 2x (the FP64 was not the limit:
the block-per-row structure is), or `large_offset_small_variance` failing
(the pivot argument is wrong in practice).

### Change

- `Welford`, `welford_add`, `welford_merge` (Chan), `welford_warp_reduce`
  (xor butterfly), `welford_block_reduce` and `layernorm_welford<T>` in
  `src/layernorm/layernorm_kernels.cuh`; `CCML_LAYERNORM_WELFORD` dispatches
  it (it was reserved and returned NOT_IMPLEMENTED).
- Tests: `layernorm.every_algorithm_on_every_case` runs both algorithms on 16
  cases (1 to 16384 columns, awkward widths, 4099 rows, and four offset cases:
  mean 10000 / stddev 0.01, 1000 / 1, -5000 / 0.5, 100 / 0.001) at the FP32
  tolerance; `layernorm.outlier_pivot` makes the pivot an outlier (first
  element 1000 stddevs from the row's mean); `dtypes.{F16,BF16}_layernorm` now
  run every algorithm.
- Bench: `ccml_bench_layernorm --dtype f32,f16,bf16` (FP16/BF16 rounded
  inputs, widened tolerance); `bench_torch.py --layernorm-dtypes`; the Kaggle
  job runs all three dtypes for layernorm.

### Evidence

**Correctness (sm_89):** worst normalized error over the 16 cases: TWO_PASS
0.011, **WELFORD 0.062**, both within tolerance; mean 10000 / stddev 0.01
passes at the FP32 tolerance. Outlier pivot: WELFORD 0.090 (TWO_PASS 0.001).
9/9 suites pass.

**Timings, local sm_89 (indicative), warm L2, median of 200**
(`results/sm_89/layernorm_welford.csv`, peaks `peaks_layernorm.json`: DRAM
114 GB/s measured):

| Shape (rows x cols) | two_pass f32 | **welford f32** | welford GB/s | two_pass f16 | welford f16 |
|---|---|---|---|---|---|
| 8192 x 768 | 2,781 us | **432 us (6.4x)** | 117 | 2,780 | 332 (8.4x) |
| 8192 x 4096 | 8,522 | **2,372 (3.6x)** | 113 | 8,526 | 1,238 (6.9x) |
| 2048 x 1024 | 811 | **90.1 (9.0x)** | 186 (L2) | 812 | 85.0 (9.6x) |
| 512 x 2048 | 319 | **32.7 (9.7x)** | 257 (L2) | 319 | 31.7 (10.0x) |
| 128 x 768 | 51.2 | **10.2 (5.0x)** | 77 | 51.2 | 9.2 (5.6x) |
| 16 x 4096 | 32.8 | **8.2 (4.0x)** | 68 | 32.8 | 9.2 (3.6x) |
| 1 x 768 | 9.2 | 5.1 (1.8x) | 2 | 9.2 | 5.1 (1.8x) |

On the large shapes WELFORD runs at **99-103% of the measured DRAM peak**
(113-117 GB/s against 114; the byte count is the minimum, 2 x rows x cols,
and the second read of a row is served by L2). Smaller tensors fit the 32 MB
L2 and exceed DRAM bandwidth.

**Nsight Compute, local sm_89, FP32** (`profiles/sm_89/layernorm/two_pass_vs_welford_*`,
`scripts/ncu_matmul.ps1 -Name two_pass_vs_welford -Kernel "layernorm_[tw][we]" -Count 2 -Dtype f32 -Bench ccml_bench_layernorm -Op layernorm -Shapes h768.rows8192,h4096.rows16`):

| Metric (8192 x 768 / 16 x 4096) | two_pass | welford |
|---|---|---|
| Duration under the profiler | 3.21 ms / 35.7 us | **0.44 ms / 9.5 us** |
| **FP64 pipe active** | **87.1% / 78.4%** | **0 / 0** |
| FP64 instructions | 5.39 M / 32.2 K | 0 / 0 |
| DRAM throughput, % of peak | 12.6% / 6.9% | **76.9% / 48.4%** |
| Instructions executed | 30.2 M / 109 K | 40.2 M / 138 K (Welford's per-element division) |
| Registers; warps active | 37; 98.7% | 38; 94.5% |

### Verdict

**Kept**, sm_89 evidence; the T4 decides prediction 1's numbers.

1. 3-6x on large shapes, >= 50% of DRAM peak: **met on sm_89** (3.6-6.4x
   FP32 at rows = 8192, 99-103% of measured DRAM peak; up to 11x on shapes
   that fit L2).
2. Correct at the FP32 tolerance including mean 10000 / stddev 0.01, with no
   other tolerance widened: **met.**
3. NCU: **met**: TWO_PASS is FP64-bound (FP64 pipe 87% busy, DRAM 13%);
   WELFORD has no FP64 and is limited by memory (DRAM 77%). This answers the
   roadmap's open question: on GPUs with throttled FP64 the "obviously
   correct" FP64 reductions cost 4-11x, and the pivot shift buys the same
   accuracy class in FP32.
4. rows <= 16 barely change: **falsified**: 4.0x at 16 x 4096 and 1.8x at
   1 x 768. The FP64 cost was per element, not per launch, so it showed even
   at 16 rows.

---

## 10. Layernorm: warp per row, row resident in registers (`CCML_LAYERNORM_WARP_PER_ROW`)

**Status:** implemented and correct (FP32, FP16, BF16; cols <= 1024). Local
sm_89 evidence below; T4 result at the end of the entry when it lands.

### Hypothesis

Entry 9's WELFORD gives each row a whole block: 256 threads, so at the common
hidden sizes (768, 1024) a thread holds 3-4 values, and the row costs two
block-wide barriers, a shared-memory merge, a per-element FP32 division
(Welford's update; NCU: 33% more instructions than TWO_PASS), and a second
read of the row for the output pass. For cols <= 1024 one warp is enough: 32
lanes x up to 32 values each, in registers.

**Change:** `CCML_LAYERNORM_WARP_PER_ROW`, cols <= 1024 (wider rows return
`CCML_ERROR_UNSUPPORTED_SHAPE`):
- one warp per row, 4 rows per 128-thread block, no `__syncthreads`;
- the row is read **once** into registers (pivot-shifted as in entry 9);
- exact two-pass statistics over the registers: mean of d by a warp-shuffle
  sum, then sum of (d - mean)^2 by a second shuffle sum; no division per
  element;
- the output is written from the registers. The number of values per lane is
  a template parameter chosen from the column count (1, 2, 4, 8, 16, 24, 32),
  so short rows do not carry 32 registers.

**Predictions** (FP32 and FP16, warm L2; T4 official, sm_89 NCU):

1. rows >= 2048, cols <= 1024: **1.1-1.4x faster than WELFORD**; WELFORD is
   already near the DRAM limit there on sm_89, so the room is the second read
   and the instructions.
2. rows 128-512, cols 768-1024 (mid sizes, partly L2-resident): **at least
   1.5x faster than WELFORD**: barriers and the per-element division dominate
   when each thread holds 3-4 values.
3. NCU: **instructions executed fall by at least 2x** against WELFORD, DRAM
   (or L2) throughput becomes the top limiter, and the global load count per
   row halves (one pass instead of two).
4. Correctness unchanged: every case of entry 9's tests at the FP32 tolerance,
   for every per-lane bucket (cols 1 ... 1024).

**What would falsify it:** no gain at rows >= 2048 on the T4 would mean the
second read was free (served by L1) and the kernel was already at the DRAM
limit; that would be recorded, and the variant kept only for mid sizes.

### Change

- `warp_sum` and `layernorm_warp_per_row<T, kPer>` in
  `src/layernorm/layernorm_kernels.cuh`; `launch_warp_per_row` picks kPer from
  {1, 2, 4, 8, 16, 24, 32} values per lane; cols > 1024 returns
  `CCML_ERROR_UNSUPPORTED_SHAPE`.
- Tests: `layernorm.warp_per_row_every_bucket` (23 widths at both edges of
  every bucket, odd widths with a +300 offset; the 1025-column refusal);
  WARP_PER_ROW joins `every_algorithm_on_every_case`, `outlier_pivot` and the
  FP16/BF16 dtype tests. Worst normalized error 0.115 (outlier pivot).

### Evidence

**Timings, local sm_89 (indicative), warm L2, median of 200**
(`results/sm_89/layernorm_warp_per_row.csv`):

| Shape | welford f32 | **warp_per_row f32** | GB/s | welford f16 | warp_per_row f16 |
|---|---|---|---|---|---|
| 8192 x 768 | 431 us | 439 (0.98x) | 115 | 332 | **218 (1.52x)** |
| 8192 x 1024 | 561 | 586 (0.96x) | 115 | 377 | **292 (1.29x)** |
| 4096 x 1000 | 347 | **282 (1.23x)** | 116 | 167 | **61.4 (2.72x)** |
| 2048 x 768 | 89.1 | **23.6 (3.78x)** | 535 (L2) | 77.8 | **24.6 (3.17x)** |
| 512 x 1024 | 25.6 | **13.3 (1.92x)** | 316 (L2) | 24.6 | **14.3 (1.71x)** |
| 128 x 768 | 10.2 | **7.2 (1.43x)** | 111 | 9.2 | **7.2 (1.29x)** |
| 16 x 1024 | 5.1 | 8.2 (**0.62x**) | 17 | 6.1 | 8.2 (**0.75x**) |
| 1 x 768 | 6.1 | 8.2 (**0.75x**) | 1.5 | 5.1 | 7.2 (**0.71x**) |

**Nsight Compute, local sm_89, FP32** (`profiles/sm_89/layernorm/welford_vs_warp_per_row_*`;
under the profiler caches are flushed between passes, so these are cold):

| Metric (2048 x 768 / 8192 x 768) | welford | warp_per_row |
|---|---|---|
| Duration under the profiler | 105 / 442 us | **61.5 / 347 us** |
| Instructions executed | 10.1 M / 40.2 M | **1.75 M / 7.00 M (5.75x fewer)** |
| DRAM bytes read | 6.32 / 26.0 MB | 6.32 / 25.2 MB (the same: WELFORD's second read hit L2) |
| DRAM throughput, % of peak | 47.2% / 78.1% | **80.5% / 89.4%** |
| Stall barrier (per issue) | 0.68 / 0.84 | 0 / 0 |
| Registers; warps active | 38; 94% | **80; 48%** |
| Stall long_scoreboard | 2.96 / 4.65 | **17.3 / 24.7** (fewer warps waiting on more loads each) |

### Verdict

**Kept** for 128 <= rows and cols <= 1024 on sm_89; the T4 sets AUTO's
thresholds.

1. rows >= 2048: 1.1-1.4x: **mixed**: FP16 1.29-1.52x, FP32 0.96-0.98x warm
   (both kernels at the DRAM peak, 115 GB/s) but 1.27x cold under the
   profiler. The falsification clause applies to FP32 warm: WELFORD's second
   read was free (L2 hits; DRAM bytes equal), so there was no DRAM traffic to
   save, only instructions.
2. Mid sizes at least 1.5x: **met** (1.7-3.8x at 512-2048 rows; 1.3-1.4x at
   128).
3. NCU: instructions down at least 2x (**5.75x**), DRAM the top limiter
   (**89%**), loads per row halved: **the instruction and DRAM parts met; the
   load part was moot** because L2 already absorbed the second read.
4. Correctness across every bucket: **met.**

**A loss, not predicted:** at 1-16 rows WARP_PER_ROW is 0.62-0.75x of
WELFORD: 1-16 warps on a 14-SM GPU, each serially issuing up to 32 loads, take
longer than 1-16 blocks of 8 warps sharing the row. AUTO must keep small row
counts on WELFORD. The next limit, from NCU: 80 registers halve occupancy and
`long_scoreboard` dominates; 128-bit loads (entry 11) cut the load count 4x
(FP32) / 8x (16-bit).

---

## 11. Layernorm: 128-bit loads and stores on warp-per-row (`CCML_LAYERNORM_VECTORIZED`)

**Status:** implemented and correct (FP32, FP16, BF16; cols <= 1024).
Local sm_89 evidence below; T4 result at the end of the entry when it lands.

### Hypothesis

Entry 10's profile names the limiter: WARP_PER_ROW holds up to 32 values per
lane in 80 registers, so only 48% of warps are active, and the top stall is
`long_scoreboard` (24.7 per issue at 8192 x 768): each lane issues up to 32
separate 32-bit (or 16-bit) loads per row and waits on them with half the
usual number of warps around to hide it. FP16 makes it worse: a 16-bit load
moves half the bytes per instruction.

**Change:** `CCML_LAYERNORM_VECTORIZED`: WARP_PER_ROW with every global
access 128 bits wide: x, gamma and beta loaded and y stored as chunks of
16 / sizeof(T) elements (4 FP32, 8 FP16/BF16); lane l owns chunks
l, l + 32, ... Same register-resident two-pass statistics. It applies when
cols is a multiple of the chunk and every pointer is 16-byte aligned; otherwise
the call runs WARP_PER_ROW's scalar kernel (the result is identical, so the
fallback is safe).

**Predictions** (warm L2; T4 official, sm_89 NCU):

1. Global load instructions per row fall **4x (FP32) / 8x (FP16, BF16)**, and
   `long_scoreboard` per issue falls by at least half.
2. FP16/BF16, rows >= 2048: **1.2-1.6x faster than WARP_PER_ROW**, reaching
   the DRAM peak where WARP_PER_ROW does not.
3. FP32 shapes already at the DRAM peak (sm_89 rows = 8192): **no change**
   (+-5%); mid sizes (L2-resident): 1.1-1.3x.
4. Correctness unchanged, including the scalar fallback (misaligned views,
   widths not a multiple of the chunk).

**What would falsify it:** load instructions falling as predicted with no
time saved at the 16-bit large shapes, which would mean WARP_PER_ROW's 16-bit
rows were not latency-bound but limited elsewhere (occupancy via registers).

### Change

- `Chunk16<T>` and `layernorm_vectorized<T, kChunks>` in
  `src/layernorm/layernorm_kernels.cuh`; `launch_vectorized` checks
  cols % (16 / sizeof(T)) and 16-byte alignment of every pointer, picks 1, 2,
  4 or 8 chunks per lane, and otherwise runs WARP_PER_ROW's scalar kernel.
- Tests: `layernorm.vectorized_vector_and_fallback_paths` (16 widths at the
  edges of every chunk bucket, 4 non-multiple widths, misaligned x/y/gamma/beta
  views); VECTORIZED joins every shared case and the FP16/BF16 tests. 17/17
  layernorm cases pass; worst normalized error 0.151.

### Evidence

**Timings, local sm_89 (indicative), warm L2, median of 200**
(`results/sm_89/layernorm_vectorized.csv`):

| Shape | warp_per_row f16 | **vectorized f16** | GB/s | warp_per_row f32 | vectorized f32 |
|---|---|---|---|---|---|
| 8192 x 768 | 243 us | 232 (1.05x) | 108 | 440 | 438 (1.00x) |
| 8192 x 1024 | 292 | 291 (1.00x) | 115 | 586 | 582 (1.01x) |
| 4096 x 1000 | 61.4 | **28.7 (2.14x)** | 572 (L2) | 322 | 325 (0.99x) |
| 2048 x 1024 | 33.8 | **15.4 (2.20x)** | 546 (L2) | 46.1 | **37.9 (1.22x)** |
| 512 x 768 | 11.3 | **7.2 (1.57x)** | 220 | 11.3 | **9.2 (1.22x)** |
| 128 x 1024 | 9.2 | **5.7 (1.61x)** | 92 | 9.2 | **6.1 (1.50x)** |
| 16 x 1024 | 8.2 | **5.1 (1.60x)** | 14 | 8.2 | **5.1 (1.60x)** |

At 1-16 rows VECTORIZED now equals WELFORD (5.1 us), removing entry 10's
small-row loss.

**Nsight Compute, local sm_89, FP16** (`profiles/sm_89/layernorm/warp_per_row_vs_vectorized_*`):

| Metric (2048 x 768 / 8192 x 768) | warp_per_row | vectorized |
|---|---|---|
| Instructions executed | 1.95 M / 7.80 M | **1.02 M / 4.08 M (1.9x fewer)** |
| Registers; warps active | 78; 47% | **55; 68%** |
| Stall lg_throttle (LSU queue full) | 0.149 / 0.236 | **0.004 / 0.001** |
| Stall long_scoreboard (per issue) | 6.33 / 7.11 | 22.9 / 27.4 |
| DRAM throughput, % of peak | 69.1% / 88.5% | 89.2% / **93.1%** |
| Duration under the profiler | 36.1 / 153 us | 40.9 / **132 us** |
| DRAM bytes read / written | 3.18 / 0 MB; 12.6 / 4.7 MB | 3.17 / 1.49 MB; 12.6 / 3.1 MB |

### Verdict

**Kept**; it supersedes WARP_PER_ROW wherever its preconditions hold.

1. Load instructions down 4x / 8x, long_scoreboard down by half: **loads
   met in effect** (instructions 1.9x fewer in total, the LSU-queue stall
   `lg_throttle` gone); **long_scoreboard falsified as stated**: per issue it
   rose 3.6x. It is a ratio: the same memory wait divided over half as many
   instructions. Per-issue stall ratios are the wrong yardstick when the
   instruction count itself changes; time and DRAM throughput are the right
   ones (DRAM 88.5% -> 93.1%, 153 -> 132 us).
2. FP16 rows >= 2048: 1.2-1.6x: **exceeded where L2-resident** (2.1-2.2x at
   2048-4096 rows), **not at 8192 rows**, where WARP_PER_ROW was already at the
   DRAM peak warm (1.00-1.05x). Cold under the profiler, 1.16x.
3. FP32 at the DRAM peak unchanged, mid sizes 1.1-1.3x: **met** (1.00x;
   1.22x).
4. Correctness including the fallback: **met.**

The cold 2048 x 768 profile is slower (36.1 -> 40.9 us) because of an
accounting artifact, not the kernel: WARP_PER_ROW's 1.5 MB output was still in
L2 when its measurement ended (0 bytes written to DRAM), while VECTORIZED's
write-back landed inside its own. The warm bench (2.2x) and the 8192-row cold
profile (equal bytes, 1.16x) are the comparisons that hold.

---

## 12. Layernorm: fused residual + bias (enhancement F2; `ccml_layernorm_residual_<dtype>`)

**Status:** implemented and correct (FP32, FP16, BF16). Local sm_89 evidence
below (mixed); the T4, whose 4 MB L2 cannot hold the intermediate, decides.

### Hypothesis

In a pre-LN transformer block every layer norm is preceded by
`s = x + residual + bias` (the attention or MLP output plus its bias, added to
the residual stream), and `s` is needed twice: normalized now, and as the
residual of the next block. Unfused, that is an elementwise kernel (read x, r,
b; write s) followed by the layer norm (read s; write y): **5 row-sized
transfers** (x, r, s, s, y). Fused, one kernel reads x and r once and writes s
and y: **4**, with one launch instead of two. For a memory-bound operator the
expected saving is about one fifth of the traffic plus a launch.

**Change:** `ccml_layernorm_residual_f32/f16/bf16(y, sum_out, x, residual,
bias, gamma, beta, rows, cols, eps, algo, stream)`: s = x + residual + bias
(bias [cols] or NULL; sum_out NULL to skip writing s), rounded to the storage
type, then y = LN(s). The norm is of the **rounded** s, so the fused result is
bit-for-bit what the unfused pipeline's layer norm sees. `algo` VECTORIZED
(the warp-per-row 128-bit kernel, cols <= 1024) or WELFORD (block per row, any
width); AUTO picks by width.

**Predictions** (warm and cold L2; T4 official, sm_89 indicative):

1. Fused vs our own separate add kernel + VECTORIZED / WELFORD, on shapes
   where both are DRAM-bound (rows >= 2048): **1.2-1.3x** (5 -> 4 transfers
   plus a launch).
2. Small shapes (rows <= 128): **1.3-1.8x**, the second launch being a large
   share of 5-10 us.
3. vs PyTorch eager (`F.layer_norm(x + r + b)`: two or three kernels): at least
   the same margin as against our separate pipeline; vs torch.compile, which
   fuses the add into its generated layer norm: **no better than VECTORIZED's
   ratio to torch.compile on the plain layer norm.**
4. `sum_out` is **bitwise equal** to the separate add kernel's output, and y
   meets the layer norm tolerance against fp64 LN of the rounded sum.

**What would falsify it:** fused no faster than separate at DRAM-bound sizes,
which would mean the extra residual read and sum write per element push the
fused kernel off the bandwidth limit (registers, occupancy) or the separate
pipeline's s was served by L2 (only possible when s fits in L2).

### Change

- `residual_sum`, `layernorm_residual_welford<T>` (block per row, any width)
  and `layernorm_residual_vectorized<T, kChunks>` (VECTORIZED with the sum in
  front) in `src/layernorm/layernorm_kernels.cuh`;
  `ccml_layernorm_residual_f32/f16/bf16` in `include/ccml/layernorm.h`.
  VECTORIZED falls back to the Welford kernel when its width or alignment
  preconditions fail; AUTO picks VECTORIZED for cols <= 1024.
- Tests: new suite `layernorm_residual` (10 shapes x {bias, no bias} x
  {sum_out, none} x {VECTORIZED, WELFORD, AUTO} x 3 dtypes, plus offset and
  misaligned cases; argument checks). `sum_out` is checked **bitwise** against
  the host's FP32 (x + r) + b rounded once; y against fp64 LN of that rounded
  s. Worst normalized error 0.040 (FP32), 0.49 (FP16, BF16).
- Bench: `ccml_bench_layernorm_residual` (fused vs `residual_add` kernel +
  plain layer norm); `bench_torch.py --ops layernorm_residual` (eager:
  add + `F.layer_norm`; torch.compile).

### Evidence

**Timings, local sm_89 (indicative), median of 200**
(`results/sm_89/layernorm_residual.csv`; ratio = best separate / best fused,
> 1 means fused is faster):

| Shape | FP16 warm | FP16 cold | FP32 warm | FP32 cold |
|---|---|---|---|---|
| 8192 x 768 | **1.17x** | **1.15x** | **1.25x** | (warm only) |
| 8192 x 4096 | 1.17x | - | **1.25x** | - |
| 4096 x 1000 | 1.07x | 1.03x | **1.23x** | **1.19x** |
| 2048 x 768 | **2.71x** | 0.97x | **0.91x (loss)** | **0.94x (loss)** |
| 512 x 768 | **2.12x** | 1.18x | **2.08x** | **0.94x (loss)** |
| 128 x 768 | **1.80x** | **1.58x** | **1.83x** | 1.20x |
| 128 x 3079 | 1.33x | 0.96x | 1.43x | **0.89x (loss)** |
| 16 x 4096 | 1.32x | 1.05x | 1.33x | **0.83x (loss)** |
| 1 x 768 | 1.40x | 1.50x | 1.33x | 1.29x |

**Nsight Compute, local sm_89, FP32** (`profiles/sm_89/layernorm/fused_vs_separate_*`;
the separate pipeline's layer norm kernel shown; its add kernel is not in the
profile):

| Metric (512 x 768 / 8192 x 768) | fused vectorized | fused welford | plain vectorized (separate's 2nd kernel) |
|---|---|---|---|
| Duration under the profiler | 34.9 / 900 us | 36.6 / **789 us** | 18.6 / 337 us |
| DRAM bytes read | 3.18 / 59.0 MB | 3.18 / 50.4 MB | 1.60 / 25.2 MB |
| Registers; warps active | 62; 60% | 40; **93%** | 64; 61% |
| Stall long_scoreboard (per issue) | 93.2 / 140 | 5.0 / 13.8 | 51.0 / 54.7 |
| DRAM throughput, % of peak | 87.4% / 87.8% | 68.1% / 91.5% | 67.4% / 90.9% |

### Verdict

**Kept**, with a measured caveat; AUTO for the fused op is set from the T4.

1. DRAM-bound rows >= 2048: 1.2-1.3x: **met where the intermediate leaves
   L2** (8192 and 4096 rows: 1.17-1.31x warm, 1.15-1.19x cold), **falsified
   where it does not**: FP32 2048 x 768/1024, 6-8 MB per tensor, fits sm_89's
   32 MB L2, so the separate pipeline's s never reaches DRAM and fusion only
   saves a launch (0.91-0.98x). This is the falsification clause's second
   case, and it is specific to a large L2: on the T4 (4 MB) the same shapes
   spill s to DRAM.
2. Small shapes 1.3-1.8x: **met warm** (1.3-2.7x); **cold, FP32 loses at
   16-512 rows** (0.83-0.94x). With a flushed L2, the fused kernel must fetch x
   and r at once with 60% of warps active (long_scoreboard 93 per issue);
   the separate add kernel is a simpler streaming kernel at full occupancy.
4. sum_out bitwise equal: **met.**
3. (vs PyTorch) is measured in the T4 run only.

**Two findings for AUTO:** (a) at DRAM-bound sizes the fused **WELFORD**
kernel beats the fused VECTORIZED one (789 vs 900 us at 8192 x 768: 93% of
warps active against 60%), so the fused op should not simply follow the
plain op's choice; (b) the fused VECTORIZED kernel reads 59 MB where 50 MB is
the minimum (x + r at 8192 x 768); the extra 9 MB is **not explained** yet.
