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

---

## 4. Matmul: Tensor Cores through WMMA (`CCML_MATMUL_WMMA`)

**Status:** implemented and correct (FP16 everywhere; BF16 on sm_80+, SKIP on
the T4; all 3 Tensor Core configurations). Local sm_89 evidence below; the
official T4 timings are pending (Kaggle, `results/sm_75/matmul.csv`).

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

**Kept** (FP16 everywhere, BF16 on sm_80+), pending the T4 run, which is the
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

---

## 5. Matmul: `mma.sync` m16n8k8 + `ldmatrix` on a swizzled layout (`CCML_MATMUL_MMA_SYNC`)

**Status:** implemented and correct (FP16 from sm_75, BF16 from sm_80, SKIP on
the T4; all 3 Tensor Core configurations). Local sm_89 evidence below, after
**two corrections during the step**; the official T4 timings are pending.

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

**Kept**, with configuration 1 as its default; T4 pending.

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

---

## 6. Matmul: split-K for decode shapes (enhancement S2; `CCML_MATMUL_SPLIT_K`, `CCML_MATMUL_SPLIT_K_ATOMIC`)

**Status:** hypothesis committed before the code.

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
