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

**Status:** hypothesis committed; implementation and measurement in progress.

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

*(filled in with the implementation)*

### Evidence

*(filled in from `results/sm_75/matmul.csv` and `profiles/sm_89/matmul/`)*

### Verdict

*(pending)*
