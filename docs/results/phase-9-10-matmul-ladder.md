# Phases 9-10: the matmul ladder (register blocking, vectorized loads, double buffering)

**Dates:** 2026-11-17 to 2026-11-18
**Branch:** `matmul-ladder`
**Outcome:** three rungs of the reference.md section 3 ladder, each a
one-change step with a hypothesis committed before the code and Nsight Compute
evidence (docs/optimization-log.md, entries 1-3). On the Tesla T4 our best
matmul went from **0.17x to 0.79x** of the best baseline (median over 76
shapes). It reaches **parity with cuBLAS on large shapes** (44.6% vs 44.6% of
peak), and **beats cuBLAS on two target shapes** with the vendor kernel
running at a higher clock. It still loses on decode shapes. All tests and
sanitizer checks pass.

## Environment

Kaggle, one session per job at commit **`12cb762`**
(`results/sm_75/matmul_ladder/`). Tesla T4 (sm_75), driver 580.178.04, CUDA
13.0, torch 2.11.0+cu130, triton 3.6.0. Measured peaks this session: 7.81
TFLOPS FP32 (at 1545 MHz) and 227.6 GB/s. Nsight Compute evidence is from the
local RTX 3050 A Laptop (sm_89), because the counters are blocked on Kaggle
(docs/local.md L3).

## What was added

| Rung | Variant | One change |
|---|---|---|
| 3 | `CCML_MATMUL_REGISTER_TILED` | Each thread computes an 8x8 patch of C from registers |
| 4 | `CCML_MATMUL_VECTORIZED` | 128-bit global and shared access, conflict-free B layout, padded A; occupancy floor |
| 5 | `CCML_MATMUL_DOUBLE_BUFFERED` | Two shared buffers, next tile prefetched into registers, one barrier per step |

Plus six tile configurations, the tuning API (`include/ccml/tuning.h`) and the
tile sweep (`--tile-sweep`, reference.md section 5).

### Tests written

- Every tile configuration of every tiled variant on 15 awkward shapes (M, N
  or K = 1; nothing a multiple of any tile), on the vector and the scalar
  global paths.
- Every k-step count for the double-buffered pipeline (K = 1 ... 1000).
- Misaligned pointer views; FP16 and BF16 for every variant.
- Tile-config API rejections.

## Correctness and sanitizer (verbatim summary)

`results/sm_75/matmul_ladder/`: ctest `100% tests passed, 0 tests failed out
of 7`; pytest `68 passed`; compute-sanitizer **24/24 exit 0**: memcheck,
racecheck, initcheck and synccheck on every suite. That includes racecheck on
the double-buffered single-barrier pipeline, the most race-prone code written
so far. The infrastructure OOM is the known deliberate error.

## Benchmark results (T4, FP32, warm L2, one session)

### Speedup per rung (median over shapes in each class)

| Step | Decode (M <= 16) | Mid (M 128-512) | Large (M >= 1000) |
|---|---|---|---|
| tiled -> register_tiled | 0.38x | 3.28x | **5.02x** |
| register_tiled -> vectorized | 1.16x | 1.37x | 1.43x |
| vectorized -> double_buffered | 1.35x | 1.22x | 1.05x |

### Median % of measured peak

| Class | tiled | register_tiled | vectorized | double_buffered | cuBLAS sgemm |
|---|---|---|---|---|---|
| Decode | 2.5 | 0.7 | 0.8 | 1.1 | 6.0 |
| Mid | 6.7 | 24.4 | 35.8 | 41.6 | 52.1 |
| Large | 6.1 | 31.2 | 43.6 | **44.6** | **44.6** |

### Selected shapes (median us, % of peak, SM clock during the run)

| Shape | tiled | register_tiled | vectorized | double_buffered | best tile | cuBLAS | torch.mm (cuBLAS) | Triton |
|---|---|---|---|---|---|---|---|---|
| gpt2s qkv 16x2304x768 | 76.0 | 243.0 | 214.8 | 151.6 | db 32x64x16: 41.9 (17.3%) | **36.9 (19.7%)** @1590 | 36.9 | 48.9 |
| gpt2s ffn_up 512x3072x768 | 4,541 | 1,142 | 705 | 609 (50.8%) @1200 | | 567 (54.5%) @885 | **532** @1020 | 642 |
| gpt2m ffn_up 2048x4096x1024 | 36,066 | 7,183 | 5,112 | 4,935 (44.6%) @885 | | **4,501 (48.9%)** @675 | 4,502 @975 | 4,685 |
| **gpt2s ffn_up 2048x3072x768** | 20,448 | 3,945 | 2,731 | **2,639 (46.9%) @930** | | 2,773 @615 | 2,826 @975 | 2,724 |
| **llama1b attn_out 2048x2048x2048** | 36,363 | 6,525 | 4,925 | **4,641 (47.4%) @990** | | 5,347 @705 | 5,345 @1020 | 4,786 |

### Against the best baseline, shape by shape

Best ccml kernel against the best of cuBLAS, cuBLASLt, PyTorch eager,
torch.compile and Triton, per shape: **median 0.79x** (Phase 7: 0.17x).
Decode 0.70x, mid 0.80x, large 0.91x. Our kernel is faster on 4 of 76 shapes,
all with separated 95% confidence intervals:

| Shape | Ours | Best baseline | Speedup | Target shape? |
|---|---|---|---|---|
| llama1b attn_out m2048 | double_buffered 4,641 us | Triton 4,786 us | 1.031x | **yes** |
| gpt2s ffn_up m2048 | double_buffered 2,639 us | cuBLASLt 2,714 us | 1.028x | **yes** |
| odd 33x33x33 | naive 4.58 us | Triton 6.66 us | 1.455x | no |
| odd 1x1x1 | naive 3.94 us | cuBLAS 4.10 us | 1.041x | no |

**The clock question, answered with the data we have.** Clocks cannot be
locked on Kaggle, and under the T4's 70 W cap each kernel's clock is partly
its own doing. Our winning rows ran at 930-990 MHz; the direct cuBLAS rows
ran at 615-705 MHz. That alone would not settle it. But the same cuBLAS
kernels, reached through `torch.mm` in the same session, ran at **975-1020
MHz, higher than ours, and were still slower**: 2,826 vs 2,639 us (1.07x) and
5,345 vs 4,641 us (1.15x). **So on these two target shapes our kernel beats
cuBLAS with the vendor at the higher clock.** That meets the reference.md
section 9 criterion without the clock caveat. We do not claim more than these
two shapes.

### Tile sweep: which configuration wins where

| Class | Best configuration (count of shapes) |
|---|---|
| Decode M <= 16 | **db 32x64x16 (4x4)**: 31 of 32 |
| Mid | db 128x128x8 (8x8): 11, db 32x64x16: 8, db 128x64x8: 7, other: 3 |
| Large | **db 128x128x8 (8x8)**: 13 of 15 |

This is the data the measured AUTO dispatch table (enhancement S4) will use.
No single tile wins everywhere.

## What did not transfer from the laptop

On sm_89 the 32x64x16 tile beat cuBLAS on a decode shape (1.08x). On the T4
cuBLAS is **1.14x faster** on the same shape (36.9 vs 41.9 us, same 1590
MHz). An Ada result is not a Turing result: that is why the official numbers
come from the T4, and why `results/sm_89/` stays separate.

## What this proves, and what it does not

**Proves:** each rung's mechanism (NCU, entries 1-3) and its speedup on the
target GPU; correctness and race-freedom of every variant; a measured
clock-fair win over cuBLAS on two target shapes.

**Does not prove:** a decode-shape win (we lose 1.14-3x there; split-K and
stream-K, enhancement S2 in Phase 11, target exactly that); anything about
Tensor Cores (Phase 11); that AUTO picks well (AUTO still resolves to `TILED`
until the dispatch table exists, and that is deliberate).

## How to reproduce

    python cloud/kaggle/kaggle_run.py --any-gpu --jobs sanitize,matmul
