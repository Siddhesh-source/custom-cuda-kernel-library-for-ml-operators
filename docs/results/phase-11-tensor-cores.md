# Phase 11: Tensor Cores vs cuBLAS (+ S1, S2, F1, B2)

**Date:** 2026-11-17
**Branch:** `tensor-core-matmul` (worktree), merged into `dev`
**Outcome:** the last rung of the reference.md section 3 ladder (Tensor Cores,
both WMMA and `mma.sync`), plus split-K (S2), threadblock swizzle (S1) and a
fused bias/GELU/residual epilogue (F1) compared against cuBLASLt's own (B2).
On the Tesla T4, FP16, **ccml beats cuBLAS on 19 target decode shapes** (M <=
16; non-overlapping 95% CIs, equal or clock-normalized clocks; up to 1.69x)
and **loses on large shapes** (median 0.61x of cuBLAS). cuBLAS FP16 itself
**fails the fp64 correctness gate on 20 of 76 shapes**. All tests and every
compute-sanitizer tool pass; BF16 Tensor Core variants SKIP on the T4 and pass
on the local sm_89 GPU.

## Environment

Kaggle, two sessions at commit **`5edb43d`** (`ccml-sanitize-tc`,
`ccml-matmul-tc`; `results/sm_75/tensor_cores/`). Tesla T4 (sm_75), driver
580.178.04, CUDA 13.0, torch 2.11.0+cu130, triton 3.6.0. Measured peaks this
session (`peaks_matmul.json`): FP32 7.65 TFLOPS, **FP16 Tensor Cores 48.4
TFLOPS** (WMMA microbenchmark; 65.1 theoretical), DRAM 227.6 GB/s. Tensor
Core kernels, ours and cuBLAS's, run at the T4's **70 W power cap**: NVML
reason 0x4 on 73% of rows, SM clocks 375-1590 MHz, no thermal or hardware
throttling (max 78 C). Candidates are interleaved, so comparisons within a
shape are fair; absolute %peak on large shapes is power-limited. Nsight
Compute evidence is from the local RTX 3050 A Laptop (sm_89), because counters
are blocked on Kaggle (docs/local.md L3).

## What was added

| Step | Variant / API | One change | Log entry |
|---|---|---|---|
| Harness | `--dtype f32,f16,bf16`, `cublasGemmEx`, cuBLASLt 16-bit, Tensor Core peaks, `bench_torch.py --matmul-dtypes` | FP16/BF16 measured like FP32 | - |
| Ladder rung 6 | `CCML_MATMUL_WMMA` | `wmma` 16x16x16 on the double-buffered tiling | 4 |
| Ladder rung 6 | `CCML_MATMUL_MMA_SYNC` | `mma.sync` m16n8k8 + `ldmatrix`, XOR-swizzled shared memory | 5 |
| S2 | `CCML_MATMUL_SPLIT_K`, `CCML_MATMUL_SPLIT_K_ATOMIC` | K split across CTAs for decode; the atomic variant is non-deterministic | 6 |
| S1 | `CCML_MATMUL_MMA_SYNC_SWIZZLED` | group-M rasterization | 7 |
| F1 | `ccml_matmul_epilogue_f16/bf16` | bias / ReLU / GELU / residual on the accumulators | 8 |

Plus three Tensor Core tile configurations (`ccml_matmul_tc_config_*`; bench
rows `wmma_*`, `mma_*`, `splitk_*`), the `ccml_bench_matmul_epilogue`
benchmark, and NCU tooling (`scripts/ncu_matmul.ps1`, `scripts/ncu_table.py`).

### Tests written

- `matmul_tc` (11 cases): every Tensor Core variant x every configuration on
  17 awkward shapes (M, N or K = 1; nothing a tile multiple; vector and scalar
  paths), every k-step count (K = 1 ... 1000), misaligned views (offsets 1, 2,
  4), target shapes, NaN-poisoned outputs, FP32 refused, config API bounds,
  and **split-K determinism (C3)**: SPLIT_K bitwise identical over 20 runs,
  SPLIT_K_ATOMIC differed in 19 of 19 (sm_89).
- `matmul_epilogue` (5 cases): 7 epilogue combinations x 8 shapes x
  {MMA_SYNC, SPLIT_K}, misaligned views, argument checks.
- `harness.compare_per_element_allowance` (the gate for unfused pipelines).

## Verbatim results

ctest (`ctest.txt`): `100% tests passed, 0 tests failed out of 9`.
`matmul_tc`: `7 passed, 0 failed, 4 skipped` (the 4 BF16 cases);
`matmul_epilogue`: `3 passed, 0 failed, 2 skipped` (BF16).
compute-sanitizer (`sanitizer/summary.txt`): memcheck, racecheck, initcheck
and synccheck **clean on all 9 suites** (plus the one deliberate OOM in
`infrastructure`, as before).

Locally (sm_89, `scripts/local_build.ps1`): 9/9 suites, including the BF16
Tensor Core cases (`matmul_tc` 11/11, `matmul_epilogue` 5/5).

## Per-step speedups (T4, FP16, warm L2, median over shapes)

`python scripts/step_summary.py results/sm_75/tensor_cores/matmul.csv --dtype f16 --pairs double_buffered:wmma,wmma:mma_sync,mma_sync:split_k,mma_sync:mma_sync_swizzled`

| Step | large (M >= 512) | mid (M = 128) | decode (M <= 16) | vs best cuBLAS: large / mid / decode |
|---|---|---|---|---|
| double_buffered -> wmma | 2.83x | 3.04x | 3.66x | 0.40x / 0.47x / 0.47x |
| wmma -> mma_sync | 1.53x | 2.01x | 2.47x | 0.58x / 0.77x / 0.97x |
| mma_sync -> split_k | 0.95x | 1.00x | 1.18x | 0.57x / 0.62x / **1.10x** |
| mma_sync -> swizzled | 1.00x | 1.01x | 1.01x | 0.57x / 0.78x / 0.95x |

Representative shapes (the same script prints the 5-line table per step):

| Shape | double_buffered | wmma | mma_sync | best ccml | best cuBLAS |
|---|---|---|---|---|---|
| gpt2m ffn_up 2048x4096x1024 | 4,587 us | 1,585 | 1,401 | 1,376 (swizzled) | 662 |
| gpt2s ffn_up 512x3072x768 | 629 | 249 | 138 | 135 (swizzled) | 107 |
| gpt2s qkv 128x2304x768 | 202 | 61.4 | 26.0 | 26.0 | 25.1 |
| gpt2s qkv 16x2304x768 | 170 | 40.7 | 15.5 | **15.2** (swizzled) | 17.3 |
| gpt2s ffn_down 1x768x3072 | 723 | 185 | 59.4 | **25.4** (split-K) | 26.4 |
| llama1b ffn_up 2048x8192x2048 | 19,078 | 6,328 | 8,948 | 5,847 (swizzled) | 2,863 |

## Shape-by-shape vs cuBLAS (reference.md section 9)

Best ccml row (any variant or tile configuration) against the faster of
cuBLAS GemmEx / SGEMM and cuBLASLt's heuristic kernel, warm L2, one session.
WIN / LOSS only with non-overlapping bootstrap 95% CIs. VENDOR FAIL: both
cuBLAS rows failed the fp64 gate, so no vendor timing exists.
`python scripts/vendor_table.py results/sm_75/tensor_cores/matmul.csv --dtype f16` (and `--dtype f32`).

**Totals.** FP16 warm: **WIN 24, LOSS 32, VENDOR FAIL 20**; cold: WIN 7, LOSS
26, TIE 1, VENDOR FAIL 12. FP32 warm: WIN 3, LOSS 71, TIE 2; cold: WIN 2,
LOSS 25. BF16 warm (no BF16 Tensor Cores on the T4; cuBLAS emulates): WIN 37,
LOSS 39. Median best-vendor / best-ccml (`scripts/dtype_summary.py`): FP16
0.94x overall (decode 1.13x, mid 0.80x, large 0.61x); FP32 0.73x; BF16 0.96x
(large 1.54x, decode 0.62x).

**Clock check of the wins.** Of the 21 FP16 wins on target shapes, 16 ran at
the same clock as the vendor row (decode shapes, 1575-1590 MHz) and 5 had the
vendor at a lower clock. Clock-normalized (vendor time x vendor clock / ccml
clock), gpt2m lm_head 16x50257x1024 (1.12x), llama1b qkv 1x6144x2048 (1.16x)
and llama1b ffn_down 1x2048x8192 (1.69x) remain wins; **llama1b ffn_up
16x8192x2048 (1.00x) and llama1b lm_head 16x32000x2048 (0.97x) are not
claimed.** That leaves **19 target-shape wins**, all at decode sizes. The
three odd-shape wins (1x1x1, 33x33x33, 7x3079x1025) are launch-overhead and
split-K effects on non-target shapes.

### FP16

| shape | M x N x K | best ccml | us | best vendor | us | vendor/ccml | ccml %peak | verdict (95% CI) |
|---|---|---|---|---|---|---|---|---|
| gpt2s.qkv.m1 | 1x2304x768 | split_k | 11.7 | gemmex | 18.5 | 1.58x | 0.5% | WIN |
| gpt2s.qkv.m16 | 16x2304x768 | mma_sync_swizzled | 15.2 | heuristic_top1 | 17.3 | 1.13x | 5.7% | WIN |
| gpt2s.qkv.m128 | 128x2304x768 | mma_64x64x32_w32x32 | 25.8 | heuristic_top1 | 25.1 | 0.97x | 27.0% | LOSS |
| gpt2s.qkv.m512 | 512x2304x768 | mma_64x64x32_w32x32 | 94.8 | heuristic_top1 | 52.8 | 0.56x | 29.4% | LOSS |
| gpt2s.qkv.m2048 | 2048x2304x768 | mma_sync | 397.3 | heuristic_top1 | 248.5 | 0.63x | 28.0% | LOSS |
| gpt2s.attn_out.m1 | 1x768x768 | split_k | 9.2 | gemmex | 10.2 | 1.11x | 0.2% | WIN |
| gpt2s.attn_out.m16 | 16x768x768 | split_k | 9.8 | heuristic_top1 | 9.2 | 0.95x | 3.0% | LOSS |
| gpt2s.attn_out.m128 | 128x768x768 | splitk_64x64x32_w32x32 | 16.4 | FAIL | - | - | 14.2% | VENDOR FAIL |
| gpt2s.attn_out.m512 | 512x768x768 | mma_64x64x32_w32x32 | 32.4 | heuristic_top1 | 26.3 | 0.81x | 28.7% | LOSS |
| gpt2s.attn_out.m2048 | 2048x768x768 | split_k | 127.2 | heuristic_top1 | 104.3 | 0.82x | 29.2% | LOSS |
| gpt2s.ffn_up.m1 | 1x3072x768 | mma_16x128x32_w16x32 | 24.1 | gemmex | 33.6 | 1.39x | 0.3% | WIN |
| gpt2s.ffn_up.m16 | 16x3072x768 | mma_16x128x32_w16x32 | 24.6 | FAIL | - | - | 4.7% | VENDOR FAIL |
| gpt2s.ffn_up.m128 | 128x3072x768 | mma_64x64x32_w32x32 | 36.6 | heuristic_top1 | 34.6 | 0.95x | 25.3% | LOSS |
| gpt2s.ffn_up.m512 | 512x3072x768 | mma_sync_swizzled | 135.2 | heuristic_top1 | 106.5 | 0.79x | 27.4% | LOSS |
| gpt2s.ffn_up.m2048 | 2048x3072x768 | mma_128x128x32_w64x32 | 637.3 | heuristic_top1 | 333.8 | 0.52x | 23.3% | LOSS |
| gpt2s.ffn_down.m1 | 1x768x3072 | splitk_64x64x32_w32x32 | 25.4 | heuristic_top1 | 26.4 | 1.04x | 0.3% | WIN |
| gpt2s.ffn_down.m16 | 16x768x3072 | splitk_64x64x32_w32x32 | 28.1 | FAIL | - | - | 4.1% | VENDOR FAIL |
| gpt2s.ffn_down.m128 | 128x768x3072 | split_k_atomic | 44.7 | FAIL | - | - | 20.8% | VENDOR FAIL |
| gpt2s.ffn_down.m512 | 512x768x3072 | mma_sync | 122.8 | heuristic_top1 | 100.0 | 0.81x | 30.2% | LOSS |
| gpt2s.ffn_down.m2048 | 2048x768x3072 | mma_sync | 506.1 | FAIL | - | - | 29.3% | VENDOR FAIL |
| gpt2s.lm_head.m1 | 1x50257x768 | split_k_atomic | 366.4 | heuristic_top1 | 358.0 | 0.98x | 0.3% | LOSS |
| gpt2s.lm_head.m16 | 16x50257x768 | split_k_atomic | 446.8 | heuristic_top1 | 507.2 | 1.14x | 4.2% | WIN |
| gpt2s.lm_head.m128 | 128x50257x768 | mma_128x128x32_w64x32 | 1443.8 | heuristic_top1 | 864.5 | 0.60x | 10.5% | LOSS |
| gpt2m.qkv.m1 | 1x3072x1024 | mma_16x128x32_w16x32 | 30.7 | heuristic_top1 | 42.4 | 1.38x | 0.3% | WIN |
| gpt2m.qkv.m16 | 16x3072x1024 | mma_16x128x32_w16x32 | 31.4 | FAIL | - | - | 4.9% | VENDOR FAIL |
| gpt2m.qkv.m128 | 128x3072x1024 | split_k | 45.1 | gemmex | 42.2 | 0.94x | 27.4% | LOSS |
| gpt2m.qkv.m512 | 512x3072x1024 | mma_sync_swizzled | 192.5 | heuristic_top1 | 143.9 | 0.75x | 25.7% | LOSS |
| gpt2m.qkv.m2048 | 2048x3072x1024 | mma_128x128x32_w64x32 | 850.2 | heuristic_top1 | 462.8 | 0.54x | 23.3% | LOSS |
| gpt2m.attn_out.m1 | 1x1024x1024 | splitk_16x128x32_w16x32 | 9.9 | heuristic_top1 | 13.8 | 1.39x | 0.3% | WIN |
| gpt2m.attn_out.m16 | 16x1024x1024 | split_k | 10.5 | heuristic_top1 | 12.1 | 1.15x | 4.9% | WIN |
| gpt2m.attn_out.m128 | 128x1024x1024 | mma_sync_swizzled | 20.5 | FAIL | - | - | 20.1% | VENDOR FAIL |
| gpt2m.attn_out.m512 | 512x1024x1024 | mma_sync_swizzled | 57.3 | heuristic_top1 | 35.4 | 0.62x | 28.8% | LOSS |
| gpt2m.attn_out.m2048 | 2048x1024x1024 | mma_sync | 220.4 | heuristic_top1 | 148.0 | 0.67x | 29.9% | LOSS |
| gpt2m.ffn_up.m1 | 1x4096x1024 | mma_16x128x32_w16x32 | 36.2 | heuristic_top1 | 40.7 | 1.12x | 0.4% | WIN |
| gpt2m.ffn_up.m16 | 16x4096x1024 | mma_16x128x32_w16x32 | 36.9 | heuristic_top1 | 39.4 | 1.07x | 5.6% | WIN |
| gpt2m.ffn_up.m128 | 128x4096x1024 | split_k_atomic | 61.4 | gemmex | 49.2 | 0.80x | 26.8% | LOSS |
| gpt2m.ffn_up.m512 | 512x4096x1024 | mma_sync_swizzled | 278.7 | heuristic_top1 | 161.8 | 0.58x | 23.7% | LOSS |
| gpt2m.ffn_up.m2048 | 2048x4096x1024 | mma_128x128x32_w64x32 | 1202.3 | heuristic_top1 | 661.5 | 0.55x | 21.9% | LOSS |
| gpt2m.ffn_down.m1 | 1x1024x4096 | splitk_64x64x32_w32x32 | 37.9 | gemmex | 41.0 | 1.08x | 0.3% | WIN |
| gpt2m.ffn_down.m16 | 16x1024x4096 | splitk_64x64x32_w32x32 | 40.8 | FAIL | - | - | 5.1% | VENDOR FAIL |
| gpt2m.ffn_down.m128 | 128x1024x4096 | split_k | 63.5 | FAIL | - | - | 26.0% | VENDOR FAIL |
| gpt2m.ffn_down.m512 | 512x1024x4096 | mma_64x64x32_w32x32 | 244.9 | FAIL | - | - | 26.9% | VENDOR FAIL |
| gpt2m.ffn_down.m2048 | 2048x1024x4096 | mma_sync | 1147.1 | FAIL | - | - | 23.0% | VENDOR FAIL |
| gpt2m.lm_head.m1 | 1x50257x1024 | split_k_atomic | 503.5 | heuristic_top1 | 610.3 | 1.21x | 0.3% | WIN |
| gpt2m.lm_head.m16 | 16x50257x1024 | split_k_atomic | 573.1 | gemmex | 678.9 | 1.18x | 4.4% | WIN |
| gpt2m.lm_head.m128 | 128x50257x1024 | splitk_128x128x32_w64x32 | 1802.1 | heuristic_top1 | 1203.6 | 0.67x | 11.2% | LOSS |
| llama1b.qkv.m1 | 1x6144x2048 | mma_sync_swizzled | 98.3 | gemmex | 122.9 | 1.25x | 0.4% | WIN |
| llama1b.qkv.m16 | 16x6144x2048 | mma_16x128x32_w16x32 | 99.3 | FAIL | - | - | 6.2% | VENDOR FAIL |
| llama1b.qkv.m128 | 128x6144x2048 | mma_sync_swizzled | 201.5 | FAIL | - | - | 24.5% | VENDOR FAIL |
| llama1b.qkv.m512 | 512x6144x2048 | mma_128x128x32_w64x32 | 973.0 | heuristic_top1 | 495.6 | 0.51x | 20.3% | LOSS |
| llama1b.qkv.m2048 | 2048x6144x2048 | mma_128x128x32_w64x32 | 4134.2 | heuristic_top1 | 2091.2 | 0.51x | 19.1% | LOSS |
| llama1b.attn_out.m1 | 1x2048x2048 | splitk_64x64x32_w32x32 | 38.0 | heuristic_top1 | 39.8 | 1.05x | 0.3% | WIN |
| llama1b.attn_out.m16 | 16x2048x2048 | splitk_64x64x32_w32x32 | 41.0 | FAIL | - | - | 5.0% | VENDOR FAIL |
| llama1b.attn_out.m128 | 128x2048x2048 | mma_sync | 58.4 | FAIL | - | - | 28.2% | VENDOR FAIL |
| llama1b.attn_out.m512 | 512x2048x2048 | mma_sync | 231.8 | heuristic_top1 | 154.8 | 0.67x | 28.4% | LOSS |
| llama1b.attn_out.m2048 | 2048x2048x2048 | mma_128x128x32_w64x32 | 1159.0 | heuristic_top1 | 663.6 | 0.57x | 22.8% | LOSS |
| llama1b.ffn_up.m1 | 1x8192x2048 | mma_16x128x32_w16x32 | 125.2 | heuristic_top1 | 176.4 | 1.41x | 0.4% | WIN |
| llama1b.ffn_up.m16 | 16x8192x2048 | mma_16x128x32_w16x32 | 127.0 | heuristic_top1 | 139.1 | 1.10x | 6.5% | WIN |
| llama1b.ffn_up.m128 | 128x8192x2048 | mma_sync_swizzled | 286.0 | heuristic_top1 | 180.0 | 0.63x | 23.1% | LOSS |
| llama1b.ffn_up.m512 | 512x8192x2048 | mma_sync | 1328.7 | heuristic_top1 | 674.0 | 0.51x | 19.9% | LOSS |
| llama1b.ffn_up.m2048 | 2048x8192x2048 | mma_128x128x32_w64x32 | 5837.7 | heuristic_top1 | 2863.3 | 0.49x | 18.1% | LOSS |
| llama1b.ffn_down.m1 | 1x2048x8192 | split_k | 131.8 | heuristic_top1 | 229.4 | 1.74x | 0.4% | WIN |
| llama1b.ffn_down.m16 | 16x2048x8192 | split_k | 137.2 | FAIL | - | - | 6.0% | VENDOR FAIL |
| llama1b.ffn_down.m128 | 128x2048x8192 | mma_sync | 227.3 | FAIL | - | - | 29.0% | VENDOR FAIL |
| llama1b.ffn_down.m512 | 512x2048x8192 | mma_128x128x32_w64x32 | 1120.9 | FAIL | - | - | 23.5% | VENDOR FAIL |
| llama1b.ffn_down.m2048 | 2048x2048x8192 | mma_128x128x32_w64x32 | 5071.6 | FAIL | - | - | 20.8% | VENDOR FAIL |
| llama1b.lm_head.m1 | 1x32000x2048 | split_k_atomic | 534.5 | gemmex | 733.2 | 1.37x | 0.4% | WIN |
| llama1b.lm_head.m16 | 16x32000x2048 | split_k_atomic | 551.8 | heuristic_top1 | 557.1 | 1.01x | 5.8% | WIN |
| llama1b.lm_head.m128 | 128x32000x2048 | mma_128x128x32_w64x32 | 1334.5 | heuristic_top1 | 727.0 | 0.54x | 19.3% | LOSS |
| odd.1x1x1 | 1x1x1 | naive | 3.6 | gemmex | 4.1 | 1.12x | 0.0% | WIN |
| odd.7x3079x1025 | 7x3079x1025 | splitk_16x128x32_w16x32 | 37.7 | heuristic_top1 | 38.9 | 1.03x | 1.8% | WIN |
| odd.33x33x33 | 33x33x33 | naive | 4.6 | gemmex | 6.1 | 1.33x | 0.0% | WIN |
| odd.127x769x1023 | 127x769x1023 | splitk_64x64x32_w32x32 | 44.7 | FAIL | - | - | 6.9% | VENDOR FAIL |
| odd.1000x1000x1000 | 1000x1000x1000 | split_k | 120.8 | heuristic_top1 | 73.1 | 0.60x | 25.4% | LOSS |
| odd.1025x4097x33 | 1025x4097x33 | split_k_atomic | 116.4 | heuristic_top1 | 77.7 | 0.67x | 3.7% | LOSS |
| odd.2047x2049x127 | 2047x2049x127 | splitk_128x128x32_w64x32 | 190.7 | heuristic_top1 | 121.0 | 0.63x | 8.6% | LOSS |

f16 warm: WIN 24, LOSS 32, TIE 0, VENDOR FAIL 20

### FP32

FP32 has no Tensor Core path on sm_75; these rows are the Phase 9-10
CUDA-core kernels, re-measured in this session.

| shape | M x N x K | best ccml | us | best vendor | us | vendor/ccml | ccml %peak | verdict (95% CI) |
|---|---|---|---|---|---|---|---|---|
| gpt2s.qkv.m1 | 1x2304x768 | naive | 39.7 | heuristic_top1 | 31.7 | 0.80x | 1.1% | LOSS |
| gpt2s.qkv.m16 | 16x2304x768 | tiled | 75.8 | sgemm | 36.9 | 0.49x | 9.2% | LOSS |
| gpt2s.qkv.m128 | 128x2304x768 | double_buffered | 175.7 | heuristic_top1 | 80.6 | 0.46x | 31.7% | LOSS |
| gpt2s.qkv.m512 | 512x2304x768 | double_buffered | 395.3 | sgemm | 353.3 | 0.89x | 56.3% | LOSS |
| gpt2s.qkv.m2048 | 2048x2304x768 | double_buffered | 1945.4 | heuristic_top1 | 1793.2 | 0.92x | 45.8% | LOSS |
| gpt2s.attn_out.m1 | 1x768x768 | naive | 17.7 | sgemm | 7.9 | 0.44x | 0.8% | LOSS |
| gpt2s.attn_out.m16 | 16x768x768 | tiled | 38.8 | sgemm | 16.3 | 0.42x | 6.0% | LOSS |
| gpt2s.attn_out.m128 | 128x768x768 | double_buffered | 167.9 | sgemm | 45.1 | 0.27x | 11.0% | LOSS |
| gpt2s.attn_out.m512 | 512x768x768 | double_buffered | 188.4 | sgemm | 110.2 | 0.58x | 39.4% | LOSS |
| gpt2s.attn_out.m2048 | 2048x768x768 | double_buffered | 638.7 | sgemm | 503.8 | 0.79x | 46.5% | LOSS |
| gpt2s.ffn_up.m1 | 1x3072x768 | naive | 65.5 | sgemm | 41.0 | 0.62x | 0.9% | LOSS |
| gpt2s.ffn_up.m16 | 16x3072x768 | tiled | 100.4 | heuristic_top1 | 48.0 | 0.48x | 9.2% | LOSS |
| gpt2s.ffn_up.m128 | 128x3072x768 | double_buffered | 188.4 | sgemm | 153.6 | 0.82x | 39.4% | LOSS |
| gpt2s.ffn_up.m512 | 512x3072x768 | double_buffered | 636.9 | sgemm | 501.5 | 0.79x | 46.6% | LOSS |
| gpt2s.ffn_up.m2048 | 2048x3072x768 | double_buffered | 2723.6 | sgemm | 2428.6 | 0.89x | 43.6% | LOSS |
| gpt2s.ffn_down.m1 | 1x768x3072 | naive | 98.2 | sgemm | 44.9 | 0.46x | 0.6% | LOSS |
| gpt2s.ffn_down.m16 | 16x768x3072 | tiled | 159.7 | sgemm | 51.2 | 0.32x | 5.8% | LOSS |
| gpt2s.ffn_down.m128 | 128x768x3072 | double_buffered | 666.2 | sgemm | 116.0 | 0.17x | 11.1% | LOSS |
| gpt2s.ffn_down.m512 | 512x768x3072 | double_buffered | 811.4 | sgemm | 439.1 | 0.54x | 36.6% | LOSS |
| gpt2s.ffn_down.m2048 | 2048x768x3072 | double_buffered | 2909.4 | heuristic_top1 | 2386.0 | 0.82x | 40.8% | LOSS |
| gpt2s.lm_head.m1 | 1x50257x768 | naive | 728.6 | heuristic_top1 | 606.6 | 0.83x | 1.3% | LOSS |
| gpt2s.lm_head.m16 | 16x50257x768 | tiled | 2146.6 | sgemm | 729.4 | 0.34x | 7.1% | LOSS |
| gpt2s.lm_head.m128 | 128x50257x768 | double_buffered | 3604.5 | heuristic_top1 | 3176.7 | 0.88x | 33.7% | LOSS |
| gpt2m.qkv.m1 | 1x3072x1024 | naive | 87.1 | sgemm | 53.2 | 0.61x | 0.9% | LOSS |
| gpt2m.qkv.m16 | 16x3072x1024 | tiled | 145.4 | heuristic_top1 | 60.5 | 0.42x | 8.5% | LOSS |
| gpt2m.qkv.m128 | 128x3072x1024 | double_buffered | 249.9 | sgemm | 151.4 | 0.61x | 39.6% | LOSS |
| gpt2m.qkv.m512 | 512x3072x1024 | double_buffered | 877.2 | sgemm | 704.5 | 0.80x | 45.1% | LOSS |
| gpt2m.qkv.m2048 | 2048x3072x1024 | double_buffered | 3592.3 | sgemm | 3385.4 | 0.94x | 44.1% | LOSS |
| gpt2m.attn_out.m1 | 1x1024x1024 | naive | 25.2 | sgemm | 8.7 | 0.35x | 1.0% | LOSS |
| gpt2m.attn_out.m16 | 16x1024x1024 | tiled | 56.0 | heuristic_top1 | 27.9 | 0.50x | 7.4% | LOSS |
| gpt2m.attn_out.m128 | 128x1024x1024 | double_buffered | 216.8 | heuristic_top1 | 56.2 | 0.26x | 15.2% | LOSS |
| gpt2m.attn_out.m512 | 512x1024x1024 | double_buffered | 268.4 | sgemm | 195.6 | 0.73x | 49.1% | LOSS |
| gpt2m.attn_out.m2048 | 2048x1024x1024 | double_buffered | 1181.6 | sgemm | 1030.2 | 0.87x | 44.6% | LOSS |
| gpt2m.ffn_up.m1 | 1x4096x1024 | naive | 96.3 | heuristic_top1 | 69.1 | 0.72x | 1.1% | LOSS |
| gpt2m.ffn_up.m16 | 16x4096x1024 | tiled | 193.8 | sgemm | 86.0 | 0.44x | 8.5% | LOSS |
| gpt2m.ffn_up.m128 | 128x4096x1024 | double_buffered | 269.4 | sgemm | 202.8 | 0.75x | 49.0% | LOSS |
| gpt2m.ffn_up.m512 | 512x4096x1024 | double_buffered | 1209.7 | sgemm | 1032.1 | 0.85x | 43.6% | LOSS |
| gpt2m.ffn_up.m2048 | 2048x4096x1024 | double_buffered | 4717.6 | heuristic_top1 | 4151.3 | 0.88x | 44.7% | LOSS |
| gpt2m.ffn_down.m1 | 1x1024x4096 | naive | 135.3 | sgemm | 70.8 | 0.52x | 0.8% | LOSS |
| gpt2m.ffn_down.m16 | 16x1024x4096 | tiled | 250.2 | sgemm | 77.8 | 0.31x | 6.6% | LOSS |
| gpt2m.ffn_down.m128 | 128x1024x4096 | double_buffered | 855.8 | sgemm | 172.4 | 0.20x | 15.4% | LOSS |
| gpt2m.ffn_down.m512 | 512x1024x4096 | double_buffered | 1234.9 | sgemm | 988.3 | 0.80x | 42.7% | LOSS |
| gpt2m.ffn_down.m2048 | 2048x1024x4096 | double_buffered | 5157.2 | sgemm | 4532.4 | 0.88x | 40.9% | LOSS |
| gpt2m.lm_head.m1 | 1x50257x1024 | naive | 999.6 | heuristic_top1 | 807.4 | 0.81x | 1.3% | LOSS |
| gpt2m.lm_head.m16 | 16x50257x1024 | tiled | 2972.2 | sgemm | 992.7 | 0.33x | 6.8% | LOSS |
| gpt2m.lm_head.m128 | 128x50257x1024 | double_buffered | 4751.4 | heuristic_top1 | 4320.9 | 0.91x | 34.1% | LOSS |
| llama1b.qkv.m1 | 1x6144x2048 | naive | 257.0 | heuristic_top1 | 214.4 | 0.83x | 1.2% | LOSS |
| llama1b.qkv.m16 | 16x6144x2048 | tiled | 680.3 | sgemm | 232.4 | 0.34x | 7.3% | LOSS |
| llama1b.qkv.m128 | 128x6144x2048 | double_buffered | 1077.9 | sgemm | 676.2 | 0.63x | 36.7% | LOSS |
| llama1b.qkv.m512 | 512x6144x2048 | double_buffered | 3669.5 | sgemm | 3299.3 | 0.90x | 43.1% | LOSS |
| llama1b.qkv.m2048 | 2048x6144x2048 | double_buffered | 15045.5 | sgemm | 15783.9 | 1.05x | 42.1% | WIN |
| llama1b.attn_out.m1 | 1x2048x2048 | naive | 92.8 | sgemm | 80.6 | 0.87x | 1.1% | LOSS |
| llama1b.attn_out.m16 | 16x2048x2048 | tiled | 220.4 | heuristic_top1 | 84.0 | 0.38x | 7.5% | LOSS |
| llama1b.attn_out.m128 | 128x2048x2048 | double_buffered | 447.2 | sgemm | 176.4 | 0.39x | 29.5% | LOSS |
| llama1b.attn_out.m512 | 512x2048x2048 | double_buffered | 1238.0 | sgemm | 1017.2 | 0.82x | 42.6% | LOSS |
| llama1b.attn_out.m2048 | 2048x2048x2048 | double_buffered | 4834.9 | sgemm | 4902.9 | 1.01x | 43.6% | TIE |
| llama1b.ffn_up.m1 | 1x8192x2048 | naive | 373.9 | heuristic_top1 | 272.1 | 0.73x | 1.1% | LOSS |
| llama1b.ffn_up.m16 | 16x8192x2048 | tiled | 905.9 | heuristic_top1 | 326.7 | 0.36x | 7.3% | LOSS |
| llama1b.ffn_up.m128 | 128x8192x2048 | double_buffered | 1269.7 | sgemm | 1145.5 | 0.90x | 41.5% | LOSS |
| llama1b.ffn_up.m512 | 512x8192x2048 | double_buffered | 5056.5 | sgemm | 4573.0 | 0.90x | 41.7% | LOSS |
| llama1b.ffn_up.m2048 | 2048x8192x2048 | double_buffered | 19752.8 | heuristic_top1 | 19566.7 | 0.99x | 42.7% | TIE |
| llama1b.ffn_down.m1 | 1x2048x8192 | naive | 386.8 | sgemm | 297.7 | 0.77x | 1.1% | LOSS |
| llama1b.ffn_down.m16 | 16x2048x8192 | tiled | 1032.5 | sgemm | 272.1 | 0.26x | 6.4% | LOSS |
| llama1b.ffn_down.m128 | 128x2048x8192 | double_buffered | 1962.8 | sgemm | 801.0 | 0.41x | 26.9% | LOSS |
| llama1b.ffn_down.m512 | 512x2048x8192 | double_buffered | 5464.9 | heuristic_top1 | 4588.0 | 0.84x | 38.6% | LOSS |
| llama1b.ffn_down.m2048 | 2048x2048x8192 | double_buffered | 19839.9 | heuristic_top1 | 19384.7 | 0.98x | 42.5% | LOSS |
| llama1b.lm_head.m1 | 1x32000x2048 | naive | 1498.2 | heuristic_top1 | 1068.8 | 0.71x | 1.1% | LOSS |
| llama1b.lm_head.m16 | 16x32000x2048 | tiled | 3930.4 | sgemm | 1257.5 | 0.32x | 6.6% | LOSS |
| llama1b.lm_head.m128 | 128x32000x2048 | double_buffered | 4941.9 | sgemm | 4626.4 | 0.94x | 41.7% | LOSS |
| odd.1x1x1 | 1x1x1 | naive | 3.8 | sgemm | 4.1 | 1.07x | 0.0% | WIN |
| odd.7x3079x1025 | 7x3079x1025 | naive | 106.5 | heuristic_top1 | 63.5 | 0.60x | 5.1% | LOSS |
| odd.33x33x33 | 33x33x33 | naive | 4.6 | sgemm | 6.8 | 1.48x | 0.2% | WIN |
| odd.127x769x1023 | 127x769x1023 | double_buffered | 238.3 | sgemm | 53.3 | 0.22x | 10.3% | LOSS |
| odd.1000x1000x1000 | 1000x1000x1000 | double_buffered | 524.2 | sgemm | 419.4 | 0.80x | 46.9% | LOSS |
| odd.1025x4097x33 | 1025x4097x33 | double_buffered | 184.3 | sgemm | 141.2 | 0.77x | 18.5% | LOSS |
| odd.2047x2049x127 | 2047x2049x127 | double_buffered | 345.3 | sgemm | 241.0 | 0.70x | 37.9% | LOSS |

f32 warm: WIN 3, LOSS 71, TIE 2, VENDOR FAIL 0

## Fusion vs separate kernels (reference.md section 5; F1, F4, B2)

`python scripts/epilogue_summary.py results/sm_75/tensor_cores/matmul_epilogue.csv`;
ratio = competitor time / ccml fused time (> 1: fused ccml faster), warm L2.
Unfused pipelines, and cuBLASLt's epilogue (whose error measured equal to
theirs on sm_89), are gated with a per-element allowance for their
intermediate FP16 rounding (log entry 8); ccml fused is held to the strict
bound.

large ours+separate: median 1.13x (n=24)  cublas+separate: median 0.77x (n=19)  cublaslt_epilogue: median 0.53x (n=19)
mid ours+separate: median 1.09x (n=12)  cublas+separate: median 1.05x (n=5)  cublaslt_epilogue: median 0.88x (n=5)
decode ours+separate: median 1.03x (n=24)  cublas+separate: median 1.23x (n=17)  cublaslt_epilogue: median 1.20x (n=12)
| shape | ccml fused us | ours+separate | cublas+separate | cublaslt_epilogue |
|---|---|---|---|---|
| gpt2s.qkv.m1 | 13.8 | 1.04x | 1.50x | 1.37x |
| gpt2s.qkv.m16 | 16.5 | 1.08x | 1.20x | 1.07x |
| gpt2s.qkv.m128 | 28.5 | 1.13x | 1.15x | 0.93x |
| gpt2s.qkv.m512 | 104.1 | 1.22x | 0.77x | 0.53x |
| gpt2s.qkv.m2048 | 541.2 | 1.21x | 0.77x | 0.48x |
| gpt2s.attn_out.m1 | 10.0 | 1.14x | 1.23x | 1.03x |
| gpt2s.attn_out.m16 | 10.2 | 1.18x | 1.12x | FAIL |
| gpt2s.attn_out.m128 | 17.9 | 1.13x | FAIL | FAIL |
| gpt2s.attn_out.m512 | 38.9 | 1.05x | 0.90x | 0.71x |
| gpt2s.attn_out.m2048 | 163.4 | 1.13x | 1.00x | 0.69x |
| gpt2s.ffn_up.m1 | 26.3 | 1.03x | 1.36x | 1.41x |
| gpt2s.ffn_up.m16 | 28.7 | 1.00x | FAIL | FAIL |
| gpt2s.ffn_up.m128 | 40.9 | 1.13x | 1.09x | 1.06x |
| gpt2s.ffn_up.m512 | 145.4 | 1.32x | 1.08x | 0.86x |
| gpt2s.ffn_up.m2048 | 790.0 | 1.28x | 0.76x | 0.58x |
| gpt2s.ffn_down.m1 | 27.0 | 1.10x | 1.06x | FAIL |
| gpt2s.ffn_down.m16 | 30.9 | 1.09x | FAIL | FAIL |
| gpt2s.ffn_down.m128 | 47.1 | 1.05x | FAIL | FAIL |
| gpt2s.ffn_down.m512 | 141.3 | 1.08x | 0.81x | 0.72x |
| gpt2s.ffn_down.m2048 | 737.3 | 1.05x | FAIL | FAIL |
| gpt2m.qkv.m1 | 32.8 | 1.06x | 1.38x | 1.44x |
| gpt2m.qkv.m16 | 34.6 | 1.01x | FAIL | FAIL |
| gpt2m.qkv.m128 | 48.6 | 1.10x | 1.05x | 0.88x |
| gpt2m.qkv.m512 | 201.0 | 1.26x | 0.93x | 0.69x |
| gpt2m.qkv.m2048 | 1132.4 | 1.22x | 0.63x | 0.43x |
| gpt2m.attn_out.m1 | 10.9 | 1.13x | 1.49x | 1.30x |
| gpt2m.attn_out.m16 | 11.1 | 1.13x | 1.26x | FAIL |
| gpt2m.attn_out.m128 | 26.3 | 0.95x | FAIL | FAIL |
| gpt2m.attn_out.m512 | 68.3 | 1.09x | 0.78x | 0.60x |
| gpt2m.attn_out.m2048 | 270.6 | 1.27x | 0.86x | 0.58x |
| gpt2m.ffn_up.m1 | 38.1 | 1.02x | 1.13x | 1.17x |
| gpt2m.ffn_up.m16 | 40.2 | 1.02x | 1.07x | 1.01x |
| gpt2m.ffn_up.m128 | 67.2 | 1.13x | 0.92x | 0.88x |
| gpt2m.ffn_up.m512 | 300.3 | 1.27x | 0.79x | 0.61x |
| gpt2m.ffn_up.m2048 | 1748.7 | 1.16x | 0.60x | 0.46x |
| gpt2m.ffn_down.m1 | 41.6 | 1.04x | 1.07x | FAIL |
| gpt2m.ffn_down.m16 | 44.8 | 1.05x | FAIL | FAIL |
| gpt2m.ffn_down.m128 | 69.1 | 1.04x | FAIL | FAIL |
| gpt2m.ffn_down.m512 | 300.3 | 1.01x | FAIL | FAIL |
| gpt2m.ffn_down.m2048 | 1431.0 | 1.02x | FAIL | FAIL |
| llama1b.qkv.m1 | 100.4 | 1.01x | 1.25x | 1.22x |
| llama1b.qkv.m16 | 102.4 | 1.02x | FAIL | FAIL |
| llama1b.qkv.m128 | 244.0 | 1.07x | FAIL | FAIL |
| llama1b.qkv.m512 | 1104.4 | 1.17x | 0.60x | 0.44x |
| llama1b.qkv.m2048 | 6921.4 | 1.04x | 0.36x | 0.30x |
| llama1b.attn_out.m1 | 43.9 | 1.03x | 1.03x | 0.95x |
| llama1b.attn_out.m16 | 44.9 | 1.05x | FAIL | FAIL |
| llama1b.attn_out.m128 | 70.0 | 1.03x | FAIL | FAIL |
| llama1b.attn_out.m512 | 273.7 | 1.18x | 0.72x | 0.53x |
| llama1b.attn_out.m2048 | 1686.6 | 1.06x | 0.55x | 0.40x |
| llama1b.ffn_up.m1 | 131.4 | 1.00x | 1.38x | 1.42x |
| llama1b.ffn_up.m16 | 133.1 | 1.02x | 1.08x | 1.10x |
| llama1b.ffn_up.m128 | 315.4 | 1.11x | 0.71x | 0.66x |
| llama1b.ffn_up.m512 | 1613.9 | 1.12x | 0.57x | 0.46x |
| llama1b.ffn_up.m2048 | 9124.9 | 1.04x | 0.38x | 0.34x |
| llama1b.ffn_down.m1 | 133.1 | 1.02x | 1.74x | FAIL |
| llama1b.ffn_down.m16 | 138.4 | 1.03x | FAIL | FAIL |
| llama1b.ffn_down.m128 | 317.9 | 1.03x | FAIL | FAIL |
| llama1b.ffn_down.m512 | 1563.0 | 1.00x | FAIL | FAIL |
| llama1b.ffn_down.m2048 | 8384.9 | 1.01x | FAIL | FAIL |

## What this proves

- The Tensor Core rungs work and are correct for every shape class, dtype
  contract and pointer alignment tested, and pass all four sanitizer tools.
- On the T4, the ladder from double buffering to `mma.sync` is worth about
  **4.3x on large FP16 shapes** (2.83 x 1.53) and **9x at decode sizes**.
- **Hand-written kernels beat cuBLAS on the decode shapes of every target
  model** (19 shapes, 1.04-1.69x): the shapes reference.md section 11 points
  at, skinny weight streams where a 16-row Tensor Core tile plus split-K fills
  the GPU.
- cuBLAS FP16 (GemmEx and the Lt heuristic, and so PyTorch eager and
  torch.compile, which fail the same 32 rows) loses accuracy on 20 shapes,
  mostly long-K FFN-down (K = 3072-8192) and some M = 128 shapes, by more than
  FP32 accumulation allows; every ccml kernel passes the same gate. The cause
  (partial sums rounded to FP16 in a split-K reduction) is inferred, not
  verified.
- Fusing the epilogue saves 1.13x (median) over our own unfused pipeline on
  large shapes and beats cuBLAS + separate kernel and cuBLASLt's fused
  epilogue at decode sizes (1.20-1.23x).

## What it does not prove

- **Large shapes are a loss**: MMA_SYNC is 0.58x of cuBLAS at the median
  (M >= 512). Missing: a multistage `cp.async` pipeline (sm_80+ only,
  docs/local.md L2), larger warp tiles without spills (configuration 0 spills
  at the 128-register floor), and the Turing-specific tuning cuBLAS has. The
  fused epilogue cannot recover that gap (0.77x of cuBLAS + separate, 0.53x
  of cuBLASLt's epilogue on large shapes).
- Split-K's slice heuristic hurts some shapes (gpt2s qkv 16x2304x768: 15.5 ->
  18.1 us; some M = 128 grids up to 1.5x slower). AUTO does not yet choose per
  shape (enhancement S4); the wins above are each shape's best row.
- Swizzling helps only where operands dwarf L2 and the grid spans many waves
  (llama1b ffn_up M = 2048: 1.53x); elsewhere it is neutral.
- BF16 Tensor Core numbers exist only for sm_89 (indicative, log entries 4-8);
  they are not mixed into these sm_75 tables.
- No end-to-end model latency yet (Phase 14).

## How to reproduce

    git checkout 5edb43d
    python cloud/kaggle/kaggle_run.py --jobs sanitize,matmul --slug-suffix tc
    python cloud/kaggle/kaggle_run.py --fetch --jobs sanitize,matmul --slug-suffix tc
    # the fetch writes results/sm_75/; the files were moved to results/sm_75/tensor_cores/
    python scripts/vendor_table.py results/sm_75/tensor_cores/matmul.csv --dtype f16
    python scripts/step_summary.py results/sm_75/tensor_cores/matmul.csv --dtype f16 --pairs double_buffered:wmma,wmma:mma_sync,mma_sync:split_k,mma_sync:mma_sync_swizzled
    python scripts/epilogue_summary.py results/sm_75/tensor_cores/matmul_epilogue.csv
    python scripts/dtype_summary.py results/sm_75/tensor_cores/matmul.csv

Local (sm_89; BF16 Tensor Cores and NCU): `powershell -File scripts/local_build.ps1`,
then the `scripts/ncu_matmul.ps1` commands quoted in log entries 4-8, from an
elevated prompt.
