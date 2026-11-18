# Phase 13: Softmax, online and fused (+ S4, F3)

**Date:** 2026-11-18
**Branch:** `softmax-variants` (worktree), merged into `dev` after Phase 12
**Outcome:** every reference.md section 3 softmax item is built and measured
on the Tesla T4: a numerically stable online softmax, warp-per-row for short
rows, block-per-row for long rows (register-resident, entry 17), a row split
across blocks for few very long rows, and a fused scale + mask (F3), with AUTO
chosen per dtype and width from the measurements (S4). **Short and long
attention rows run at the DRAM peak** (211-253 GB/s, 93-111% of the measured
copy peak) and are **at parity with the best of cuDNN, PyTorch eager,
torch.compile (max-autotune) and Triton** (median 0.94-0.99x over all shapes;
on attention rows of 256-2048 columns, wins of up to 15% and losses of up to
14%). **The fused masked
softmax beats every PyTorch baseline on 91 of 96 warm shapes** (median 1.96x
FP32, 1.97x FP16, 3.05x BF16; all wins clock-checked), including a compile
baseline that builds the causal mask in-graph. **Losses:** 128-column rows
(4-29%: launch and tail latency on sub-MB tensors), and vocabulary rows
of 30522+ columns: 128 rows lose 23-41% in every dtype, 16 rows lose 16-40%
in FP32 and BF16 (FP16 ties); these are past the register limit and run the
two-read ONLINE kernel.

## Environment

Kaggle, two sessions at commit **`a56eb2b`** (`ccml-softmax-sm`,
`ccml-sanitize-sm`; `results/sm_75/softmax_variants/`). Tesla T4 (sm_75),
driver 580.178.04, CUDA 13.0, torch 2.11.0+cu130, triton 3.6.0
(`env_softmax.txt`). Measured DRAM peak this session 227.3 GB/s
(`peaks_softmax.json`; 320.1 theoretical). SM clocks 585-1590 MHz, NVML
reason 0x4 (power cap) on 978 of 2287 rows, no thermal or hardware
throttling (`throttled` false on every row, max 81 C). Candidates are
interleaved within a shape. Because the T4 drops clocks at its 70 W cap, a win
over a baseline that ran >5% slower counts only if it survives clock
normalization (`scripts/rowwise_summary.py`; as in Phase 11); losses are
reported raw. Nsight Compute evidence is from the local RTX 3050 A Laptop
(sm_89; counters are blocked on Kaggle, docs/local.md L3), and only for
entry 13: the elevated prompt for entries 14, 16 and 17 was declined.

An earlier full run at commit `01c3555` (before entry 17 and the in-graph
causal baseline) is in git history (commit 56f5f1c); every number here comes
from the `a56eb2b` run, so all of Phase 13 rests on one session.

## What was added

| Step | Variant / API | One change | Log entry |
|---|---|---|---|
| Online | `CCML_SOFTMAX_ONLINE` | single pass running (max, sum) in FP32 (Milakov & Gimelshein 2018), two reads instead of three, no FP64 | 13 |
| Short rows | `CCML_SOFTMAX_WARP_PER_ROW` | one warp per row, row in registers, 128-bit loads, one exp per element; cols <= 1024 | 14 |
| Few long rows | `CCML_SOFTMAX_SPLIT_ROW` | a row split across blocks, two kernels, deterministic merge | 15 |
| F3 | `ccml_softmax_masked_<dtype>` | scale + additive mask + bottom-right causal mask fused into the load; fully masked rows are zeros | 16 |
| Long rows | `CCML_SOFTMAX_BLOCK_RESIDENT` | entry 14's design at block scope: cols <= 8192 FP32, 16384 16-bit (aligned) | 17 |
| S4 | AUTO | per dtype and width, from this run (`resolve_auto`, src/softmax/softmax.cu) | 13-17 |

Harness: `ccml_bench_softmax --dtype`, cuDNN softmax for FP16/BF16,
`ccml_bench_softmax_masked` with our own unfused pipeline
(`bench/baselines/scale_mask.cu` + the same variant) as the fusion baseline,
`bench_torch.py` masked baselines (eager, torch.compile with a materialized
mask, torch.compile with the causal mask built in-graph), and
`scripts/rowwise_summary.py` / `scripts/rowwise_steps.py`.

### Tests written

- `softmax` (17 cases): every algorithm on 13 shared cases (1 to 50257
  columns, +-5000 offsets, one-hot, constant rows), the special-values policy
  (partial and full -inf rows, NaN poisoning, a dominant element), every
  WARP_PER_ROW chunk bucket on vector and scalar paths with a misaligned
  view, every BLOCK_RESIDENT bucket edge with both refusals, SPLIT_ROW
  segment boundaries, -inf and NaN across segments, and bitwise determinism.
- `softmax_masked` (5 cases): padding and causal masks, causal with
  q < k (decode), mask broadcasting, fully masked rows are exactly zero,
  argument checks.
- `dtypes`: FP16 and BF16 for every softmax algorithm against the fp64
  reference with the storage-rounding allowance.

## Verbatim results

ctest (`ctest.txt`): `100% tests passed, 0 tests failed out of 11`.
`softmax`: `17 passed, 0 failed, 0 skipped`; `softmax_masked`: `5 passed, 0
failed, 0 skipped`; `dtypes`: `13 passed, 0 failed, 0 skipped`.
compute-sanitizer (`sanitizer/summary.txt`): memcheck, racecheck, initcheck
and synccheck **clean on all 11 suites** (plus the one deliberate OOM in
`infrastructure`, as before). Every row of `softmax.csv`,
`softmax_masked.csv`, `torch_softmax.csv` and `torch_softmax_masked.csv` is
PASS, or NOT_RUN for a variant's documented refusal (WARP_PER_ROW past 1024
columns, BLOCK_RESIDENT past its register limit, cuDNN BF16 on sm_75).

Locally (sm_89, `scripts/local_build.ps1`): 11/11 suites.

## Per-step speedups (T4, warm L2, median over shapes; rows >= 1000 unless noted)

| Step | Shapes | FP32 | FP16 | BF16 | Reaches |
|---|---|---|---|---|---|
| BLOCK_PER_ROW -> ONLINE (13) | cols >= 1000 | 2.26x | 2.71x | 4.58x | 140-211 GB/s FP32, 85-137 16-bit |
| ONLINE -> WARP_PER_ROW (14) | cols 128-512 | 2.48x | 3.63x | 3.29x | 211-240 GB/s beyond L2 |
| ONLINE -> WARP_PER_ROW (14) | cols 1024 | 1.15x | 2.09x | 2.09x | 228-234 GB/s |
| ONLINE -> SPLIT_ROW (15) | 1 row x 30522-50257 | 3.35x | 4.67x | 4.70x | 9-28 GB/s |
| ONLINE -> SPLIT_ROW (15) | 16 rows x 30522-50257 | 2.08x | 3.04x | 3.01x | 93-184 GB/s |
| ONLINE -> BLOCK_RESIDENT (17) | rows > 16, cols 1025-4097 | 1.31x | 2.25x | 2.35x | 177-253 GB/s |
| separate -> fused mask (16) | attention, beyond L2 | 1.97x | 2.28x | 2.49x | |

`python scripts/rowwise_steps.py results/sm_75/softmax_variants/softmax.csv
--pairs PREV:NEW` with the filters of log entries 13-17. Each entry's
predictions are judged there, including the ones the T4 falsified (entry 13
prediction 4, entry 16 predictions 2 and, for BF16, 3).

## Shape-by-shape vs the baselines (reference.md section 9)

Best ccml variant per shape against the best of cuDNN
(`softmax_accurate_instance`), PyTorch eager, torch.compile max-autotune and
Triton (`online_looped`), warm L2. Verdict from non-overlapping 95% CIs;
CLOCK = a win that does not survive clock normalization (not claimed).
`python scripts/rowwise_summary.py results/sm_75/softmax_variants/softmax.csv
results/sm_75/softmax_variants/torch_softmax.csv --dtype <dt> --table`.

**Totals, plain softmax.** Warm: FP32 WIN 9, LOSS 20, TIE 3; FP16 WIN 7,
LOSS 17, TIE 5, CLOCK 3; BF16 WIN 10, LOSS 19, TIE 1, CLOCK 2 (median
baseline/ccml 0.96x, 0.99x, 0.94x). Cold L2: FP32 WIN 1, LOSS 14, TIE 2;
FP16 WIN 6, LOSS 12, TIE 3, CLOCK 2; BF16 WIN 9, LOSS 11, TIE 1, CLOCK 2.
**No ccml row fails the gate; no baseline row fails it either.**

**Totals, fused scale + mask** (against eager and both compile baselines;
cuDNN has no masked softmax). Warm: FP32 WIN 31, CLOCK 1; FP16 WIN 28, LOSS
4; BF16 WIN 32 (median 1.96x, 1.97x, 3.05x). Cold: WIN 11, 15, 15, no losses.
The 4 FP16 losses (3-10%) are 2048-column rows, where the fused op still runs
ONLINE (no register-resident masked variant yet).

### Plain softmax, F16

| shape | rows x cols | best ccml | us | GB/s | % measured peak | best baseline | us | baseline/ccml | verdict (95% CI) | SM MHz ccml/base |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2s.attn.s128 | 1536 x 128 | warp_per_row | 6.3 | 124 | 55% | cudnn softmax_accurate_instance | 5.2 | 0.82x | LOSS | 1590/1590 |
| gpt2s.attn.s256 | 3072 x 256 | warp_per_row | 8.4 | 375 | 165% | cudnn softmax_accurate_instance | 8.0 | 0.96x | LOSS | 1590/1590 |
| gpt2s.attn.s512 | 6144 x 512 | warp_per_row | 57.3 | 219 | 97% | triton online_looped | 57.7 | 1.01x | CLOCK | 1575/1200 |
| gpt2s.attn.s1024 | 12288 x 1024 | block_resident | 211.7 | 238 | 105% | triton online_looped | 206.8 | 0.98x | LOSS | 1320/1365 |
| gpt2s.attn.s2048 | 24576 x 2048 | block_resident | 811.0 | 248 | 109% | triton online_looped | 806.6 | 0.99x | LOSS | 1095/1095 |
| gpt2m.attn.s128 | 2048 x 128 | warp_per_row | 7.0 | 150 | 66% | cudnn softmax_accurate_instance | 5.8 | 0.82x | LOSS | 1590/1590 |
| gpt2m.attn.s256 | 4096 x 256 | warp_per_row | 10.2 | 410 | 180% | cudnn softmax_accurate_instance | 10.2 | 1.00x | TIE | 1590/1590 |
| gpt2m.attn.s512 | 8192 x 512 | warp_per_row | 75.9 | 221 | 97% | cudnn softmax_accurate_instance | 77.3 | 1.02x | CLOCK | 1560/1440 |
| gpt2m.attn.s1024 | 16384 x 1024 | warp_per_row | 290.5 | 231 | 102% | triton online_looped | 272.4 | 0.94x | LOSS | 1065/1320 |
| gpt2m.attn.s2048 | 32768 x 2048 | block_resident | 1071.1 | 251 | 110% | triton online_looped | 1084.9 | 1.01x | WIN | 975/1035 |
| llama1b.attn.s128 | 4096 x 128 | warp_per_row | 10.2 | 206 | 91% | cudnn softmax_accurate_instance | 7.4 | 0.73x | LOSS | 1590/1590 |
| llama1b.attn.s256 | 8192 x 256 | warp_per_row | 39.7 | 211 | 93% | cudnn softmax_accurate_instance | 39.6 | 1.00x | TIE | 1440/1470 |
| llama1b.attn.s512 | 16384 x 512 | warp_per_row | 147.1 | 228 | 100% | triton online_looped | 140.5 | 0.96x | LOSS | 1260/1485 |
| llama1b.attn.s1024 | 32768 x 1024 | warp_per_row | 574.3 | 234 | 103% | triton online_looped | 537.8 | 0.94x | LOSS | 945/1155 |
| llama1b.attn.s2048 | 65536 x 2048 | block_resident | 2118.5 | 253 | 112% | triton online_looped | 2430.8 | 1.15x | WIN | 915/885 |
| gpt2s.attn.b8.s512 | 49152 x 512 | warp_per_row | 422.3 | 238 | 105% | triton online_looped | 418.8 | 0.99x | LOSS | 975/1185 |
| vocab30522.rows1 | 1 x 30522 | split_row | 13.3 | 9 | 4% | triton online_looped | 23.2 | 1.74x | WIN | 1590/1170 |
| vocab30522.rows16 | 16 x 30522 | split_row | 20.6 | 95 | 42% | triton online_looped | 30.8 | 1.50x | TIE | 1590/1485 |
| vocab30522.rows128 | 128 x 30522 | split_row | 174.0 | 90 | 40% | triton online_looped | 124.7 | 0.72x | LOSS | 1545/1395 |
| vocab32000.rows1 | 1 x 32000 | split_row | 13.5 | 10 | 4% | triton online_looped | 25.6 | 1.90x | WIN | 1590/1335 |
| vocab32000.rows16 | 16 x 32000 | split_row | 20.8 | 99 | 43% | triton online_looped | 26.6 | 1.28x | TIE | 1590/1410 |
| vocab32000.rows128 | 128 x 32000 | split_row | 171.3 | 96 | 42% | triton online_looped | 108.5 | 0.63x | LOSS | 1545/1395 |
| vocab50257.rows1 | 1 x 50257 | split_row | 14.6 | 14 | 6% | triton online_looped | 30.7 | 2.11x | WIN | 1590/1305 |
| vocab50257.rows16 | 16 x 50257 | split_row | 28.7 | 112 | 49% | triton online_looped | 31.9 | 1.11x | CLOCK | 1590/1350 |
| vocab50257.rows128 | 128 x 50257 | split_row | 293.4 | 88 | 39% | pytorch_eager torch.softmax | 204.8 | 0.70x | LOSS | 1500/1440 |
| odd.c1.rows1000 | 1000 x 1 | warp_per_row | 4.2 | 1 | 0% | pytorch_eager torch.softmax | 3.7 | 0.89x | LOSS | 1590/1590 |
| odd.c7.rows1000 | 1000 x 7 | warp_per_row | 4.2 | 7 | 3% | cudnn softmax_accurate_instance | 3.9 | 0.94x | LOSS | 1590/1590 |
| odd.c33.rows1000 | 1000 x 33 | warp_per_row | 4.3 | 30 | 13% | cudnn softmax_accurate_instance | 4.4 | 1.00x | TIE | 1590/1590 |
| odd.c1000.rows1000 | 1000 x 1000 | warp_per_row | 10.4 | 383 | 169% | triton online_looped | 26.7 | 2.56x | WIN | 1590/1500 |
| odd.c1025.rows1000 | 1000 x 1025 | block_resident | 21.2 | 194 | 85% | cudnn softmax_accurate_instance | 30.0 | 1.42x | WIN | 1590/1590 |
| odd.c3079.rows1000 | 1000 x 3079 | online | 92.2 | 134 | 59% | triton online_looped | 71.2 | 0.77x | LOSS | 1560/1560 |
| odd.c4097.rows1000 | 1000 x 4097 | online | 119.8 | 137 | 60% | cudnn softmax_accurate_instance | 93.4 | 0.78x | LOSS | 1530/1515 |

### Plain softmax, F32

| shape | rows x cols | best ccml | us | GB/s | % measured peak | best baseline | us | baseline/ccml | verdict (95% CI) | SM MHz ccml/base |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2s.attn.s128 | 1536 x 128 | warp_per_row | 5.6 | 280 | 123% | cudnn softmax_accurate_instance | 5.2 | 0.93x | LOSS | 1590/1590 |
| gpt2s.attn.s256 | 3072 x 256 | warp_per_row | 30.0 | 210 | 92% | cudnn softmax_accurate_instance | 25.7 | 0.86x | LOSS | 1590/1590 |
| gpt2s.attn.s512 | 6144 x 512 | block_resident | 106.0 | 237 | 104% | triton online_looped | 106.5 | 1.00x | WIN | 1530/1485 |
| gpt2s.attn.s1024 | 12288 x 1024 | block_resident | 416.9 | 241 | 106% | cudnn softmax_accurate_instance | 420.0 | 1.01x | WIN | 1410/1500 |
| gpt2s.attn.s2048 | 24576 x 2048 | block_resident | 1688.6 | 238 | 105% | triton online_looped | 1706.6 | 1.01x | WIN | 1035/1140 |
| gpt2m.attn.s128 | 2048 x 128 | warp_per_row | 6.1 | 345 | 152% | cudnn softmax_accurate_instance | 5.9 | 0.96x | LOSS | 1590/1590 |
| gpt2m.attn.s256 | 4096 x 256 | warp_per_row | 38.9 | 216 | 95% | triton online_looped | 38.6 | 0.99x | LOSS | 1590/1290 |
| gpt2m.attn.s512 | 8192 x 512 | block_resident | 139.3 | 241 | 106% | triton online_looped | 139.4 | 1.00x | TIE | 1380/1560 |
| gpt2m.attn.s1024 | 16384 x 1024 | block_resident | 549.0 | 244 | 108% | cudnn softmax_accurate_instance | 559.2 | 1.02x | WIN | 1110/1170 |
| gpt2m.attn.s2048 | 32768 x 2048 | block_resident | 2244.6 | 239 | 105% | triton online_looped | 2265.1 | 1.01x | WIN | 960/1005 |
| llama1b.attn.s128 | 4096 x 128 | warp_per_row | 8.5 | 493 | 217% | cudnn softmax_accurate_instance | 8.1 | 0.95x | LOSS | 1590/1590 |
| llama1b.attn.s256 | 8192 x 256 | warp_per_row | 75.6 | 222 | 98% | cudnn softmax_accurate_instance | 74.9 | 0.99x | LOSS | 1530/1335 |
| llama1b.attn.s512 | 16384 x 512 | block_resident | 274.1 | 245 | 108% | triton online_looped | 274.4 | 1.00x | TIE | 1050/1395 |
| llama1b.attn.s1024 | 32768 x 1024 | block_resident | 1087.5 | 247 | 109% | cudnn softmax_accurate_instance | 1097.4 | 1.01x | WIN | 945/1095 |
| llama1b.attn.s2048 | 65536 x 2048 | block_resident | 4477.9 | 240 | 106% | triton online_looped | 4499.4 | 1.00x | WIN | 870/1035 |
| gpt2s.attn.b8.s512 | 49152 x 512 | warp_per_row | 857.6 | 235 | 103% | triton online_looped | 806.9 | 0.94x | LOSS | 945/1200 |
| vocab30522.rows1 | 1 x 30522 | split_row | 13.5 | 18 | 8% | cudnn softmax_accurate_instance | 12.6 | 0.93x | LOSS | 1590/1590 |
| vocab30522.rows16 | 16 x 30522 | split_row | 22.3 | 175 | 77% | cudnn softmax_accurate_instance | 14.3 | 0.64x | LOSS | 1590/1590 |
| vocab30522.rows128 | 128 x 30522 | online | 284.7 | 110 | 48% | cudnn softmax_accurate_instance | 167.9 | 0.59x | LOSS | 1485/1500 |
| vocab32000.rows1 | 1 x 32000 | split_row | 13.6 | 19 | 8% | cudnn softmax_accurate_instance | 13.1 | 0.96x | LOSS | 1590/1590 |
| vocab32000.rows16 | 16 x 32000 | split_row | 22.3 | 184 | 81% | cudnn softmax_accurate_instance | 14.3 | 0.64x | LOSS | 1590/1590 |
| vocab32000.rows128 | 128 x 32000 | split_row | 239.1 | 137 | 60% | cudnn softmax_accurate_instance | 153.6 | 0.64x | LOSS | 1530/1515 |
| vocab50257.rows1 | 1 x 50257 | split_row | 14.5 | 28 | 12% | cudnn softmax_accurate_instance | 17.7 | 1.23x | WIN | 1590/1590 |
| vocab50257.rows16 | 16 x 50257 | split_row | 61.7 | 104 | 46% | cudnn softmax_accurate_instance | 36.8 | 0.60x | LOSS | 1590/1590 |
| vocab50257.rows128 | 128 x 50257 | split_row | 443.8 | 116 | 51% | cudnn softmax_accurate_instance | 342.8 | 0.77x | LOSS | 1305/1410 |
| odd.c1.rows1000 | 1000 x 1 | warp_per_row | 4.2 | 2 | 1% | cudnn softmax_accurate_instance | 3.8 | 0.91x | LOSS | 1590/1590 |
| odd.c7.rows1000 | 1000 x 7 | warp_per_row | 4.2 | 13 | 6% | cudnn softmax_accurate_instance | 3.9 | 0.95x | LOSS | 1590/1590 |
| odd.c33.rows1000 | 1000 x 33 | warp_per_row | 4.4 | 61 | 27% | cudnn softmax_accurate_instance | 4.3 | 0.99x | TIE | 1590/1590 |
| odd.c1000.rows1000 | 1000 x 1000 | block_resident | 36.9 | 217 | 95% | cudnn softmax_accurate_instance | 37.1 | 1.01x | WIN | 1590/1590 |
| odd.c1025.rows1000 | 1000 x 1025 | block_resident | 41.2 | 199 | 88% | cudnn softmax_accurate_instance | 39.2 | 0.95x | LOSS | 1590/1590 |
| odd.c3079.rows1000 | 1000 x 3079 | online | 138.1 | 178 | 78% | cudnn softmax_accurate_instance | 112.7 | 0.82x | LOSS | 1560/1530 |
| odd.c4097.rows1000 | 1000 x 4097 | online | 200.5 | 163 | 72% | cudnn softmax_accurate_instance | 151.6 | 0.76x | LOSS | 1485/1545 |

### Plain softmax, BF16

| shape | rows x cols | best ccml | us | GB/s | % measured peak | best baseline | us | baseline/ccml | verdict (95% CI) | SM MHz ccml/base |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2s.attn.s128 | 1536 x 128 | warp_per_row | 7.2 | 110 | 48% | pytorch_eager torch.softmax | 6.0 | 0.83x | LOSS | 1590/1590 |
| gpt2s.attn.s256 | 3072 x 256 | warp_per_row | 9.9 | 317 | 140% | pytorch_eager torch.softmax | 10.2 | 1.03x | WIN | 1590/1590 |
| gpt2s.attn.s512 | 6144 x 512 | warp_per_row | 57.4 | 219 | 96% | triton online_looped | 55.3 | 0.96x | LOSS | 1545/1590 |
| gpt2s.attn.s1024 | 12288 x 1024 | warp_per_row | 220.4 | 228 | 100% | triton online_looped | 207.1 | 0.94x | LOSS | 1275/1380 |
| gpt2s.attn.s2048 | 24576 x 2048 | block_resident | 807.1 | 249 | 110% | triton online_looped | 838.2 | 1.04x | WIN | 1050/1005 |
| gpt2m.attn.s128 | 2048 x 128 | warp_per_row | 7.9 | 132 | 58% | pytorch_eager torch.softmax | 6.3 | 0.80x | LOSS | 1590/1590 |
| gpt2m.attn.s256 | 4096 x 256 | warp_per_row | 12.3 | 341 | 150% | pytorch_eager torch.softmax | 12.5 | 1.02x | WIN | 1590/1590 |
| gpt2m.attn.s512 | 8192 x 512 | warp_per_row | 76.6 | 219 | 96% | triton online_looped | 72.3 | 0.94x | LOSS | 1560/1575 |
| gpt2m.attn.s1024 | 16384 x 1024 | warp_per_row | 291.8 | 230 | 101% | triton online_looped | 273.8 | 0.94x | LOSS | 1110/1290 |
| gpt2m.attn.s2048 | 32768 x 2048 | block_resident | 1065.2 | 252 | 111% | triton online_looped | 1261.9 | 1.18x | CLOCK | 960/780 |
| llama1b.attn.s128 | 4096 x 128 | warp_per_row | 12.0 | 174 | 77% | torch_compile max-autotune | 8.5 | 0.71x | LOSS | 1590/1590 |
| llama1b.attn.s256 | 8192 x 256 | warp_per_row | 38.9 | 216 | 95% | torch_compile max-autotune | 40.9 | 1.05x | WIN | 1485/1590 |
| llama1b.attn.s512 | 16384 x 512 | warp_per_row | 145.5 | 231 | 101% | triton online_looped | 140.9 | 0.97x | LOSS | 1200/1455 |
| llama1b.attn.s1024 | 32768 x 1024 | warp_per_row | 575.5 | 233 | 103% | triton online_looped | 539.0 | 0.94x | LOSS | 975/1065 |
| llama1b.attn.s2048 | 65536 x 2048 | block_resident | 2205.7 | 243 | 107% | triton online_looped | 2553.8 | 1.16x | CLOCK | 945/765 |
| gpt2s.attn.b8.s512 | 49152 x 512 | warp_per_row | 420.1 | 240 | 105% | triton online_looped | 440.0 | 1.05x | WIN | 990/1185 |
| vocab30522.rows1 | 1 x 30522 | split_row | 13.5 | 9 | 4% | pytorch_eager torch.softmax | 16.4 | 1.21x | WIN | 1590/1590 |
| vocab30522.rows16 | 16 x 30522 | split_row | 21.1 | 93 | 41% | torch_compile max-autotune | 17.8 | 0.84x | LOSS | 1590/1590 |
| vocab30522.rows128 | 128 x 30522 | split_row | 175.2 | 89 | 39% | pytorch_eager torch.softmax | 110.6 | 0.63x | LOSS | 1515/1575 |
| vocab32000.rows1 | 1 x 32000 | split_row | 13.8 | 9 | 4% | triton online_looped | 14.2 | 1.03x | WIN | 1590/1590 |
| vocab32000.rows16 | 16 x 32000 | split_row | 21.7 | 94 | 41% | triton online_looped | 14.5 | 0.67x | LOSS | 1590/1590 |
| vocab32000.rows128 | 128 x 32000 | online | 173.2 | 95 | 42% | triton online_looped | 108.2 | 0.62x | LOSS | 1545/1590 |
| vocab50257.rows1 | 1 x 50257 | split_row | 14.7 | 14 | 6% | pytorch_eager torch.softmax | 22.4 | 1.52x | WIN | 1590/1590 |
| vocab50257.rows16 | 16 x 50257 | split_row | 29.2 | 110 | 48% | pytorch_eager torch.softmax | 23.1 | 0.79x | LOSS | 1590/1590 |
| vocab50257.rows128 | 128 x 50257 | online | 294.8 | 87 | 38% | pytorch_eager torch.softmax | 190.5 | 0.65x | LOSS | 1500/1515 |
| odd.c1.rows1000 | 1000 x 1 | warp_per_row | 4.2 | 1 | 0% | torch_compile max-autotune | 3.8 | 0.92x | LOSS | 1590/1590 |
| odd.c7.rows1000 | 1000 x 7 | warp_per_row | 4.2 | 7 | 3% | torch_compile max-autotune | 4.0 | 0.97x | TIE | 1590/1590 |
| odd.c33.rows1000 | 1000 x 33 | warp_per_row | 4.5 | 29 | 13% | torch_compile max-autotune | 4.6 | 1.03x | WIN | 1590/1590 |
| odd.c1000.rows1000 | 1000 x 1000 | warp_per_row | 12.1 | 330 | 145% | pytorch_eager torch.softmax | 13.6 | 1.13x | WIN | 1590/1590 |
| odd.c1025.rows1000 | 1000 x 1025 | block_resident | 23.1 | 177 | 78% | torch_compile max-autotune | 18.8 | 0.81x | LOSS | 1590/1590 |
| odd.c3079.rows1000 | 1000 x 3079 | split_row | 94.7 | 130 | 57% | triton online_looped | 68.8 | 0.73x | LOSS | 1560/1590 |
| odd.c4097.rows1000 | 1000 x 4097 | online | 122.9 | 133 | 59% | triton online_looped | 94.2 | 0.77x | LOSS | 1515/1545 |

### Fused scale + mask, F16

| shape | rows x cols | best ccml | us | GB/s | % measured peak | best baseline | us | baseline/ccml | verdict (95% CI) | SM MHz ccml/base |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2s.attn.s128.causal | 1536 x 128 | fused_warp_per_row | 6.9 | 114 | 50% | torch_compile max-autotune in-graph causal | 19.5 | 2.82x | WIN | 1590/1230 |
| gpt2s.attn.s128.padding | 1536 x 128 | fused_warp_per_row | 7.5 | 105 | 46% | pytorch_eager softmax(x*s+mask) | 26.6 | 3.54x | WIN | 1590/1065 |
| gpt2s.attn.s256.causal | 3072 x 256 | fused_warp_per_row | 9.8 | 320 | 141% | torch_compile max-autotune in-graph causal | 36.1 | 3.67x | WIN | 1590/1335 |
| gpt2s.attn.s256.padding | 3072 x 256 | fused_warp_per_row | 11.1 | 283 | 125% | pytorch_eager softmax(x*s+mask) | 55.2 | 4.97x | WIN | 1590/600 |
| gpt2s.attn.s512.causal | 6144 x 512 | fused_warp_per_row | 57.3 | 219 | 97% | torch_compile max-autotune in-graph causal | 112.1 | 1.95x | WIN | 1440/1500 |
| gpt2s.attn.s512.padding | 6144 x 512 | fused_warp_per_row | 57.4 | 219 | 96% | torch_compile max-autotune | 117.5 | 2.05x | WIN | 1380/1485 |
| gpt2s.attn.s1024.causal | 12288 x 1024 | fused_warp_per_row | 219.6 | 229 | 101% | torch_compile max-autotune in-graph causal | 427.1 | 1.94x | WIN | 1170/1410 |
| gpt2s.attn.s1024.padding | 12288 x 1024 | fused_warp_per_row | 219.5 | 229 | 101% | torch_compile max-autotune | 424.2 | 1.93x | WIN | 1140/1305 |
| gpt2s.attn.s2048.causal | 24576 x 2048 | fused_online | 1811.0 | 111 | 49% | torch_compile max-autotune in-graph causal | 1875.7 | 1.04x | WIN | 1020/1200 |
| gpt2s.attn.s2048.padding | 24576 x 2048 | fused_online | 2195.5 | 92 | 40% | torch_compile max-autotune | 2068.5 | 0.94x | LOSS | 975/1065 |
| gpt2m.attn.s128.causal | 2048 x 128 | fused_warp_per_row | 7.9 | 132 | 58% | pytorch_eager softmax(x*s+mask) | 27.5 | 3.47x | WIN | 1590/1455 |
| gpt2m.attn.s128.padding | 2048 x 128 | fused_warp_per_row | 8.3 | 127 | 56% | pytorch_eager softmax(x*s+mask) | 29.9 | 3.62x | WIN | 1590/1335 |
| gpt2m.attn.s256.causal | 4096 x 256 | fused_warp_per_row | 12.0 | 350 | 154% | torch_compile max-autotune in-graph causal | 38.9 | 3.24x | WIN | 1590/1350 |
| gpt2m.attn.s256.padding | 4096 x 256 | fused_warp_per_row | 13.9 | 302 | 133% | torch_compile max-autotune | 62.3 | 4.49x | WIN | 1575/765 |
| gpt2m.attn.s512.causal | 8192 x 512 | fused_warp_per_row | 75.9 | 221 | 97% | torch_compile max-autotune in-graph causal | 148.4 | 1.95x | WIN | 1320/1425 |
| gpt2m.attn.s512.padding | 8192 x 512 | fused_warp_per_row | 76.2 | 220 | 97% | torch_compile max-autotune | 147.2 | 1.93x | WIN | 1260/855 |
| gpt2m.attn.s1024.causal | 16384 x 1024 | fused_warp_per_row | 291.2 | 230 | 101% | torch_compile max-autotune in-graph causal | 573.4 | 1.97x | WIN | 1095/1335 |
| gpt2m.attn.s1024.padding | 16384 x 1024 | fused_warp_per_row | 290.5 | 231 | 102% | torch_compile max-autotune | 565.5 | 1.95x | WIN | 1080/1140 |
| gpt2m.attn.s2048.causal | 32768 x 2048 | fused_online | 2451.5 | 110 | 48% | torch_compile max-autotune in-graph causal | 2491.7 | 1.02x | WIN | 990/1260 |
| gpt2m.attn.s2048.padding | 32768 x 2048 | fused_online | 2837.1 | 95 | 42% | torch_compile max-autotune | 2759.4 | 0.97x | LOSS | 960/1035 |
| llama1b.attn.s128.causal | 4096 x 128 | fused_warp_per_row | 11.5 | 182 | 80% | torch_compile max-autotune in-graph causal | 22.5 | 1.96x | WIN | 1590/1170 |
| llama1b.attn.s128.padding | 4096 x 128 | fused_warp_per_row | 12.3 | 171 | 75% | torch_compile max-autotune | 32.4 | 2.65x | WIN | 1590/1140 |
| llama1b.attn.s256.causal | 8192 x 256 | fused_warp_per_row | 39.4 | 213 | 94% | torch_compile max-autotune in-graph causal | 75.8 | 1.93x | WIN | 1290/975 |
| llama1b.attn.s256.padding | 8192 x 256 | fused_warp_per_row | 39.5 | 212 | 93% | torch_compile max-autotune | 89.0 | 2.25x | WIN | 1365/1305 |
| llama1b.attn.s512.causal | 16384 x 512 | fused_warp_per_row | 145.4 | 231 | 102% | torch_compile max-autotune in-graph causal | 292.0 | 2.01x | WIN | 1140/1440 |
| llama1b.attn.s512.padding | 16384 x 512 | fused_warp_per_row | 145.4 | 231 | 102% | torch_compile max-autotune | 286.2 | 1.97x | WIN | 1140/1350 |
| llama1b.attn.s1024.causal | 32768 x 1024 | fused_warp_per_row | 573.6 | 234 | 103% | torch_compile max-autotune in-graph causal | 1130.7 | 1.97x | WIN | 1005/1290 |
| llama1b.attn.s1024.padding | 32768 x 1024 | fused_warp_per_row | 570.6 | 235 | 104% | torch_compile max-autotune | 1124.2 | 1.97x | WIN | 990/1095 |
| llama1b.attn.s2048.causal | 65536 x 2048 | fused_online | 5140.4 | 104 | 46% | torch_compile max-autotune in-graph causal | 4942.0 | 0.96x | LOSS | 930/1140 |
| llama1b.attn.s2048.padding | 65536 x 2048 | fused_online | 6012.5 | 89 | 39% | torch_compile max-autotune | 5433.3 | 0.90x | LOSS | 930/1050 |
| gpt2s.attn.b8.s512.causal | 49152 x 512 | fused_warp_per_row | 422.8 | 238 | 105% | torch_compile max-autotune in-graph causal | 860.2 | 2.03x | WIN | 1020/1275 |
| gpt2s.attn.b8.s512.padding | 49152 x 512 | fused_warp_per_row | 419.8 | 240 | 106% | torch_compile max-autotune | 838.2 | 2.00x | WIN | 975/1050 |

### Fused scale + mask, F32

| shape | rows x cols | best ccml | us | GB/s | % measured peak | best baseline | us | baseline/ccml | verdict (95% CI) | SM MHz ccml/base |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2s.attn.s128.causal | 1536 x 128 | fused_warp_per_row | 6.1 | 256 | 113% | pytorch_eager softmax(x*s+mask) | 27.0 | 4.40x | WIN | 1590/1485 |
| gpt2s.attn.s128.padding | 1536 x 128 | fused_warp_per_row | 6.8 | 233 | 102% | pytorch_eager softmax(x*s+mask) | 27.4 | 4.06x | WIN | 1590/585 |
| gpt2s.attn.s256.causal | 3072 x 256 | fused_warp_per_row | 30.0 | 210 | 92% | torch_compile max-autotune in-graph causal | 58.4 | 1.95x | WIN | 1590/1485 |
| gpt2s.attn.s256.padding | 3072 x 256 | fused_warp_per_row | 30.1 | 209 | 92% | torch_compile max-autotune | 65.3 | 2.17x | CLOCK | 1545/585 |
| gpt2s.attn.s512.causal | 6144 x 512 | fused_warp_per_row | 111.4 | 226 | 99% | torch_compile max-autotune in-graph causal | 219.1 | 1.97x | WIN | 1365/1500 |
| gpt2s.attn.s512.padding | 6144 x 512 | fused_warp_per_row | 111.0 | 227 | 100% | torch_compile max-autotune | 248.4 | 2.24x | WIN | 1335/1560 |
| gpt2s.attn.s1024.causal | 12288 x 1024 | fused_warp_per_row | 438.3 | 230 | 101% | torch_compile max-autotune in-graph causal | 858.2 | 1.96x | WIN | 1080/1410 |
| gpt2s.attn.s1024.padding | 12288 x 1024 | fused_warp_per_row | 439.5 | 229 | 101% | torch_compile max-autotune | 854.7 | 1.94x | WIN | 1050/1320 |
| gpt2s.attn.s2048.causal | 24576 x 2048 | fused_online | 2310.1 | 174 | 77% | torch_compile max-autotune in-graph causal | 3365.5 | 1.46x | WIN | 1035/1275 |
| gpt2s.attn.s2048.padding | 24576 x 2048 | fused_online | 2525.2 | 159 | 70% | torch_compile max-autotune | 3502.1 | 1.39x | WIN | 960/1230 |
| gpt2m.attn.s128.causal | 2048 x 128 | fused_warp_per_row | 6.5 | 323 | 142% | pytorch_eager softmax(x*s+mask) | 30.6 | 4.71x | WIN | 1470/1365 |
| gpt2m.attn.s128.padding | 2048 x 128 | fused_warp_per_row | 7.5 | 279 | 123% | pytorch_eager softmax(x*s+mask) | 30.7 | 4.08x | WIN | 1590/945 |
| gpt2m.attn.s256.causal | 4096 x 256 | fused_warp_per_row | 38.9 | 216 | 95% | torch_compile max-autotune in-graph causal | 74.7 | 1.92x | WIN | 1560/1500 |
| gpt2m.attn.s256.padding | 4096 x 256 | fused_warp_per_row | 38.9 | 216 | 95% | torch_compile max-autotune | 72.2 | 1.86x | WIN | 1530/1035 |
| gpt2m.attn.s512.causal | 8192 x 512 | fused_warp_per_row | 147.9 | 227 | 100% | torch_compile max-autotune in-graph causal | 291.7 | 1.97x | WIN | 1320/1500 |
| gpt2m.attn.s512.padding | 8192 x 512 | fused_warp_per_row | 147.5 | 228 | 100% | torch_compile max-autotune | 290.8 | 1.97x | WIN | 1230/1545 |
| gpt2m.attn.s1024.causal | 16384 x 1024 | fused_warp_per_row | 582.6 | 230 | 101% | torch_compile max-autotune in-graph causal | 1140.2 | 1.96x | WIN | 1050/1365 |
| gpt2m.attn.s1024.padding | 16384 x 1024 | fused_warp_per_row | 583.5 | 230 | 101% | torch_compile max-autotune | 1126.3 | 1.93x | WIN | 1005/1245 |
| gpt2m.attn.s2048.causal | 32768 x 2048 | fused_online | 3087.4 | 174 | 76% | torch_compile max-autotune in-graph causal | 4481.0 | 1.45x | WIN | 1020/1260 |
| gpt2m.attn.s2048.padding | 32768 x 2048 | fused_online | 3420.1 | 157 | 69% | torch_compile max-autotune | 4465.6 | 1.31x | WIN | 930/1185 |
| llama1b.attn.s128.causal | 4096 x 128 | fused_warp_per_row | 9.7 | 433 | 191% | torch_compile max-autotune in-graph causal | 51.9 | 5.36x | WIN | 1590/1500 |
| llama1b.attn.s128.padding | 4096 x 128 | fused_warp_per_row | 10.5 | 401 | 176% | pytorch_eager softmax(x*s+mask) | 53.2 | 5.09x | WIN | 1590/585 |
| llama1b.attn.s256.causal | 8192 x 256 | fused_warp_per_row | 75.6 | 222 | 98% | torch_compile max-autotune in-graph causal | 145.0 | 1.92x | WIN | 1350/1260 |
| llama1b.attn.s256.padding | 8192 x 256 | fused_warp_per_row | 75.6 | 222 | 98% | torch_compile max-autotune | 148.3 | 1.96x | WIN | 1335/1500 |
| llama1b.attn.s512.causal | 16384 x 512 | fused_warp_per_row | 288.9 | 232 | 102% | torch_compile max-autotune in-graph causal | 579.6 | 2.01x | WIN | 1080/1395 |
| llama1b.attn.s512.padding | 16384 x 512 | fused_warp_per_row | 288.5 | 233 | 102% | torch_compile max-autotune | 572.0 | 1.98x | WIN | 1050/1320 |
| llama1b.attn.s1024.causal | 32768 x 1024 | fused_warp_per_row | 1152.9 | 233 | 102% | torch_compile max-autotune in-graph causal | 2279.0 | 1.98x | WIN | 1020/1350 |
| llama1b.attn.s1024.padding | 32768 x 1024 | fused_warp_per_row | 1153.2 | 233 | 102% | torch_compile max-autotune | 2232.1 | 1.94x | WIN | 945/1215 |
| llama1b.attn.s2048.causal | 65536 x 2048 | fused_online | 6457.3 | 166 | 73% | torch_compile max-autotune in-graph causal | 8916.9 | 1.38x | WIN | 960/1215 |
| llama1b.attn.s2048.padding | 65536 x 2048 | fused_online | 7143.0 | 150 | 66% | torch_compile max-autotune | 9196.8 | 1.29x | WIN | 870/1200 |
| gpt2s.attn.b8.s512.causal | 49152 x 512 | fused_warp_per_row | 858.3 | 235 | 103% | torch_compile max-autotune in-graph causal | 1681.8 | 1.96x | WIN | 1020/1320 |
| gpt2s.attn.b8.s512.padding | 49152 x 512 | fused_warp_per_row | 855.6 | 235 | 104% | torch_compile max-autotune | 1689.2 | 1.97x | WIN | 960/1230 |

### Fused scale + mask, BF16

| shape | rows x cols | best ccml | us | GB/s | % measured peak | best baseline | us | baseline/ccml | verdict (95% CI) | SM MHz ccml/base |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2s.attn.s128.causal | 1536 x 128 | fused_warp_per_row | 7.7 | 102 | 45% | torch_compile max-autotune | 12.1 | 1.57x | WIN | 1590/1590 |
| gpt2s.attn.s128.padding | 1536 x 128 | fused_warp_per_row | 8.2 | 96 | 42% | pytorch_eager softmax(x*s+mask) | 14.5 | 1.77x | WIN | 1590/1380 |
| gpt2s.attn.s256.causal | 3072 x 256 | fused_warp_per_row | 10.9 | 288 | 127% | pytorch_eager softmax(x*s+mask) | 33.1 | 3.03x | WIN | 1590/1590 |
| gpt2s.attn.s256.padding | 3072 x 256 | fused_warp_per_row | 12.3 | 256 | 113% | pytorch_eager softmax(x*s+mask) | 29.1 | 2.37x | WIN | 1590/1590 |
| gpt2s.attn.s512.causal | 6144 x 512 | fused_warp_per_row | 57.7 | 218 | 96% | torch_compile max-autotune | 197.0 | 3.41x | WIN | 1425/1545 |
| gpt2s.attn.s512.padding | 6144 x 512 | fused_warp_per_row | 58.2 | 216 | 95% | pytorch_eager softmax(x*s+mask) | 177.8 | 3.05x | WIN | 1395/1500 |
| gpt2s.attn.s1024.causal | 12288 x 1024 | fused_warp_per_row | 229.3 | 220 | 97% | pytorch_eager softmax(x*s+mask) | 759.8 | 3.31x | WIN | 1170/1305 |
| gpt2s.attn.s1024.padding | 12288 x 1024 | fused_warp_per_row | 230.1 | 219 | 96% | torch_compile max-autotune | 700.4 | 3.04x | WIN | 1140/1260 |
| gpt2s.attn.s2048.causal | 24576 x 2048 | fused_online | 1907.9 | 106 | 46% | pytorch_eager softmax(x*s+mask) | 3077.9 | 1.61x | WIN | 1020/1200 |
| gpt2s.attn.s2048.padding | 24576 x 2048 | fused_online | 2238.5 | 90 | 40% | torch_compile max-autotune | 2963.5 | 1.32x | WIN | 960/1110 |
| gpt2m.attn.s128.causal | 2048 x 128 | fused_warp_per_row | 8.5 | 124 | 54% | pytorch_eager softmax(x*s+mask) | 13.0 | 1.54x | WIN | 1590/1590 |
| gpt2m.attn.s128.padding | 2048 x 128 | fused_warp_per_row | 9.1 | 115 | 51% | pytorch_eager softmax(x*s+mask) | 14.0 | 1.54x | WIN | 1590/1590 |
| gpt2m.attn.s256.causal | 4096 x 256 | fused_warp_per_row | 13.8 | 305 | 134% | pytorch_eager softmax(x*s+mask) | 47.5 | 3.45x | WIN | 1590/1590 |
| gpt2m.attn.s256.padding | 4096 x 256 | fused_warp_per_row | 15.2 | 277 | 122% | torch_compile max-autotune | 42.1 | 2.78x | WIN | 1560/1500 |
| gpt2m.attn.s512.causal | 8192 x 512 | fused_warp_per_row | 76.4 | 219 | 97% | pytorch_eager softmax(x*s+mask) | 259.8 | 3.40x | WIN | 1350/1485 |
| gpt2m.attn.s512.padding | 8192 x 512 | fused_warp_per_row | 76.7 | 219 | 96% | pytorch_eager softmax(x*s+mask) | 234.5 | 3.06x | WIN | 1320/1470 |
| gpt2m.attn.s1024.causal | 16384 x 1024 | fused_warp_per_row | 303.6 | 221 | 97% | pytorch_eager softmax(x*s+mask) | 1003.5 | 3.31x | WIN | 1110/1080 |
| gpt2m.attn.s1024.padding | 16384 x 1024 | fused_warp_per_row | 306.7 | 219 | 96% | pytorch_eager softmax(x*s+mask) | 936.0 | 3.05x | WIN | 1065/1185 |
| gpt2m.attn.s2048.causal | 32768 x 2048 | fused_online | 2572.3 | 104 | 46% | torch_compile max-autotune | 4081.7 | 1.59x | WIN | 990/1185 |
| gpt2m.attn.s2048.padding | 32768 x 2048 | fused_online | 3020.5 | 89 | 39% | torch_compile max-autotune | 3993.3 | 1.32x | WIN | 945/1065 |
| llama1b.attn.s128.causal | 4096 x 128 | fused_warp_per_row | 13.0 | 161 | 71% | torch_compile max-autotune | 18.0 | 1.39x | WIN | 1590/1590 |
| llama1b.attn.s128.padding | 4096 x 128 | fused_warp_per_row | 13.9 | 151 | 67% | pytorch_eager softmax(x*s+mask) | 22.5 | 1.63x | WIN | 1590/1380 |
| llama1b.attn.s256.causal | 8192 x 256 | fused_warp_per_row | 39.4 | 213 | 94% | pytorch_eager softmax(x*s+mask) | 133.9 | 3.40x | WIN | 1380/1590 |
| llama1b.attn.s256.padding | 8192 x 256 | fused_warp_per_row | 38.9 | 216 | 95% | torch_compile max-autotune | 121.1 | 3.11x | WIN | 1425/1440 |
| llama1b.attn.s512.causal | 16384 x 512 | fused_warp_per_row | 145.7 | 230 | 101% | pytorch_eager softmax(x*s+mask) | 504.6 | 3.46x | WIN | 1200/1335 |
| llama1b.attn.s512.padding | 16384 x 512 | fused_warp_per_row | 145.9 | 230 | 101% | torch_compile max-autotune | 462.4 | 3.17x | WIN | 1170/1350 |
| llama1b.attn.s1024.causal | 32768 x 1024 | fused_warp_per_row | 592.1 | 227 | 100% | pytorch_eager softmax(x*s+mask) | 1989.9 | 3.36x | WIN | 1035/1185 |
| llama1b.attn.s1024.padding | 32768 x 1024 | fused_warp_per_row | 590.9 | 227 | 100% | torch_compile max-autotune | 1906.7 | 3.23x | WIN | 975/1095 |
| llama1b.attn.s2048.causal | 65536 x 2048 | fused_online | 5395.8 | 99 | 44% | pytorch_eager softmax(x*s+mask) | 8122.2 | 1.51x | WIN | 945/1185 |
| llama1b.attn.s2048.padding | 65536 x 2048 | fused_online | 6279.1 | 86 | 38% | torch_compile max-autotune | 8004.1 | 1.27x | WIN | 915/1020 |
| gpt2s.attn.b8.s512.causal | 49152 x 512 | fused_warp_per_row | 421.3 | 239 | 105% | pytorch_eager softmax(x*s+mask) | 1484.1 | 3.52x | WIN | 1035/1215 |
| gpt2s.attn.b8.s512.padding | 49152 x 512 | fused_warp_per_row | 421.7 | 239 | 105% | torch_compile max-autotune | 1407.0 | 3.34x | WIN | 1005/1080 |

## What this proves

- Every softmax item of reference.md section 3 exists, is correct in FP32,
  FP16 and BF16 against the fp64 reference, and is measured on the target GPU:
  stable, online, warp-per-row for short rows, block-per-row for long rows,
  fused scale and mask.
- **Attention softmax runs at the DRAM peak** for every row length from 256
  to 2048 columns once the tensor leaves L2 (211-253 GB/s), which is where
  cuDNN, Inductor and Triton also are: the result is parity (wins of up to 15%
  and losses of up to 14% depending on shape and dtype), not a margin.
- **Fusion is the win** (reference.md section 5, "fusion vs separate
  kernels"): the fused scale + mask softmax is 1.5-3.1x faster than the same
  kernel after a separate scale-mask pass, and beats PyTorch, compiled or not,
  by about 2x in FP32 and FP16 and 3x in BF16 on 91 of 96 shapes. Against a
  compile baseline that builds the causal mask in-graph the FP32/FP16 margin
  equals the plain-softmax margin (entry 16): the gain over compile is in the
  softmax kernel, not in avoiding a mask read.
- Decode-time vocabulary softmax (one row) beats every baseline in FP16 and
  BF16 (1.03-2.11x) by splitting the row across blocks.
- Each optimization's gain is attributed: per-step ratios above, predictions
  and their outcomes in log entries 13-17, including the falsified ones.

## What it does not prove

- **Not a vendor-beating softmax on short rows**: 128-column attention rows
  lose 4-29% (sub-MB tensors, where launch and tail latency dominate; cuDNN is
  faster there).
- **Vocabulary softmax with 16-128 rows loses** (128 rows: 23-41% in every
  dtype; 16 rows: 16-40% in FP32 and BF16, FP16 ties): 30522-50257 columns
  exceed BLOCK_RESIDENT's register limit, so they run ONLINE (128 rows) or
  SPLIT_ROW (16 rows), whose segment kernels have ONLINE's scalar, two-read
  design. A looped, vectorized block kernel would be the next step.
- FP32 vocabulary rows: one row wins only at 50257 columns; cuDNN wins the
  other two by 4-7%.
- The fused masked op has no register-resident long-row variant: its
  2048-column rows lose 3-10% to torch.compile.
- Nsight Compute evidence exists only for entry 13 (sm_89); entries 14, 16 and
  17 rest on timings and on the bandwidth they reach, not on counters.
- AUTO was chosen from this run and changed afterwards (commit cb54800), so
  the AUTO rows in these CSVs are the previous AUTO; the variants it now picks
  are each measured here directly.

## How to reproduce

    git checkout a56eb2b
    python cloud/kaggle/kaggle_run.py --jobs softmax,sanitize --slug-suffix sm --any-gpu
    python cloud/kaggle/kaggle_run.py --fetch --jobs softmax,sanitize --slug-suffix sm
    # the fetch writes results/sm_75/; the files were moved to results/sm_75/softmax_variants/
    python scripts/rowwise_summary.py results/sm_75/softmax_variants/softmax.csv results/sm_75/softmax_variants/torch_softmax.csv --dtype f16 --table --buckets
    python scripts/rowwise_summary.py results/sm_75/softmax_variants/softmax_masked.csv results/sm_75/softmax_variants/torch_softmax_masked.csv --dtype f16 --table
    python scripts/rowwise_steps.py results/sm_75/softmax_variants/softmax.csv --pairs block_per_row:online,online:warp_per_row,online:split_row,online:block_resident

Local (sm_89, indicative; NCU for entry 13): `powershell -File
scripts/local_build.ps1`, then the `scripts/ncu_matmul.ps1 -Op softmax`
commands quoted in log entry 13, from an elevated prompt.
