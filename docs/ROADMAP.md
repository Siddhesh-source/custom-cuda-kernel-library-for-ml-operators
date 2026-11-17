# Roadmap

**Authority:** [`docs/reference.md`](reference.md) is the specification. Every
item below traces to a clause in it (cited as **[R§n]**). Where any other
document in this repository disagrees with `reference.md`, `reference.md` wins
and the other document is wrong. Enhancements (marked **[E]**, detailed in
[enhancements.md](enhancements.md)) are additive and never displace a
required item.

Last updated: 2026-11-16. Progress is **phase-gated, not calendar-based**: each phase runs from its prompt in [`prompt.md`](../prompt.md), and the next starts only when the current one meets its ACCEPT checks. The phase order follows the deliverable sequence of [R§8].

---

## 1. Compliance audit — where the repository stands against the spec

Status key: ✅ compliant · 🟡 partial · ❌ missing · ⚠️ contradicts spec

| # | Requirement | Clause | Status | Gap |
|---|---|---|---|---|
| 1 | Env: `nvidia-smi`, `nvcc`, `ncu`, `nsys` verified; `.venv` with CUDA PyTorch, triton, numpy, pandas, matplotlib, pytest | Setup | ✅ | `scripts/setup_env.sh`; recorded in `results/sm_75/env_*.txt` ([phase 7](results/phase-7-measurement-infrastructure.md)). Kaggle cannot bootstrap a venv, so the image interpreter is used and recorded |
| 2 | Dirs `src bench tests results docs scripts` | Setup | ✅ | `scripts/` created |
| 3 | Repo dir `profiles/` with small `.ncu-rep` files | R§7 | ✅ | Dir created; `.gitignore` exception added |
| 4 | Target-shape list from real models (M,N,K, hidden, seq, batch) | R§3 | ✅ | `bench/shapes.json` (GPT-2 s/m, Llama-1B, BERT vocab; 146 shapes, odd included). Production shapes not supplied: recorded assumption, replaceable |
| 5 | Correctness tests before speed; compute-sanitizer memcheck + racecheck | Method | ✅ | In-run fp64 gate; memcheck, racecheck, initcheck, synccheck clean on all suites (one deliberate API error, recognized) ([phase 7](results/phase-7-measurement-infrastructure.md)) |
| 6 | Warmup + CUDA events; **median** and spread of **100+** iters | Method | ✅ | Warmup 20, 100-200 guarded per-iteration events, median/p10/p90/IQR/CI, batched cross-check |
| 7 | Identical GPU/clocks/dtype/shapes; watch throttling & other processes | Method, R§11 | ✅ | One session per operator (same GPU); NVML clocks/temp/reasons per row, sampled mid-run; other processes logged. Clocks cannot be locked on Kaggle: recorded instead (T4 power cap documented) |
| 8 | Vendor baselines **cuBLAS / cuDNN**, PyTorch eager, **torch.compile (Inductor)**, naive, best-per-shape | R§4 | ✅ | cuBLAS, cuBLASLt, cuDNN softmax, eager, compile, Triton; naive; best-per-shape analysis. cuDNN layernorm exists only in the graph API: eager is that vendor baseline (documented) |
| 9 | Percent of hardware peak (TFLOPS, GB/s) | Method, R§6 | ✅ | TFLOPS / GB/s and % of theoretical and measured peak in every row |
| 10 | Nsight Systems timelines, Nsight Compute metrics per major optimization | Method, R§9 | ❌ | Never run |
| 11 | `CUDA_CHECK` on every call, `cudaGetLastError` on every launch | Method | ✅ | `src/common/check.h` — re-verify for each new kernel |
| 12 | Matmul ladder: naive → smem tiling → register blocking → float4 → double buffering → Tensor Cores (WMMA or mma.sync); speedup recorded per step | R§3 | 🟡 | 5 of 6 rungs: naive, smem tiled, register blocking, float4/vectorized, double buffering, each with its speedup and NCU evidence ([phases 9-10](results/phase-9-10-matmul-ladder.md)). Tensor Cores: Phase 11 |
| 13 | Layernorm: block- or warp-per-row, Welford or two-pass, warp shuffle, vectorized, optional residual/bias fusion | R§3 | 🟡 | Two-pass block-per-row only |
| 14 | Softmax: stable, online, warp-per-row short / block-per-row long, optional fused scale/mask | R§3 | 🟡 | Block-per-row three-pass only |
| 15 | Correctness vs PyTorch **float64** with explicit tolerances for **FP32, FP16, BF16** | R§3, R§9 | ✅ | All operators, all three dtypes, two fp64 references, per-dtype tolerance table; T4 verified ([phase 8](results/phase-8-multi-dtype.md)). BF16 Tensor Core kernels follow in Phase 11 |
| 16 | Benchmark harness: CSV, TFLOPS, GB/s | R§3 | ✅ | `bench/`, `scripts/bench_torch.py`; one CSV schema (docs/benchmarking.md) |
| 17 | PyTorch extension via **`torch.utils.cpp_extension`**, droppable into a model | R§3, R§9 | ❌ | ctypes binding only (conflicting docs removed) |
| 18 | Experiments: tile sweep, shape sweep (+odd shapes), dtype comparison, occupancy vs registers, vectorized vs scalar, fusion vs separate, end-to-end model latency | R§5 | 🟡 | Done for matmul: tile sweep (16 configs x 76 shapes), shape sweep incl. odd, occupancy vs registers (NCU, entry 2), vectorized vs scalar (entry 2). Remaining: dtype comparison, fusion vs separate, end-to-end latency |
| 19 | Roofline plots | R§10 | ❌ | — |
| 20 | Technical report + demo notebook | R§10 | ❌ | — |
| 21 | ≥1 operator beats vendor on a target shape, or documented why not | R§9 | ✅ | matmul beats cuBLAS on 2 target shapes on the T4 with the vendor at a higher clock (1.15x, 1.07x) ([phases 9-10](results/phase-9-10-matmul-ladder.md)). Phase 7 baseline was: ours vs best baseline median 0.17x (matmul), 0.31x (layernorm), 0.23x (softmax); only 2 tiny odd matmul shapes faster |
| 22 | Kernel library with build system | R§10 | ✅ | CMake, presets, `find_package` |
| 23 | Never report speed for a kernel failing tests | R§11 | ✅ | Enforced in code: a non-PASS row carries no timing (tested). It caught a real bug in our Triton baseline |

**Removed as non-compliant (2026-11-16):** `docs/project-vision.md` and
`docs/milestone-40.md` declared Tensor Cores, fusion, FP16/BF16 and
`cpp_extension` out of scope, which contradicts the spec. Both were deleted.
This roadmap and [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) replace them. The
Modal backend was removed too, because all GPU work now runs on Kaggle.

---

## 2. Hardware plan

**All GPU work runs on Kaggle Tesla T4 (sm_75, 15 GB)** through
`cloud/kaggle/kaggle_run.py` (machine shape pinned to T4). That covers the
build, tests, compute-sanitizer, benchmarks, Nsight Systems, the PyTorch
extension and the model integration. The T4 is the benchmark target, so every
tuning decision is made from T4 measurements.

A Kaggle probe on 2026-11-16 found:

| Capability on Kaggle T4 | Status |
|---|---|
| nvcc 13.0, CMake 3.31, Ninja, GCC | ✅ |
| torch 2.11+cu130 (CUDA), torch.compile, Triton 3.6, pynvml, `CUDA_HOME` set | ✅ |
| FP16 Tensor Cores (WMMA, `mma.sync` m16n8k8) | ✅ (sm_75) |
| BF16 I/O and correctness with FP32 math | ✅ |
| compute-sanitizer (memcheck, racecheck, initcheck, synccheck) | ✅ |
| Nsight Systems (`nsys profile`) | ✅ |
| Nsight Compute counters | ❌ `ERR_NVGPUCTRPERM` (no `CAP_SYS_ADMIN`), so this is local: [local.md](local.md) L3 |
| FP16 Tensor Core throughput (torch 2048³) | 19.45 TFLOPS; BF16 emulated 2.5 TFLOPS |
| Clock locking (`nvidia-smi -lgc`) | ❌ denied, so clocks are recorded per row instead |
| BF16 Tensor Cores, `cp.async` | ❌ need sm_80+ |

Anything Kaggle cannot run is listed in **[local.md](local.md)** and is the
only work that needs a local GPU. Results are filed under `results/sm_75/`, and
results from other architectures are never mixed with them.

Every result file records GPU, arch, driver, CUDA, clocks, temperature and
throttle reasons.

---

## 3. Phase plan (order follows [R§8])

Each phase ends with tests in `tests/` and `docs/results/phase-N-*.md`
(existing workflow). No phase reports a speed number for a kernel failing its
tests [R§11].

### Phase 7: Environment, shapes, references, harness, baselines
*Local: none.*
- `scripts/setup_env.sh` (runs inside the Kaggle kernel): tool version check (`nvidia-smi`, `nvcc`, `ncu`, `nsys`), `.venv` on top of the image's CUDA PyTorch + `triton numpy pandas matplotlib pytest`. Output → `docs/results/phase-7-env.md`.
- Create `scripts/`, `profiles/`; `.gitignore` exception for `profiles/**/*.ncu-rep`.
- **`bench/shapes.json`**: single source of truth for target and odd shapes (§4).
- Harness rework [R Method, R§3]: ≥100 timed iters (default 200), warmup ≥20, **median, p10/p90, IQR, min**; TFLOPS, GB/s, %peak columns; CSV to `results/sm_75/<op>.csv`; GPU clock/temp/throttle captured per row via NVML **[E-M1..M4]**.
- Direct vendor baselines in C++: **cuBLAS** (`cublasGemmEx`, FP32/FP16/BF16, plus `cublasLt` heuristics top-1), **cuDNN** layernorm/softmax (cuDNN softmax forward; cuDNN graph/frontend layernorm where available, else `torch.nn.functional` as documented substitute).
- Python baselines: PyTorch eager, **torch.compile (Inductor)**, **Triton** **[E-B1]**.
- Measured peaks: FMA microbench + STREAM-copy microbench → `results/sm_75/peaks.json` **[E-M5]**.
- **compute-sanitizer** memcheck + racecheck (+ initcheck, synccheck **[E-C1]**) on all existing kernels → `scripts/sanitize.sh`, logs in `results/sm_75/sanitizer/`.

### Phase 8: Multi-dtype foundation + naive kernels
*Local: none.*
- Public API: `ccml_<op>_f16`, `ccml_<op>_bf16` alongside `_f32` (same signature shape, `__half`/`__nv_bfloat16` I/O, FP32 accumulation). BF16 works on all architectures through FP32 math, so it is tested on the T4. Only the BF16 Tensor Core variants are gated to sm_80+.
- Naive kernels for all three ops in all three dtypes.
- Correctness vs **PyTorch float64** reference with explicit per-dtype tolerances [R§3] — table committed in `docs/correctness.md`. Starting policy (to be justified by measured error): FP32 `atol 1e-5, rtol 1e-5`; FP16 `atol 1e-3, rtol 1e-2`; BF16 `atol 1e-2, rtol 5e-2`; matmul tolerance scaled by `sqrt(K)`.
- Report max abs and max rel error per test [R§6].

### Phase 9: Matmul — smem tiling (exists) + register blocking
*Local: **L3 Nsight Compute**. The reboot that enables counters must be done before this phase.*
- `REGISTER_TILED`: block tile BM×BN×BK, thread tile TM×TN (start 128×128×8, 8×8), templated. Tile sweep [R§5].
- NCU before/after; speedup vs previous rung recorded in `docs/optimization-log.md`.

### Phase 10: Matmul — float4 + double buffering + first NCU analysis
*Local: L3 Nsight Compute and **L2 `cp.async`** variant.*
- `VECTORIZED` (float4 / 128-bit loads, transposed A in smem, padding/swizzle for bank conflicts).
- `DOUBLE_BUFFERED` (register-prefetch double buffering, which works on the T4). The `cp.async` multistage version needs sm_80+ and is in [local.md](local.md).
- Experiment: vectorized vs scalar [R§5]. NCU: achieved occupancy, memory throughput, warp stall reasons [R§6] → `profiles/matmul/`.

### Phase 11: Matmul — Tensor Cores vs cuBLAS
*Local: **L1 BF16 Tensor Cores** and L3 Nsight Compute.*
- `WMMA` (FP16 in, FP32 accumulate), then `MMA_SYNC` (`mma.sync` m16n8k8 + `ldmatrix`, register-prefetch pipeline) on the T4. The BF16 Tensor Core variants need sm_80+ and are in [local.md](local.md).
- **[E-S1]** threadblock swizzle, **[E-S2]** split-K / stream-K for skinny shapes, **[E-F1]** fused bias/GELU/residual epilogue.
- Shape-by-shape table vs cuBLAS on target shapes [R§9]. Occupancy vs register usage sweep via `__launch_bounds__` [R§5].

### Phase 12: Layernorm — warp shuffle, vectorized, residual fusion
*Local: L3 Nsight Compute.*
- `WARP_PER_ROW` (cols ≤ 1024), `WELFORD` (single pass, warp-shuffle Welford merge), `VECTORIZED` (128-bit), `FUSED_RESIDUAL_BIAS` (`y = LN(x + residual + bias)`, also writes the pre-norm sum).
- Resolve open question: cost of fp64 accumulation (measured).
- Fusion vs separate kernels experiment [R§5].

### Phase 13: Softmax — online and fused
*Local: L3 Nsight Compute.*
- `WARP_PER_ROW` (short rows, register-resident), `ONLINE` (single pass running max/sum, Milakov & Gimelshein 2018), `FUSED_SCALE_MASK` (scale + additive/causal mask), block-per-row for long rows (≥ 4096, e.g. vocab 32000/50257).
- AUTO dispatch thresholds chosen by measurement **[E-S4]**.

### Phase 14: PyTorch extension + real model
*Local: none (optional local CUDA PyTorch only to drive L1 from Python).*
- `python/ccml_ext/` built with **`torch.utils.cpp_extension`** (`CUDAExtension` + `setup.py`, and JIT `load()` path). Uses PyTorch's current stream, dispatch on dtype, rejects non-contiguous inputs.
- Register as `torch.library` custom ops with fake/meta impls so **torch.compile can trace through them** **[E-P1]**.
- Drop-in `nn.Module` replacements (`CcmlLinear`, `CcmlLayerNorm`, `CcmlSoftmax`); swap into a real model (GPT-2 small, 124M, on the Kaggle T4) and measure **end-to-end latency** [R§5]. Logit/perplexity parity check against the unmodified model.
- ctypes binding kept as the C-ABI test path; cpp_extension is the deliverable.

### Phase 15: Full sweep, ablations, roofline
*Local: L3 Nsight Compute for ablation metrics.*
- Every op × variant × dtype × shape in `bench/shapes.json`, against every baseline [R§4]. Raw CSV committed.
- Ablations: each optimization turned off individually.
- Roofline plots per op/dtype with measured and theoretical ceilings (`scripts/roofline.py`, matplotlib) [R§10].
- **[E-G1]** CUDA Graphs and **[E-G2]** multi-stream results for the transformer block.

### Phase 16: Freeze, report, demo
*Local: L3 for any missing NCU evidence in the DoD audit.*
- Tag `v1.0`. `docs/report.md` (technical report), `notebooks/demo.ipynb` (build ext, run correctness, run model with swapped kernels, plot results).
- Final shape-by-shape comparison table [R§2, R§9] with honest losses [R§11].
- Definition-of-done audit (§5) with every box linked to evidence.

---

## 4. Target shapes (`bench/shapes.json`)

No sponsor shapes have been provided. [R§3] requires shapes from real models,
so these come from published architectures that fit the 15 GB Kaggle T4, and
the assumption is recorded. Replace or extend them when real production
shapes arrive.

| Model | hidden | FFN | heads | vocab |
|---|---|---|---|---|
| GPT-2 small | 768 | 3072 | 12 | 50257 |
| BERT-base | 768 | 3072 | 12 | 30522 |
| GPT-2 medium | 1024 | 4096 | 16 | 50257 |
| Llama-style 1B | 2048 | 8192 | 32 | 32000 |

- **Matmul (M,N,K):** M = batch·seq ∈ {1, 8, 16, 64, 128, 512, 2048}. (N,K) ∈ {QKV (3H,H), out-proj (H,H), FFN-up (4H,H), FFN-down (H,4H), LM-head (V,H)}. Decode shapes (M ≤ 16) are the skinny shapes where [R§11] says wins are most likely.
- **Layernorm:** rows = batch·seq ∈ {1 … 8192}, cols = H ∈ {768, 1024, 2048, 4096}.
- **Softmax:** attention rows = batch·heads·seq, cols = seq ∈ {128, 256, 512, 1024, 2048}; vocab rows = batch·seq, cols ∈ {30522, 32000, 50257}.
- **Odd shapes** [R§5]: 1, 7, 33, 127, 1000, 1023, 1025, 4097, and primes such as 769 and 3079.
- Each shape must fit within 12 GB including baselines (VRAM guard in the harness; the T4 has 15 GB).

---

## 5. Definition of done [R§9, R§10]: audit checklist

- [ ] All kernels pass correctness for FP32, FP16, BF16 vs PyTorch fp64, with explicit tolerances (Kaggle T4; BF16 Tensor Core variants per local.md)
- [ ] compute-sanitizer memcheck + racecheck clean on all kernels, logs committed
- [ ] Shape-by-shape table vs cuBLAS / cuDNN / PyTorch eager / torch.compile / naive / best-ours
- [ ] ≥1 operator beats the vendor baseline on a target shape **or** a documented explanation of why not
- [ ] Nsight Compute evidence (`profiles/`) for each major optimization
- [ ] Python bindings via `torch.utils.cpp_extension`, used inside a real model
- [ ] All seven R§5 experiments run, results committed
- [ ] Metrics: latency µs, TFLOPS + %peak (matmul), GB/s + %peak BW (row-wise), speedup, max abs/rel error, occupancy, memory throughput, stall reasons [R§6]
- [ ] Raw CSVs, roofline plots, technical report, demo notebook
- [ ] Every comparison on same GPU, dtype and clocks, recorded [R§11]

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Kaggle GPU quota (~30 h, resets weekly) | Batch each phase into one kernel run; compile-only check with `--no-gpu` first |
| Kaggle queue delays | `--push` and fetch later; never block the work on one run |
| Nsight Compute counters unavailable on Kaggle (confirmed) | Local GPU for `profiles/` evidence only (local.md L3); timings stay on the T4 |
| T4 pin queues for a long time (seen 2026-11-16: 45+ min) | `kaggle_run.py --any-gpu` drops the pin; the log records the actual GPU and a non-T4 run is filed by its own arch |
| Clocks cannot be locked | Record clocks, temperature and throttle reasons per row; interleave variants; reject throttled samples **[E-M3]** |
| No win vs cuBLAS | Focus on skinny/decode and fused ops; R§9 allows a documented explanation |
| Enhancements crowd out required work | An enhancement starts only after the phase's required items pass |
