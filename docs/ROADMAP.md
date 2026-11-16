# Roadmap

**Authority:** [`docs/reference.md`](reference.md) is the specification. Every
item below traces to a clause in it (cited as **[R§n]**). Where any other
document in this repository disagrees with `reference.md`, `reference.md` wins
and the other document is wrong. Enhancements (marked **[E]**, detailed in
[enhancements.md](enhancements.md)) are additive and never displace a
required item.

Last updated: 2026-11-16. Schedule: 10 weeks, 2026-11-16 → 2027-01-24 [R§8].

---

## 1. Compliance audit — where the repository stands against the spec

Status key: ✅ compliant · 🟡 partial · ❌ missing · ⚠️ contradicts spec

| # | Requirement | Clause | Status | Gap |
|---|---|---|---|---|
| 1 | Env: `nvidia-smi`, `nvcc`, `ncu`, `nsys` verified; `.venv` with CUDA PyTorch, triton, numpy, pandas, matplotlib, pytest | Setup | ❌ | Local PyTorch is CPU-only; no venv recorded; Windows SDK missing |
| 2 | Dirs `src bench tests results docs scripts` | Setup | 🟡 | `scripts/` missing |
| 3 | Repo dir `profiles/` with small `.ncu-rep` files | R§7 | ⚠️ | Dir missing; `.gitignore` excludes `*.ncu-rep` |
| 4 | Target-shape list from real models (M,N,K, hidden, seq, batch) | R§3 | ❌ | No shape list; "sponsor shapes unknown" |
| 5 | Correctness tests before speed; compute-sanitizer memcheck + racecheck | Method | 🟡 | Tests exist; sanitizer **never run** |
| 6 | Warmup + CUDA events; **median** and spread of **100+** iters | Method | 🟡 | Harness reports mean/min/max/stddev; iteration count to verify |
| 7 | Identical GPU/clocks/dtype/shapes; watch throttling & other processes | Method, R§11 | ❌ | Clocks, temps, throttle reasons not recorded |
| 8 | Vendor baselines **cuBLAS / cuDNN**, PyTorch eager, **torch.compile (Inductor)**, naive, best-per-shape | R§4 | 🟡 | Only PyTorch eager, via case study; no direct cuBLAS/cuDNN; no Inductor |
| 9 | Percent of hardware peak (TFLOPS, GB/s) | Method, R§6 | 🟡 | Case study only; harness CSV lacks TFLOPS/GB/s/%peak |
| 10 | Nsight Systems timelines, Nsight Compute metrics per major optimization | Method, R§9 | ❌ | Never run |
| 11 | `CUDA_CHECK` on every call, `cudaGetLastError` on every launch | Method | ✅ | `src/common/check.h` — re-verify for each new kernel |
| 12 | Matmul ladder: naive → smem tiling → register blocking → float4 → double buffering → Tensor Cores (WMMA or mma.sync); speedup recorded per step | R§3 | 🟡 | 2 of 6 rungs |
| 13 | Layernorm: block- or warp-per-row, Welford or two-pass, warp shuffle, vectorized, optional residual/bias fusion | R§3 | 🟡 | Two-pass block-per-row only |
| 14 | Softmax: stable, online, warp-per-row short / block-per-row long, optional fused scale/mask | R§3 | 🟡 | Block-per-row three-pass only |
| 15 | Correctness vs PyTorch **float64** with explicit tolerances for **FP32, FP16, BF16** | R§3, R§9 | ⚠️ | FP32 only; docs call FP16/BF16 "out of scope" |
| 16 | Benchmark harness: CSV, TFLOPS, GB/s | R§3 | 🟡 | CSV yes; no throughput columns |
| 17 | PyTorch extension via **`torch.utils.cpp_extension`**, droppable into a model | R§3, R§9 | ⚠️ | ctypes binding; docs call cpp_extension "out of scope" |
| 18 | Experiments: tile sweep, shape sweep (+odd shapes), dtype comparison, occupancy vs registers, vectorized vs scalar, fusion vs separate, end-to-end model latency | R§5 | ❌ | None run as sweeps |
| 19 | Roofline plots | R§10 | ❌ | — |
| 20 | Technical report + demo notebook | R§10 | ❌ | — |
| 21 | ≥1 operator beats vendor on a target shape, or documented why not | R§9 | ❌ | All 2–11× slower |
| 22 | Kernel library with build system | R§10 | ✅ | CMake, presets, `find_package` |
| 23 | Never report speed for a kernel failing tests | R§11 | ✅ | Enforced by policy; enforce in harness too ([E] gate) |

**Superseded statements.** `docs/project-vision.md` §2 and
`docs/milestone-40.md` list Tensor Cores/WMMA, double buffering, vectorized
loads, fused operators, mixed precision and `torch.utils.cpp_extension` as out
of scope. Under `reference.md` all of these are **required**. Those lists are
superseded by this roadmap.

---

## 2. Hardware plan

| Machine | Arch | Use | Limits |
|---|---|---|---|
| Local RTX 3050 A Laptop | Ada sm_89, ~4 GB (≈2.9 GB free), WDDM | **Primary**: NCU counters, BF16, cp.async, final numbers | Needs Windows SDK; laptop clocks/thermals vary; NCU counters need admin |
| Kaggle T4 | Turing sm_75 | Correctness, FP16 Tensor Core (WMMA/mma.sync m16n8k8) | No BF16, no cp.async; perf counters usually blocked [R§1 prereqs] |
| Modal L4 | Ada sm_89 | Fallback for sm_89 numbers | Needs payment card |

**Critical path item #0:** install the Windows SDK so the local Ada GPU builds.
[R§1] says hardware-counter profiling needs a local or rented GPU, and the
DoD requires Nsight Compute evidence. Without a local build, R§9 cannot be met.

Every result file records GPU, arch, driver, CUDA, clocks, temperature and
throttle reasons. Turing and Ada numbers are never mixed in one table.

---

## 3. Week-by-week plan (mapped 1:1 to [R§8])

Each phase ends with tests in `tests/` and `docs/results/phase-N-*.md`
(existing workflow). No phase reports a speed number for a kernel failing its
tests [R§11].

### Week 1 — Phase 7: Environment, shapes, references, harness, baselines
- `scripts/setup_env.{sh,ps1}`: tool version check (`nvidia-smi`, `nvcc`, `ncu`, `nsys`), `.venv`, CUDA PyTorch + `triton numpy pandas matplotlib pytest`. Output → `docs/results/phase-7-env.md`.
- Create `scripts/`, `profiles/`; `.gitignore` exception for `profiles/**/*.ncu-rep`.
- **`bench/shapes.json`**: single source of truth for target and odd shapes (§4).
- Harness rework [R Method, R§3]: ≥100 timed iters (default 200), warmup ≥20, **median, p10/p90, IQR, min**; TFLOPS, GB/s, %peak columns; CSV to `results/<arch>/<op>.csv`; GPU clock/temp/throttle captured per row via NVML **[E-M1..M4]**.
- Direct vendor baselines in C++: **cuBLAS** (`cublasGemmEx`, FP32/FP16/BF16, plus `cublasLt` heuristics top-1), **cuDNN** layernorm/softmax (cuDNN softmax forward; cuDNN graph/frontend layernorm where available, else `torch.nn.functional` as documented substitute).
- Python baselines: PyTorch eager, **torch.compile (Inductor)**, **Triton** **[E-B1]**.
- Measured peaks: FMA microbench + STREAM-copy microbench → `results/<arch>/peaks.json` **[E-M5]**.
- **compute-sanitizer** memcheck + racecheck (+ initcheck, synccheck **[E-C1]**) on all existing kernels → `scripts/sanitize.sh`, logs in `results/<arch>/sanitizer/`.

### Week 2 — Phase 8: Multi-dtype foundation + naive kernels
- Public API: `ccml_<op>_f16`, `ccml_<op>_bf16` alongside `_f32` (same signature shape, `__half`/`__nv_bfloat16` I/O, FP32 accumulation). BF16 compiled only for sm_80+, returns `CCML_ERROR_NOT_IMPLEMENTED` below.
- Naive kernels for all three ops in all three dtypes.
- Correctness vs **PyTorch float64** reference with explicit per-dtype tolerances [R§3] — table committed in `docs/correctness.md`. Starting policy (to be justified by measured error): FP32 `atol 1e-5, rtol 1e-5`; FP16 `atol 1e-3, rtol 1e-2`; BF16 `atol 1e-2, rtol 5e-2`; matmul tolerance scaled by `sqrt(K)`.
- Report max abs and max rel error per test [R§6].

### Week 3 — Phase 9: Matmul — smem tiling (exists) + register blocking
- `REGISTER_TILED`: block tile BM×BN×BK, thread tile TM×TN (start 128×128×8, 8×8), templated. Tile sweep [R§5].
- NCU before/after; speedup vs previous rung recorded in `docs/optimization-log.md`.

### Week 4 — Phase 10: Matmul — float4 + double buffering + first NCU analysis
- `VECTORIZED` (float4 / 128-bit loads, transposed A in smem, padding/swizzle for bank conflicts).
- `DOUBLE_BUFFERED` (register prefetch on sm_75; `cp.async` multi-stage on sm_80+).
- Experiment: vectorized vs scalar [R§5]. NCU: achieved occupancy, memory throughput, warp stall reasons [R§6] → `profiles/matmul/`.

### Week 5 — Phase 11: Matmul — Tensor Cores vs cuBLAS
- `WMMA` (FP16/BF16 in, FP32 accumulate), then `MMA_SYNC` (`mma.sync` + `ldmatrix`, multistage `cp.async`).
- **[E-S1]** threadblock swizzle, **[E-S2]** split-K / stream-K for skinny shapes, **[E-F1]** fused bias/GELU/residual epilogue.
- Shape-by-shape table vs cuBLAS on target shapes [R§9]. Occupancy vs register usage sweep via `__launch_bounds__` [R§5].

### Week 6 — Phase 12: Layernorm — warp shuffle, vectorized, residual fusion
- `WARP_PER_ROW` (cols ≤ 1024), `WELFORD` (single pass, warp-shuffle Welford merge), `VECTORIZED` (128-bit), `FUSED_RESIDUAL_BIAS` (`y = LN(x + residual + bias)`, also writes the pre-norm sum).
- Resolve open question: cost of fp64 accumulation (measured).
- Fusion vs separate kernels experiment [R§5].

### Week 7 — Phase 13: Softmax — online and fused
- `WARP_PER_ROW` (short rows, register-resident), `ONLINE` (single pass running max/sum, Milakov & Gimelshein 2018), `FUSED_SCALE_MASK` (scale + additive/causal mask), block-per-row for long rows (≥ 4096, e.g. vocab 32000/50257).
- AUTO dispatch thresholds chosen by measurement **[E-S4]**.

### Week 8 — Phase 14: PyTorch extension + real model
- `python/ccml_ext/` built with **`torch.utils.cpp_extension`** (`CUDAExtension` + `setup.py`, and JIT `load()` path). Uses PyTorch's current stream, dispatch on dtype, rejects non-contiguous inputs.
- Register as `torch.library` custom ops with fake/meta impls so **torch.compile can trace through them** **[E-P1]**.
- Drop-in `nn.Module` replacements (`CcmlLinear`, `CcmlLayerNorm`, `CcmlSoftmax`); swap into a real model (GPT-2 small, 124M, fits 4 GB in FP16) and measure **end-to-end latency** [R§5]. Logit/perplexity parity check against the unmodified model.
- ctypes binding kept as the C-ABI test path; cpp_extension is the deliverable.

### Week 9 — Phase 15: Full sweep, ablations, roofline
- Every op × variant × dtype × shape in `bench/shapes.json`, against every baseline [R§4]. Raw CSV committed.
- Ablations: each optimization turned off individually.
- Roofline plots per op/dtype with measured and theoretical ceilings (`scripts/roofline.py`, matplotlib) [R§10].
- **[E-G1]** CUDA Graphs and **[E-G2]** multi-stream results for the transformer block.

### Week 10 — Phase 16: Freeze, report, demo
- Tag `v1.0`. `docs/report.md` (technical report), `notebooks/demo.ipynb` (build ext, run correctness, run model with swapped kernels, plot results).
- Final shape-by-shape comparison table [R§2, R§9] with honest losses [R§11].
- Definition-of-done audit (§5) with every box linked to evidence.

---

## 4. Target shapes (`bench/shapes.json`)

No sponsor shapes have been provided. [R§3] requires shapes from real models,
so these come from published architectures that fit the 4 GB local GPU, and
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
- Each shape must fit within 2.5 GB including baselines (VRAM guard in harness).

---

## 5. Definition of done [R§9, R§10]: audit checklist

- [ ] All kernels pass correctness for FP32, FP16, BF16 (BF16 on sm_80+), vs PyTorch fp64, explicit tolerances
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
| Windows SDK / local build stays broken | Week-1 blocker; fallback Modal L4. Without local or rented GPU, NCU evidence is impossible |
| NCU counters blocked (WDDM, non-admin) | Enable "allow access to GPU performance counters to all users" in NVIDIA Control Panel; run elevated |
| Laptop thermal throttling skews medians | NVML throttle-reason capture, cooldown between runs, interleaved A/B ordering, reject throttled samples **[E-M3]** |
| 4 GB VRAM | VRAM guard; shapes sized to device |
| No win vs cuBLAS | Focus on skinny/decode + fused ops; R§9 allows documented explanation |
| Enhancements eat schedule | Enhancements start only after the week's required items pass |
