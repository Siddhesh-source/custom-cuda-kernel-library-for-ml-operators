# prompt.md: prompts for building ccml

How to use these:
- All GPU work runs on a **Kaggle T4** (`python cloud/kaggle/kaggle_run.py`). Only `docs/local.md` items run locally.
- Run **one prompt per session**, and `/clear` between phases. CLAUDE.md loads automatically, so the prompts never repeat its rules.
- The prompts reference files instead of pasting them. Claude reads only what it needs.
- Each prompt has five parts: **GOAL** (the outcome), **SPEC** (the clauses of `docs/reference.md` it serves), **DO** (the ordered steps), **ACCEPT** (checks that can be verified mechanically), and **OUT** (what to report). This keeps them short and leaves nothing to guess.
- Fill in `{placeholders}` before sending.
- The phase prompts follow `docs/ROADMAP.md` §3.

---

## 0. Session bootstrap (prepend only if CLAUDE.md is not loaded)
```
Obey CLAUDE.md. docs/reference.md is binding. Read docs/ROADMAP.md §{n} only. Terse output, diffs only.
```

---

## Phase 7 · Environment, shapes, harness, baselines
```
GOAL: Phase 7 per docs/ROADMAP.md §3.
SPEC: reference.md Setup, Method, §3 shape list + harness, §4 baselines.
DO:
1. scripts/setup_env.{sh,ps1}: verify nvidia-smi/nvcc/ncu/nsys, create .venv, CUDA torch + triton numpy pandas matplotlib pytest; print versions.
2. Create scripts/, profiles/; .gitignore: allow profiles/**/*.ncu-rep, *.nsys-rep.
3. bench/shapes.json from ROADMAP §4 (target + odd shapes, per op).
4. Harness: warmup>=20, iters>=200 default, median/p10/p90/IQR/min + bootstrap 95% CI; FLOPs/bytes per op -> TFLOPS, GB/s, %peak (theoretical + measured); NVML clocks/temp/throttle per row; refuse to time unless correctness check passed this run; CSV -> results/sm_75/<op>.csv.
5. Vendor baselines in bench/: cuBLAS (cublasGemmEx + cublasLt top-1), cuDNN softmax/layernorm (else documented torch fallback).
6. scripts/bench_torch.py: PyTorch eager, torch.compile(mode="max-autotune"), Triton tutorial kernels, same shapes, same CSV schema.
7. scripts/peaks.cu: FMA + STREAM-copy microbench -> results/sm_75/peaks.json.
8. scripts/sanitize.sh: memcheck, racecheck, initcheck, synccheck over all test binaries; logs -> results/sm_75/sanitizer/.
ACCEPT: build -Werror clean; all ctest pass; sanitizer 0 errors; CSV has all columns; docs/results/phase-7-*.md written.
OUT: files changed (1 line each), test/sanitizer totals, blockers. Commit+push.
```

## Phase 8 · Multiple dtypes and naive kernels
```
GOAL: FP16 + BF16 for all 3 ops, naive variants, fp64-reference correctness.
SPEC: reference.md §3 correctness suite, §9 "all kernels pass for each dtype".
DO:
1. Public API: ccml_<op>_f16/_bf16 mirroring _f32 (__half/__nv_bfloat16 I/O, FP32 accumulate). BF16 via FP32 math runs on T4; only BF16 Tensor Core variants are gated sm_80+ (docs/local.md).
2. Template kernels on T (load/store converters in src/common/dtype.cuh); no copy-paste per dtype.
3. Tests: compare to torch float64 reference; per-dtype atol/rtol table in docs/correctness.md; matmul tol scaled by sqrt(K); log max abs + max rel error per case.
4. Adversarial inputs: -inf rows, constant rows, ±65504, NaN policy.
ACCEPT: every op x dtype x shape passes (BF16 skipped only on sm<80, reported as SKIP); sanitizer clean.
OUT: pass/fail/skip table per op x dtype; worst errors. Commit+push.
```

## Phases 9–11 · Matmul optimization step (reuse for every rung)
Rungs: Phase 9 REGISTER_TILED · Phase 10 VECTORIZED, DOUBLE_BUFFERED · Phase 11 WMMA, MMA_SYNC. Requires the counter-enabling reboot (docs/local.md) before Phase 9.
```
GOAL: Add matmul variant {REGISTER_TILED|VECTORIZED|DOUBLE_BUFFERED|WMMA|MMA_SYNC} as one isolated step.
SPEC: reference.md §3 matmul ladder; §5 tile sweep, vectorized vs scalar, occupancy vs regs; §9 NCU evidence.
DO:
1. Write hypothesis (mechanism + predicted effect) to docs/optimization-log.md BEFORE coding.
2. Implement in src/matmul/matmul_kernels.cuh, templated on tile params; add algo enum + dispatch case. Previous rungs untouched.
3. Tests: all shapes x dtypes in tests/test_matmul.cpp pass + sanitizer clean.
4. Sweep tile configs on target shapes; keep top-3.
5. LOCAL (docs/local.md L3; scripts/local_build.ps1 first): ncu --set full on 2 representative shapes (one target, one skinny) for prev rung vs new -> profiles/matmul/<algo>_<shape>.ncu-rep; extract occupancy, mem throughput, top-3 stall reasons, regs/thread.
6. Bench vs previous rung, cuBLAS, torch eager, torch.compile.
ACCEPT: tests+sanitizer green; log entry has hypothesis, NCU deltas, median+CI, %peak, verdict (kept/reverted).
OUT: 5-line table {shape, prev µs, new µs, cuBLAS µs, %peak}. Commit+push.
```

## Phase 11 · Tensor Cores and scheduling enhancements
```
GOAL: FP16 WMMA then mma.sync m16n8k8+ldmatrix with register-prefetch pipeline (T4); then S1 swizzle, S2 split-K/stream-K, F1 fused epilogue. BF16 TC + cp.async variants: implement, compile-check for sm_80, leave the run to docs/local.md.
SPEC: reference.md §3 (Tensor Cores), §9 table vs cuBLAS, §11 focus skinny/fused shapes. docs/enhancements.md S1,S2,F1.
DO: Follow "Matmul optimization step" per item. For S2 target M<=16 decode shapes; mark atomic split-K non-deterministic. For F1 compare against cuBLASLt epilogue (B2) and cuBLAS+separate kernel.
ACCEPT: shape-by-shape table vs cuBLAS in docs/results/phase-11-*.md, wins AND losses listed.
OUT: shapes where we win (with CI), shapes where we lose and why (NCU-backed).
```

## Phase 12 · Layernorm
```
GOAL: Layernorm variants WARP_PER_ROW, WELFORD, VECTORIZED (128-bit), FUSED_RESIDUAL_BIAS (y=LN(x+res+bias), also writes sum); persistent variant S3.
SPEC: reference.md §3 layernorm; §5 fusion vs separate, vectorized vs scalar.
DO: one variant per step (log hypothesis -> implement -> tests+sanitizer -> NCU -> bench vs cuDNN/torch/Triton/compile). Measure fp64-accumulation cost vs fp32-Welford. Set AUTO thresholds from data (tuning table S4).
ACCEPT: all dtypes pass; GB/s + %peak BW reported; fusion vs separate table.
OUT: best variant per cols bucket with %peak BW. Commit+push.
```

## Phase 13 · Softmax
```
GOAL: Softmax WARP_PER_ROW (short rows, register-resident), ONLINE (single pass running max/sum), FUSED_SCALE_MASK (scale + additive/causal mask), block-per-row for long rows (vocab 32000/50257), persistent S3.
SPEC: reference.md §3 softmax; §5 fusion experiment.
DO: same per-variant loop as layernorm. Test fully-masked rows (-inf) -> define and test output policy. AUTO thresholds measured.
ACCEPT: all dtypes pass; rows sum to 1 within tol; GB/s + %peak; fused vs separate table.
OUT: best variant per cols bucket. Commit+push.
```

## Phase 14 · PyTorch extension and model integration
```
GOAL: torch.utils.cpp_extension binding usable in a real model; end-to-end latency.
SPEC: reference.md §3 PyTorch extension, §5 end-to-end model latency, §9 bindings usable in a model.
DO:
1. python/ccml_ext/: CUDAExtension setup.py + JIT load(); ops take torch tensors, dispatch on dtype, use at::cuda::getCurrentCUDAStream(), TORCH_CHECK contiguity/device/dtype.
2. Register via torch.library custom_op + register_fake so torch.compile traces through (P1).
3. Drop-ins CcmlLinear/CcmlLayerNorm/CcmlSoftmax + ccml.patch_model(model) (P2).
4. GPT-2 small FP16: logits max err + perplexity delta vs stock (C4); latency: eager, eager+ccml, compile, compile+ccml, CUDA Graph+ccml (G1); batch {1,8}, seq {128,512}.
5. pytest suite in python/tests/.
ACCEPT: pytest green; parity within tolerance; latency table committed.
OUT: latency table + parity numbers. Commit+push.
```

## Phase 15 · Full sweep, ablations, roofline
```
GOAL: All R§5 experiments + roofline.
SPEC: reference.md §5 all seven experiments, §6 metrics, §10 roofline plots.
DO: scripts/sweep.py runs every op x variant x dtype x shape x baseline (cold+warm L2, interleaved order, throttle-flagged) -> results/sm_75/. Ablations: disable each optimization singly. scripts/roofline.py -> results/sm_75/plots/*.png (measured + theoretical ceilings). G2 multi-stream nsys timeline -> profiles/.
ACCEPT: every R§5 experiment has a CSV + plot + 3-line finding in docs/results/phase-15-*.md.
OUT: list of experiments with one-line finding each.
```

## Phase 16 · Freeze, report and demo
```
GOAL: v1.0 freeze.
SPEC: reference.md §9 definition of done, §10 deliverables.
DO: docs/report.md (question, method, shape-by-shape table vs all baselines, per-optimization attribution w/ NCU, roofline, honest losses, threats to validity); notebooks/demo.ipynb (build, correctness, model swap, plots); audit ROADMAP §5 linking evidence per box; tag v1.0.
ACCEPT: every DoD box checked with a link, or explicitly explained.
OUT: DoD audit table. Commit, tag, push.
```

---

## Utility prompts

**Spec compliance audit** (run at the end of each phase)
```
Audit repo vs docs/reference.md clause by clause. Output only: clause | status (✅/🟡/❌/⚠️) | evidence path | fix. Update docs/ROADMAP.md §1. No prose.
```

**Fix a failing test**
```
Test {name} fails: {1-line symptom}. Find root cause (read only implicated files), minimal fix, rerun that test + sanitizer. Output: cause (1 line), diff, result.
```

**Read an NCU report**
```
Analyse profiles/{file}.ncu-rep (ncu --import ... --page details --csv). Output: occupancy (achieved/theoretical), DRAM/L2/SMEM throughput %, top-3 stall reasons, limiter (compute/memory/latency), 1 next optimization with predicted gain. ≤10 lines.
```

**Write a phase results document**
```
Write docs/results/phase-{n}-{name}.md per docs/results/README.md template from {log/CSV paths}. Copy numbers verbatim; absent = "not measured". Include "what this does NOT prove".
```

**Run on a remote GPU**
```
Run phase {n} build+tests+sanitizer+bench+nsys in ONE Kaggle T4 run via cloud/kaggle/kaggle_run.py; fetch outputs into results/sm_75/ and profiles/; never merge with other arch tables. Output: pass/fail totals + CSV paths.
```

**Pre-commit review**
```
Review staged diff for: CUDA_CHECK/cudaGetLastError gaps, races/missing __syncthreads, OOB on non-multiple shapes, dtype accumulation errors, spec violations. Findings only, ranked. Then commit+push (no attribution lines).
```
