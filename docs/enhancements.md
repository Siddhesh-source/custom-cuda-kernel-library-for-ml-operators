# Enhancements

These additions go beyond the minimum that [`reference.md`](reference.md)
requires. Each one is chosen because it plausibly improves a measured result or
makes the results more trustworthy. The rules:

1. **Additive only.** An enhancement starts only after that week's required items pass.
2. **Hypothesis first.** Each is logged in `docs/optimization-log.md` with its
   predicted effect and mechanism. It is kept or reverted on measurement, and a
   failure is still logged.
3. **Behind an `algo` value or a flag.** The plain variant the spec requires
   stays available, so an ablation can always switch an enhancement off.

IDs are re`ferenced from [ROADMAP.md](ROADMAP.md).

---

## G — GPU scheduling (block, stream, launch level)

| ID | Enhancement | Why it should help | Where |
|---|---|---|---|
| **S1** | **Threadblock swizzle / rasterization.** Remap `blockIdx` so that concurrently resident CTAs share A row-panels and B column-panels (CUTLASS-style group-M swizzle) | Raises the L2 hit rate on large M·N grids; costs nothing when the grid is small | matmul |
| **S2** | **Split-K and Stream-K.** Split-K partitions K across CTAs, with an FP32 reduction or atomics. Stream-K (Osama et al., 2023) gives each persistent CTA an equal share of MAC iterations | For skinny/decode shapes (M ≤ 16, large K), a plain tiling launches fewer CTAs than there are SMs. This is the most likely place to beat cuBLAS [R§11] | matmul |
| **S3** | **Persistent row-wise kernels.** Launch exactly `SMs × occupancy` blocks and loop over rows with a grid-stride | Removes block scheduling overhead when there are many short rows (rows ≥ 8192, cols ≤ 1024) | layernorm, softmax |
| **S4** | **Measured AUTO dispatch table.** An offline autotuner writes `src/<op>/tuning_<arch>.inc`, mapping (dtype, shape bucket) → (algo, tile/block config). AUTO looks the shape up; it does not guess | "Your best kernel per shape" [R§4] becomes a real code path rather than a table in a report | all |
| **G1** | **CUDA Graphs.** Capture the transformer-block sequence once and replay it | Decode-size kernels run for single-digit µs, so launch overhead is a large share of their time. A graph removes most of the CPU-side launch cost | python ext, bench |
| **G2** | **Multi-stream overlap.** Run independent ops (the Q/K/V projections, or two microbatches) on separate streams joined by events. Measure the overlap with nsys | Small kernels leave SMs idle; concurrency fills them | bench, demo |
| **G3** | **L2 persistence window** (`cudaAccessPolicyWindow`) for weights reused across calls | Keeps hot weights in L2 between layers. Measure warm and cold | matmul |
| **G4** | **Occupancy-aware launch.** Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and `__launch_bounds__` variants, with the register cap swept | This *is* the spec's experiment "occupancy vs register usage" [R§5], run systematically | all |
| **G5** | **Remote job queue.** `scripts/run_remote.py` dispatches a phase to Kaggle (sm_75) or Modal (sm_89) and tags the results with the architecture | Correctness and performance runs stay separated by architecture and reproducible | cloud |

Excluded: Programmatic Dependent Launch, TMA and wgmma all need sm_90 and
cannot run on the Ada target.

## F — Fusion

| ID | Enhancement | Why |
|---|---|---|
| **F1** | Matmul epilogue fusion: `+bias`, GELU/ReLU, `+residual`, dtype cast | Avoids one or two full round-trips of the output through HBM. The usual way a hand-written kernel beats cuBLAS + separate ops |
| **F2** | `residual + bias + LayerNorm`, which also emits the pre-norm sum needed by the next residual | Required as optional by the spec [R§3]; implemented in full |
| **F3** | Softmax with fused `scale` and additive/causal mask (`softmax(x·s + m)`) | Required as optional by the spec [R§3]; this is the shape attention needs |
| **F4** | Fused vs separate comparison for every fused kernel | The spec's experiment "fusion vs separate kernels" [R§5] |

## M — Measurement rigor

| ID | Enhancement | Why |
|---|---|---|
| **M1** | Report median with a **bootstrap 95% CI**, plus p10/p90/IQR, over ≥100 iterations (default 200) | The spec asks for median and spread [R Method]. The CI makes "A beats B" a statistical claim |
| **M2** | **Cold and warm L2.** Flush L2 between iterations with a write to a buffer larger than L2, and report both | Small shapes fit entirely in L2. Without a flush, vendor and ours can both look unrealistically fast |
| **M3** | Thermal and clock guard: log SM clock, memory clock, temperature and throttle reasons through NVML for each run; lock clocks where permitted (`nvidia-smi -lgc`); **interleave A/B/A/B** variant order; flag runs that throttled | The spec requires same clocks and watching throttling [R Method, R§11]. Laptop GPUs drift |
| **M4** | Check that no other process is on the GPU before a run (NVML process list) | The spec says to watch for other processes [R Method] |
| **M5** | **Measured peaks**: FMA-throughput and STREAM-copy microbenchmarks, plus theoretical peaks. Report %peak against both | Without these, "% of peak" is relative to a spec sheet the laptop never reaches |
| **M6** | **Regression gate**: `scripts/compare.py` fails when a median regresses by more than 3% with non-overlapping CIs | Stops an optimization from quietly breaking another shape |
| **M7** | The harness **refuses to time a kernel that has not passed its correctness check** in the same run | Enforces [R§11] in code rather than relying on discipline |

## C — Correctness

| ID | Enhancement | Why |
|---|---|---|
| **C1** | compute-sanitizer `initcheck` + `synccheck` in addition to memcheck/racecheck | Finds reads of uninitialized shared memory and invalid barriers, which appear when tiles are padded |
| **C2** | Adversarial inputs: rows of all `-inf` (fully masked), constant rows, ±65504 (FP16 max), BF16 subnormals, NaN propagation policy | Fused masks and online softmax fail on exactly these |
| **C3** | Bitwise determinism check: two runs must give identical bytes. Split-K with atomics is explicitly marked non-deterministic | Makes split-K's trade-off visible |
| **C4** | Model-level parity: GPT-2 logits max error and perplexity delta with the kernels swapped in | Shows the kernels are usable in a model [R§9] |

## P — Python and integration

| ID | Enhancement | Why |
|---|---|---|
| **P1** | `torch.library.custom_op` registration with fake (meta) kernels | Lets torch.compile capture graphs containing ccml ops. Compare "Inductor" against "Inductor + ccml" |
| **P2** | `nn.Module` drop-ins and a `ccml.patch_model(model)` helper | "Dropped into a model" [R§3] in one line |

## B — Baselines

| ID | Enhancement | Why |
|---|---|---|
| **B1** | **Triton** baselines (fused softmax, layernorm, matmul in the style of the Triton tutorials) | `reference.md` installs triton. Triton is the realistic alternative to hand-written CUDA, so comparing against it strengthens the research question |
| **B2** | cuBLASLt with heuristic top-k and epilogue fusion as a second vendor baseline for F1 | A fair vendor comparison for fused matmuls |

---

## Priority if time runs short

S2 → F1 → M2/M3/M7 → S4 → F2/F3 → G1 → P1 → B1 → S1 → rest.

The reasoning: S2 and F1 are the most likely sources of a real win over the
vendor baseline [R§9]. M2, M3 and M7 decide whether any result is believable.
