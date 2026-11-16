# CLAUDE.md

## Authority (read first)
1. `docs/reference.md` is the specification and is binding. Comply with it exactly. Never weaken, skip or reinterpret a clause.
2. `docs/ROADMAP.md` is the plan, `docs/enhancements.md` holds optional additions, and `docs/PROJECT_OVERVIEW.md` is the summary.
3. If any other doc, comment or older "out of scope" list conflicts with (1), (1) wins. Fix the conflicting doc.
4. An enhancement never displaces a required item. Start one only after the phase's required items pass. Work is phase-gated by `prompt.md`, never by calendar.

## Hard rules
- Correctness before speed. Never time, report or commit a speed number for a kernel that fails its tests.
- Every CUDA API call is wrapped in `CUDA_CHECK` (`src/common/check.h`). Every launch is followed by `cudaGetLastError`.
- Timing: CUDA events, warmup ≥20, ≥100 timed iterations, report median + spread. Never use wall-clock time without synchronization.
- Comparisons use the same GPU, clocks, dtype and shapes. Record GPU, arch, driver, CUDA version, clocks, temperature and throttle state.
- Report % of hardware peak (TFLOPS or GB/s), not only speedup. Report losses honestly.
- Correctness is checked against a PyTorch float64 reference with explicit tolerances for each dtype (FP32, FP16, BF16).
- Run compute-sanitizer (memcheck and racecheck) on every new kernel.
- Never invent a number. Every figure comes from a committed script and its committed output.
- One optimization per step. Log it in `docs/optimization-log.md` with its hypothesis, NCU evidence (`profiles/`) and measured delta. Failed attempts are logged too.
- Never mix results from different architectures in one table. The target is the Kaggle T4 (sm_75).

## Phase workflow
Each phase ends with tests in `tests/`, a results document `docs/results/phase-N-<name>.md` (environment, verbatim output, what it proves and what it does not, how to reproduce), an update to the ROADMAP checklist, and then a commit and push.

## Code layout
- Public C ABI `include/ccml/`: `ccml_<op>_<f32|f16|bf16>(out, in..., dims..., params..., algo, stream)`. A new variant is a new `algo` enum value and a new `case`. An unbuilt variant returns `CCML_ERROR_NOT_IMPLEMENTED`.
- `src/<op>/<op>.cu` holds the host side (validation and dispatch). `src/<op>/<op>_kernels.cuh` holds device code only. Operators never include each other. Shared helpers go in `src/common/`.
- `tests/` checks correctness only and never times. `bench/` times only. `profiles/` holds small reports. `results/sm_75/` holds CSVs. `scripts/` holds automation.

## Build and test
```
cmake --workflow --preset default        # configure + build + ctest (sm_75)
python cloud/kaggle/kaggle_run.py          # run everything on a Kaggle T4
ctest --preset default -R <op>
```
Hardware: **every GPU task runs on a Kaggle T4 (sm_75)** via `cloud/kaggle/`. Credentials are already configured. Batch the work into one kernel run per phase to save quota. Only the items in `docs/local.md` need a local GPU; never plan local work outside that list. The local machine cannot build CUDA (no Windows SDK) and its PyTorch is CPU-only.

## Git
- **Branches:** `main` holds only finished work and receives exactly one merge, `dev` → `main`, when the whole project is complete. `dev` is the integration branch. All work happens on a **task branch** cut from `dev`. A task branch may cover one or several phases, and it is merged back into `dev` once its phases pass ACCEPT.
- **Branch names describe the task, never the phase number**: e.g. `measurement-infrastructure`, `matmul-register-tiling`, `rowwise-fusion`. Never `phase-7` or `p9-10`.
- Commit and push using the repo's configured git identity (`Siddhesh-source`). Push the current task branch after every commit. Never commit directly to `main`.
- **No attribution or watermark lines**: no `Co-Authored-By`, no "Generated with", no AI mentions in commits, PRs or code.
- Commit messages are imperative and scoped, for example `matmul: add register-blocked kernel (8x8 thread tile)`.

## Output style
Be terse. Show diffs, not whole files. Don't restate the plan or paste unchanged code. Point to `file:line`.
