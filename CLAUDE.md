# CLAUDE.md

## Authority (read first)
1. `docs/reference.md` is the specification and is binding. Comply with it exactly. Never weaken, skip or reinterpret a clause.
2. `docs/ROADMAP.md` is the plan, `docs/enhancements.md` holds optional additions, and `docs/PROJECT_OVERVIEW.md` is the summary.
3. If any other doc, comment or older "out of scope" list conflicts with (1), (1) wins. Fix the conflicting doc.
4. An enhancement never displaces a required item. Start one only after the week's required items pass.

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
- Never mix Turing (sm_75) and Ada (sm_89) results in one table.

## Phase workflow
Each phase ends with tests in `tests/`, a results document `docs/results/phase-N-<name>.md` (environment, verbatim output, what it proves and what it does not, how to reproduce), an update to the ROADMAP checklist, and then a commit and push.

## Code layout
- Public C ABI `include/ccml/`: `ccml_<op>_<f32|f16|bf16>(out, in..., dims..., params..., algo, stream)`. A new variant is a new `algo` enum value and a new `case`. An unbuilt variant returns `CCML_ERROR_NOT_IMPLEMENTED`.
- `src/<op>/<op>.cu` holds the host side (validation and dispatch). `src/<op>/<op>_kernels.cuh` holds device code only. Operators never include each other. Shared helpers go in `src/common/`.
- `tests/` checks correctness only and never times. `bench/` times only. `profiles/` holds small reports. `results/<arch>/` holds CSVs. `scripts/` holds automation.

## Build and test
```
cmake --workflow --preset default        # configure + build + ctest (sm_89)
cmake -B build/t4 -G Ninja -DCCML_CUDA_ARCHITECTURES=75 && cmake --build build/t4
ctest --preset default -R <op>
```
Hardware: the local machine has an RTX 3050 A Laptop (sm_89, 4 GB, about 2.9 GB free). The Windows SDK is missing, so CUDA does not build locally yet, and local PyTorch is CPU-only. Remote options: Kaggle T4 (sm_75, correctness only) and Modal L4 (sm_89). Use `cloud/`.

## Git
- Commit and push using the repo's configured git identity (`Siddhesh-source`). Push to `origin main` after every commit.
- **No attribution or watermark lines**: no `Co-Authored-By`, no "Generated with", no AI mentions in commits, PRs or code.
- Commit messages are imperative and scoped, for example `matmul: add register-blocked kernel (8x8 thread tile)`.

## Output style
Be terse. Show diffs, not whole files. Don't restate the plan or paste unchanged code. Point to `file:line`.
