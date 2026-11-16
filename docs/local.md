# Local GPU components

**Everything in this project runs on a Kaggle Tesla T4** (`python
cloud/kaggle/kaggle_run.py`), except the items below. These cannot run on Kaggle
at all. Each one was checked with a probe kernel on Kaggle on 2026-11-16, not
assumed.

Set these up locally. Nothing else needs a local GPU.

## What needs a local GPU

| # | Component | Spec clause | Why it fails on Kaggle (probe evidence) | Local requirement |
|---|---|---|---|---|
| L1 | **BF16 Tensor Core kernels**: WMMA / `mma.sync` with `__nv_bfloat16` operands | reference.md §3 (Tensor Cores), §5 dtype comparison, prerequisites ("BF16 needs 8.0+") | The T4 is sm_75. BF16 WMMA fragments do not compile for sm_75; they need sm_80+ | GPU with compute capability ≥ 8.0. The RTX 3050 A Laptop (sm_89) qualifies. On the T4, torch BF16 matmul on the T4 runs emulated at 2.5 TFLOPS, against 19.45 for FP16 |
| L2 | **`cp.async` multistage double buffering** (matmul ladder, sm_80+ form) | reference.md §3 (double buffering) | The `cp.async` instruction does not exist on sm_75. The register-prefetch form of double buffering **does** run on the T4 and satisfies the spec there | Same GPU as L1 |
| L3 | **Nsight Compute hardware counters** | reference.md Method, §6 (occupancy, throughput, stall reasons), §9 (NCU evidence) | `ncu` exits with **`ERR_NVGPUCTRPERM`**. The container runs as root but has `CAP_SYS_ADMIN` dropped, and the driver has `RmProfilingAdminOnly=1`, so counters cannot be enabled from inside the kernel | Local GPU with performance-counter access enabled for your user (see setup below) |

**Not local-only (verified on Kaggle T4):** the build (nvcc 13.0, CMake 3.31,
Ninja); all correctness tests including BF16 with FP32 math; FP16 Tensor Cores;
compute-sanitizer memcheck, racecheck, initcheck and synccheck (all ran clean on the probe); Nsight Systems timelines (`nsys profile` wrote a 110 KB `.nsys-rep`); PyTorch eager, torch.compile and Triton
baselines (torch 2.11+cu130, triton 3.6); `torch.utils.cpp_extension`
(`CUDA_HOME` set); NVML via pynvml.

**Clock locking** (`nvidia-smi -lgc`) is denied on Kaggle. It is not
local-only: the spec requires the *same* clocks within a comparison, and the
harness records clocks per row and flags drift. Locking clocks locally is
optional.

## Local setup: one time

1. **Windows SDK.** Without it nvcc cannot compile anything on Windows.
   In the Visual Studio Installer, choose Modify → Desktop development with C++
   → tick *Windows 11 SDK*. Check it with `echo %WindowsSdkDir%` in a Developer
   Command Prompt.
2. **Ninja:** `pip install ninja`. Configure from a Developer Command Prompt
   (`vcvars64.bat`) so that Ninja finds `cl.exe`.
3. **Build for sm_89:**
   ```
   cmake -B build/local -G Ninja -DCCML_CUDA_ARCHITECTURES=89
   cmake --build build/local
   ctest --test-dir build/local
   ```
4. **GPU performance counters (L3).** In the NVIDIA Control Panel, open Desktop
   → Enable Developer Settings, then Developer → Manage GPU Performance
   Counters → *Allow access to the GPU performance counters to all users*.
   Reboot. Check with
   `ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed build/local/bin/ccml_test_softmax`.
5. **Optional:** a CUDA build of PyTorch in `.venv`, only if you want to run L1
   or L2 through Python.

## Workflow for L3 (Nsight Compute)

Kaggle produces the kernels, the timings and the Nsight Systems timelines.
Locally, only `ncu` runs, on the same kernels and shapes, and its reports go to
`profiles/sm_89/`. The optimization log then cites the T4 timing next to the
local NCU metrics, with the architecture of each named. NCU evidence explains
*why* a change helps, and the T4 timing measures *how much*. The two are never
merged into one number.

## How the local results are kept separate

- Local results go in `results/sm_89/` and `profiles/sm_89/`. They are never
  mixed into `results/sm_75/` tables.
- L1 and L2 variants are compiled on every build. On sm_75 they return
  `CCML_ERROR_NOT_IMPLEMENTED`, so the Kaggle suite reports them as **SKIP**,
  never as PASS.
- The local run command for each phase is given in that phase's results
  document.
