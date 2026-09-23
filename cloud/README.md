# Remote build and test

The development machine cannot compile CUDA locally: it has no Windows SDK, and
the CUDA toolkit installed there has no Visual Studio integration. Until that is
fixed, the build and every GPU run happen remotely.

There are two backends. Pick by what you need:

| | [Kaggle](kaggle/README.md) | Modal (this file) |
|---|---|---|
| Cost | Free, ~30 GPU-hours/week | **Requires a payment method for any GPU** |
| GPUs | P100 (sm_60), T4 (sm_75) | T4 through H100, incl. **L4 (sm_89)** |
| Matches the dev GPU (Ada 8.9) | No | Yes, with `CCML_GPU=L4` |
| Good for | Correctness | Correctness **and** tuning |

**Use Kaggle for correctness work now; use Modal with `CCML_GPU=L4` once
performance matters**, because tuning numbers from Pascal or Turing do not
transfer to Ada.

Modal's CPU worker still compiles everything for free — see the table below.

---

## Modal

Runs on a [Modal](https://modal.com) worker.

## Setup

    pip install modal
    python -m modal setup        # opens a browser to create an API token

`python -m modal profile current` prints the active profile if you are already
authenticated.

## Running

On Windows, set the console encoding first. The Modal CLI prints Unicode that
the default cp1252 encoding cannot represent, and it fails with
`'charmap' codec can't encode character`:

    set PYTHONIOENCODING=utf-8

Then:

    CCML_GPU=none modal run cloud/modal_app.py    # compile only, no GPU needed
    modal run cloud/modal_app.py                  # build and test on a GPU (T4)
    CCML_GPU=L4 modal run cloud/modal_app.py      # Ada, matches the dev GPU
    modal run cloud/modal_app.py --benchmarks     # also run the benchmarks

## Why CCML_GPU is an environment variable, not a flag

Modal validates a function's GPU request when the app is **registered**, which
happens before any command line flag is parsed. A `--no-gpu` flag therefore
cannot prevent the error from an account without GPU access. Reading the
environment at import time means `CCML_GPU=none` registers a plain CPU function
and never asks for a GPU at all.

## What each mode verifies

| | CPU (`CCML_GPU=none`) | GPU |
|---|---|---|
| CMake configure and generate | yes | yes |
| Every header and source compiles, warnings-as-errors | yes | yes |
| Linking, shared library, all executables | yes | yes |
| `install()` and `find_package` export rules | yes | yes |
| Kernels actually run | no | yes |
| Correctness tests | no | yes |
| Benchmarks | no | yes |

nvcc does not need a GPU to compile device code, which is what makes the CPU
mode useful. On a CPU worker `ccml_device_query` is expected to report
`cudaErrorInsufficientDriver` -- that it runs at all is what proves linking is
correct.

## GPU choice

`T4` (Turing, sm_75) is the default because it is the cheapest.

**For anything performance related, use `L4`** (Ada, sm_89), which is the same
architecture as the development GPU. Turing lacks async shared-memory copies
(`cp.async`, sm_80+), splits shared memory and L1 differently, and has different
occupancy limits, so a tile size chosen from T4 measurements says nothing about
the Ada target. T4 is fine for correctness, which is all this milestone needs.

The build detects the worker's compute capability at runtime rather than
assuming one, so switching GPUs needs no source change.

## Cost

GPU functions require a payment method on the Modal account; without one, every
GPU type is refused, including T4. CPU functions work on the free tier.

## Caveat on the verified build

The remote build uses GCC on Linux. MSVC `/W4` on Windows warns about different
things, so a clean remote build does not guarantee a clean local one once the
Windows SDK is installed.
