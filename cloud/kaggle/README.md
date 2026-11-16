# Remote build and test on Kaggle

Kaggle gives roughly 30 GPU-hours per week at no cost and needs **no payment
method**. It is the backend for all GPU work in this project; the few
exceptions are listed in [`../../docs/local.md`](../../docs/local.md).

## Credentials

1. Sign in at [kaggle.com](https://www.kaggle.com).
2. Go to [kaggle.com/settings](https://www.kaggle.com/settings), find the **API**
   section, and click **Create New Token**. A `kaggle.json` downloads.
3. Move it to `C:\Users\<you>\.kaggle\kaggle.json` (or `~/.kaggle/kaggle.json`).

Verify with:

    python cloud/kaggle/kaggle_run.py --push

If the credentials are missing it says so and does nothing else.

**Phone verification:** Kaggle requires a verified phone number before a kernel
may use a GPU accelerator. If the run comes back with no GPU attached, that is
usually why — check [kaggle.com/settings](https://www.kaggle.com/settings).

## Running

    python cloud/kaggle/kaggle_run.py            # push, wait, print the log
    python cloud/kaggle/kaggle_run.py --push     # push and return immediately
    python cloud/kaggle/kaggle_run.py --fetch    # fetch the last run's log
    python cloud/kaggle/kaggle_run.py --no-gpu   # CPU kernel, compile check only

The kernel is pushed **private**. It appears at
`kaggle.com/code/<username>/ccml-build`.

## How the source gets there

The whole repository is tarred, gzipped, base64-encoded and embedded directly in
the generated kernel script — about 66 KB for 67 files. The kernel unpacks
itself into `/kaggle/working/ccml` and builds there.

This is deliberate. The alternatives each add a moving part:

| Alternative | Why not |
|---|---|
| `git clone` in the kernel | Needs the repo pushed to GitHub, and needs internet enabled on the kernel |
| Kaggle Dataset | A second upload to keep in sync with the source |
| Manual notebook paste | Not reproducible, not scriptable |

Embedding means the kernel runs with `enable_internet: false` and still has
everything. Kaggle's image already ships the CUDA toolkit, numpy, pytest and
torch, so nothing needs installing.

`kaggle_run.py` fails loudly if the payload ever grows past 800 KB, which is the
point at which a Dataset becomes the right answer.

## Files

| File | Role |
|---|---|
| `kaggle_run.py` | Local driver: packages, pushes, polls, fetches |
| `kernel_template.py` | What runs on Kaggle. **Edit this**, not the copy in `_build/` |
| `_build/` | Generated push directory (gitignored) |
| `_output/` | Fetched kernel output (gitignored) |

The template carries a `__CCML_PAYLOAD__` marker that the driver substitutes.

## Toolchain handling

Kaggle's image is not guaranteed to have what this project needs, so the kernel
script adapts:

- **cmake** — the project requires 3.24+. If the image has something older the
  script pip-installs a newer one, and fails with a clear message if there is no
  internet to do that with.
- **ninja** — used if present, otherwise it falls back to `Unix Makefiles`.
- **architecture** — detected from `nvidia-smi` at runtime rather than assumed,
  so it is correct whether Kaggle hands out a P100 (sm_60) or a T4 (sm_75).

## Target GPU: Tesla T4 (sm_75)

The kernel metadata requests `machine_shape: NvidiaTeslaT4`, so every run gets
a T4 rather than a P100. The T4 is the **benchmark and tuning target** for this
project. All tile sizes, block sizes and AUTO thresholds are chosen from T4
measurements.

What the T4 cannot do, as measured by a probe run on 2026-11-16: native BF16
Tensor Cores and `cp.async` (both need sm_80+), and clock locking
(`nvidia-smi -lgc` is denied). Nsight Compute counters are blocked
(`ERR_NVGPUCTRPERM`), so NCU is a local item in [`../../docs/local.md`](../../docs/local.md).
compute-sanitizer and Nsight Systems work.

If the T4 pin leaves a run `QUEUED` for more than about 10 minutes, cancel it
and push with `--any-gpu`. The log records which GPU actually ran. BF16 *correctness* (with FP32 math)
does run on the T4.

## Getting the log

`--fetch` downloads `ccml_build_log.txt`, which the kernel writes to
`/kaggle/working` as it runs, and prints it. That file exists even when a step
fails, so a failed build is still diagnosable without opening the browser.
