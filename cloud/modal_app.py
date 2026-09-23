"""Build and run ccml on a Modal worker.

The development machine cannot compile CUDA at all (no Windows SDK, and the
local toolkit has no Visual Studio integration), so the build and every GPU run
happen remotely instead.

Usage:

    CCML_GPU=none modal run cloud/modal_app.py    # compile only, no GPU needed
    modal run cloud/modal_app.py                  # build and test on a GPU
    modal run cloud/modal_app.py --benchmarks     # also run the benchmarks
    CCML_GPU=L4 modal run cloud/modal_app.py      # pick a different GPU

The GPU is selected by the CCML_GPU environment variable rather than a command
line flag because Modal validates a function's GPU request when the app is
registered, before any flag is read. Setting CCML_GPU=none registers a plain CPU
function, which is what makes the compile-only path work on an account with no
GPU access.

GPU functions require a payment method on the Modal account. Without one,
CCML_GPU=none still verifies that everything compiles, links and installs,
because nvcc does not need a GPU to compile device code.

On Windows, set PYTHONIOENCODING=utf-8 first -- the Modal CLI prints Unicode
that the default cp1252 console encoding cannot represent.
"""

from __future__ import annotations

import os
import pathlib

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Read-only mount of the repository, and a writable working copy. They must be
# different paths: the mount cannot be built into.
REMOTE_SOURCE = "/opt/ccml-source"
REMOTE_WORK = "/workspace"

# T4 is Turing, compute capability 7.5, the cheapest GPU Modal offers.
#
# ARCHITECTURE CAVEAT: the development GPU is Ada (8.9). Turing lacks async
# shared-memory copies (cp.async, sm_80+), splits shared memory and L1
# differently, and has different occupancy limits. That is fine for
# CORRECTNESS, which is what this milestone needs, and it is why the build
# detects the architecture at runtime instead of assuming sm_89.
#
# It is NOT fine for tuning: a tile size chosen from T4 measurements says
# nothing about the Ada target. Use L4 (also 8.9) for anything performance
# related, so that occupancy and tiling decisions actually transfer.
DEFAULT_GPU = "T4"

# Architecture to compile for when no GPU is present to ask. Matches the
# development machine so the compile-only path exercises the real target.
FALLBACK_ARCHITECTURE = "89"

# CUDA 12.4 is comfortably new enough for sm_89 and this tag is reliably
# available. The local toolkit is 13.2; that difference does not matter for the
# source, but any timing taken here must record which toolkit produced it.
CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"

_requested = os.environ.get("CCML_GPU", DEFAULT_GPU).strip()
GPU = None if _requested.lower() in ("", "none", "cpu", "no") else _requested


def _ignore(path: pathlib.Path) -> bool:
    """Keep build trees, VCS metadata and caches out of the image."""
    excluded = {".git", "build", "out", "__pycache__", ".pytest_cache", ".vs"}
    return bool(set(path.parts) & excluded) or path.suffix in {".pyc", ".pyd"}


image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    # cmake from apt on Ubuntu 22.04 is 3.22, older than this project requires.
    .pip_install("cmake", "ninja", "numpy", "pytest")
    # CPU-only torch: it is the reference oracle, never the thing under test.
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(REPO_ROOT, REMOTE_SOURCE, ignore=_ignore)
)

app = modal.App("ccml-build", image=image)


def _run(label, command, cwd=None, check=True):
    """Run one step, echo it, and return (exit code, combined output)."""
    import subprocess

    print("")
    print("=" * 72)
    print(label)
    print("$ " + " ".join(command))
    print("=" * 72)

    completed = subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    print(completed.stdout, end="")

    if check and completed.returncode != 0:
        raise RuntimeError(label + " failed with exit code "
                           + str(completed.returncode))
    return completed.returncode, completed.stdout


def _detect_gpu():
    """Return (name, architecture) for this worker, or (None, None) on a CPU box."""
    import subprocess

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except FileNotFoundError:
        return None, None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None, None

    first_line = completed.stdout.strip().splitlines()[0]
    name, compute_cap = (part.strip() for part in first_line.split(","))
    return name, compute_cap.replace(".", "")


def _configure_and_build(architecture, results):
    """Copy the read-only mount somewhere writable, configure and build.

    Returns (source directory, build directory).
    """
    import shutil

    work = REMOTE_WORK + "/ccml"
    build = REMOTE_WORK + "/build"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(REMOTE_SOURCE, work)

    _, configure_output = _run("Configure", [
        "cmake", "-S", work, "-B", build,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCCML_CUDA_ARCHITECTURES=" + architecture,
        # On deliberately: this code has never been compiled anywhere, so any
        # warning is a finding rather than noise.
        "-DCCML_WARNINGS_AS_ERRORS=ON",
    ])
    results["configure"] = configure_output

    _, build_output = _run("Build", ["cmake", "--build", build, "--parallel"])
    results["build"] = build_output

    _run("Install rules", ["cmake", "--install", build,
                           "--prefix", REMOTE_WORK + "/install"])
    return work, build


@app.function(gpu=GPU, timeout=3600)
def build_and_run(run_benchmarks: bool = False,
                  run_python_tests: bool = True) -> dict:
    """Configure, build, and exercise the library as far as the worker allows.

    On a GPU worker this runs the full correctness suite. On a CPU worker it
    still compiles and links everything, which verifies CMake, every header,
    every source under warnings-as-errors, and the install/export rules -- just
    not anything that requires running a kernel.
    """
    results = {}

    _, nvcc_version = _run("CUDA toolkit", ["nvcc", "--version"])
    results["nvcc"] = nvcc_version

    gpu_name, architecture = _detect_gpu()
    if gpu_name is None:
        architecture = FALLBACK_ARCHITECTURE
        results["mode"] = "cpu-build-only"
        print("")
        print("No GPU on this worker. Compiling for sm_" + architecture
              + "; kernels will not be run.")
    else:
        results["mode"] = "gpu"
        results["gpu"] = gpu_name
        _run("GPU and driver", ["nvidia-smi"])
        print("")
        print("Detected " + gpu_name + ", building for sm_" + architecture)

    results["architecture"] = architecture

    work, build = _configure_and_build(architecture, results)

    # On a CPU worker this is expected to report no device. That it runs at all
    # is what proves linking is correct.
    _, device_query = _run("Device properties",
                           [build + "/bin/ccml_device_query"], check=False)
    results["device_query"] = device_query

    _run("Built artifacts", ["ls", "-la", build + "/bin", build + "/lib"],
         check=False)

    if gpu_name is None:
        print("")
        print("Skipping tests and benchmarks: no GPU on this worker.")
        return results

    code, ctest_output = _run("Correctness tests",
                              ["ctest", "--test-dir", build, "--output-on-failure"],
                              check=False)
    results["ctest"] = ctest_output
    results["ctest_exit_code"] = str(code)

    # ctest reports pass/fail per executable; the PASS/FAIL/SKIP breakdown
    # inside each one is what actually matters at this stage.
    for operator in ("matmul", "layernorm", "softmax"):
        _, output = _run("Test detail: " + operator,
                         [build + "/bin/ccml_test_" + operator], check=False)
        results["test_" + operator] = output

    if run_python_tests:
        _, pytest_output = _run(
            "Python binding tests",
            ["python", "-m", "pytest", work + "/python/tests", "-v"],
            cwd=work, check=False)
        results["pytest"] = pytest_output

    if run_benchmarks:
        for operator in ("matmul", "layernorm", "softmax"):
            _, output = _run("Benchmark: " + operator,
                             [build + "/bin/ccml_bench_" + operator], check=False)
            results["bench_" + operator] = output

    return results


@app.local_entrypoint()
def main(benchmarks: bool = False, python_tests: bool = True):
    print("Requested GPU: " + str(GPU if GPU else "none (CPU worker)"))

    results = build_and_run.remote(run_benchmarks=benchmarks,
                                   run_python_tests=python_tests)

    print("")
    print("#" * 72)
    print("# SUMMARY")
    print("#" * 72)
    print("Mode           : " + str(results.get("mode")))
    print("GPU            : " + str(results.get("gpu", "none (CPU worker)")))
    print("Architecture   : sm_" + str(results.get("architecture")))
    if "ctest_exit_code" in results:
        print("ctest exit code: " + str(results.get("ctest_exit_code")))
    print("")
    print("Device properties:")
    print(results.get("device_query", "<unavailable>"))
