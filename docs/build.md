# Building

## Requirements

| Component | Needed | Notes |
|---|---|---|
| CUDA Toolkit | 12.0 or newer | Developed against 13.2 |
| CMake | 3.25 or newer | 3.25 is what workflow presets need |
| C++ compiler | GCC 9+ (Kaggle) | MSVC + Windows SDK only for an optional local build |
| Python | 3.9+ | Optional, for the binding only |

Nothing else. The library depends on the CUDA runtime and nothing more.

## Where it builds

**On Kaggle.** `python cloud/kaggle/kaggle_run.py` packs the repository, builds
it on a Tesla T4 with GCC + Ninja, runs the tests, and fetches the log. Kaggle's
image already has CUDA, CMake 3.31, Ninja, PyTorch (CUDA build), Triton and
pynvml. Building locally on Windows is optional; see [local.md](local.md).

## Build

    cmake --preset default
    cmake --build --preset default

Outputs land in `build/default/bin` (executables and, on Windows, the DLL) and
`build/default/lib`.

The `default` preset is Release and targets `sm_75` (Kaggle T4). Change the
architecture for a different GPU:

    cmake -B build/mygpu -DCCML_CUDA_ARCHITECTURES=86
    cmake --build build/mygpu

Find your compute capability with:

    nvidia-smi --query-gpu=name,compute_cap --format=csv

or by running `ccml_device_query` once the build works.

## Options

| Option | Default | Effect |
|---|---|---|
| `CCML_CUDA_ARCHITECTURES` | `75` | Device architectures to compile for |
| `CCML_BUILD_TESTS` | `ON` | Build the correctness tests |
| `CCML_BUILD_BENCHMARKS` | `ON` | Build the benchmarks |
| `CCML_BUILD_EXAMPLES` | `ON` | Build `ccml_device_query` |
| `CCML_WARNINGS_AS_ERRORS` | `OFF` | Promote compiler warnings to errors |
| `BUILD_SHARED_LIBS` | `ON` | Shared is the default because the Python binding loads a shared library |

## Run

    build/default/bin/ccml_device_query
    ctest --preset default

`ctest` runs one executable per operator. To run a single one, or a single test
within it:

    ctest --preset default -R softmax
    build/default/bin/ccml_test_softmax large_positive

## Debug builds and compute-sanitizer

    cmake --preset debug
    cmake --build --preset debug
    compute-sanitizer --tool memcheck  build/debug/bin/ccml_test_matmul
    compute-sanitizer --tool racecheck build/debug/bin/ccml_test_matmul

The Debug preset adds device debug info (`-G`), which disables most device
optimization. Never take timings from a Debug build.

## Using ccml from another CMake project

    cmake --install build/default --prefix /path/to/install

then, downstream:

    find_package(ccml REQUIRED)
    target_link_libraries(my_target PRIVATE ccml::ccml)

## Python binding

See `python/README.md`. It requires no compilation and no CUDA-enabled PyTorch.
