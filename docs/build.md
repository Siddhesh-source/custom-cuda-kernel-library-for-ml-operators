# Building

## Requirements

| Component | Needed | Notes |
|---|---|---|
| CUDA Toolkit | 12.0 or newer | Developed against 13.2 |
| CMake | 3.25 or newer | 3.25 is what workflow presets need |
| C++ compiler | MSVC 2019+ / GCC 9+ / Clang 10+ | nvcc requires MSVC as host compiler on Windows |
| Windows SDK | required on Windows | See the note below |
| Python | 3.9+ | Optional, for the binding only |

Nothing else. The library depends on the CUDA runtime and nothing more.

## Windows: the Windows SDK

nvcc drives the MSVC host compiler, which needs the Windows SDK headers
(`corecrt.h`, `stdio.h`) and import libraries. An MSVC installation without the
SDK component cannot compile anything - not CUDA, not plain C++.

To check, in a Developer Command Prompt:

    echo %WindowsSdkDir%

An empty result means the SDK is missing. Install it through the Visual Studio
Installer (Modify, then Desktop development with C++, then tick a Windows SDK),
or from the command line:

    "C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" modify ^
      --installPath "<your VS install path>" ^
      --add Microsoft.VisualStudio.Component.Windows11SDK.26100 ^
      --includeRecommended --passive --norestart

## Windows: use the Ninja generator, not the Visual Studio generator

The presets specify `Ninja` deliberately. The Visual Studio generator drives CUDA
through MSBuild integration files (`CUDA <ver>.props` / `.targets`) that the
toolkit installs into the VS `BuildCustomizations` folder. A toolkit installed to
a custom location often does not place them there, and CMake then fails with:

    CMake Error ... No CUDA toolset found.

This is what happens on the current development machine: the toolkit has no
`extras/visual_studio_integration` directory at all. Ninja sidesteps MSBuild
entirely and calls `nvcc` directly.

Install Ninja if it is missing:

    pip install ninja

Configure from a **Developer Command Prompt** (or after running `vcvars64.bat`),
so that Ninja can find `cl.exe`:

    "<your VS install path>\VC\Auxiliary\Build\vcvars64.bat"

If you would rather use the Visual Studio generator, copy the four files from
`<cuda>/extras/visual_studio_integration/MSBuildExtensions` into
`<vs>/MSBuild/Microsoft/VC/<version>/BuildCustomizations`, or pass the toolset
explicitly:

    cmake -B build/vs -G "Visual Studio 17 2022" -T cuda="<cuda toolkit root>"

## Build

    cmake --preset default
    cmake --build --preset default

Outputs land in `build/default/bin` (executables and, on Windows, the DLL) and
`build/default/lib`.

The `default` preset is Release and targets `sm_89`. Change the architecture for
a different GPU:

    cmake -B build/mygpu -DCCML_CUDA_ARCHITECTURES=86
    cmake --build build/mygpu

Find your compute capability with:

    nvidia-smi --query-gpu=name,compute_cap --format=csv

or by running `ccml_device_query` once the build works.

## Options

| Option | Default | Effect |
|---|---|---|
| `CCML_CUDA_ARCHITECTURES` | `89` | Device architectures to compile for |
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
