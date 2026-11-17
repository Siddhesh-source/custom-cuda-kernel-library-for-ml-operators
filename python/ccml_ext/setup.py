"""Build the ccml PyTorch extension with torch.utils.cpp_extension (CUDAExtension).

The extension is a thin binding over the ccml C library, which is built by
CMake first (so the extension and the C++ test suites run the same binary):

    cmake --preset default && cmake --build build/default
    CCML_BUILD_DIR=build/default pip install --no-build-isolation ./python/ccml_ext

CCML_BUILD_DIR (default: ../../build/default relative to this file) must hold
lib/libccml.so and generated/ccml/version.h. The library directory is baked in
as an rpath, so no LD_LIBRARY_PATH is needed. The JIT alternative, with no
install step, is ccml_ext.load_jit() (ccml_ext/_loader.py).
"""

import os
import pathlib

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD = pathlib.Path(os.environ.get("CCML_BUILD_DIR", REPO / "build" / "default")).resolve()
LIB = BUILD / "lib"

setup(
    name="ccml_ext",
    version="0.1.0",
    description="ccml CUDA kernels for PyTorch",
    packages=["ccml_ext"],
    ext_modules=[
        CUDAExtension(
            name="ccml_ext._C",
            sources=[str(HERE / "csrc" / "ccml_ext.cpp")],
            include_dirs=[str(REPO / "include"), str(BUILD / "generated")],
            library_dirs=[str(LIB)],
            libraries=["ccml"],
            extra_link_args=[f"-Wl,-rpath,{LIB}"],
            extra_compile_args={"cxx": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
