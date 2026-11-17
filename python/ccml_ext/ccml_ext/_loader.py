"""Find the compiled extension: the installed ccml_ext._C (setup.py), or a JIT
build with torch.utils.cpp_extension.load() against the CMake-built library.

CCML_BUILD_DIR selects the CMake build directory for the JIT path (default:
<repo>/build/default); it must hold lib/libccml.so and generated/ccml/version.h.
"""

from __future__ import annotations

import os
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[3]


def load_jit(verbose: bool = False):
    """Compile (or reuse the cached build of) csrc/ccml_ext.cpp with
    torch.utils.cpp_extension.load and return the module."""
    from torch.utils.cpp_extension import load

    build = pathlib.Path(os.environ.get("CCML_BUILD_DIR", _REPO / "build" / "default")).resolve()
    lib = build / "lib"
    return load(
        name="ccml_ext_jit",
        sources=[str(pathlib.Path(__file__).resolve().parents[1] / "csrc" / "ccml_ext.cpp")],
        extra_include_paths=[str(_REPO / "include"), str(build / "generated")],
        extra_ldflags=[f"-L{lib}", "-lccml", f"-Wl,-rpath,{lib}"],
        extra_cflags=["-O3"],
        verbose=verbose,
    )


def load_backend():
    """The installed extension if present, otherwise the JIT build."""
    try:
        from . import _C  # type: ignore[attr-defined]

        return _C
    except ImportError:
        return load_jit()
