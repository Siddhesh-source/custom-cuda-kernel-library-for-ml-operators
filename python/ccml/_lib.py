"""Locate and bind the ccml shared library via ctypes.

ctypes is used rather than pybind11 or torch.utils.cpp_extension on purpose:

* it needs no compiler at install time, so the Python layer cannot break the
  C++ build;
* it introduces no dependency beyond the standard library and numpy;
* it works against the same C ABI that any other language binding would use,
  so the ABI gets exercised rather than bypassed.

The cost is that array marshalling is explicit. That is acceptable for a
validation and convenience layer.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# --- status codes, mirroring include/ccml/status.h -------------------------

CCML_SUCCESS = 0
CCML_ERROR_INVALID_ARGUMENT = 1
CCML_ERROR_UNSUPPORTED_SHAPE = 2
CCML_ERROR_NOT_IMPLEMENTED = 3
CCML_ERROR_CUDA = 4
CCML_ERROR_OUT_OF_MEMORY = 5


class CcmlError(RuntimeError):
    """A ccml call returned a non-success status."""

    def __init__(self, status: int, detail: str, call: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"{call} failed: status={status} ({detail})")


class CcmlNotImplementedError(CcmlError, NotImplementedError):
    """The operator exists in the ABI but has no kernel behind it yet."""


def _library_filename() -> str:
    if sys.platform == "win32":
        return "ccml.dll"
    if sys.platform == "darwin":
        return "libccml.dylib"
    return "libccml.so"


def _candidate_paths() -> list[Path]:
    """Places the built library is plausibly found, most specific first."""
    name = _library_filename()
    candidates: list[Path] = []

    override = os.environ.get("CCML_LIBRARY_PATH")
    if override:
        candidates.append(Path(override))

    repo_root = Path(__file__).resolve().parents[2]
    for build_dir in ("build/default", "build/debug", "build"):
        base = repo_root / build_dir
        candidates.append(base / "bin" / name)
        candidates.append(base / "lib" / name)
        # Multi-config generators nest by configuration.
        for config in ("Release", "Debug", "RelWithDebInfo"):
            candidates.append(base / "bin" / config / name)
            candidates.append(base / "lib" / config / name)

    return candidates


def load_library() -> ctypes.CDLL:
    """Load ccml, raising a message that says what to do if it is missing."""
    tried = []
    for path in _candidate_paths():
        tried.append(str(path))
        if path.is_file():
            return ctypes.CDLL(str(path))

    raise FileNotFoundError(
        "could not find the ccml shared library.\n"
        "Build it first:\n"
        "    cmake --preset default\n"
        "    cmake --build --preset default\n"
        "Or set CCML_LIBRARY_PATH to the built library.\n"
        "Looked in:\n  " + "\n  ".join(tried)
    )


def _declare(lib: ctypes.CDLL) -> None:
    """Attach argtypes and restype to every entry point we call.

    Without this ctypes assumes int arguments and returns, which silently
    corrupts pointers on 64-bit builds.
    """
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_void_pp = ctypes.POINTER(ctypes.c_void_p)

    lib.ccml_status_string.argtypes = [ctypes.c_int]
    lib.ccml_status_string.restype = ctypes.c_char_p

    lib.ccml_last_error_string.argtypes = []
    lib.ccml_last_error_string.restype = ctypes.c_char_p

    lib.ccml_version_string.argtypes = []
    lib.ccml_version_string.restype = ctypes.c_char_p

    lib.ccml_device_malloc.argtypes = [c_void_pp, ctypes.c_size_t]
    lib.ccml_device_malloc.restype = ctypes.c_int

    lib.ccml_device_free.argtypes = [ctypes.c_void_p]
    lib.ccml_device_free.restype = ctypes.c_int

    lib.ccml_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ccml_memcpy_h2d.restype = ctypes.c_int

    lib.ccml_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ccml_memcpy_d2h.restype = ctypes.c_int

    lib.ccml_device_synchronize.argtypes = []
    lib.ccml_device_synchronize.restype = ctypes.c_int

    lib.ccml_matmul_f32.argtypes = [
        c_float_p, c_float_p, c_float_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p,
    ]
    lib.ccml_matmul_f32.restype = ctypes.c_int

    lib.ccml_layernorm_f32.argtypes = [
        c_float_p, c_float_p, c_float_p, c_float_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_int, ctypes.c_void_p,
    ]
    lib.ccml_layernorm_f32.restype = ctypes.c_int

    lib.ccml_softmax_f32.argtypes = [
        c_float_p, c_float_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p,
    ]
    lib.ccml_softmax_f32.restype = ctypes.c_int

    # FP16 and BF16 entry points take ccml_f16* / ccml_bf16*: 16-bit structs,
    # passed as untyped device pointers.
    vp = ctypes.c_void_p
    for suffix in ("f16", "bf16"):
        mm = getattr(lib, "ccml_matmul_" + suffix)
        mm.argtypes = [vp, vp, vp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, vp]
        mm.restype = ctypes.c_int
        ln = getattr(lib, "ccml_layernorm_" + suffix)
        ln.argtypes = [vp, vp, vp, vp, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                       ctypes.c_int, vp]
        ln.restype = ctypes.c_int
        sm = getattr(lib, "ccml_softmax_" + suffix)
        sm.argtypes = [vp, vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, vp]
        sm.restype = ctypes.c_int


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    """The loaded library, loading it on first use."""
    global _lib
    if _lib is None:
        loaded = load_library()
        _declare(loaded)
        _lib = loaded
    return _lib


def check(status: int, call: str) -> None:
    """Turn a status code into an exception, preserving the C-side detail."""
    if status == CCML_SUCCESS:
        return
    detail = lib().ccml_last_error_string().decode("utf-8", "replace")
    if status == CCML_ERROR_NOT_IMPLEMENTED:
        raise CcmlNotImplementedError(status, detail, call)
    raise CcmlError(status, detail, call)
