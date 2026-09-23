"""numpy-facing wrappers over the ccml C API.

Each function takes host numpy arrays, moves them to the device, runs the
kernel, and brings the result back. That is not how the library would be used
inside a real training loop -- it is how it is used for validation and for a
quick interactive check, which is what this layer is for.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from ._lib import check, lib

# Algorithm enums, mirroring the public headers.
MATMUL_AUTO, MATMUL_NAIVE, MATMUL_TILED = 0, 1, 2
LAYERNORM_AUTO, LAYERNORM_TWO_PASS, LAYERNORM_WELFORD = 0, 1, 2
SOFTMAX_AUTO, SOFTMAX_BLOCK_PER_ROW, SOFTMAX_WARP_PER_ROW = 0, 1, 2

_FLOAT_P = ctypes.POINTER(ctypes.c_float)


def version() -> str:
    return lib().ccml_version_string().decode()


def _as_f32(array: np.ndarray, name: str) -> np.ndarray:
    """Contiguous float32 view, copying only if necessary."""
    result = np.ascontiguousarray(array, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    return result


class _Device:
    """RAII device allocation, so a failed call cannot leak GPU memory."""

    def __init__(self, nbytes: int) -> None:
        raw = ctypes.c_void_p()
        check(lib().ccml_device_malloc(ctypes.byref(raw), nbytes),
              "ccml_device_malloc")
        self.ptr = raw
        self.nbytes = nbytes

    def upload(self, host: np.ndarray) -> None:
        check(lib().ccml_memcpy_h2d(self.ptr, host.ctypes.data_as(ctypes.c_void_p),
                                    self.nbytes), "ccml_memcpy_h2d")

    def download(self, host: np.ndarray) -> None:
        check(lib().ccml_memcpy_d2h(host.ctypes.data_as(ctypes.c_void_p), self.ptr,
                                    self.nbytes), "ccml_memcpy_d2h")

    def as_float_p(self) -> "ctypes._Pointer":
        return ctypes.cast(self.ptr, _FLOAT_P)

    def __enter__(self) -> "_Device":
        return self

    def __exit__(self, *exc) -> None:
        lib().ccml_device_free(self.ptr)


def matmul(a: np.ndarray, b: np.ndarray, algo: int = MATMUL_AUTO) -> np.ndarray:
    """C = A @ B in float32."""
    a = _as_f32(a, "a")
    b = _as_f32(b, "b")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"shapes {a.shape} and {b.shape} are not multipliable")

    m, k = a.shape
    n = b.shape[1]
    out = np.empty((m, n), dtype=np.float32)

    with _Device(a.nbytes) as da, _Device(b.nbytes) as db, _Device(out.nbytes) as dc:
        da.upload(a)
        db.upload(b)
        check(lib().ccml_matmul_f32(dc.as_float_p(), da.as_float_p(),
                                    db.as_float_p(), m, n, k, algo, None),
              "ccml_matmul_f32")
        check(lib().ccml_device_synchronize(), "ccml_device_synchronize")
        dc.download(out)
    return out


def layernorm(x: np.ndarray,
              gamma: Optional[np.ndarray] = None,
              beta: Optional[np.ndarray] = None,
              eps: float = 1e-5,
              algo: int = LAYERNORM_AUTO) -> np.ndarray:
    """Layer normalization over the last dimension."""
    x = _as_f32(x, "x")
    rows, cols = x.shape
    out = np.empty_like(x)

    gamma_arr = None if gamma is None else np.ascontiguousarray(gamma, np.float32)
    beta_arr = None if beta is None else np.ascontiguousarray(beta, np.float32)
    for name, arr in (("gamma", gamma_arr), ("beta", beta_arr)):
        if arr is not None and arr.shape != (cols,):
            raise ValueError(f"{name} must have shape ({cols},), got {arr.shape}")

    with _Device(x.nbytes) as dx, _Device(out.nbytes) as dy:
        dx.upload(x)

        dg = _Device(gamma_arr.nbytes) if gamma_arr is not None else None
        dbeta = _Device(beta_arr.nbytes) if beta_arr is not None else None
        try:
            if dg is not None:
                dg.upload(gamma_arr)
            if dbeta is not None:
                dbeta.upload(beta_arr)

            check(lib().ccml_layernorm_f32(
                dy.as_float_p(), dx.as_float_p(),
                dg.as_float_p() if dg is not None else None,
                dbeta.as_float_p() if dbeta is not None else None,
                rows, cols, ctypes.c_float(eps), algo, None),
                "ccml_layernorm_f32")
            check(lib().ccml_device_synchronize(), "ccml_device_synchronize")
            dy.download(out)
        finally:
            if dg is not None:
                lib().ccml_device_free(dg.ptr)
            if dbeta is not None:
                lib().ccml_device_free(dbeta.ptr)
    return out


def softmax(x: np.ndarray, algo: int = SOFTMAX_AUTO) -> np.ndarray:
    """Softmax over the last dimension."""
    x = _as_f32(x, "x")
    rows, cols = x.shape
    out = np.empty_like(x)

    with _Device(x.nbytes) as dx, _Device(out.nbytes) as dy:
        dx.upload(x)
        check(lib().ccml_softmax_f32(dy.as_float_p(), dx.as_float_p(),
                                     rows, cols, algo, None),
              "ccml_softmax_f32")
        check(lib().ccml_device_synchronize(), "ccml_device_synchronize")
        dy.download(out)
    return out
