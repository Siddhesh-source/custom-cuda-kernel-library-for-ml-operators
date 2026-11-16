"""Minimal PyTorch interface over the ccml CUDA kernels (interim).

reference.md section 3 requires a torch.utils.cpp_extension extension; that is
Phase 14 (docs/ROADMAP.md) and replaces this module. Until then this ctypes
layer is the zero-copy path: validate the tensor, pass its data_ptr() to the C
ABI, let the kernel write into a tensor PyTorch allocated, on PyTorch's current
stream.

float32, float16 and bfloat16 tensors dispatch to ccml_<op>_f32 / _f16 / _bf16.
Tensors that are not on the GPU, not one of those dtypes, or not contiguous are
rejected with an explanation rather than silently converted -- a hidden
.cpu(), .float() or .contiguous() is a cost that should be the caller's choice.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import torch

from ._lib import check, lib

__all__ = ["custom_matmul", "custom_layernorm", "custom_softmax"]

_FLOAT_P = ctypes.POINTER(ctypes.c_float)

# Algorithm selectors, mirroring the public headers.
_MATMUL_AUTO = 0
_LAYERNORM_AUTO = 0
_SOFTMAX_AUTO = 0


_SUFFIX = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}


def _ptr(tensor: Optional[torch.Tensor]):
    """Device pointer typed for the entry point of this tensor's dtype."""
    if tensor is None:
        return None
    if tensor.dtype is torch.float32:
        return _float_ptr(tensor)
    return ctypes.c_void_p(tensor.data_ptr())


def _entry(op: str, dtype: torch.dtype):
    name = "ccml_" + op + "_" + _SUFFIX[dtype]
    return getattr(lib(), name), name


def _float_ptr(tensor: Optional[torch.Tensor]):
    """The tensor's device pointer, typed for the C ABI. None stays NULL."""
    if tensor is None:
        return None
    return ctypes.cast(ctypes.c_void_p(tensor.data_ptr()), _FLOAT_P)


def _stream(device: torch.device) -> ctypes.c_void_p:
    """PyTorch's current stream for this device.

    Running on it is what makes the operator compose with surrounding PyTorch
    work: the kernel is ordered behind whatever produced its input and ahead of
    whatever consumes its output, with no synchronisation and no copy. Using the
    default stream instead would be a race against anything PyTorch has queued.
    """
    return ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)


def _validate(tensor, name: str, min_dims: int = 1) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")

    if not tensor.is_cuda:
        raise ValueError(
            f"{name} must be a CUDA tensor, but it is on {tensor.device}. "
            f"ccml operates on GPU memory and will not copy for you; "
            f"move it explicitly with {name}.cuda()."
        )

    if tensor.dtype not in _SUFFIX:
        raise TypeError(
            f"{name} must be float32, float16 or bfloat16, got {tensor.dtype}; "
            f"cast explicitly if that is what you want."
        )

    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous. ccml assumes a row-major contiguous "
            f"layout; call {name}.contiguous() explicitly, which costs a copy."
        )

    if tensor.dim() < min_dims:
        raise ValueError(
            f"{name} must have at least {min_dims} dimension(s), got {tensor.dim()}"
        )

    if tensor.numel() == 0:
        raise ValueError(f"{name} is empty; every dimension must be positive")


def _rows_and_cols(tensor: torch.Tensor) -> tuple:
    """Collapse leading dimensions, as the row-wise operators expect.

    A [batch, seq, hidden] tensor reduced over hidden is rows = batch*seq,
    cols = hidden. The tensor is contiguous, so this is arithmetic on the shape
    and moves nothing.
    """
    cols = tensor.shape[-1]
    return tensor.numel() // cols, cols


def custom_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension of a contiguous CUDA tensor."""
    _validate(x, "x", min_dims=1)
    rows, cols = _rows_and_cols(x)

    out = torch.empty_like(x)
    fn, name = _entry("softmax", x.dtype)
    with torch.cuda.device(x.device):
        check(fn(_ptr(out), _ptr(x), rows, cols, _SOFTMAX_AUTO, _stream(x.device)),
              name)
    return out


def custom_layernorm(x: torch.Tensor,
                     weight: Optional[torch.Tensor] = None,
                     bias: Optional[torch.Tensor] = None,
                     eps: float = 1e-5) -> torch.Tensor:
    """Layer normalization over the last dimension.

    `weight` and `bias` are the gamma and beta of the usual formulation and are
    independently optional. Named to match torch.nn.functional.layer_norm.
    """
    _validate(x, "x", min_dims=1)
    rows, cols = _rows_and_cols(x)

    for tensor, name in ((weight, "weight"), (bias, "bias")):
        if tensor is None:
            continue
        _validate(tensor, name, min_dims=1)
        if tensor.shape != (cols,):
            raise ValueError(
                f"{name} must have shape ({cols},) to match the normalized "
                f"dimension, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != x.dtype:
            raise TypeError(f"{name} is {tensor.dtype} but x is {x.dtype}")
        if tensor.device != x.device:
            raise ValueError(
                f"{name} is on {tensor.device} but x is on {x.device}"
            )

    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    out = torch.empty_like(x)
    fn, name = _entry("layernorm", x.dtype)
    with torch.cuda.device(x.device):
        check(fn(_ptr(out), _ptr(x), _ptr(weight), _ptr(bias), rows, cols,
                 ctypes.c_float(eps), _LAYERNORM_AUTO, _stream(x.device)),
              name)
    return out


def custom_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix product of two 2-D contiguous CUDA tensors of the same dtype.

    No batching and no broadcasting: [M, K] @ [K, N] -> [M, N].
    """
    _validate(a, "a", min_dims=2)
    _validate(b, "b", min_dims=2)

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"custom_matmul takes 2-D tensors, got {a.dim()}-D and {b.dim()}-D. "
            f"Batched products are not implemented."
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"shapes {tuple(a.shape)} and {tuple(b.shape)} are not multipliable"
        )
    if a.device != b.device:
        raise ValueError(f"a is on {a.device} but b is on {b.device}")
    if a.dtype != b.dtype:
        raise TypeError(f"a is {a.dtype} but b is {b.dtype}")

    m, k = a.shape
    n = b.shape[1]

    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    fn, name = _entry("matmul", a.dtype)
    with torch.cuda.device(a.device):
        check(fn(_ptr(out), _ptr(a), _ptr(b), m, n, k, _MATMUL_AUTO, _stream(a.device)),
              name)
    return out
