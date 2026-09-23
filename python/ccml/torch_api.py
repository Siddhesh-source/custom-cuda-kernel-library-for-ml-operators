"""Minimal PyTorch interface over the ccml CUDA kernels.

Why ctypes over tensor.data_ptr() and not torch.utils.cpp_extension
-------------------------------------------------------------------
`cpp_extension` is the conventional route, but it needs a compiler and a
CUDA-enabled PyTorch whose CUDA version matches the local toolkit, *at import
time*. The development machine here has PyTorch 2.1.1+cpu and cannot compile
CUDA at all, so that route cannot even be attempted, and on any other machine it
adds a second build system beside the CMake one.

The library already exposes a flat C ABI taking device pointers. A PyTorch
tensor's `data_ptr()` is exactly such a pointer. So the binding is: validate the
tensor, hand its pointer to the C function, and let the kernel write into a
tensor PyTorch allocated. No compiler, no second build, and no copy.

What this costs: no autograd, and no torch dispatcher integration. Neither is in
scope at this milestone. If a backward pass is ever needed, a `cpp_extension`
wrapper can be added over the same C ABI without changing any CUDA code.

Nothing here is copied to the host. Tensors that are not already on the GPU, not
float32, or not contiguous are rejected with an explanation rather than silently
converted -- a hidden `.cpu()` or `.contiguous()` is exactly the sort of cost
that should be the caller's decision.
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

    if tensor.dtype is not torch.float32:
        raise TypeError(
            f"{name} must be float32, got {tensor.dtype}. ccml implements float32 "
            f"only; cast explicitly with {name}.float() if that is what you want."
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
    """Softmax over the last dimension of a contiguous float32 CUDA tensor."""
    _validate(x, "x", min_dims=1)
    rows, cols = _rows_and_cols(x)

    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        check(
            lib().ccml_softmax_f32(_float_ptr(out), _float_ptr(x), rows, cols,
                                   _SOFTMAX_AUTO, _stream(x.device)),
            "ccml_softmax_f32",
        )
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
        if tensor.device != x.device:
            raise ValueError(
                f"{name} is on {tensor.device} but x is on {x.device}"
            )

    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        check(
            lib().ccml_layernorm_f32(_float_ptr(out), _float_ptr(x),
                                     _float_ptr(weight), _float_ptr(bias),
                                     rows, cols, ctypes.c_float(eps),
                                     _LAYERNORM_AUTO, _stream(x.device)),
            "ccml_layernorm_f32",
        )
    return out


def custom_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix product of two 2-D contiguous float32 CUDA tensors.

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

    m, k = a.shape
    n = b.shape[1]

    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    with torch.cuda.device(a.device):
        check(
            lib().ccml_matmul_f32(_float_ptr(out), _float_ptr(a), _float_ptr(b),
                                  m, n, k, _MATMUL_AUTO, _stream(a.device)),
            "ccml_matmul_f32",
        )
    return out
