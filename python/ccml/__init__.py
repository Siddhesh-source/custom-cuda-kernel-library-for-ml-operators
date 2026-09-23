"""Optional Python binding for the ccml CUDA kernel library.

Deleting this package does not affect the C++ build in any way -- it discovers
the built shared library at runtime and has no build-system edge to it.

Two layers, for two different jobs:

PyTorch (``custom_*``) -- operates directly on CUDA tensors with no copy, on
PyTorch's current stream. This is the interface to use.

    import torch, ccml
    y = ccml.custom_softmax(torch.randn(64, 1024, device="cuda"))

numpy (``softmax``, ``layernorm``, ``matmul``) -- takes host arrays, moves them
to the device and back. Convenient for a quick check, and used by the test
suite, but every call costs two transfers. Not for use in a training loop.

    import numpy as np, ccml
    y = ccml.softmax(np.random.randn(64, 1024).astype(np.float32))
"""

from ._lib import CcmlError, CcmlNotImplementedError
from .ops import (
    LAYERNORM_AUTO,
    LAYERNORM_TWO_PASS,
    LAYERNORM_WELFORD,
    MATMUL_AUTO,
    MATMUL_NAIVE,
    MATMUL_TILED,
    SOFTMAX_AUTO,
    SOFTMAX_BLOCK_PER_ROW,
    SOFTMAX_WARP_PER_ROW,
    layernorm,
    matmul,
    softmax,
    version,
)

# The PyTorch layer is optional: the numpy layer must keep working without it.
try:
    from .torch_api import custom_layernorm, custom_matmul, custom_softmax

    HAVE_TORCH = True
except ImportError as _torch_import_error:  # pragma: no cover - environment dependent
    HAVE_TORCH = False

    def _needs_torch(*_args, **_kwargs):
        raise ImportError(
            "ccml.custom_* requires PyTorch, which failed to import: "
            f"{_torch_import_error}\n"
            "Install it, or use the numpy entry points (ccml.softmax, "
            "ccml.layernorm, ccml.matmul) instead."
        )

    custom_layernorm = custom_matmul = custom_softmax = _needs_torch

__all__ = [
    "CcmlError",
    "CcmlNotImplementedError",
    "HAVE_TORCH",
    # PyTorch interface
    "custom_matmul",
    "custom_layernorm",
    "custom_softmax",
    # numpy interface
    "layernorm",
    "matmul",
    "softmax",
    "version",
    # algorithm selectors
    "MATMUL_AUTO",
    "MATMUL_NAIVE",
    "MATMUL_TILED",
    "LAYERNORM_AUTO",
    "LAYERNORM_TWO_PASS",
    "LAYERNORM_WELFORD",
    "SOFTMAX_AUTO",
    "SOFTMAX_BLOCK_PER_ROW",
    "SOFTMAX_WARP_PER_ROW",
]
