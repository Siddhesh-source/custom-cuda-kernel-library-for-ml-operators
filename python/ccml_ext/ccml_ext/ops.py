"""ccml kernels as torch.library custom ops (enhancement P1).

Each op calls the compiled extension (ccml_ext._C, or a JIT build) and has a
fake (meta) implementation that only describes the output, so torch.compile
and torch.export can trace a model that uses them without running the kernel.
Use them directly (torch.ops.ccml.<name> or the functions below) or through
the nn.Module drop-ins in ccml_ext.modules.

Algorithm numbers are the C enums (include/ccml/*.h); 0 is AUTO everywhere.
"""

from __future__ import annotations

from typing import Optional

import torch

from ._loader import load_backend

_C = load_backend()

# include/ccml/matmul.h
MATMUL_AUTO = 0
MATMUL_MMA_SYNC = 7
MATMUL_SPLIT_K = 8
# include/ccml/matmul.h ccml_activation
ACT_NONE, ACT_RELU, ACT_GELU_TANH = 0, 1, 2
# include/ccml/layernorm.h, include/ccml/softmax.h
AUTO = 0

# Decode-sized products (M <= 16) go to split-K, which fills the GPU there
# (optimization log entry 6); larger ones to MMA_SYNC's fused epilogue.
SPLIT_K_MAX_ROWS = 16


@torch.library.custom_op("ccml::matmul", mutates_args=())
def matmul(a: torch.Tensor, b: torch.Tensor, algo: int = MATMUL_AUTO) -> torch.Tensor:
    return _C.matmul(a, b, algo)


@matmul.register_fake
def _(a, b, algo=MATMUL_AUTO):
    return a.new_empty((a.shape[0], b.shape[1]))


@torch.library.custom_op("ccml::linear", mutates_args=())
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           residual: Optional[torch.Tensor] = None, act: int = ACT_NONE) -> torch.Tensor:
    """act(x @ weight + bias) + residual; weight is [in, out] (K x N)."""
    rows = x.numel() // x.shape[-1]
    algo = MATMUL_SPLIT_K if rows <= SPLIT_K_MAX_ROWS else MATMUL_MMA_SYNC
    return _C.linear(x, weight, bias, residual, act, algo)


@linear.register_fake
def _(x, weight, bias=None, residual=None, act=ACT_NONE):
    return x.new_empty((*x.shape[:-1], weight.shape[1]))


@torch.library.custom_op("ccml::layernorm", mutates_args=())
def layernorm(x: torch.Tensor, weight: Optional[torch.Tensor] = None,
              bias: Optional[torch.Tensor] = None, eps: float = 1e-5,
              algo: int = AUTO) -> torch.Tensor:
    return _C.layernorm(x, weight, bias, eps, algo)


@layernorm.register_fake
def _(x, weight=None, bias=None, eps=1e-5, algo=AUTO):
    return torch.empty_like(x)


@torch.library.custom_op("ccml::layernorm_residual", mutates_args=())
def layernorm_residual(x: torch.Tensor, residual: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       weight: Optional[torch.Tensor] = None,
                       ln_bias: Optional[torch.Tensor] = None, eps: float = 1e-5,
                       algo: int = AUTO) -> tuple[torch.Tensor, torch.Tensor]:
    """(LN(x + residual + bias), x + residual + bias)."""
    return tuple(_C.layernorm_residual(x, residual, bias, weight, ln_bias, eps, algo))


@layernorm_residual.register_fake
def _(x, residual, bias=None, weight=None, ln_bias=None, eps=1e-5, algo=AUTO):
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op("ccml::softmax", mutates_args=())
def softmax(x: torch.Tensor, algo: int = AUTO) -> torch.Tensor:
    """softmax over the last dimension."""
    return _C.softmax(x, algo)


@softmax.register_fake
def _(x, algo=AUTO):
    return torch.empty_like(x)


@torch.library.custom_op("ccml::softmax_masked", mutates_args=())
def softmax_masked(x: torch.Tensor, mask: Optional[torch.Tensor] = None, scale: float = 1.0,
                   causal_q: int = 0, algo: int = AUTO) -> torch.Tensor:
    """softmax(x * scale + mask) over the last dimension; causal_q > 0 adds a
    bottom-right causal mask over blocks of causal_q rows. Fully masked rows
    are zeros (include/ccml/softmax.h)."""
    return _C.softmax_masked(x, mask, scale, causal_q, algo)


@softmax_masked.register_fake
def _(x, mask=None, scale=1.0, causal_q=0, algo=AUTO):
    return torch.empty_like(x)
