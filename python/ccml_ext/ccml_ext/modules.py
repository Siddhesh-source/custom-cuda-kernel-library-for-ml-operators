"""nn.Module drop-ins over the ccml custom ops, and patch_model() for Hugging
Face GPT-2 (enhancement P2; ROADMAP Phase 14).

  CcmlLinear     act(x @ W + b): FP16/BF16 through the fused matmul epilogue
                 (MMA_SYNC, or split-K for <= 16 rows); FP32 through ccml matmul
                 with the bias and activation applied by PyTorch.
  CcmlLayerNorm  ccml layer norm (AUTO).
  CcmlSoftmax    ccml softmax over the last dimension (AUTO).

patch_model(model) swaps, in a transformers GPT2LMHeadModel / GPT2Model:
  ln_1, ln_2, ln_f        -> CcmlLayerNorm
  attn.c_attn, attn.c_proj, mlp.c_proj -> CcmlLinear
  mlp.c_fc + mlp.act      -> CcmlLinear with the GELU fused, act -> Identity
  attention softmax       -> a registered "ccml" attention function: scores by
                             torch.matmul, then ccml softmax_masked with the
                             scale and the causal mask fused.
Residual additions stay in Hugging Face's block code (the fused residual
kernels are benchmarked separately, log entries 8 and 12).

Inputs to the modules are made contiguous explicitly (a documented copy when
it happens); the ops themselves reject non-contiguous tensors.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from . import ops

#: How many times the ccml attention function ran (patch_model reports and the
#: model benchmark checks it, so a patch that silently did not apply is caught).
ATTENTION_CALLS = 0


class CcmlLinear(nn.Module):
    """y = act(x @ weight + bias); weight stored [in, out] (K x N) contiguous."""

    def __init__(self, weight_kn: torch.Tensor, bias: Optional[torch.Tensor] = None,
                 act: int = ops.ACT_NONE):
        super().__init__()
        self.weight = nn.Parameter(weight_kn.detach().contiguous(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach().contiguous(),
                                                           requires_grad=False)
        self.act = act

    @classmethod
    def from_linear(cls, linear: nn.Linear, act: int = ops.ACT_NONE) -> "CcmlLinear":
        """nn.Linear stores [out, in]; transposed once here."""
        return cls(linear.weight.t(), linear.bias, act)

    @classmethod
    def from_conv1d(cls, conv, act: int = ops.ACT_NONE) -> "CcmlLinear":
        """transformers' Conv1D already stores [in, out]."""
        return cls(conv.weight, conv.bias, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if x.dtype == torch.float32:
            y = ops.matmul(x.reshape(-1, x.shape[-1]), self.weight).reshape(
                *x.shape[:-1], self.weight.shape[1])
            if self.bias is not None:
                y = y + self.bias
            if self.act == ops.ACT_RELU:
                y = torch.relu(y)
            elif self.act == ops.ACT_GELU_TANH:
                y = nn.functional.gelu(y, approximate="tanh")
            return y
        return ops.linear(x, self.weight, self.bias, None, self.act)


class CcmlLayerNorm(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                 eps: float = 1e-5):
        super().__init__()
        self.weight = None if weight is None else nn.Parameter(weight.detach().contiguous(),
                                                               requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach().contiguous(),
                                                           requires_grad=False)
        self.eps = eps

    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm) -> "CcmlLayerNorm":
        return cls(ln.weight, ln.bias, ln.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.layernorm(x.contiguous(), self.weight, self.bias, self.eps)


class CcmlSoftmax(nn.Module):
    """softmax over the last dimension (dim=-1 only)."""

    def __init__(self, dim: int = -1):
        super().__init__()
        if dim != -1:
            raise ValueError("CcmlSoftmax normalizes the last dimension only (dim=-1)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.softmax(x.contiguous())


def ccml_attention(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    """GPT-2 attention with the ccml softmax: same contract as transformers'
    gpt2.eager_attention_forward (query/key/value [b, h, q|k, d]; returns
    (output [b, q, h, d], None))."""
    global ATTENTION_CALLS
    ATTENTION_CALLS += 1
    q_len, k_len, d = query.shape[-2], key.shape[-2], value.shape[-1]
    scale = 1.0
    if getattr(module, "scale_attn_weights", True):
        scale = 1.0 / math.sqrt(d)
    if getattr(module, "scale_attn_by_inverse_layer_idx", False):
        scale /= float(getattr(module, "layer_idx", 0) + 1)
    scores = torch.matmul(query, key.transpose(-1, -2)).contiguous()
    if attention_mask is not None:
        # An explicit (additive) mask from transformers: applied by PyTorch, the
        # causal mask (always valid for GPT-2 self-attention) fused in ccml.
        mask = attention_mask[:, :, :, :k_len]
        if mask.dtype == torch.bool:  # True = attend
            mask = torch.zeros_like(mask, dtype=scores.dtype).masked_fill(~mask, float("-inf"))
        scores = (scores * scale + mask).to(value.dtype).contiguous()
        probs = ops.softmax_masked(scores, None, 1.0, q_len)
    else:
        probs = ops.softmax_masked(scores, None, scale, q_len)
    if module.training:
        probs = module.attn_dropout(probs)
    if head_mask is not None:
        probs = probs * head_mask
    out = torch.matmul(probs, value).transpose(1, 2)
    return out, None


def _register_attention() -> bool:
    """Register "ccml" with transformers' attention (and mask) interfaces.
    Returns False if this transformers version has no such registry."""
    try:
        from transformers import AttentionInterface
    except ImportError:
        return False
    AttentionInterface.register("ccml", ccml_attention)
    try:  # the mask builder for custom implementations (newer transformers)
        from transformers.masking_utils import AttentionMaskInterface, eager_mask

        AttentionMaskInterface.register("ccml", eager_mask)
    except ImportError:
        pass
    return True


def patch_model(model: nn.Module, attention: bool = True) -> dict:
    """Swap GPT-2's layers for ccml ones in place. Returns a report of what was
    replaced (counts, and whether attention was registered)."""
    transformer = getattr(model, "transformer", model)
    report = {"layernorm": 0, "linear": 0, "linear_gelu": 0, "attention": False}
    for block in transformer.h:
        block.ln_1 = CcmlLayerNorm.from_layernorm(block.ln_1)
        block.ln_2 = CcmlLayerNorm.from_layernorm(block.ln_2)
        report["layernorm"] += 2
        block.attn.c_attn = CcmlLinear.from_conv1d(block.attn.c_attn)
        block.attn.c_proj = CcmlLinear.from_conv1d(block.attn.c_proj)
        block.mlp.c_proj = CcmlLinear.from_conv1d(block.mlp.c_proj)
        report["linear"] += 3
        act = getattr(model.config, "activation_function", "gelu_new")
        if act in ("gelu_new", "gelu_pytorch_tanh"):
            block.mlp.c_fc = CcmlLinear.from_conv1d(block.mlp.c_fc, ops.ACT_GELU_TANH)
            block.mlp.act = nn.Identity()
            report["linear_gelu"] += 1
        else:
            block.mlp.c_fc = CcmlLinear.from_conv1d(block.mlp.c_fc)
            report["linear"] += 1
    transformer.ln_f = CcmlLayerNorm.from_layernorm(transformer.ln_f)
    report["layernorm"] += 1
    if attention and _register_attention():
        if hasattr(model, "set_attn_implementation"):  # newer transformers
            model.set_attn_implementation("ccml")
        else:
            model.config._attn_implementation = "ccml"
        report["attention"] = True
    return report
