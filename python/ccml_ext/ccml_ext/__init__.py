"""ccml CUDA kernels for PyTorch: torch.utils.cpp_extension bindings,
torch.library custom ops (traceable by torch.compile), and nn.Module drop-ins
with a patch_model() helper for Hugging Face GPT-2.

    import ccml_ext
    y = ccml_ext.ops.layernorm(x, w, b)            # or torch.ops.ccml.layernorm
    model = ccml_ext.patch_model(model)            # GPT-2: linears, norms, attention
"""

from . import ops
from .modules import CcmlLayerNorm, CcmlLinear, CcmlSoftmax, patch_model

__all__ = ["ops", "CcmlLinear", "CcmlLayerNorm", "CcmlSoftmax", "patch_model"]
