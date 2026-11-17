# ccml_ext: the PyTorch extension

`torch.utils.cpp_extension` bindings for the ccml kernels (reference.md
section 3), registered as `torch.library` custom ops so `torch.compile` traces
through them (enhancement P1), with `nn.Module` drop-ins and a GPT-2
`patch_model()` (P2). It replaces the interim ctypes binding in `python/ccml/`
as the deliverable; the ctypes package stays as the C-ABI test path.

## Build

The extension links the CMake-built library, so the C++ suites and Python run
the same binary:

    cmake --preset default && cmake --build build/default
    CCML_BUILD_DIR=build/default pip install --no-build-isolation ./python/ccml_ext

or, with no install step, a JIT build on first import (`torch.utils.cpp_extension.load`):

    CCML_BUILD_DIR=build/default PYTHONPATH=python/ccml_ext python -c "import ccml_ext"

Needs a CUDA build of PyTorch. The local Windows machine has none
(docs/local.md), so it is built and tested on Kaggle: `python
cloud/kaggle/kaggle_run.py --jobs model --any-gpu`.

## Use

```python
import torch, ccml_ext
from ccml_ext import ops

y = ops.layernorm(x, weight, bias, 1e-5)               # also torch.ops.ccml.layernorm
h = ops.linear(x16, w_kn, bias, None, ops.ACT_GELU_TANH)  # gelu(x @ W + b), fused
p = ops.softmax_masked(scores, None, 0.125, causal_q=q)   # causal attention softmax

model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").half().cuda()
report = ccml_ext.patch_model(model)   # norms, linears (+GELU), attention softmax
```

| Op | Contract |
|---|---|
| `matmul(a, b)` | `a [M, K] @ b [K, N]`, any of f32/f16/bf16 |
| `linear(x, w, bias, residual, act)` | `act(x @ w + bias) + residual`, f16/bf16, fused epilogue (split-K for <= 16 rows) |
| `layernorm(x, w, b, eps)` | over the last dimension |
| `layernorm_residual(x, r, bias, w, b, eps)` | `(LN(x + r + bias), x + r + bias)` |
| `softmax(x)` | over the last dimension |
| `softmax_masked(x, mask, scale, causal_q)` | `softmax(x * scale + mask)`, bottom-right causal; fully masked rows are zeros |

Every op runs on PyTorch's current stream and rejects, rather than silently
copies, tensors that are on the CPU, non-contiguous, mixed-dtype or of an
unsupported dtype. The modules call `.contiguous()` explicitly.

## Tests

`python/ccml_ext/tests/test_ext.py` (pytest, CUDA): every op against a float64
reference at the per-dtype tolerances of docs/correctness.md, input validation,
the current stream, `torch.library.opcheck` (schema and fake tensors),
`torch.compile(fullgraph=True)` through the ops, and `patch_model` on a small
config-built GPT-2. GPT-2 small parity and latency: `scripts/model_bench.py`.
