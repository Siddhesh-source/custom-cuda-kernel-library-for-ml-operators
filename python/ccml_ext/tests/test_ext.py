"""Tests for the ccml PyTorch extension (ROADMAP Phase 14): every op against a
float64 PyTorch reference with the project's per-dtype tolerances
(docs/correctness.md, scripts/benchlib.py), input validation, the current
stream, torch.library schema / fake checks (opcheck), torch.compile tracing
through the custom ops, and the nn.Module drop-ins on a small GPT-2 built from
a config (no download).

Needs a CUDA GPU and the extension (installed, or JIT-built from
CCML_BUILD_DIR). BF16 Tensor Core ops SKIP below sm_80, never PASS.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import benchlib as bl  # noqa: E402

try:  # the installed extension (setup.py); else the source tree (JIT build)
    import ccml_ext  # noqa: E402
except ImportError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import ccml_ext  # noqa: E402
from ccml_ext import ops  # noqa: E402

DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
SM80 = torch.cuda.get_device_capability() >= (8, 0)


def _check(got: torch.Tensor, ref: torch.Tensor, tol: tuple[float, float], what: str):
    ok, mabs, mrel = bl.compare(got, ref, *tol)
    assert ok, f"{what}: max_abs={mabs:.3e} max_rel={mrel:.3e} tol={tol}"


def _randn(*shape, dtype=torch.float32, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale).to(dtype)


# --------------------------------------------------------------- ops -----

@pytest.mark.parametrize("dt", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 33, 65), (128, 768, 768), (1, 2304, 768)])
def test_matmul(dt, shape):
    M, N, K = shape
    a, b = _randn(M, K, dtype=DTYPES[dt], seed=1), _randn(K, N, dtype=DTYPES[dt], seed=2)
    ref = a.double() @ b.double()
    tol = bl.widen_for_storage(bl.tolerance_dot(K, ref.pow(2).mean().sqrt().item()), dt)
    _check(ops.matmul(a, b), ref, tol, f"matmul {dt} {shape}")


@pytest.mark.parametrize("dt", ["f16", "bf16"])
@pytest.mark.parametrize("rows", [1, 16, 130])
@pytest.mark.parametrize("act", [ops.ACT_NONE, ops.ACT_GELU_TANH])
@pytest.mark.parametrize("residual", [False, True])
def test_linear_fused_epilogue(dt, rows, act, residual):
    if dt == "bf16" and not SM80:
        pytest.skip("BF16 Tensor Cores need sm_80+ (docs/local.md L1)")
    K, N = 256, 384
    x = _randn(rows, K, dtype=DTYPES[dt], seed=3)
    w = _randn(K, N, dtype=DTYPES[dt], seed=4)
    b = _randn(N, dtype=DTYPES[dt], seed=5, scale=math.sqrt(K))
    r = _randn(rows, N, dtype=DTYPES[dt], seed=6, scale=math.sqrt(K)) if residual else None
    prod = x.double() @ w.double()
    ref = prod + b.double()
    if act == ops.ACT_GELU_TANH:
        ref = F.gelu(ref, approximate="tanh")
    if r is not None:
        ref = ref + r.double()
    abs_t, rel_t = bl.tolerance_dot(K, prod.pow(2).mean().sqrt().item())
    tol = bl.widen_for_storage((2 * abs_t, rel_t), dt)
    _check(ops.linear(x, w, b, r, act), ref, tol, f"linear {dt} rows={rows} act={act}")


@pytest.mark.parametrize("dt", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1), (7, 33), (64, 768), (3, 2, 1025), (16, 4096)])
@pytest.mark.parametrize("offset", [0.0, 100.0])
def test_layernorm(dt, shape, offset):
    cols = shape[-1]
    x = _randn(*shape, dtype=DTYPES[dt], seed=7) + offset
    x = x.to(DTYPES[dt])
    w = (_randn(cols, seed=8) * 0.5 + 1.0).to(DTYPES[dt])
    b = _randn(cols, dtype=DTYPES[dt], seed=9)
    ref = F.layer_norm(x.double(), (cols,), w.double(), b.double(), 1e-5)
    tol = bl.widen_for_storage(bl.tolerance_reduction(cols), dt)
    _check(ops.layernorm(x, w, b, 1e-5), ref, tol, f"layernorm {dt} {shape} offset={offset}")


@pytest.mark.parametrize("dt", ["f32", "f16", "bf16"])
def test_layernorm_residual(dt):
    x, r = _randn(37, 768, dtype=DTYPES[dt], seed=10), _randn(37, 768, dtype=DTYPES[dt], seed=11)
    bias = _randn(768, dtype=DTYPES[dt], seed=12)
    w, lb = _randn(768, dtype=DTYPES[dt], seed=13), _randn(768, dtype=DTYPES[dt], seed=14)
    y, s = ops.layernorm_residual(x, r, bias, w, lb, 1e-5)
    s_expected = (x.float() + r.float() + bias.float()).to(DTYPES[dt])
    assert torch.equal(s, s_expected), "sum must be the FP32 (x + r) + b rounded once"
    ref = F.layer_norm(s_expected.double(), (768,), w.double(), lb.double(), 1e-5)
    _check(y, ref, bl.widen_for_storage(bl.tolerance_reduction(768), dt), f"ln_residual {dt}")


@pytest.mark.parametrize("dt", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("shape", [(8, 1), (33, 517), (4, 128, 256), (3, 32000)])
def test_softmax(dt, shape):
    x = _randn(*shape, dtype=DTYPES[dt], seed=15, scale=3.0)
    ref = torch.softmax(x.double(), dim=-1)
    tol = bl.widen_for_storage(bl.tolerance_softmax(shape[-1]), dt)
    _check(ops.softmax(x), ref, tol, f"softmax {dt} {shape}")


@pytest.mark.parametrize("dt", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("q,k", [(128, 128), (1, 77), (16, 300)])
def test_softmax_masked_causal(dt, q, k):
    x = _randn(2, 4, q, k, dtype=DTYPES[dt], seed=16, scale=3.0)
    keep = torch.arange(k, device="cuda")[None, :] <= (torch.arange(q, device="cuda")[:, None]
                                                       + (k - q))
    logits = x.double() * 0.125
    ref = torch.softmax(logits.masked_fill(~keep, float("-inf")), dim=-1)
    tol = bl.widen_for_storage(bl.tolerance_softmax(k), dt)
    _check(ops.softmax_masked(x, None, 0.125, q), ref, tol, f"softmax_masked {dt} q={q} k={k}")


def test_softmax_masked_fully_masked_rows_are_zero():
    x = _randn(3, 64, seed=17)
    mask = torch.full((3, 64), float("-inf"), device="cuda")
    mask[1] = 0.0
    y = ops.softmax_masked(x, mask, 1.0, 0)
    assert torch.equal(y[0], torch.zeros_like(y[0])) and torch.equal(y[2], torch.zeros_like(y[2]))
    assert torch.allclose(y[1].sum(), torch.tensor(1.0, device="cuda"))


# -------------------------------------------------------- validation -----

def test_rejects_what_it_cannot_take_without_a_copy():
    x = _randn(8, 16)
    with pytest.raises(RuntimeError, match="contiguous"):
        ops.layernorm(x.t())
    with pytest.raises(RuntimeError, match="CUDA"):
        ops.softmax(x.cpu())
    with pytest.raises(RuntimeError, match="dtype"):
        ops.matmul(x, _randn(16, 4, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="float16 or bfloat16"):
        ops.linear(x, _randn(16, 4))
    with pytest.raises(RuntimeError):
        ops.softmax(x.to(torch.float64))


def test_runs_on_the_current_stream():
    x = _randn(256, 1024, seed=18)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y = ops.softmax(x)
        z = y * 2.0  # same stream: ordered after the kernel
    s.synchronize()
    assert torch.allclose(z, 2.0 * torch.softmax(x, dim=-1), atol=1e-6)


# ------------------------------------------------- custom ops / compile -----

def test_opcheck_schemas_and_fakes():
    x16 = _randn(16, 64, dtype=torch.float16, seed=19)
    w16 = _randn(64, 32, dtype=torch.float16, seed=20)
    x = _randn(16, 64, seed=21)
    cases = [
        (torch.ops.ccml.matmul.default, (x, _randn(64, 8), 0)),
        (torch.ops.ccml.linear.default, (x16, w16, None, None, ops.ACT_GELU_TANH)),
        (torch.ops.ccml.layernorm.default, (x, None, None, 1e-5, 0)),
        (torch.ops.ccml.layernorm_residual.default, (x, x.clone(), None, None, None, 1e-5, 0)),
        (torch.ops.ccml.softmax.default, (x, 0)),
        (torch.ops.ccml.softmax_masked.default, (x, None, 0.5, 16, 0)),
    ]
    for op, args in cases:
        # Not differentiable (inference kernels): skip the autograd checks.
        torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))


def test_torch_compile_traces_through_the_ops():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = ccml_ext.CcmlLayerNorm(torch.ones(256, device="cuda", dtype=torch.float16),
                                             torch.zeros(256, device="cuda",
                                                         dtype=torch.float16))
            self.fc = ccml_ext.CcmlLinear(_randn(256, 512, dtype=torch.float16, seed=22),
                                          _randn(512, dtype=torch.float16, seed=23),
                                          ops.ACT_GELU_TANH)

        def forward(self, x):
            return ops.softmax(self.fc(self.ln(x)))

    m = Block()
    x = _randn(64, 256, dtype=torch.float16, seed=24)
    eager = m(x)
    torch._dynamo.reset()
    compiled = torch.compile(m, fullgraph=True)
    assert torch.equal(compiled(x), eager), "compiled graph must call the same kernels"


# ------------------------------------------------------------ modules -----

def test_patch_model_on_a_small_gpt2():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(n_layer=2, n_embd=128, n_head=4, n_positions=256,
                                  vocab_size=1000)
    torch.manual_seed(0)
    stock = transformers.GPT2LMHeadModel(cfg).to("cuda", torch.float16).eval()
    patched = transformers.GPT2LMHeadModel(cfg).to("cuda", torch.float16).eval()
    patched.load_state_dict(stock.state_dict())
    report = ccml_ext.patch_model(patched)
    assert report["layernorm"] == 5 and report["linear"] + report["linear_gelu"] == 8
    ids = torch.randint(0, 1000, (2, 96), device="cuda")
    ccml_ext.modules.ATTENTION_CALLS = 0
    with torch.no_grad():
        ref = stock(ids).logits.float()
        got = patched(ids).logits.float()
    if report["attention"]:
        assert ccml_ext.modules.ATTENTION_CALLS > 0, "attention was registered but never ran"
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    # FP16 end to end, different summation orders: bounded by a few FP16 ulps
    # of the logit scale; the full-size parity with real weights is
    # scripts/model_bench.py.
    assert err <= 2e-2 * max(scale, 1.0), f"logits differ by {err} (scale {scale})"
