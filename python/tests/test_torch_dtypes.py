"""FP16 and BF16 against a PyTorch float64 reference (reference.md section 3:
"compare to PyTorch float64 reference with explicit tolerances per dtype").

Inputs are created directly in the 16-bit dtype, so .double() of them is exact
and the reference sees precisely the values the kernel saw. The tolerances are
the C++ policy (support/compare.h, docs/correctness.md):

    rel = 8 * eps_f32 * sqrt(terms) + 2 * unit_roundoff(dtype)
    abs = the f32 absolute part (matmul: scaled by rms of the result) + floor(dtype)
    softmax: relative only, absolute floor at the dtype's underflow

Needs a CUDA-enabled PyTorch; skips cleanly without one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ccml = pytest.importorskip("ccml")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the PyTorch interface requires a CUDA-enabled torch",
)

EPS32 = float(torch.finfo(torch.float32).eps)
FLT_MIN = float(torch.finfo(torch.float32).tiny)
UNIT = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}
FLOOR = {torch.float16: 2.0 ** -24, torch.bfloat16: FLT_MIN}
DTYPES = [torch.float16, torch.bfloat16]


@pytest.fixture(scope="module", autouse=True)
def require_library():
    try:
        ccml.version()
    except FileNotFoundError as exc:
        pytest.skip(f"ccml shared library not built: {exc}")


def _reduction(terms: int) -> float:
    return 8.0 * EPS32 * (max(terms, 1) ** 0.5)


def _assert_within(got, ref, atol, rtol, what):
    """Element-wise |got - ref| <= atol + rtol * |ref|, reporting the worst case."""
    diff = (got.double() - ref).abs()
    bound = atol + rtol * ref.abs()
    worst = (diff / bound).max().item()
    assert torch.isfinite(got).all(), f"{what}: non-finite output"
    assert worst <= 1.0, (f"{what}: worst normalized error {worst:.3f} "
                          f"(max abs {diff.max().item():.3e}, atol {atol:.3e}, rtol {rtol:.3e})")


@pytest.mark.parametrize("dtype", DTYPES, ids=["f16", "bf16"])
@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (17, 33, 65), (128, 128, 128),
                                   (64, 257, 1023), (3, 768, 768)])
def test_matmul_matches_torch_fp64(dtype, m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    got = ccml.custom_matmul(a, b)
    assert got.dtype is dtype
    ref = a.double() @ b.double()
    scale = ref.pow(2).mean().sqrt().item()
    _assert_within(got, ref, _reduction(k) * max(scale, 1e-30) + FLOOR[dtype],
                   _reduction(k) + 2 * UNIT[dtype], f"matmul {dtype} {m}x{n}x{k}")


@pytest.mark.parametrize("dtype", DTYPES, ids=["f16", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1), (7, 33), (4, 16, 768), (128, 1025), (3, 32000)])
def test_layernorm_matches_torch_fp64(dtype, shape):
    cols = shape[-1]
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = (torch.rand(cols, device="cuda") + 0.5).to(dtype)
    b = torch.randn(cols, device="cuda").to(dtype)
    got = ccml.custom_layernorm(x, w, b, eps=1e-5)
    ref = torch.nn.functional.layer_norm(x.double(), (cols,), w.double(), b.double(), 1e-5)
    _assert_within(got, ref, _reduction(cols) + FLOOR[dtype],
                   _reduction(cols) + 2 * UNIT[dtype], f"layernorm {dtype} {shape}")


@pytest.mark.parametrize("dtype", DTYPES, ids=["f16", "bf16"])
@pytest.mark.parametrize("rows,cols,scale", [(1, 1, 1.0), (64, 1024, 1.0), (33, 517, 1.0),
                                             (4, 32000, 1.0), (32, 512, 30.0)])
def test_softmax_matches_torch_fp64(dtype, rows, cols, scale):
    x = (torch.randn(rows, cols, device="cuda") * scale).to(dtype)
    got = ccml.custom_softmax(x)
    ref = torch.softmax(x.double(), dim=-1)
    _assert_within(got, ref, FLT_MIN + FLOOR[dtype], _reduction(cols) + 2 * UNIT[dtype],
                   f"softmax {dtype} {rows}x{cols} scale={scale}")


@pytest.mark.parametrize("dtype", DTYPES, ids=["f16", "bf16"])
def test_masking_and_nan_policy_matches_torch(dtype):
    inf = float("inf")
    x = torch.tensor([[1.0, -inf, 2.0, -inf], [-inf] * 4, [0.0, float("nan"), 1.0, 2.0]],
                     device="cuda").to(dtype)
    got = ccml.custom_softmax(x)
    ref = torch.softmax(x.double(), dim=-1)
    assert torch.equal(got[0].isnan(), ref[0].isnan())
    assert (got[0][[1, 3]] == 0).all(), "masked entries are exactly zero"
    assert got[1].isnan().all() and ref[1].isnan().all(), "fully masked row is NaN, as in torch"
    assert got[2].isnan().all() and ref[2].isnan().all(), "NaN poisons the row, as in torch"


def test_mixed_dtypes_are_rejected():
    a = torch.randn(4, 4, device="cuda", dtype=torch.float16)
    b = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="bfloat16"):
        ccml.custom_matmul(a, b)
    x = torch.randn(2, 4, device="cuda", dtype=torch.float16)
    with pytest.raises(TypeError, match="float32"):
        ccml.custom_layernorm(x, torch.ones(4, device="cuda"))
