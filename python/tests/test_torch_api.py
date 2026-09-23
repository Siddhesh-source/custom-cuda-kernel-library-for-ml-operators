"""Tests for the PyTorch interface.

These need a CUDA-enabled PyTorch and skip cleanly without one, which is the
normal case on the development machine (torch 2.1.1+cpu).

The separate suite in test_ops.py covers the numpy entry points, which move data
through the host. This one covers the zero-copy path: PyTorch allocates, ccml
writes into that allocation, nothing is transferred.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ccml = pytest.importorskip("ccml")
torch = pytest.importorskip("torch")

from ccml import CcmlNotImplementedError  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the PyTorch interface requires a CUDA-enabled torch",
)


@pytest.fixture(scope="module", autouse=True)
def require_library():
    try:
        ccml.version()
    except FileNotFoundError as exc:
        pytest.skip(f"ccml shared library not built: {exc}")


def _tolerance(reduction_length: int) -> float:
    """Matches the C++ policy: 8 * eps_f32 * sqrt(terms). See docs/correctness.md."""
    return 8.0 * float(torch.finfo(torch.float32).eps) * (reduction_length ** 0.5)


# --- softmax --------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(64, 1024), (33, 517), (128, 1), (1, 4096)])
def test_softmax_matches_torch(rows, cols):
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

    got = ccml.custom_softmax(x)
    expected = torch.softmax(x.double(), dim=-1)

    tol = _tolerance(cols)
    torch.testing.assert_close(got.double(), expected, rtol=tol, atol=tol)


def test_softmax_handles_extreme_inputs():
    x = torch.randn(16, 256, device="cuda", dtype=torch.float32) * 10.0 + 5000.0

    got = ccml.custom_softmax(x)

    assert torch.isfinite(got).all(), "softmax overflowed"
    torch.testing.assert_close(got.sum(dim=-1),
                               torch.ones(16, device="cuda"), rtol=0, atol=1e-5)


def test_softmax_accepts_higher_rank_tensors():
    """Leading dimensions collapse into rows; the view costs nothing."""
    x = torch.randn(4, 16, 768, device="cuda", dtype=torch.float32)

    got = ccml.custom_softmax(x)
    expected = torch.softmax(x.double(), dim=-1)

    assert got.shape == x.shape
    tol = _tolerance(768)
    torch.testing.assert_close(got.double(), expected, rtol=tol, atol=tol)


# --- layernorm ------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(64, 768), (33, 517), (1, 1024)])
def test_layernorm_matches_torch(rows, cols):
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    weight = torch.rand(cols, device="cuda", dtype=torch.float32) + 0.5
    bias = torch.randn(cols, device="cuda", dtype=torch.float32)

    got = ccml.custom_layernorm(x, weight, bias)
    expected = torch.nn.functional.layer_norm(
        x.double(), (cols,), weight.double(), bias.double(), 1e-5)

    tol = _tolerance(cols)
    torch.testing.assert_close(got.double(), expected, rtol=tol, atol=tol)


def test_layernorm_affine_is_optional():
    x = torch.randn(64, 768, device="cuda", dtype=torch.float32)

    got = ccml.custom_layernorm(x)
    expected = torch.nn.functional.layer_norm(x.double(), (768,))

    tol = _tolerance(768)
    torch.testing.assert_close(got.double(), expected, rtol=tol, atol=tol)


def test_layernorm_survives_large_offset():
    """The case a naive E[x^2] - E[x]^2 variance gets badly wrong."""
    x = torch.randn(32, 512, device="cuda", dtype=torch.float32) * 0.01 + 10000.0

    got = ccml.custom_layernorm(x)
    expected = torch.nn.functional.layer_norm(x.double(), (512,))

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.double(), expected, rtol=1e-4, atol=1e-4)


# --- matmul ---------------------------------------------------------------

def test_matmul_matches_torch_or_reports_unimplemented():
    a = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    b = torch.randn(64, 32, device="cuda", dtype=torch.float32)

    try:
        got = ccml.custom_matmul(a, b)
    except CcmlNotImplementedError:
        pytest.skip("matmul has no kernel behind it yet")

    expected = a.double() @ b.double()
    tol = _tolerance(64)
    torch.testing.assert_close(got.double(), expected, rtol=tol, atol=tol)


# --- validation -----------------------------------------------------------

def test_cpu_tensor_is_rejected_not_copied():
    with pytest.raises(ValueError, match="CUDA tensor"):
        ccml.custom_softmax(torch.randn(4, 8))


def test_wrong_dtype_is_rejected_not_cast():
    x = torch.randn(4, 8, device="cuda", dtype=torch.float64)
    with pytest.raises(TypeError, match="float32"):
        ccml.custom_softmax(x)


def test_non_contiguous_is_rejected_not_copied():
    x = torch.randn(8, 8, device="cuda", dtype=torch.float32).t()
    assert not x.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        ccml.custom_softmax(x)


def test_layernorm_rejects_wrong_weight_shape():
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(4, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        ccml.custom_layernorm(x, weight)


def test_layernorm_rejects_nonpositive_eps():
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="eps"):
        ccml.custom_layernorm(x, eps=0.0)


def test_matmul_rejects_mismatched_shapes():
    a = torch.randn(2, 3, device="cuda", dtype=torch.float32)
    b = torch.randn(4, 5, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="multipliable"):
        ccml.custom_matmul(a, b)


def test_matmul_rejects_batched_input():
    a = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)
    b = torch.randn(2, 4, 5, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="2-D"):
        ccml.custom_matmul(a, b)


# --- no hidden transfers --------------------------------------------------

def test_output_stays_on_the_input_device():
    x = torch.randn(16, 256, device="cuda", dtype=torch.float32)

    out = ccml.custom_softmax(x)

    assert out.is_cuda, "the result must stay on the GPU"
    assert out.device == x.device
    assert out.dtype == x.dtype


def test_result_is_ordered_against_torch_work_on_the_same_stream():
    """No explicit synchronisation: the kernel runs on PyTorch's current stream,
    so work queued before it is visible to it and work queued after sees its
    output."""
    x = torch.randn(64, 512, device="cuda", dtype=torch.float32)

    scaled = x * 2.0              # queued on the current stream
    got = ccml.custom_softmax(scaled)   # must see the scaling
    total = got.sum(dim=-1)       # must see the softmax

    torch.testing.assert_close(total, torch.ones(64, device="cuda"),
                               rtol=0, atol=1e-5)
