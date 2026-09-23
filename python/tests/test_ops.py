"""Second, independent correctness check against PyTorch CPU and numpy.

The C++ tests compare against hand-written double-precision oracles in
tests/support/reference.cpp. These compare against PyTorch instead. Two
independent references disagreeing is far more informative than one reference
agreeing with itself, which is why both exist.

PyTorch here is used on the CPU only -- no CUDA build of torch is required, and
none is assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ccml = pytest.importorskip("ccml")
from ccml import CcmlNotImplementedError  # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(scope="session", autouse=True)
def require_library():
    """Skip the whole module cleanly if the C++ library has not been built."""
    try:
        ccml.version()
    except FileNotFoundError as exc:
        pytest.skip(f"ccml shared library not built: {exc}")


def _skip_if_unimplemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except CcmlNotImplementedError as exc:
        pytest.skip(str(exc))


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- matmul ---------------------------------------------------------------

@pytest.mark.parametrize("m,n,k", [(128, 128, 128), (67, 53, 41), (1, 1, 1)])
def test_matmul_matches_torch(m, n, k):
    rng = _rng(1)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)

    got = _skip_if_unimplemented(ccml.matmul, a, b)
    expected = (torch.from_numpy(a).double() @ torch.from_numpy(b).double()).numpy()

    # Tolerance scales with sqrt(k); see docs/correctness.md.
    tol = 8.0 * np.finfo(np.float32).eps * np.sqrt(k)
    np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)


# --- layernorm ------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(64, 768), (33, 517), (1, 1024)])
def test_layernorm_matches_torch(rows, cols):
    rng = _rng(2)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    gamma = rng.uniform(0.5, 1.5, cols).astype(np.float32)
    beta = rng.uniform(-1.0, 1.0, cols).astype(np.float32)
    eps = 1e-5

    got = _skip_if_unimplemented(ccml.layernorm, x, gamma, beta, eps)
    expected = torch.nn.functional.layer_norm(
        torch.from_numpy(x).double(), (cols,),
        torch.from_numpy(gamma).double(), torch.from_numpy(beta).double(), eps
    ).numpy()

    tol = 8.0 * np.finfo(np.float32).eps * np.sqrt(cols)
    np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)


def test_layernorm_large_offset_small_variance():
    """The case a naive E[x^2] - E[x]^2 variance gets wrong."""
    rng = _rng(3)
    x = (10000.0 + 0.01 * rng.standard_normal((32, 512))).astype(np.float32)

    got = _skip_if_unimplemented(ccml.layernorm, x)
    expected = torch.nn.functional.layer_norm(
        torch.from_numpy(x).double(), (512,)
    ).numpy()

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


# --- softmax --------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(64, 1024), (37, 519), (128, 1)])
def test_softmax_matches_torch(rows, cols):
    rng = _rng(4)
    x = rng.standard_normal((rows, cols), dtype=np.float32)

    got = _skip_if_unimplemented(ccml.softmax, x)
    expected = torch.softmax(torch.from_numpy(x).double(), dim=-1).numpy()

    tol = 8.0 * np.finfo(np.float32).eps * np.sqrt(cols)
    np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)


def test_softmax_rows_sum_to_one():
    rng = _rng(5)
    x = rng.standard_normal((64, 1024), dtype=np.float32)

    got = _skip_if_unimplemented(ccml.softmax, x)

    np.testing.assert_allclose(got.sum(axis=-1), np.ones(64), rtol=0, atol=1e-4)
    assert (got >= 0.0).all() and (got <= 1.0).all()


def test_softmax_survives_large_inputs():
    """Without the max subtraction, exp() overflows and these rows become NaN."""
    x = (5000.0 + 10.0 * _rng(6).standard_normal((16, 256))).astype(np.float32)

    got = _skip_if_unimplemented(ccml.softmax, x)

    assert np.isfinite(got).all(), "softmax overflowed on large inputs"
    np.testing.assert_allclose(got.sum(axis=-1), np.ones(16), rtol=0, atol=1e-4)


# --- argument validation (passes even with no kernels built) --------------

def test_matmul_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        ccml.matmul(np.zeros((2, 3), np.float32), np.zeros((4, 5), np.float32))


def test_layernorm_rejects_wrong_gamma_shape():
    with pytest.raises(ValueError):
        ccml.layernorm(np.zeros((2, 8), np.float32), np.zeros(4, np.float32))
