"""Tests for scripts/benchlib.py: the Python measurement helpers must agree with
the C++ ones (tests/test_bench_stats.cpp checks the same values)."""

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import benchlib as bl  # noqa: E402


def test_percentiles_match_numpy_linear():
    v = [1.0, 2.0, 3.0, 4.0]
    expected = {0.0: 1.0, 0.10: 1.3, 0.25: 1.75, 0.5: 2.5, 0.9: 3.7, 1.0: 4.0}
    for q, want in expected.items():
        assert bl.percentile_sorted(v, q) == pytest.approx(want, abs=1e-12)
    np = pytest.importorskip("numpy")
    data = [math.sin(i) for i in range(57)]
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert bl.percentile_sorted(sorted(data), q) == pytest.approx(
            float(np.percentile(data, 100 * q)), abs=1e-12)


def test_summary_matches_cpp_test_values():
    s = bl.summarize([float(i) for i in range(1, 102)], resamples=500)
    assert s.valid and s.n == 101
    assert (s.median, s.min, s.max, s.mean) == (51.0, 1.0, 101.0, 51.0)
    assert (s.p10, s.p90, s.iqr) == (11.0, 91.0, 50.0)
    assert s.ci_lo <= s.median <= s.ci_hi


def test_bootstrap_deterministic_and_flat():
    data = [math.sin(i) * 10 + 100 for i in range(200)]
    assert bl.bootstrap_median_ci(data, 300, 42) == bl.bootstrap_median_ci(data, 300, 42)
    assert bl.bootstrap_median_ci([3.0] * 150, 100, 1) == (3.0, 3.0)
    assert not bl.summarize([]).valid


def test_iteration_floor():
    assert bl.choose_iterations(0.001) == 200
    assert bl.choose_iterations(20.0) == 150
    assert bl.choose_iterations(500.0) == 100
    assert bl.choose_iterations(0.0) == 200


def test_flop_byte_and_throttle():
    assert bl.matmul_flops(2, 3, 4) == 48.0
    assert bl.matmul_bytes(2, 3, 4, 4) == (8 + 12 + 6) * 4.0
    assert bl.layernorm_bytes(10, 768, 4) == (2 * 7680 + 2 * 768) * 4.0
    assert bl.softmax_bytes(3, 5, 4) == 120.0
    assert not bl.is_throttled(0x4) and not bl.is_throttled(0x1)
    assert bl.is_throttled(0x20) and bl.is_throttled(0x40) and bl.is_throttled(0x08)


def test_tolerance_formula_matches_cpp():
    abs_t, rel_t = bl.tolerance_reduction(100)
    assert abs_t == rel_t == pytest.approx(8 * bl.FLOAT32_EPS * 10)
    abs_d, rel_d = bl.tolerance_dot(100, 3.0)
    assert abs_d == pytest.approx(3 * abs_t) and rel_d == rel_t


def test_widen_for_storage_matches_cpp():
    base = (1e-4, 2e-5)
    assert bl.widen_for_storage(base, "f32") == base
    a, r = bl.widen_for_storage(base, "f16")
    assert a == pytest.approx(1e-4 + 2 ** -24) and r == pytest.approx(2e-5 + 2 * 2 ** -11)
    a, r = bl.widen_for_storage(base, "bf16")
    assert r == pytest.approx(2e-5 + 2 * 2 ** -8)


def test_softmax_tolerance_is_relative():
    abs_t, rel_t = bl.tolerance_softmax(32000)
    assert abs_t < 1e-37  # a typical output in a 32000-wide row is ~3e-5
    assert rel_t == bl.tolerance_reduction(32000)[1]


def _row(gate):
    m = bl.Measured(summary=bl.Summary(n=200, median=10.0, valid=True), valid=True)
    ctx = {"gpu": "Test GPU", "cc": "7.5", "driver": "1", "cuda": "13.0", "commit": "x",
           "fp32_tflops_theory": 8.0, "fp32_tflops_measured": 0.0,
           "bandwidth_gbps_theory": 320.0, "bandwidth_gbps_measured": 0.0}
    return bl.csv_row(op="matmul", impl="pytorch_eager", algo="torch.mm", dtype="f32",
                      shape={"tag": "t", "M": 64, "N": 64, "K": 64}, cold=False,
                      gate=gate, max_abs=1e-6, max_rel=1e-6,
                      flops=bl.matmul_flops(64, 64, 64),
                      nbytes=bl.matmul_bytes(64, 64, 64, 4), measured=m, tel={}, ctx=ctx)


def test_matmul_peak_follows_dtype():
    p = {"fp32_tflops_theory": 8.0, "fp32_tflops_measured": 7.0,
         "tc_f16_tflops_theory": 65.0, "tc_f16_tflops_measured": 40.0}
    assert bl.matmul_peak_tflops(p, "f32", False) == 8.0
    assert bl.matmul_peak_tflops(p, "f16", False) == 65.0
    assert bl.matmul_peak_tflops(p, "f16", True) == 40.0
    assert bl.matmul_peak_tflops(p, "bf16", True) == 7.0  # no BF16 Tensor Cores
    p.update(tc_bf16_tflops_theory=65.0, tc_bf16_tflops_measured=39.0)
    assert bl.matmul_peak_tflops(p, "bf16", True) == 39.0


def test_csv_gate_rule_and_column_count():
    failed, passed = _row("FAIL"), _row("PASS")
    assert ",FAIL," in failed and ",200,10," not in failed
    assert ",PASS," in passed and ",200,10," in passed
    n = bl.CSV_HEADER.count(",")
    assert failed.count(",") == n and passed.count(",") == n
