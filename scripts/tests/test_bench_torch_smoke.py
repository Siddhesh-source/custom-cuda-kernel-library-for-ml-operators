"""End-to-end smoke test of scripts/bench_torch.py: every op runs one tiny
shape through gate, timing and CSV writing. It checks that the script works,
not what it measures (these rows are never results). Needs CUDA; runs in the
Kaggle sanitize job. Added after a variable named `out` shadowed the CSV file
handle and a full T4 session lost its PyTorch baselines."""

import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "bench_torch.py"


@pytest.mark.parametrize("op,flt", [("layernorm", "h768.rows16"),
                                    ("layernorm_residual", "h768.rows16"),
                                    ("softmax", "gpt2s.attn.s128"),
                                    ("matmul", "gpt2s.qkv.m16")])
def test_bench_torch_runs_end_to_end(tmp_path, op, flt):
    r = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(tmp_path), "--ops", op,
                        "--impls", "eager,compile", "--filter", flt, "--cache", "warm",
                        "--compile-mode", "default", "--max-iters", "100",
                        "--layernorm-dtypes", "f32,f16,bf16"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rows = [l for l in (tmp_path / f"torch_{op}.csv").read_text().splitlines()
            if l and not l.startswith("#")]
    assert len(rows) > 1, "header only: no shape matched or nothing was written"
    assert all(",FAIL," not in l and ",NOT_RUN," not in l for l in rows[1:]), rows[1:]
