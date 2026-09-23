#!/usr/bin/env python3
"""Compare every ccml operator against its PyTorch equivalent on the GPU.

    python python/compare_with_torch.py

Creates CUDA tensors, runs each custom operator and the PyTorch reference on the
same inputs, and reports the maximum absolute and relative error. Nothing is
timed: this measures agreement, not speed.

The reference is evaluated in float64 on the GPU so that the comparison is
against the mathematics rather than against another float32 implementation's
rounding. Both see identical float32 inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import ccml  # noqa: E402
from ccml import CcmlNotImplementedError  # noqa: E402


def errors(got: torch.Tensor, reference: torch.Tensor) -> tuple:
    """Maximum absolute and relative error between a result and a float64 reference."""
    got64 = got.double()
    difference = (got64 - reference).abs()

    max_abs = difference.max().item()

    # Relative error is undefined where the reference is zero; those elements
    # are judged by the absolute figure instead.
    magnitude = reference.abs()
    nonzero = magnitude > 0
    max_rel = (difference[nonzero] / magnitude[nonzero]).max().item() if nonzero.any() else 0.0

    return max_abs, max_rel


def report(name: str, shape: str, max_abs: float, max_rel: float) -> None:
    print(f"  {name:<22} {shape:<22} {max_abs:>11.3e} {max_rel:>11.3e}")


def check_softmax(device: torch.device) -> None:
    print("\ncustom_softmax vs torch.softmax")
    print(f"  {'case':<22} {'shape':<22} {'max_abs':>11} {'max_rel':>11}")

    for rows, cols in ((64, 1024), (33, 517), (128, 1), (8, 16384)):
        x = torch.randn(rows, cols, device=device, dtype=torch.float32)

        got = ccml.custom_softmax(x)
        expected = torch.softmax(x.double(), dim=-1)

        max_abs, max_rel = errors(got, expected)
        report("normal input", f"{rows} x {cols}", max_abs, max_rel)

    # Without the max subtraction every output here would be NaN.
    x = torch.randn(16, 256, device=device, dtype=torch.float32) * 10.0 + 5000.0
    got = ccml.custom_softmax(x)
    expected = torch.softmax(x.double(), dim=-1)
    max_abs, max_rel = errors(got, expected)
    report("mean +5000", "16 x 256", max_abs, max_rel)
    assert torch.isfinite(got).all(), "softmax overflowed on large inputs"

    # A 3-D tensor: leading dimensions collapse into rows.
    x = torch.randn(4, 16, 768, device=device, dtype=torch.float32)
    got = ccml.custom_softmax(x)
    expected = torch.softmax(x.double(), dim=-1)
    max_abs, max_rel = errors(got, expected)
    report("3-D input", "4 x 16 x 768", max_abs, max_rel)


def check_layernorm(device: torch.device) -> None:
    print("\ncustom_layernorm vs torch.nn.functional.layer_norm")
    print(f"  {'case':<22} {'shape':<22} {'max_abs':>11} {'max_rel':>11}")

    for rows, cols in ((64, 768), (33, 517), (1, 1024), (8, 16384)):
        x = torch.randn(rows, cols, device=device, dtype=torch.float32)
        weight = torch.rand(cols, device=device, dtype=torch.float32) + 0.5
        bias = torch.randn(cols, device=device, dtype=torch.float32)

        got = ccml.custom_layernorm(x, weight, bias)
        expected = torch.nn.functional.layer_norm(
            x.double(), (cols,), weight.double(), bias.double(), 1e-5)

        max_abs, max_rel = errors(got, expected)
        report("with affine", f"{rows} x {cols}", max_abs, max_rel)

    # No affine at all.
    x = torch.randn(64, 768, device=device, dtype=torch.float32)
    got = ccml.custom_layernorm(x)
    expected = torch.nn.functional.layer_norm(x.double(), (768,))
    max_abs, max_rel = errors(got, expected)
    report("no affine", "64 x 768", max_abs, max_rel)

    # The case a naive E[x^2] - E[x]^2 variance gets badly wrong.
    x = torch.randn(32, 512, device=device, dtype=torch.float32) * 0.01 + 10000.0
    got = ccml.custom_layernorm(x)
    expected = torch.nn.functional.layer_norm(x.double(), (512,))
    max_abs, max_rel = errors(got, expected)
    report("mean 1e4, sd 0.01", "32 x 512", max_abs, max_rel)

    # A 3-D tensor: leading dimensions collapse into rows.
    x = torch.randn(4, 16, 768, device=device, dtype=torch.float32)
    got = ccml.custom_layernorm(x)
    expected = torch.nn.functional.layer_norm(x.double(), (768,))
    max_abs, max_rel = errors(got, expected)
    report("3-D input", "4 x 16 x 768", max_abs, max_rel)


def check_matmul(device: torch.device) -> None:
    print("\ncustom_matmul vs torch.matmul")
    print(f"  {'case':<22} {'shape':<22} {'max_abs':>11} {'max_rel':>11}")

    for m, k, n in ((128, 128, 128), (67, 41, 53)):
        a = torch.randn(m, k, device=device, dtype=torch.float32)
        b = torch.randn(k, n, device=device, dtype=torch.float32)
        try:
            got = ccml.custom_matmul(a, b)
        except CcmlNotImplementedError as exc:
            print(f"  {'NOT IMPLEMENTED':<22} {f'{m} x {k} @ {k} x {n}':<22}  {exc}")
            return
        expected = a.double() @ b.double()
        max_abs, max_rel = errors(got, expected)
        report("normal input", f"{m} x {k} @ {k} x {n}", max_abs, max_rel)


def check_validation(device: torch.device) -> None:
    """The errors a caller should get, rather than a silent copy or a crash."""
    print("\nvalidation")

    cases = [
        ("CPU tensor rejected",
         lambda: ccml.custom_softmax(torch.randn(4, 8))),
        ("float64 rejected",
         lambda: ccml.custom_softmax(torch.randn(4, 8, device=device,
                                                 dtype=torch.float64))),
        ("non-contiguous rejected",
         lambda: ccml.custom_softmax(
             torch.randn(8, 8, device=device, dtype=torch.float32).t())),
        ("bad weight shape rejected",
         lambda: ccml.custom_layernorm(
             torch.randn(4, 8, device=device, dtype=torch.float32),
             torch.randn(4, device=device, dtype=torch.float32))),
        ("mismatched matmul rejected",
         lambda: ccml.custom_matmul(
             torch.randn(2, 3, device=device, dtype=torch.float32),
             torch.randn(4, 5, device=device, dtype=torch.float32))),
    ]

    for label, call in cases:
        try:
            call()
        except (ValueError, TypeError) as exc:
            message = str(exc).splitlines()[0]
            print(f"  {label:<28} {message[:90]}")
        else:
            print(f"  {label:<28} NOT REJECTED -- this is a bug")


def main() -> int:
    if not torch.cuda.is_available():
        print("No CUDA device available to PyTorch.")
        print(f"torch {torch.__version__}, CUDA build: {torch.version.cuda}")
        print("\nThese operators run on the GPU and will not fall back to the CPU.")
        return 1

    device = torch.device("cuda")
    print(f"ccml {ccml.version()}")
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    print("\nReference is evaluated in float64 on the same float32 inputs.")

    check_softmax(device)
    check_layernorm(device)
    check_matmul(device)
    check_validation(device)

    print("\nNo timing was taken. These figures are agreement, not speed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
