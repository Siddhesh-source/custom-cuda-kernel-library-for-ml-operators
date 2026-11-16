"""Triton baselines (enhancement B1): row softmax, layernorm and FP32 matmul.

Triton is the realistic alternative to hand-written CUDA, so it is measured as
an extra baseline next to PyTorch eager, torch.compile and cuBLAS/cuDNN. The
kernels follow the standard structure of the Triton tutorials, written for
arbitrary row widths (vocabulary softmax rows are 50257 wide) and with IEEE
FP32 dot products (no TF32), so the comparison with FP32 CUDA-core kernels is
like for like.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(y_ptr, x_ptr, cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride
    y_row = y_ptr + row * stride
    # Online max/sum in one pass over x, then a normalizing pass.
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < cols, other=float("-inf"))
        m_new = tl.maximum(m, x)
        # A lane that has only ever seen -inf (padding past `cols`, or a masked
        # input) has m == m_new == -inf, and exp(-inf - -inf) is NaN. Such a
        # lane contributes exactly 0. Without this guard any row narrower than
        # BLOCK came out NaN: the first Kaggle run's correctness gate failed
        # 9 Triton rows for exactly that reason.
        alive = m_new > float("-inf")
        s = tl.where(alive, s * tl.exp(m - m_new) + tl.exp(x - m_new), 0.0)
        m = m_new
    row_max = tl.max(m, 0)
    row_sum = tl.sum(tl.where(m > float("-inf"), s * tl.exp(m - row_max), 0.0), 0)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < cols, other=float("-inf"))
        tl.store(y_row + offs, tl.exp(x - row_max) / row_sum, mask=offs < cols)


def softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    y = torch.empty_like(x)
    block = min(4096, triton.next_power_of_2(cols))
    _softmax_kernel[(rows,)](y, x, cols, x.stride(0), BLOCK=block,
                             num_warps=8 if block >= 2048 else 4)
    return y


@triton.jit
def _layernorm_kernel(y_ptr, x_ptr, w_ptr, b_ptr, cols, stride, eps,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride
    y_row = y_ptr + row * stride
    acc = tl.zeros([BLOCK], tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x_row + offs, mask=offs < cols, other=0.0)
    mean = tl.sum(acc, 0) / cols
    acc = tl.zeros([BLOCK], tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < cols, other=0.0)
        d = tl.where(offs < cols, x - mean, 0.0)
        acc += d * d
    rstd = 1.0 / tl.sqrt(tl.sum(acc, 0) / cols + eps)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        x = tl.load(x_row + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(y_row + offs, (x - mean) * rstd * w + b, mask=mask)


def layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
              eps: float) -> torch.Tensor:
    rows, cols = x.shape
    y = torch.empty_like(x)
    block = min(4096, triton.next_power_of_2(cols))
    _layernorm_kernel[(rows,)](y, x, w, b, cols, x.stride(0), eps, BLOCK=block,
                               num_warps=8 if block >= 2048 else 4)
    return y


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 16, "G": 8}, num_stages=2, num_warps=8),
        triton.Config({"BM": 128, "BN": 64, "BK": 16, "G": 8}, num_stages=2, num_warps=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 16, "G": 8}, num_stages=2, num_warps=4),
        triton.Config({"BM": 64, "BN": 64, "BK": 32, "G": 8}, num_stages=2, num_warps=4),
        triton.Config({"BM": 32, "BN": 64, "BK": 32, "G": 8}, num_stages=2, num_warps=4),
        triton.Config({"BM": 16, "BN": 128, "BK": 32, "G": 8}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   G: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    # Grouped ordering for L2 reuse (the same idea as enhancement S1).
    group = pid // (G * num_n)
    first_m = group * G
    size_m = min(num_m - first_m, G)
    pid_m = first_m + (pid % size_m)
    pid_n = (pid % (G * num_n)) // size_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * sam + (k + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)  # noqa: E731
    _matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                         b.stride(1), c.stride(0), c.stride(1))
    return c
