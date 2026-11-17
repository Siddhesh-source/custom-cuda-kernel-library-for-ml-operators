# Design

Status: **living document**. It records the design rationale for every kernel
variant. Scope and requirements come from [`reference.md`](reference.md), which
is binding, and the plan is in [ROADMAP.md](ROADMAP.md).

## Scope

Three operators in FP32, FP16 and BF16 (FP32 accumulation), row-major and
contiguous [reference.md §3]:

| Operator | Signature | Header |
|---|---|---|
| Matrix multiply | `C[M,N] = A[M,K] * B[K,N]` | `include/ccml/matmul.h` |
| Layer norm | over the last dimension, optional affine | `include/ccml/layernorm.h` |
| Softmax | over the last dimension | `include/ccml/softmax.h` |

Excluded because `reference.md` does not ask for them: strides, transpose
flags, broadcasting, alpha/beta accumulation, backward passes. Each could be
added later behind the same C ABI.

Dtype entry points follow `ccml_<op>_<f32|f16|bf16>`. BF16 I/O works on every
architecture. BF16 Tensor Core paths need sm_80+ and return
`CCML_ERROR_NOT_IMPLEMENTED` below that.

## API shape and why

**Flat C ABI, one function per operator.** Every entry point returns a
`ccml_status`; nothing throws and nothing aborts. This is what makes the ctypes
binding possible without a compiler, and it keeps the library usable from C.

**Device pointers in, device pointers out.** The library never allocates the
caller's tensors and never copies to the host. `ccml/memory.h` offers allocation
helpers, but only so that a binding with no other CUDA access can use the
library; C++ callers should ignore them.

**An `algo` enum per operator.** This is the extension point for all later
optimization work. A new kernel variant is a new enum value plus a case in the
dispatch switch, with no public signature change, so tests and benchmarks can
compare variants directly. `AUTO` resolves to the current best default and is
what callers should normally pass.

**Kernels separated from host wrappers.** `src/<op>/<op>.cu` validates
arguments, chooses launch configuration and dispatches.
`src/<op>/<op>_kernels.cuh` contains only `__global__` code. Validation is
therefore written once regardless of how many kernel variants exist, and an
optimization pass touches exactly one file.

## Hardware the defaults are aimed at

**Target: Tesla T4 on Kaggle (Turing, sm_75).** All GPU work runs there, and
every tile size, block size and AUTO threshold is chosen from T4 measurements.
The exceptions that need sm_80+ hardware are listed in [local.md](local.md).

T4 constraints that shape the design: 1024 threads/SM, 64 KB shared memory/SM
(48 KB per block by default), 4 MB L2, 320 GB/s theoretical bandwidth, FP16
Tensor Cores (WMMA m16n16k16, `mma.sync` m16n8k8), **no `cp.async`** (so double
buffering uses register prefetch), **no BF16 Tensor Cores**.

### Measured: Tesla T4 (Kaggle)

From `ccml_device_query` on a Kaggle kernel, 2026-09-23. Recorded because it is
real measured data, and because the contrast with the target matters.

| Property | Tesla T4 |
|---|---|
| Compute capability | 7.5 (Turing) |
| SM count | 40 |
| Max threads / block | 1024 |
| Max threads / SM | 1024 |
| Shared memory / block | 49152 bytes |
| Shared memory / SM | 65536 bytes |
| Registers / block | 65536 |
| Global memory | 14.56 GiB |
| L2 cache | 4096 KiB |
| Memory bus width | 256 bits |
| Memory clock | 5001 MHz |
| SM clock | 1590 MHz |
| Theoretical peak bandwidth | 320.1 GB/s (ceiling, never achieved) |

## Matrix multiplication

**Naive** (`CCML_MATMUL_NAIVE`) - one thread per output element, accumulating
over K in a register. No data reuse between threads, so it is bandwidth-bound by
construction. It is kept permanently, not as a stepping stone: it is the
correctness baseline that every faster variant is differenced against.

**Tiled** (`CCML_MATMUL_TILED`) - each block cooperatively stages a square tile
of A and of B into shared memory, so each loaded element is reused once per row
of the tile. This is the standard structure, not an optimization, and it is what
later work (vectorized loads, register tiling, double buffering) extends.

Tile dimension is currently 16, chosen provisionally. It must be revisited
against measured occupancy and the sponsor target shapes before any statement is
made about its performance.

Correctness detail that matters more than the tiling: when M, N or K is not a
multiple of the tile dimension, out-of-range elements must be **zero-filled**
during staging rather than skipped, because every thread in the block has to
reach the same `__syncthreads()` calls.

TODO: block dimension, elements per thread, whether shared memory is padded to
avoid bank conflicts. All pending measurement.

## Layer normalization

One block per row, two reduction passes: mean, then variance about that mean,
then a normalize-and-scale write.

Two passes rather than the single sum/sum-of-squares pass on purpose. The
single-pass form computes variance as `E[x^2] - E[x]^2`, which loses
catastrophic precision when the mean is large relative to the spread, which is
exactly what `layernorm.large_offset_small_variance` tests. Welford's method is
the numerically sound single-pass alternative; it is a separate algorithm
(`CCML_LAYERNORM_WELFORD`, Phase 12) rather than folded in here, so that the two
are compared rather than one silently replacing the other.

Variance is the biased (1/cols) estimator, matching
`torch.nn.functional.layer_norm`. `gamma` and `beta` are independently optional.

**Reductions accumulate in double**, which contradicts the float32 default in
`types.h` and is worth the exception. The output divides by the standard
deviation, so any error in the mean is magnified by `1/stddev`. A float32 mean
carries roughly half an ulp of error; at magnitude 10000 that is about 5e-4,
which against a standard deviation of 0.01 is a **5% error in every output
element**. No summation order fixes this — the mean has to be held to better
than float32 precision and carried in that precision through the subtraction.
The cost is fp64 arithmetic, which consumer GPUs run at a fraction of fp32
rate. **Measured in Phase 12:** it is not hidden. Nsight Compute shows
`TWO_PASS` with the FP64 pipe 87% busy and DRAM at 13% (sm_89), and the FP32
variants run 3.6-11x faster. They keep the same accuracy class without FP64 by
subtracting a pivot (the row's first element) before accumulating, which
removes the large intermediate instead of carrying it in double precision;
they meet the same FP32 tolerance, including mean 10000 / stddev 0.01
([optimization-log.md](optimization-log.md) entry 9, [correctness.md](correctness.md)).
`TWO_PASS` remains as the FP64 reference variant.

Reduction strategy: warp shuffles first, then one warp reduces the per-warp
partials through shared memory. Two barriers in total, against one per level for
a full shared-memory tree.

Block size is `min(round_up(cols, 32), 256)`, floored at one warp. Narrow rows
get a smaller block rather than 256 threads of which all but a handful idle
through every barrier. This is sizing to the work, not tuning — **no block size
in this library has been chosen by measurement.**

Rows wider than the block are handled by striding, so no column count is a
special case.

Measured results: [results/phase-2-layernorm.md](results/phase-2-layernorm.md).

## Softmax

One block per row, three passes: row max, sum of `exp(x - max)`, then the
normalized write.

Subtracting the row max is a **correctness requirement, not a tuning choice**.
`exp` overflows float32 above roughly x = 88, after which the row becomes
`inf/inf = NaN`. `softmax.large_positive_inputs` exists to prove the guard is
present rather than assumed.

Subtracting the maximum also removes the division-by-zero case: the maximum
element always contributes `exp(0) = 1`, so the denominator is at least 1
however extreme the input. There is no guard to write.

**Three passes over x, recomputing the exponential, rather than caching it in
the output.** Caching would evaluate `expf` once per element instead of twice,
but moves five array transfers against four — two reads of x, two writes of y
and a read of y, against three reads of x and one write. For a memory-bound
kernel that is the wrong trade. This is reasoning from byte counts, **not
measurement**.

The sum accumulates in double, because its error compounds across the row. The
exponential is evaluated in float: `expf` is within a couple of ulp, its error
is per-element rather than accumulating, and the output is float32 anyway. This
differs from layer normalization, which needs double throughout for the reason
above.

Reduction strategy and block sizing are shared with layer normalization, via
`src/common/reduce.cuh` — the reduction takes its combining operation as a
parameter so that softmax's maximum and both operators' sums use one
implementation. Operators never include each other.

A warp-per-row variant (`CCML_SOFTMAX_WARP_PER_ROW`) is reserved for narrow
rows, where a whole block per row wastes most of its threads on the reduction.
Whether it is worth building depends on the sponsor target column counts.

Measured results: [results/phase-3-softmax.md](results/phase-3-softmax.md).

## Open questions

All of these are settled by measurement on the T4 (see ROADMAP §1 and §3):

1. Does layernorm's fp64 accumulation cost measurable time against fp32 Welford?
2. What arithmetic intensity does register blocking actually reach?
3. Below what row width does warp-per-row softmax or layernorm win?
4. Where should AUTO switch between variants, per dtype and shape bucket?
5. On which skinny (decode) shapes does split-K or stream-K beat cuBLAS?
