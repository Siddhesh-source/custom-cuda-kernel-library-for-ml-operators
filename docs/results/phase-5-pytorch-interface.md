# Phase 5 — PyTorch interface

**Date:** 2026-09-24
**Outcome:** 31 Python tests passed, 4 skipped (all matmul), 0 failed.
64 C++ tests unchanged and still passing.
**Agreement with PyTorch on CUDA tensors: worst absolute error 2.895e-08 for
softmax, 2.383e-07 for layer normalization.**

## Environment

Kaggle GPU kernel (`isiddhesh3011/ccml-build`, version 15). Tesla T4, sm_75,
CUDA 12.8.93, GCC 11.4.0, CMake 3.31.10, Ninja. Release, `-Werror`, zero
warnings.

**PyTorch 2.10.0+cu128** — the first phase in which a CUDA-enabled PyTorch was
available, so this is the first time the zero-copy path has actually run.

## The binding approach, and why

`torch.utils.cpp_extension` is the conventional route and was **not** used. It
needs a compiler and a CUDA-enabled PyTorch whose CUDA version matches the local
toolkit *at import time*. The development machine has PyTorch 2.1.1+cpu and
cannot compile CUDA at all, so that route cannot be attempted there; and
anywhere else it adds a second build system beside CMake.

The library already exposes a flat C ABI taking device pointers, and a PyTorch
tensor's `data_ptr()` is exactly such a pointer. The binding is therefore:
validate the tensor, hand its pointer to the C function, let the kernel write
into an allocation PyTorch made. No compiler, no second build, no copy.

**The kernel runs on PyTorch's current stream**
(`torch.cuda.current_stream(device).cuda_stream`). This is the part that makes
it compose rather than merely work: the operator is ordered behind whatever
produced its input and ahead of whatever consumes its output, needing no
synchronisation. Running on the default stream instead would be a race against
anything PyTorch had queued.

What this costs: **no autograd**, and no dispatcher integration. Neither is in
scope at this milestone, and a `cpp_extension` wrapper could be added later over
the same C ABI without touching any CUDA code.

## What was built

| Component | File |
|---|---|
| PyTorch interface | `python/ccml/torch_api.py` (new) |
| Package exports, optional-torch handling | `python/ccml/__init__.py` |
| Comparison script | `python/compare_with_torch.py` (new) |
| Tests | `python/tests/test_torch_api.py` (new, 22 tests) |

Public names are `custom_matmul`, `custom_layernorm`, `custom_softmax`. Python
has namespaces, so the `custom_` prefix costs nothing here — unlike in the C ABI,
where it was rejected in [phase 4](phase-4-library-refactor.md) precisely
because C has none.

## Results

### `compare_with_torch.py`, on a Tesla T4

Reference evaluated in **float64** on identical float32 inputs, so the
comparison is against the mathematics rather than another float32
implementation's rounding.

```
ccml 0.1.0
torch 2.10.0+cu128 on Tesla T4

custom_softmax vs torch.softmax
  case                   shape                      max_abs     max_rel
  normal input           64 x 1024                1.188e-09   3.917e-07
  normal input           33 x 517                 3.434e-09   3.710e-07
  normal input           128 x 1                  0.000e+00   0.000e+00
  normal input           8 x 16384                1.253e-10   5.356e-07
  mean +5000             16 x 256                 2.895e-08   1.486e-07
  3-D input              4 x 16 x 768             2.244e-09   3.783e-07

custom_layernorm vs torch.nn.functional.layer_norm
  case                   shape                      max_abs     max_rel
  with affine            64 x 768                 2.381e-07   5.923e-08
  with affine            33 x 517                 2.383e-07   5.897e-08
  with affine            1 x 1024                 2.137e-07   5.705e-08
  with affine            8 x 16384                2.377e-07   1.881e-07
  no affine              64 x 768                 1.192e-07   5.918e-08
  mean 1e4, sd 0.01      32 x 512                 1.214e-07   6.001e-08
  3-D input              4 x 16 x 768             1.701e-07   5.926e-08

custom_matmul vs torch.matmul
  NOT IMPLEMENTED        128 x 128 @ 128 x 128   ccml_matmul_f32: kernels are
                                                 not implemented in this build
```

A single-column softmax row returns **exactly** 1.0 — zero error, not merely
small error.

These figures agree with the C++ suites in
[phase 2](phase-2-layernorm.md) and [phase 3](phase-3-softmax.md), which is the
point of running both: the same kernels reached through a different binding and
compared against a different oracle give the same answer.

### Validation — the errors a caller actually gets

```
  CPU tensor rejected          x must be a CUDA tensor, but it is on cpu. ccml
                               operates on GPU memory and will not copy for you
  float64 rejected             x must be float32, got torch.float64. ccml
                               implements float32 only; cast explicitly
  non-contiguous rejected      x must be contiguous. ccml assumes a row-major
                               contiguous layout; call x.contiguous() explicitly
  bad weight shape rejected    weight must have shape (8,) to match the
                               normalized dimension, got (4,)
  mismatched matmul rejected   shapes (2, 3) and (4, 5) are not multipliable
```

Each of these is a decision with a cost — a host transfer, a cast, a copy — so
it is refused rather than performed invisibly. A binding that quietly calls
`.cuda()` or `.contiguous()` hides exactly the expense a user came to a custom
kernel to avoid.

### Test suites

```
31 passed, 4 skipped in 4.65s
```

| Suite | Passed | Skipped |
|---|---|---|
| `test_torch_api.py` (PyTorch interface) | 22 | 1 (matmul) |
| `test_ops.py` (numpy interface) | 9 | 3 (matmul) |

C++ suites unchanged: 64 passed, 6 matmul shape tests still skipped.

One test worth naming: `test_result_is_ordered_against_torch_work_on_the_same_stream`
queues a PyTorch multiply, a ccml softmax, and a PyTorch sum with **no explicit
synchronisation**, and checks the rows sum to 1. It passes, which is the
evidence that the stream handling is right — on the wrong stream this would race
and produce garbage intermittently.

## What this does NOT establish

- **No performance claim.** Nothing was timed. "Zero copy" is a statement about
  what the code does, not a measured speed.
- **No autograd.** These are forward-only functions; they will not participate
  in `backward()`.
- **Matmul still has no kernel.**
- **Turing (sm_75), not the Ada target.**
- **Not tested on the development machine** — its PyTorch is CPU-only, so the
  `custom_*` path skips there entirely.
- **`compute-sanitizer` still has not been run** on any kernel. Unchanged since
  phase 3 and still the largest correctness gap.

## Reproducing this

```
python cloud/kaggle/kaggle_run.py
```

Or with a local CUDA-enabled PyTorch:

```
cmake --workflow --preset build
cd python && python compare_with_torch.py
```
