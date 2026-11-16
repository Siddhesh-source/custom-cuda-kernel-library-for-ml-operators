# Python binding

Optional. The C++ library does not depend on this directory in any way, and the
build system has no target for it — the package finds the built shared library
at runtime.

## Requirements

    numpy
    torch         # for the custom_* interface, and as the test oracle
    pytest        # to run the tests

A CUDA-enabled PyTorch is needed for the `custom_*` functions, since they
operate on CUDA tensors. The numpy entry points and the C++ test suite need only
a CPU build.

## Use

Build the C++ library first:

    cmake --workflow --preset build

Then:

    cd python
    python -c "import ccml; print(ccml.version())"

If the library is somewhere unusual, point at it explicitly:

    set CCML_LIBRARY_PATH=D:\path\to\ccml.dll     # Windows
    export CCML_LIBRARY_PATH=/path/to/libccml.so  # Linux

## Two interfaces

### PyTorch — `custom_*`, zero copy

Operates directly on CUDA tensors. Use this one.

```python
import torch, ccml

x = torch.randn(64, 1024, device="cuda")
y = ccml.custom_softmax(x)

h = torch.randn(4, 16, 768, device="cuda")       # leading dims collapse into rows
n = ccml.custom_layernorm(h, weight, bias, eps=1e-5)

c = ccml.custom_matmul(a, b)                     # 2-D only, no batching
```

Nothing is transferred. PyTorch allocates the output, ccml writes into that
allocation, and the kernel runs on **PyTorch's current stream** — so it is
ordered behind whatever produced its input and ahead of whatever consumes its
output, with no synchronisation.

### numpy — host arrays

```python
import numpy as np, ccml
y = ccml.softmax(np.random.randn(64, 1024).astype(np.float32))
```

Convenient for a quick check and used by the test suite, but every call costs a
transfer to the device and back. Not for use in a training loop.

## What is validated, and what is never done silently

| Condition | Result |
|---|---|
| tensor on CPU | `ValueError` — **never** copied to the GPU for you |
| dtype not float32 | `TypeError` — never cast |
| not contiguous | `ValueError` — never made contiguous for you |
| `weight`/`bias` wrong shape | `ValueError` naming the expected shape |
| `eps <= 0` | `ValueError` |
| matmul shapes not multipliable | `ValueError` |
| matmul given 3-D input | `ValueError` — batching is not implemented |
| operator has no kernel | `CcmlNotImplementedError` |

Every one of these is a decision with a cost — a transfer, a cast, a copy — so
it is left to the caller rather than made invisibly. A CUDA failure arrives as
`CcmlError` carrying the runtime's own message.

## Tests

    cd python
    pytest tests -v                   # both interfaces
    python compare_with_torch.py      # error report against PyTorch, on the GPU

`compare_with_torch.py` creates CUDA tensors, runs each operator and its PyTorch
equivalent on identical inputs, and prints the maximum absolute and relative
error. The reference is evaluated in float64 so the comparison is against the
mathematics, not against another float32 implementation's rounding. It times
nothing.

Tests skip rather than fail when the library is not built, when no CUDA device
is present, or when an operator has no kernel behind it. A skip is not a pass —
read the summary line.

## Status: interim binding

`reference.md` §3 requires a **`torch.utils.cpp_extension`** extension. That is
Phase 14 in [ROADMAP.md](../docs/ROADMAP.md), and it builds on Kaggle, which has a
CUDA PyTorch whose CUDA version matches the toolkit. This ctypes binding remains
only until the extension reaches parity: the case study and the current pytest
suite depend on it. It is **deleted** once the extension passes the same tests.
