# ccml — Custom CUDA Kernel Library for Common ML Operators

Hand-written CUDA C++ kernels for matrix multiplication, layer normalization and
softmax, packaged as a reusable library with a stable C ABI, correctness tests
against independent references, a benchmark harness, and an optional Python
binding.

## Status

| Operator | State | Correctness | Speed vs PyTorch |
|---|---|---|---|
| `ccml_matmul_f32` | working | worst scaled rel. error 7.38e-06 | **0.12x – 0.32x** |
| `ccml_layernorm_f32` | working | worst rel. error 5.96e-08 | **0.17x – 0.46x** |
| `ccml_softmax_f32` | working | worst abs. error 4.40e-08 | **0.22x – 0.52x** |

All three are validated against two independent references — a double-precision
CPU oracle and PyTorch — on a Tesla T4.

**Every operator is currently slower than PyTorch**, by 2x to 11x depending on
shape. Measured, not estimated: see the
[case study](docs/results/phase-6-case-study.md) and
[`results/`](results/). The one speedup this project claims is between two of its
own kernels — shared-memory tiling beats the naive matmul baseline by 1.38x-1.55x.

## Dependencies

The CUDA runtime. That is the entire list.

The optional Python binding adds numpy and pytest. Its `custom_*` functions
operate on CUDA tensors and so need a CUDA-enabled PyTorch; the numpy entry
points and the C++ test suite work with a CPU-only build.

## Build and test

One command builds everything:

    cmake --workflow --preset build

One command builds everything and runs every correctness test:

    cmake --workflow --preset default

Or the steps separately:

    cmake --preset default          # configure
    cmake --build --preset default  # build
    ctest --preset default          # test

Each suite runs independently:

    ctest --preset default -R softmax
    build/default/bin/ccml_test_softmax large_positive

The default preset targets `sm_75`, the Kaggle Tesla T4 that is this project's
GPU target. For another GPU:

    cmake -B build/mygpu -G Ninja -DCCML_CUDA_ARCHITECTURES=89
    cmake --build build/mygpu

**All GPU work runs on Kaggle:**

    python cloud/kaggle/kaggle_run.py

See [`cloud/`](cloud/README.md). The few components that need a local GPU are
listed in [docs/local.md](docs/local.md).

## Usage

All tensors are **device** pointers, row-major and contiguous. Every entry point
returns a `ccml_status`; nothing throws, nothing aborts, nothing prints.

```c
#include <ccml/ccml.h>

float *x, *y;            /* device pointers, [rows, cols] */
float *gamma, *beta;     /* device pointers, [cols], or NULL */

ccml_status status = ccml_softmax_f32(y, x, rows, cols,
                                      CCML_SOFTMAX_AUTO, stream);
if (status != CCML_SUCCESS) {
    fprintf(stderr, "%s: %s\n",
            ccml_status_string(status),   /* what went wrong */
            ccml_last_error_string());    /* the detail, including CUDA's own */
}

status = ccml_layernorm_f32(y, x, gamma, beta, rows, cols, 1e-5f,
                            CCML_LAYERNORM_AUTO, stream);

/* Operators are asynchronous. Reading y before synchronising is a race. */
ccml_stream_synchronize(stream);
```

`examples/usage.cpp` is this code, compiled and runnable:

    build/default/bin/ccml_usage

From PyTorch, operating directly on CUDA tensors with no copy, on PyTorch's
current stream:

```python
import torch, ccml

y = ccml.custom_softmax(torch.randn(64, 1024, device="cuda"))
z = ccml.custom_layernorm(torch.randn(4, 16, 768, device="cuda"))
```

A tensor on the CPU, of the wrong dtype, or non-contiguous is **rejected with an
explanation, never converted silently** — each of those is a cost the caller
should choose. See [python/README.md](python/README.md).

### API conventions

Every operator follows the same shape, so learning one teaches the rest:

    ccml_<operator>_f32(output, inputs..., dimensions..., parameters...,
                        algo, stream)

- **Output first**, then inputs, then shape, then operator parameters.
- **Row-major, contiguous.** No strides, no transpose flags, no broadcasting.
- **Device pointers**, never owned by the library.
- **Last dimension is the reduced one** for row-wise operators. Collapse leading
  dimensions into `rows`: a `[batch, seq, hidden]` tensor normalized over hidden
  is `rows = batch*seq, cols = hidden`.
- **`algo`** selects a kernel variant; `..._AUTO` picks the current default. An
  unbuilt variant returns `CCML_ERROR_NOT_IMPLEMENTED` rather than silently
  falling back to a different one.
- **`stream`** may be `NULL` for the default stream. The library never creates
  or owns streams.

Symbols are prefixed `ccml_` rather than `custom_` because a C library has no
namespaces and must not collide with anything else a caller links. The full set
of rules is in [`include/ccml/types.h`](include/ccml/types.h).

## Layout

| Path | Contents |
|---|---|
| `include/ccml/` | Public C ABI. The entire consumer surface |
| `src/<op>/<op>.cu` | Host wrapper: validation, launch config, dispatch |
| `src/<op>/<op>_kernels.cuh` | Device code only. The optimization surface |
| `src/common/` | Error handling, launch validation, block reduction, device query |
| `support/` | Device buffers, RNG, RAII streams and events — shared by tests and benchmarks |
| `tests/` | Correctness, one executable per suite. No timing |
| `bench/` | Timing. No assertions, not registered with CTest |
| `python/` | PyTorch extension (`torch.utils.cpp_extension`, Phase 14); interim ctypes binding |
| `examples/` | Public-API-only consumers |
| `cloud/` | Kaggle runner: all GPU builds, tests, benchmarks and profiling |
| `profiles/` | Small Nsight Compute / Nsight Systems reports |
| `scripts/` | Setup, sanitizer, sweeps, roofline, remote runs |
| `docs/` | Design rationale, correctness evidence, per-phase results |

Rules the layout enforces: the library depends on nothing but the CUDA runtime;
device code lives only in `*_kernels.cuh`; operators never include each other;
and shared device utilities live in `src/common/`, not copied between operators.

## Documentation

**Specification and plan:**

- [Reference](docs/reference.md): the binding project specification
- [Project overview](docs/PROJECT_OVERVIEW.md) · [Roadmap](docs/ROADMAP.md) ·
  [Enhancements](docs/enhancements.md)

**Start here if you are new to CUDA:**

- [Foundations](docs/foundations.md) — every concept this project uses, built up
  from nothing, with small examples. Written for someone who knows C++ but has
  never written a GPU kernel. Each idea is tied to a file or a measurement from
  this repository.

**Reference:**

- [Design](docs/design.md) — algorithm choices and the reasoning behind them
- [Correctness](docs/correctness.md) — method, tolerance policy, measured errors
- [Benchmarking](docs/benchmarking.md) — measurement method, baselines, CSV schema
- [Build](docs/build.md) — requirements, options, troubleshooting
- [Local GPU components](docs/local.md) — the only items that cannot run on Kaggle
- [Phase results](docs/results/README.md) — measured output from each phase

Every phase ends with tests under `tests/` and a results document under
`docs/results/`, recording what was observed on real hardware rather than what
was intended.

## License

MIT. See [LICENSE](LICENSE).
