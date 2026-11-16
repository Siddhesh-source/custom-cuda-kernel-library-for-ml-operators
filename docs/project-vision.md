# The finished project

> **Superseded in part (2026-11-16):** `docs/reference.md` is the binding specification. Any scope exclusion below that conflicts with it (Tensor Cores/WMMA, double buffering, vectorized loads, fusion, FP16/BF16, `torch.utils.cpp_extension`) no longer applies. See [ROADMAP.md](ROADMAP.md).

What this library looks like when it is complete, how it gets there from where
it stands today, and how anyone will be able to check the claims.

**Read this alongside** [milestone-40.md](milestone-40.md), which says what is
true *right now*. This document is about the destination. Everything here
described as a target is labelled as one — **no performance number in this
document is a measurement unless it says so and links to where it was taken.**

---

## 1. What the project is

A reusable CUDA C++ library of hand-written kernels for the three operators that
dominate Transformer inference — **matrix multiplication, layer normalization,
and softmax** — packaged so that another project can actually depend on it, and
verified so that its claims can be checked.

The point is not to beat cuBLAS. cuBLAS is decades of work by people with access
to hardware documentation we do not have. The point is to build these kernels
properly, understand exactly why they perform the way they do, and be able to
demonstrate and defend every statement made about them.

**Two things make this project unusual, and they should survive to completion:**

- Every claim traces to a committed, rerunnable script and its recorded output.
- Negative results are published as readily as positive ones. The current
  headline finding is that our kernels are 2–11× *slower* than PyTorch
  ([phase 6](results/phase-6-case-study.md)), and that is written at the top of
  the README rather than buried.

---

## 2. Scope on completion

### In scope

| Area | Final state |
|---|---|
| **Operators** | matmul, layernorm, softmax — forward pass, float32 |
| **Kernel variants** | Multiple per operator, selectable, each independently measurable |
| **Precision** | float32 throughout; fp16/bf16 only if the sponsor requires it |
| **API** | Stable C ABI; a PyTorch interface; a numpy convenience layer |
| **Verification** | Two independent references, property tests, `compute-sanitizer` clean |
| **Measurement** | Every operator benchmarked against PyTorch/cuBLAS across a shape sweep, on named hardware |
| **Documentation** | Design rationale, foundations, per-phase results, a reproducible case study |
| **Platforms** | Linux and Windows; Turing and Ada verified |

### Explicitly out of scope

Backward passes and autograd. Training. Multi-GPU. Distributed anything. A
complete Transformer model. Batched or multi-head attention as kernels
(the case study is single-head-shaped by design). Tensor Core / WMMA paths
unless time clearly permits. Inference serving, quantization, sparsity.

Saying no to these is what makes the rest defensible. A project that
half-implements ten things demonstrates less than one that fully implements
three.

---

## 3. The finished repository

The structure is already in place and is not expected to change. Additions are
marked.

```
ccml/
├── CMakeLists.txt              one command builds everything
├── CMakePresets.json           workflow presets: build, test, debug
├── README.md
│
├── include/ccml/               PUBLIC API — the entire consumer surface
│   ├── ccml.h                  umbrella
│   ├── status.h  types.h  export.h
│   ├── device.h  memory.h
│   ├── matmul.h  layernorm.h  softmax.h
│
├── src/
│   ├── common/                 shared internals, never installed
│   │   ├── check.h             error handling, CUDA detail preserved
│   │   ├── launch.h            launch validation against real device limits
│   │   ├── reduce.cuh          block reduction, shared by row-wise operators
│   │   ├── rowwise.h           shared launch configuration
│   │   └── util.h  device.cpp  memory.cpp  status.cpp
│   └── <op>/
│       ├── <op>.cu             host wrapper: validate, configure, dispatch
│       └── <op>_kernels.cuh    device code ONLY — the optimization surface
│
├── support/                    shared by tests and benchmarks
│   └── device_buffer.h  cuda_stream.h  cuda_event.h  random.h
│
├── tests/                      correctness only, no timing
│   ├── test_infrastructure.cu  launches its own kernels
│   ├── test_harness.cpp        tests the test harness
│   ├── test_matmul.cpp  test_layernorm.cpp  test_softmax.cpp
│   └── support/                harness, comparison, CPU oracles
│
├── bench/                      timing only, no assertions
│   └── bench_*.cpp  support/timer.cpp  support/report.cpp
│
├── python/
│   ├── ccml/                   torch_api.py (zero-copy), ops.py (numpy)
│   ├── case_study.py           the Transformer-shaped experiment
│   └── compare_with_torch.py
│
├── examples/                   public-API-only consumers
│   └── device_query.cpp  usage.cpp
│
├── cloud/                      remote build and test (Kaggle, Modal)
│
├── results/                    generated, committed
│   ├── case_study_results.csv
│   ├── case_study_summary.md
│   └── + per-operator benchmark sweeps            ← to add
│
└── docs/
    ├── foundations.md          every concept, from the ground up
    ├── design.md  correctness.md  benchmarking.md  build.md
    ├── project-vision.md       this file
    ├── milestone-40.md         → milestone-100.md                ← to add
    ├── optimization-log.md     each change, its hypothesis, its result  ← to add
    └── results/                one document per completed phase
```

Two additions carry most of the remaining value:

**`docs/optimization-log.md`** — for each performance change: what was tried, the
predicted effect and why, the measured effect, and whether it was kept. This is
the difference between a project that got faster and a project that can *explain*
why it got faster. Failed experiments stay in the log.

**`results/` sweeps** — machine-readable timings per operator across shapes, so
a regression is visible as a diff.

---

## 4. The finished API

The C ABI is already stable and is not expected to change shape. Every operator
follows one pattern:

```c
ccml_<operator>_f32(output, inputs..., dimensions..., parameters...,
                    algo, stream);
```

```c
ccml_status ccml_matmul_f32(float* C, const float* A, const float* B,
                            int M, int N, int K,
                            ccml_matmul_algo algo, cudaStream_t stream);

ccml_status ccml_layernorm_f32(float* y, const float* x,
                               const float* gamma, const float* beta,
                               int rows, int cols, float eps,
                               ccml_layernorm_algo algo, cudaStream_t stream);

ccml_status ccml_softmax_f32(float* y, const float* x, int rows, int cols,
                             ccml_softmax_algo algo, cudaStream_t stream);
```

**Growth happens through the `algo` enum, not through new functions.** That is
the whole design: a new kernel variant is a new enum value and a `case` in the
dispatch switch. Signatures never change, so every variant is directly
comparable in the same benchmark, and a caller passing `AUTO` silently gets the
best one.

Expected final variant sets:

| Operator | Variants |
|---|---|
| matmul | `NAIVE`, `TILED`, `REGISTER_TILED`, possibly `VECTORIZED` |
| layernorm | `TWO_PASS`, `WELFORD`, possibly `FUSED` |
| softmax | `BLOCK_PER_ROW`, `WARP_PER_ROW`, `ONLINE` (single-pass) |

`NAIVE` and `TWO_PASS` are never deleted. They are the correctness baselines
that every faster variant is differenced against, and the reference points that
make a speedup claim meaningful.

From Python, unchanged:

```python
import torch, ccml
y = ccml.custom_softmax(torch.randn(1024, 1024, device="cuda"))
```

---

## 5. Where the project stands, and what remains

| | Now | On completion |
|---|---|---|
| Operators implemented | 3 of 3 | 3 of 3 |
| Correctness verified | ✅ two references, 65 shapes | ✅ plus sanitizer-clean |
| `compute-sanitizer` | ❌ **never run** | ✅ memcheck + racecheck clean |
| Performance measured | ✅ first case study | ✅ full sweeps, per variant |
| Performance *optimized* | ❌ one tiling step | ✅ register tiling, fused row-wise |
| Builds on Windows | ❌ no SDK on dev machine | ✅ verified under MSVC |
| Verified on Ada (sm_89) | ❌ Turing only | ✅ the actual target |
| Sponsor shapes | ❌ unknown | ✅ obtained, tuned against |
| Committed to git | ❌ **zero commits** | ✅ full history |

That table is the honest project plan. The four ❌ rows at the top are worth more
than any further feature.

---

## 6. Roadmap

### Phase 7 — Verify what exists *(next, and non-negotiable)*

Run `compute-sanitizer` memcheck and racecheck on all three operators.

This comes before any optimization. Three kernels now depend on barriers that
have never been machine-checked, and two row-wise operators reuse a single
shared scratch buffer across two reductions
([foundations.md §5.5](foundations.md#55-why-this-is-the-scariest-code-in-the-library)).
A race there produces plausible wrong answers, not crashes. Optimizing on top of
an unverified barrier means any later bug has two candidate causes instead of
one.

**Done when:** both tools clean on all operators, output committed.

### Phase 8 — Establish the measurement baseline

Run the existing benchmark harness across a full shape sweep for every operator
and variant. Commit the CSVs. Compute achieved bandwidth and FLOP/s against the
device ceilings.

**Done when:** `results/` holds a per-operator sweep, and every operator has a
recorded distance from its roofline limit.

### Phase 9 — Matmul: register tiling

The measured bottleneck. Currently **8.6% of fp32 peak** against cuBLAS's
**70.7%** ([phase 6](results/phase-6-case-study.md)).

The reasoning is in [foundations.md §4.2](foundations.md#42-worked-example-why-our-matmul-is-slow):
our 16×16 shared-memory tiling gives an arithmetic intensity of 4 FLOP/byte
against a T4 ridge point of ~25, which caps it near 16% of peak no matter how
well it is written. Having each thread compute a 4×4 patch of C raises intensity
to ~16 FLOP/byte.

**Target: a substantial multiple of current matmul throughput, from raising
arithmetic intensity.** Stated as a hypothesis with a mechanism, not a promise —
it is believed when measured. If it does not materialise, that is a finding and
goes in the optimization log.

**Done when:** `REGISTER_TILED` passes every existing correctness test, is
measured across the sweep, and the result — whatever it is — is recorded.

### Phase 10 — Row-wise operators: fewer passes

The case study found PyTorch moving *less data* than we do for softmax and
layernorm — our implied bandwidth exceeded the ceiling under our own traffic
assumption, which means the assumption was wrong for PyTorch.

Our softmax makes three passes over each row. A single-pass online algorithm
reads it once. Layernorm's two-pass structure is required for numerical reasons
([foundations.md §6.4](foundations.md#64-catastrophic-cancellation)) but Welford
achieves the same stability in one pass.

This is also where the open question from phase 2 gets answered: **does
layernorm's fp64 accumulation actually cost anything?** It was chosen for a
documented correctness reason with the cost explicitly deferred to measurement.

**Done when:** `ONLINE` softmax and `WELFORD` layernorm exist, are correct, and
are measured against the current variants.

### Phase 11 — The target hardware

Everything so far is Turing (sm_75). The stated target is Ada (sm_89). Ada has
async shared-memory copy (`cp.async`), a different shared/L1 split, and different
occupancy limits — so tile and block sizes chosen on a T4 do not transfer.

Requires either the Windows SDK installed locally, or Modal with an L4.

**Done when:** the full suite and sweep have run on sm_89 and the numbers are
recorded separately from the Turing ones.

### Phase 12 — Sponsor alignment and tuning

With real target shapes, revisit tile sizes, block sizes, and the AUTO dispatch
thresholds — measured, not guessed. This is the first point at which any block
size in this library will have been chosen by measurement rather than reasoning.

### Phase 13 — Final report

Assemble the case study, sweeps, and optimization log into the project report.

---

## 7. What "complete" means

A checklist that can be audited rather than asserted.

**Correctness**
- [ ] All three operators pass against two independent references
- [ ] `compute-sanitizer` memcheck and racecheck clean, output committed
- [ ] Edge cases covered: size 1, non-power-of-two, extreme magnitudes, degenerate rows
- [ ] Tolerance policy justified by measured error, per operator
- [ ] Every correctness claim reproducible with one command

**Performance**
- [ ] Every operator and variant benchmarked across a shape sweep
- [ ] CUDA events, warmup discarded, spread reported
- [ ] Compared against PyTorch/cuBLAS on named hardware
- [ ] Achieved throughput stated as a fraction of the device's roofline limit
- [ ] Every speedup traceable to a specific change in the optimization log
- [ ] Failed optimizations recorded too

**Engineering**
- [ ] One command builds; one command tests
- [ ] `find_package(ccml)` works from a separate project
- [ ] Zero warnings under `-Wall -Wextra -Werror`
- [ ] No duplicated CUDA utility code
- [ ] Linux and Windows both verified
- [ ] Full git history

**Documentation**
- [ ] Foundations document — every concept explained
- [ ] Design document — every choice justified
- [ ] Per-phase results with verbatim output
- [ ] Reproducible case study
- [ ] An explicit list of what the project does **not** claim

---

## 8. Performance: what we will and will not say

This section exists because it is the easiest place for a project like this to
become dishonest.

**Targets, clearly labelled as targets:**

| Operator | Now (measured, T4) | Target | Mechanism |
|---|---|---|---|
| matmul | 8.6% of fp32 peak | materially closer to the ~63% the roofline allows at 16 FLOP/byte | register tiling |
| layernorm | ~18% of bandwidth ceiling | a substantially larger fraction | fewer passes, Welford |
| softmax | ~31% of bandwidth ceiling | a substantially larger fraction | single-pass online |

Deliberately no precise target percentages. A number invented now would be a
guess dressed as a goal, and the measurement is what decides.

**What will never be claimed:**

- That this library beats cuBLAS or cuDNN, unless measured and reproducible.
- That any kernel is "optimized" without a before-and-after on named hardware.
- A speedup on one shape presented as a general result.
- Timings without the conditions: GPU, driver, toolkit, build type, what else
  was running.
- That correctness on Turing implies correctness on Ada.

**The standard already in force, and staying:** every number in this repository
comes from a committed, rerunnable script, and the phase document that reports a
result also lists what that result does **not** establish.

---

## 9. Risks and open questions

| Risk | Impact | Response |
|---|---|---|
| Local machine still cannot build | Slows every cycle | Remote builds work; the Windows SDK install is a one-off fix |
| Free-tier GPU is Turing, not Ada | Tuning does not transfer | Correctness on Turing; use Modal L4 for anything performance-related |
| Sponsor shapes never arrive | Tuning targets stay generic | Tune for the case-study shapes and state the assumption |
| Register tiling underdelivers | A phase yields no speedup | Record it as a finding; the roofline analysis stays valid either way |
| `compute-sanitizer` finds a real race | Rework needed | Better found now than after three more kernels depend on it |
| Scope creep into training/autograd | Nothing finished properly | Out-of-scope list in §2 is the defence |

**Genuinely open questions**, all resolvable by measurement:

1. Does layernorm's fp64 accumulation cost measurable time? *(open since phase 2)*
2. What arithmetic intensity does register tiling actually achieve, against the
   predicted 16 FLOP/byte?
3. Is a warp-per-row softmax worth building, and below what row width?
4. Where should `AUTO` switch between variants?
5. How much of the naive-vs-tiled gap is L2 caching rather than shared memory?

---

## 10. In one paragraph

On completion this is a small, honest, well-tested CUDA library: three operators,
several measured kernel variants each, a stable C ABI, a zero-copy PyTorch
interface, and a build that works with one command on two platforms. Its
distinguishing feature is not speed — it will not beat cuBLAS — but that every
claim it makes can be reproduced from the repository, every design decision is
written down with its reasoning, and the things it got wrong are recorded next to
the things it got right. A reader should be able to start from
[foundations.md](foundations.md) knowing no CUDA, and finish understanding both
how these kernels work and exactly how fast they are not.
