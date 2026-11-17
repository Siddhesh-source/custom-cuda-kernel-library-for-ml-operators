# Phase results

One document per completed phase, recording what was actually observed rather
than what was intended.

## The rule

**Every phase ends with two things: tests under `tests/`, and a results document
here.** A phase is not finished when the code is written — it is finished when
the tests exist, they have been run on real hardware, and the output is recorded
in this directory.

## Index

| Phase | Document | Outcome |
|---|---|---|
| 1 — Common CUDA infrastructure | [phase-1-infrastructure.md](phase-1-infrastructure.md) | 35/35 passed on a Tesla T4 |
| 2 — Layer normalization | [phase-2-layernorm.md](phase-2-layernorm.md) | 14/14 over 26 shapes; worst relative error 5.96e-08 |
| 3 — Softmax | [phase-3-softmax.md](phase-3-softmax.md) | 13/13 over 28 shapes; worst absolute error 4.40e-08 |
| 4 — Library consolidation | [phase-4-library-refactor.md](phase-4-library-refactor.md) | 64/64 passed; duplication removed, one-command build and test |
| 5 — PyTorch interface | [phase-5-pytorch-interface.md](phase-5-pytorch-interface.md) | 31/31 Python tests; zero-copy CUDA tensors on torch's stream |
| 6 — Matmul + case study | [phase-6-case-study.md](phase-6-case-study.md) | All 3 operators correct; **2-11x slower than PyTorch**; matmul at 8.6% of fp32 peak |
| 7 — Measurement infrastructure | [phase-7-measurement-infrastructure.md](phase-7-measurement-infrastructure.md) | Spec harness + all baselines on T4; sanitizer clean; 9 measurement defects found and fixed; baseline: ours 0.17-0.31x of best |

## Provenance note

Phases 1-6 cite versions of the Kaggle kernel `isiddhesh3011/ccml-build`. That kernel was deleted on 2026-11-17: Kaggle has no cancel command, and deleting it was the only way to stop an obsolete run. The verbatim output of every cited version is preserved in these documents. From Phase 7 on, each job runs as its own kernel (`ccml-sanitize`, `ccml-matmul`, `ccml-layernorm`, `ccml-softmax`), and every CSV row records the commit it measured.

## What a results document must contain

1. **Environment** — GPU, compute capability, driver, CUDA toolkit, compiler,
   CMake, generator, and where it ran. A result without its conditions is not
   reproducible.
2. **What was added** — the tests written in this phase, by suite.
3. **Verbatim output** — pasted from the run, not retyped or summarised into
   prose. Numbers must be copied, never recalled.
4. **What this proves, and what it does not.** The second half is the one that
   matters: it is what stops a later reader assuming more than was measured.
5. **Issues found and how they were resolved**, including problems in the plan
   rather than only in the code.
6. **How to reproduce it** — the exact command.

## Rules for these documents

- **Never invent a number.** If a figure is not in the run output, it does not
  go in the document. An absent measurement is written as absent.
- **Distinguish PASS from SKIP.** A skipped test is not evidence of anything.
  Totals always report both.
- **No performance claims** without a committed, rerunnable script and its
  output. Timing taken on borrowed hardware records which hardware.
- **Record the architecture.** Results from Turing (Kaggle T4) do not transfer
  to any other architecture; saying so in each document keeps that from being forgotten.
