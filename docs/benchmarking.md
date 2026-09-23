# Benchmarking

Status: **skeleton**. The timing harness exists. **No performance measurement
has been taken, and no performance claim is supported by anything in this
repository.**

Read that again before quoting anything from this project in a report.

## What exists

`bench/` builds three executables that emit CSV on stdout:

    build/default/bin/ccml_bench_matmul
    build/default/bin/ccml_bench_layernorm
    build/default/bin/ccml_bench_softmax

Each prints a header recording the device, library version and build stamp, then
one row per (operator, algorithm, shape):

    op,algo,shape,samples,mean_ms,min_ms,max_ms,stddev_ms

Rows with an empty `mean_ms` mean the work did not run - almost always because
the kernel is not implemented. A missing timing is reported as missing rather
than as zero, since a zero would read as an extraordinarily fast kernel.

## Method

**CUDA events, not a host clock.** On the Windows WDDM driver model the OS
batches command submission, so a host-side wall clock around a launch can report
almost anything. Event timestamps are recorded in the stream itself.

**Warmup is discarded.** The first launch pays module load and JIT, and the GPU
clocks are still ramping. Ten warmup iterations, then fifty samples.

**Spread is reported, not just the mean.** `min`, `max` and `stddev` are in every
row. A mean without a spread hides thermal throttling, which a 60 W laptop GPU
does under sustained load.

## What is deliberately absent

**Derived figures.** No GFLOP/s, no achieved bandwidth, no ratios. Those are
analysis and belong in this document next to the conditions they were measured
under. Emitting them from the harness makes it far too easy for a number with no
provenance to end up in a report.

**cuBLAS and cuDNN comparison.** Out of scope at this milestone. cuBLAS is
available in the local toolkit; cuDNN is not installed. When that work happens it
goes in `bench/`, never in `src/` - the library itself must keep depending only
on the CUDA runtime.

**Autotuning.** Not started. Tile dimensions and block sizes in `src/` are
provisional defaults, documented as such in `design.md`.

## Constraints on this machine

The development GPU has 4 GB of VRAM with roughly 2.9 GB free in practice, and
the desktop compositor is sharing it. Consequences:

- Benchmark shapes must stay small. The current largest matmul shape is
  1024x1024x1024, which is about 12 MB of tensors.
- Another application taking VRAM changes the result. Record what else was
  running.
- The GPU is in WDDM mode, so it is also driving a display. Measurements taken
  while the desktop is busy are not comparable to ones taken while it is idle.

## Recording a run

When results are eventually taken, save the CSV under `docs/results/` with the
date and machine in the filename, and note in this document:

- driver version and CUDA toolkit version
- build type and the exact CMake preset
- what else was using the GPU
- GPU temperature and clock behaviour over the run, if throttling is suspected

A benchmark number without those conditions attached is not reproducible and
should not be reported.
