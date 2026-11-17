# Superseded: first full Phase 7 benchmark run (commit d8c676b, 2026-11-17)

Kept for the record; **not** used for any result. Three measurement defects
were found in these files and fixed in the harness before the run that
replaced them (docs/results/phase-7-measurement-infrastructure.md):

1. **Launch-gap bias.** Per-iteration timings of kernels shorter than their
   submission cost included the time the GPU waited for the CPU. Python medians
   ran up to 2.1x the batched mean below 30 us, which made small-shape
   comparisons unfair to the PyTorch, torch.compile and Triton baselines.
2. **Idle-clock telemetry.** Clocks were sampled after each timing loop, when
   the GPU had already dropped to idle (585 MHz rows with no throttle flag).
3. **Triton softmax NaN** for rows narrower than its block. The correctness gate
   failed all 9 affected rows, so none of them was timed.

The sanitizer, ctest and pytest outputs from the same run remain valid: the
library code did not change. They are in `results/sm_75/` above this folder.
