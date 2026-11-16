# Concepts, part 2: measuring, trusting and speeding up GPU kernels

[foundations.md](foundations.md) explains how a GPU kernel works. This
document explains what we learned **building on top of that** in Phases 7–9:
how to time a kernel without fooling yourself, how to know an answer is
right, what changes with 16-bit numbers, why one matmul trick gave a 4×
speedup, and the practical lessons that cost us time.

Every section answers three questions: **what** the idea is, **why** it
matters, and **how** this project does it. Every number is a real measurement,
with the GPU it came from named.

---

## Contents

1. [The one-paragraph picture](#1-the-one-paragraph-picture)
2. [Timing a kernel honestly](#2-timing-a-kernel-honestly)
3. ["Percent of peak": what are we a percent of?](#3-percent-of-peak-what-are-we-a-percent-of)
4. [Knowing the answer is right](#4-knowing-the-answer-is-right)
5. [Smaller numbers: FP16 and BF16](#5-smaller-numbers-fp16-and-bf16)
6. [Why matmul got 4× faster: register blocking](#6-why-matmul-got-4-faster-register-blocking)
7. [Reading a profiler](#7-reading-a-profiler)
8. [Practical lessons: the things that cost us time](#8-practical-lessons-the-things-that-cost-us-time)
9. [Cheat sheet](#9-cheat-sheet)

---

## 1. The one-paragraph picture

We are building fast versions of three operations (**matmul**, **layernorm**
and **softmax**) and comparing them against NVIDIA's own libraries. A speed
claim is only worth something if the answer is correct, the timing is fair, and
we can say how close we are to what the hardware can do. So, before making
anything faster, we built a **measuring instrument** (Phase 7) and taught the
kernels **three number formats** (Phase 8). Then we made the first real
optimization (Phase 9) and used a profiler to check that it worked for the
reason we predicted.

> **The rule behind everything:** a fast wrong answer is worth nothing, and an
> unfair timing is worse than none, because it misleads.

---

## 2. Timing a kernel honestly

### What

Running a kernel many times, measuring each run, and summarizing the results
in a way that is stable and comparable.

### Why it is harder than it looks

Imagine timing a sprinter with a stopwatch that you start when you *shout* "go"
and stop when you *hear* the finish. You would mostly be measuring your own
reaction time. Timing a GPU from the CPU has the same flaw: the CPU only
*queues* the work, and the GPU runs it later.

```
CPU:  [queue kernel]--[queue kernel]--[queue kernel]--> returns immediately
GPU:            [....kernel....][....kernel....][....kernel....]
                ^ the real work happens here, later
```

### How we do it, and why each piece is there

| Technique | Plain explanation | Why |
|---|---|---|
| **CUDA events** | Timestamps written by the GPU itself, in its own queue | Measures the GPU, not the CPU's view of it |
| **Warmup (20 runs, thrown away)** | Run it a few times before measuring | The first run loads code and the GPU is still waking up |
| **100–200 timed runs** | Many measurements, not one | One run is an anecdote; 200 is data |
| **Median, not mean** | The middle value | One slow outlier (the OS doing something) cannot drag it |
| **p10 / p90 / IQR** | How spread out the runs are | A tight spread means the number is trustworthy |
| **Bootstrap 95% CI** | Resample the runs 2000 times to see how much the median could wobble | Turns "A is faster than B" into a statistical statement |
| **Interleaving A, B, A, B** | Alternate between candidates instead of all of A, then all of B | If the GPU warms up and slows over time, both suffer equally |
| **Cold and warm cache** | Measure with the L2 cache flushed, and with it full | Small inputs fit in cache and look unrealistically fast |
| **Batched cross-check** | Also time 200 runs in one block (the spec's own pattern) | A second method that should agree with the first |

**Where:** `bench/support/timer.cpp` (C++), `scripts/benchlib.py` (Python).
Both are tested against the same expected values, so the two languages cannot
drift apart.

### The trap we fell into: a GPU that was asleep

An idle GPU runs slowly to save power. The T4 idles at **300 MHz** and needs
several hundred milliseconds to reach full speed. Our first "peak speed" test
warmed up for only three runs, about 20 ms, so it measured a GPU that was still
waking up:

| | Measured |
|---|---|
| Our "peak FP32" test (3-run warmup) | **3.2 TFLOPS** |
| cuBLAS, on a real matmul, in the same session | **6.1 TFLOPS** |

A real program cannot beat the hardware's peak. When it appears to, the
*peak* is wrong. The fix was to warm up for **1.5 seconds of GPU time**, and to
record the clock speed during the measurement so this can't hide again.

> **Lesson:** if a result beats a limit, suspect the limit first.

---

## 3. "Percent of peak": what are we a percent of?

### What

Speed expressed as a fraction of the most the hardware can do. "We take 0.8
ms" means little on its own; "we reach 40% of what this GPU can compute" is
meaningful on any GPU.

### Why the spec insists on it

A speedup over a slow baseline can look great while still being far from
good. Percent of peak tells you how much room is left.

### How: two kinds of limit

Every kernel is limited by one of two things:

```
             how much math?                    how much data moved?
   matmul ─────────────────────►  COMPUTE      layernorm, softmax ──►  MEMORY
   (each number is reused many times)          (each number is used ~once)
```

- **Compute-bound** (matmul): measured in **TFLOPS**, trillions of
  floating-point operations per second. A matmul of size M×N×K does
  `2·M·N·K` of them.
- **Memory-bound** (layernorm, softmax): measured in **GB/s**. We count
  only the bytes that *must* move (read the input once, write the output once),
  so the percentage can never be inflated by our own inefficiency.

### Two peaks, side by side

| | How it is obtained | Example |
|---|---|---|
| **Theoretical peak** | From the spec sheet: cores × clock × 2 | T4: 8.14 TFLOPS, 320 GB/s |
| **Measured peak** | A tiny ideal program that does nothing but math (or copying) | Laptop: 8.71 TFLOPS, 115 GB/s |

We report both: the theoretical peak is the hard ceiling, and the measured one
is what is actually reachable.

### The second trap: a spec sheet that undersold the hardware

On the laptop GPU, the measured peak (8.73 TFLOPS) came out **above** the
theoretical one (7.90). The cause: CUDA reports a clock of 2205 MHz, but
laptop GPUs boost far higher (2550 MHz observed, a 3105 MHz ceiling). Using
the low clock would make every "percent of peak" look better than it is. We
now use the GPU's real maximum boost clock (read through NVML, NVIDIA's
monitoring library) whenever it is higher.

> **Lesson:** the denominator matters as much as the numerator. Get it wrong
> and every percentage lies.

---

## 4. Knowing the answer is right

### What

Comparing each kernel's output against a **reference**: a slow, simple,
obviously-correct version computed in **double precision** (64-bit), which is
far more precise than the 32-bit or 16-bit kernels being tested.

### Why

A GPU kernel can be fast and subtly wrong, and wrong answers often *look*
plausible. So:

- **Correctness is checked before timing, in the same run.** A kernel that
  fails is never timed. This is enforced in code (the benchmark writes no
  timing for a failed check), not left to discipline.
- There are **two independent references**: our own CPU version, and PyTorch
  in float64. If one of them has a bug, the other catches it.

### How "close enough" is decided

Floating-point math is never exact, so we allow a **tolerance**:

```
  |our answer - reference|  ≤  absolute_part  +  relative_part × |reference|
```

Think of it like a scale at a shop: "within 1 gram, or within 1%, whichever
is larger." The absolute part protects values near zero; the relative part
protects large values.

Each operation needs its own tolerance because its errors behave differently:

| Operation | Tolerance model | Why |
|---|---|---|
| matmul | Absolute part scaled by the typical size of the result | Sums of positive and negative terms can cancel to near zero, where a relative test is meaningless |
| layernorm | Grows with √(row length) | Error accumulates gently over a sum |
| softmax | **Purely relative** | Explained below |

### A bug we found in our own tests

Softmax turns a row of numbers into probabilities that add up to 1. In a row
of 32,000 numbers, each output is tiny, around **0.00003**. Our original
tolerance allowed an absolute error of about **0.00017**. That is larger than
the outputs themselves:

```
  allowed error  0.00017  ████████████████████
  typical output 0.00003  ███
```

A kernel that wrote **all zeros** would have passed. Every source of error in
softmax is *relative* to each output, so the tolerance is now relative too.
All the existing kernels still passed the stricter test, so nothing was hiding
behind the loose one, but now we know that for sure.

> **Lesson:** test your tests. A check that cannot fail proves nothing.

### Small habits that catch real bugs

- **Poison the output first.** The output buffer is filled with garbage (NaN)
  before the kernel runs. If the kernel forgets to write an element, the
  garbage shows up instead of a lucky leftover value.
- **Awkward sizes.** Tests use sizes like 131 × 67 × 97, where nothing is a
  multiple of any tile size, plus sizes of exactly 1. Bugs live at edges.
- **compute-sanitizer.** NVIDIA's tool for finding out-of-bounds memory access
  and race conditions (two threads touching the same memory without
  coordination). These produce *wrong-but-plausible* answers rather than
  crashes, so the tool runs on every test suite.

---

## 5. Smaller numbers: FP16 and BF16

### What

Two 16-bit number formats, half the size of the usual 32-bit float:

```
FP32  [s][eeeeeeee][mmmmmmmmmmmmmmmmmmmmmmm]   8 exponent bits, 23 mantissa bits
FP16  [s][eeeee][mmmmmmmmmm]                   5 exponent bits, 10 mantissa bits
BF16  [s][eeeeeeee][mmmmmmm]                   8 exponent bits,  7 mantissa bits
```

- The **exponent** decides the *range* (how big or small a number can be).
- The **mantissa** decides the *precision* (how many digits are correct).

| | Largest value | Precision (unit roundoff) | Character |
|---|---|---|---|
| FP32 | ~3.4 × 10³⁸ | ~6 × 10⁻⁸ | The reference standard |
| FP16 | **65,504** | ~5 × 10⁻⁴ | Precise, but small range |
| BF16 | ~3.4 × 10³⁸ | ~4 × 10⁻³ | FP32's range, less precision |

### Why use them

Half the bytes means half the memory traffic, and on the right hardware much
faster math (Tensor Cores, Phase 11). Real models run in these formats.

### How we support them: widen, compute, round once

```
  16-bit in memory ──load──► widen to 32-bit ──► all math in 32-bit (or 64) ──► round once ──► 16-bit out
```

The kernels never do arithmetic in 16 bits. So a 16-bit result differs from
the 32-bit one **only by the final rounding**, and that is exactly what the
16-bit tolerance allows: twice the format's unit roundoff, on top of the
32-bit tolerance. The measured errors confirm it. BF16 errors sit at about
3.9 × 10⁻³, which is its unit roundoff and nothing else.

Two details that matter:

- **Fair inputs.** Test inputs are rounded to 16 bits *first*, and the
  reference is computed from those rounded values. Otherwise we would be
  blaming the kernel for the input's rounding.
- **Round once, not twice.** Layernorm computes in 64-bit. Converting
  64 → 32 → 16 bits rounds twice, which is occasionally one step off. We convert
  64 → 16 directly.

### What behaves differently (and is tested)

| Situation | Result | Same as PyTorch? |
|---|---|---|
| FP16 result above 65,504 | Becomes **infinity** (300 × 300 = 90,000 overflows) | Yes |
| Same product in BF16 | Stays finite (BF16 has FP32's range) | Yes |
| Softmax row with some −∞ entries ("masked") | Exactly **0** there; the rest sums to 1 | Yes |
| Softmax row that is *entirely* −∞ | **NaN** everywhere (0 ÷ 0) | Yes |
| A NaN anywhere in a row | The whole row becomes NaN | Yes |

### Where the format needs special hardware

BF16 *storage* works on every GPU, since it is just bit shuffling. BF16
*Tensor Core math* needs compute capability 8.0+. The T4 is 7.5, so that one
piece runs on the local GPU ([local.md](local.md)).

---

## 6. Why matmul got 4× faster: register blocking

This is the heart of Phase 9, and the most important idea in GPU performance.

### The problem, as an analogy

You are cooking, and every ingredient is in a store across town (**global
memory**). Fetching is slow. Two improvements are possible:

1. **Fridge** (*shared memory*): bring a batch of ingredients home and share
   them with your housemates. That was the old `tiled` kernel.
2. **Countertop** (*registers*): take a handful out of the fridge and cook
   *many dishes* from it before going back. That is **register blocking**.

### The idea in one picture

The old kernel: each thread computes **one** output, so for every
multiply-add it fetches two values from the fridge.

The new kernel: each thread computes an **8 × 8 patch** of outputs. For each
step it fetches 8 values of A and 8 values of B (16 fetches), then does every
pairing between them, which is 64 multiply-adds:

```
        b0 b1 b2 b3 b4 b5 b6 b7      ← 8 values of B (in registers)
   a0 [ ×  ×  ×  ×  ×  ×  ×  × ]
   a1 [ ×  ×  ×  ×  ×  ×  ×  × ]
   ...                              64 multiply-adds from just 16 fetches
   a7 [ ×  ×  ×  ×  ×  ×  ×  × ]
   ↑ 8 values of A (in registers)
```

| | Fetches per multiply-add (from shared memory) |
|---|---|
| Old `tiled` | 2 |
| New `register_tiled` | 16 / 64 = **0.25**, 8× fewer |

### The deeper reason: arithmetic intensity

**Arithmetic intensity** = math done ÷ bytes fetched. Every GPU has a
**ridge point**, the intensity at which it stops waiting for data and starts
waiting for math:

```
 speed
   │                 ┌────────────── compute ceiling (8.1 TFLOPS on T4)
   │               ╱
   │             ╱  ← memory ceiling (320 GB/s)
   │           ╱
   │         ╱  ●  old tiled: 4 FLOP/byte (stuck on the slope)
   │       ╱
   │     ╱              ● register_tiled: 32 FLOP/byte (past the ridge at ~25)
   └───────────────────────────────── arithmetic intensity
```

The old kernel sat on the slope, limited by memory no matter how well it was
written. The new one is past the ridge. This picture is called a **roofline**.

### What we predicted, and what happened

We wrote the prediction down and committed it **before** writing the code
([optimization-log.md](optimization-log.md)). That stops us from adjusting the
story to fit the result afterwards.

Measured on the local laptop GPU (indicative; the T4 run is the official one):

| Shape (M × N × K) | Old `tiled` | New `register_tiled` | Prediction |
|---|---|---|---|
| 2048 × 4096 × 1024 | 23.5 ms | **4.96 ms (4.75× faster), 40% of peak** | 3–6× ✅ |
| 512 × 3072 × 768 | 3.27 ms | **0.76 ms (4.3×), 36% of peak** | 3–6× ✅ |
| 16 × 2304 × 768 | 90 µs | 172 µs (**2× slower**) | "no gain for small M" ✅ |

### Why bigger tiles hurt small inputs

During text generation a model processes **one token at a time**, so M is 1 or
16. A 128-row tile would then be more than 87% empty: like booking a 128-seat
bus for 16 passengers. The tile sweep (six tile sizes, all tested) showed a
**32 × 64** tile winning there. So no single tile is best, and the library will
need to pick the tile per shape. That's a decision made from data, not
guesswork.

### Nothing is free: registers and occupancy

Keeping 64 results plus 16 inputs per thread uses **126 registers** (the old
kernel used 37). Registers are a fixed budget per SM, so fewer threads can
be active at once:

| | Registers per thread | Share of warp slots in use ("occupancy") |
|---|---|---|
| tiled | 37 | 99% |
| register_tiled | 126 | **31%** |

Fewer active threads means fewer chances to hide waiting time. The kernel
still won by more than 4×, but this is now the limit. Optimization is a chain
of bottlenecks: remove one, and the next becomes visible.

---

## 7. Reading a profiler

### What

**Nsight Compute** (`ncu`) runs a kernel under hardware counters and reports
what the GPU was actually doing, so a story like "it's faster because of X"
can be checked rather than assumed.

### The handful of metrics that matter here

| Metric | Plain meaning | What a bad value suggests |
|---|---|---|
| **Duration** | How long the kernel took | — |
| **Warps active %** (occupancy) | How full the GPU's thread slots were | Too few threads to hide delays |
| **Stall: MIO throttle** | Threads waiting in line for shared memory | Too many shared-memory accesses |
| **Stall: barrier** | Threads waiting at `__syncthreads()` for slower teammates | Uneven work, or too-frequent syncs |
| **Stall: long scoreboard** | Waiting for global memory | Memory-bound |
| **Shared-load wavefronts** | How many trips shared memory needed | More trips than expected means bank conflicts |
| **Bank conflicts** | Several threads hit the same memory bank at once and had to queue | Bad access pattern |

### What it told us about register blocking

| | tiled | register_tiled | Reading |
|---|---|---|---|
| Stall: MIO throttle | **18.2** | 0.33 | The predicted bottleneck is gone ✅ |
| Shared-memory trips | 56.6 M | 11.8 M | 4.8× fewer, not the predicted 8× |
| Bank conflicts | 0 | **4.7 M** | ...because of this |

**Bank conflicts, simply:** shared memory is split into 32 "checkout lanes"
(banks). If 32 threads each use a different lane, they are served in one go.
Our B-operand reads land 16 threads on only 4 lanes, so they queue four deep.
That is precisely what Phase 10's vectorized loads fix. The profiler found the
next optimization for us.

### Practical notes

- **Counters need permission.** Windows blocks them for normal users until a
  setting changes and the machine reboots. Until then, `ncu` works from an
  administrator prompt. Kaggle blocks them completely, so profiling runs on the
  local GPU.
- **Keep reports small.** A full report was 12.8 MB; collecting only the
  sections we need gave 1.4 MB, which keeps the repository usable.
- **Timings come from the T4; profiles come from the laptop.** Each document
  names both GPUs and never mixes their numbers into one figure.

---

## 8. Practical lessons: the things that cost us time

These are not about GPUs. They are about working reliably.

| What went wrong | Why | Fix now in place |
|---|---|---|
| A cloud job sat **queued for 45+ minutes** | We demanded one specific GPU type (T4) and none were free | `--any-gpu` drops the requirement; every result is filed by the GPU it actually ran on |
| Only **2 cloud jobs** could run at once; extras were *rejected*, not queued | Account limit on Kaggle | The launcher holds jobs back and submits them as slots free |
| One job **ran the wrong code** | The launcher packed files when it submitted, after we had started editing another branch | Code is now snapshotted **once, at launch**, for every job |
| Shell scripts failed with `$'\r': command not found` | Windows line endings (CRLF) in files run on Linux | `.gitattributes` forces LF everywhere; the packer also strips CRLF |
| Python benchmarks found **no PyTorch** | The cloud image's `python3` was a different install from the one with PyTorch | The job uses the exact interpreter the image runs |
| **No clock or temperature data** in the cloud | The monitoring library had a different file name there | The build searches for the driver's own copy |
| A run had to be **cancelled** but there is no cancel command | Kaggle's API limitation | Deleting the job is the only way. History is kept in our docs, not on Kaggle |

### How work is organized

```
  task branch (e.g. matmul-ladder) ──merge──►  dev  ──(once, at the end)──►  main
```

Branches are named for the **task**, not a phase number. Each phase ends with
tests, a results document and an updated roadmap. Commits carry the author's
identity only.

---

## 9. Cheat sheet

| Idea | One line |
|---|---|
| CUDA events | Let the GPU timestamp itself |
| Warmup | A cold GPU is a slow GPU; T4 needs ~1 s to wake |
| Median + CI | Robust middle, plus how sure we are |
| Interleave | A B A B, so drift hits everyone equally |
| Cold vs warm L2 | Small inputs fit in cache; measure both |
| % of peak | Report the distance to the ceiling, not just the speedup |
| Compute vs memory bound | Matmul is limited by math, softmax and layernorm by data movement |
| Arithmetic intensity | Math per byte; past the ridge point you're compute-bound |
| Correctness gate | Never time a wrong kernel; enforce it in code |
| Tolerance | Absolute + relative, chosen per operation; test that it *can* fail |
| FP16 vs BF16 | FP16: precise, small range. BF16: big range, less precise |
| Widen, compute, round once | 16-bit in memory, 32-bit math, one rounding at the end |
| Register blocking | Each thread computes an 8×8 patch: 8× less shared-memory traffic |
| Occupancy | More registers per thread means fewer threads to hide delays |
| Bank conflicts | Threads queuing for the same shared-memory lane |
| Profile, don't guess | NCU confirmed the win *and* found the next bottleneck |
| Predict first | Commit the hypothesis before the code; record failures too |

---

**Where to go next:** [optimization-log.md](optimization-log.md) for the full
evidence, [benchmarking.md](benchmarking.md) for the exact method,
[correctness.md](correctness.md) for every tolerance and measured error, and
[ROADMAP.md](ROADMAP.md) for what comes next.
