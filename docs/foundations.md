# Foundations

Every concept this project uses, built up from nothing, with small examples.

This is written for someone who knows C++ but has never written a GPU kernel.
Each idea is explained plainly, shown with a tiny example, and then tied to
somewhere it actually appears in this repository — a file, or a number we
measured. The point is not to teach CUDA in the abstract; it is to explain why
this library looks the way it does.

**Contents**

1. [Why a GPU at all](#1-why-a-gpu-at-all)
2. [The execution model: threads, warps, blocks, grids](#2-the-execution-model)
3. [The memory hierarchy](#3-the-memory-hierarchy)
4. [What actually makes a kernel fast](#4-what-actually-makes-a-kernel-fast)
5. [Cooperation: barriers, races, reductions](#5-cooperation-barriers-races-reductions)
6. [Floating point, and why it bites](#6-floating-point-and-why-it-bites)
7. [The three operators](#7-the-three-operators)
8. [Tiling, and why it is the whole game](#8-tiling-and-why-it-is-the-whole-game)
9. [Asynchrony, streams, and honest timing](#9-asynchrony-streams-and-honest-timing)
10. [How we know the answers are right](#10-how-we-know-the-answers-are-right)
11. [The software engineering around it](#11-the-software-engineering-around-it)
12. [Glossary](#12-glossary)

---

## 1. Why a GPU at all

A GPU is not a faster CPU. It is a differently shaped machine.

A CPU has a few very clever cores. Each one predicts branches, reorders your
instructions, and keeps large caches, all to finish **one** chain of work as
quickly as possible. A GPU has thousands of simple cores that are individually
much slower, and it is built to finish **an enormous number of identical small
jobs** in the same amount of time.

> **Analogy.** A CPU is four professors. A GPU is ten thousand students. Ask a
> hard, twisty question and the professors win easily. Ask "multiply these ten
> thousand pairs of numbers" and the students are finished before the professors
> have picked up a pen.

This is why machine learning runs on GPUs. Almost everything in a neural network
is the same small operation repeated across a huge array: multiply and add,
multiply and add.

The relevant number for our development GPU (measured with
`ccml_device_query`, recorded in [design.md](design.md)) and the Tesla T4 we run
experiments on:

| Tesla T4 | Value |
|---|---|
| Streaming multiprocessors (SMs) | 40 |
| Max threads resident per SM | 1024 |
| Peak fp32 arithmetic | ~8.1 trillion operations/second |
| Peak memory bandwidth | 320.1 GB/s |

Note the ratio between those last two. The chip can do about 8.1 trillion
floating-point operations per second, but can only fetch about 80 billion
floats per second from memory. **It can compute roughly 100 times faster than it
can be fed.** Almost every performance problem in this project follows from that
one fact. Section 4 makes it precise.

---

## 2. The execution model

### 2.1 Thread

A **thread** is one instance of your function. If you want to add two arrays of
a million elements, you launch a million threads, and each one adds a single
pair. A thread knows only its own index.

```cuda
__global__ void add(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // "which element am I?"
    if (i < n) out[i] = a[i] + b[i];
}
```

`__global__` marks a function that runs on the GPU and is called from the CPU.
That is a **kernel**.

### 2.2 The index formula, and the guard

That one line of index arithmetic appears in every kernel, so it is worth
reading slowly. Threads are grouped into **blocks**. `blockDim.x` is the block
size, `blockIdx.x` is which block you are in, `threadIdx.x` is your position
inside it.

With a block size of 256:

| | `blockIdx.x` | `threadIdx.x` | global index `i` |
|---|---|---|---|
| first thread of block 0 | 0 | 0 | 0 |
| last thread of block 0 | 0 | 255 | 255 |
| first thread of block 1 | 1 | 0 | 256 |
| first thread of block 3 | 3 | 0 | 768 |

The `if (i < n)` guard matters. You can only launch whole blocks. For
`n = 1000` and a block size of 256 you must launch 4 blocks — 1024 threads — so
24 threads have no element to work on. Without the guard they would write past
the end of the array.

**This is the single most common CUDA bug**, and it is why every operator in
this library is tested on deliberately awkward sizes: 517, 1023, 1025, 4097. See
`tests/test_layernorm.cpp` and `tests/test_softmax.cpp`.

### 2.3 Warp

Threads do not really run independently. The hardware runs them in groups of
**32, called a warp**, in lockstep: all 32 execute the same instruction at the
same time.

> **Analogy.** A warp is 32 rowers in one boat. They all pull together. If the
> cox says *"those on the left pull, the rest rest"*, the right side sits idle
> while the left pulls, and then they swap. **Both branches cost time.**

That is **warp divergence**. When threads in the same warp take different paths
through an `if`, the warp executes both paths in sequence, with the irrelevant
threads switched off. Divergence between *different* warps is free — only
divergence *inside* one warp costs.

This is why our kernels are careful to make branches uniform across a block
where they can. In `layernorm_kernels.cuh`:

```cuda
const int row = blockIdx.x;
if (row >= rows) return;   // whole block returns together: no divergence
```

### 2.4 Block

A **block** is a group of threads (up to 1024) that:

* run on **one** SM,
* can **share memory** with each other,
* can **wait for each other** at a barrier.

Threads in different blocks can do none of those things. Blocks are independent
by design, which is what lets the GPU schedule them in any order across any
number of SMs.

Our row-wise operators give **one block to each row** of the input. All the
threads that need to cooperate — to sum a row, or find its maximum — are in the
same block, and rows never need to talk to each other. See
`src/common/rowwise.h`.

### 2.5 Grid, and the launch

The **grid** is all the blocks. You choose both sizes at launch:

```cuda
kernel<<<grid, block>>>(arguments);
```

For 1000 elements with 256 threads per block you need `ceil(1000/256) = 4`
blocks — which is exactly what `div_up` in `src/common/util.h` computes.

Blocks and grids can be 1-, 2-, or 3-dimensional. Matmul uses 2-D because the
output is a matrix: `grid.x` covers columns, `grid.y` covers rows.

### 2.6 Putting the hierarchy together

```
grid  ──┬── block 0 ──┬── warp 0 ── 32 threads
        │             ├── warp 1 ── 32 threads
        │             └── ...       (up to 1024 threads)
        ├── block 1 ── ...
        └── block N ── ...
```

| Level | Size | Can cooperate? | Lives on |
|---|---|---|---|
| Thread | 1 | — | a core |
| Warp | 32 | implicitly, in lockstep | an SM |
| Block | ≤1024 threads | yes: shared memory + barriers | one SM |
| Grid | all blocks | **no** | the whole GPU |

---

## 3. The memory hierarchy

A GPU has several kinds of memory, differing enormously in size and speed.

> **Analogy.** Registers are what you are holding in your hands. Shared memory
> is the shelf at your desk. L2 cache is the stockroom down the corridor. Global
> memory is the warehouse across town. Every step out is roughly an order of
> magnitude slower.

| Memory | Size (Tesla T4) | Speed | Scope |
|---|---|---|---|
| Registers | 64 KB per block | fastest | one thread |
| Shared memory | 48 KB per block | ~10× slower than registers | one block |
| L2 cache | 4 MB | — | whole GPU |
| Global memory | 14.5 GB | ~100× slower than shared | whole GPU, survives launches |

Those numbers are not from a specification sheet — they are what
`ccml_device_query` printed on the machine we ran on, recorded in
[results/phase-1-infrastructure.md](results/phase-1-infrastructure.md).

### 3.1 Global memory and coalescing

Global memory is read in **chunks**, not individual floats. When a warp reads
memory, the hardware wants those 32 reads to fall in one contiguous 128-byte
chunk. Then one transaction serves all 32 threads. This is **coalescing**.

```
Coalesced   — thread i reads a[i]:
  thread:   0  1  2  3 ...  31
  address:  0  4  8 12 ... 124      one 128-byte transaction ✓

Strided     — thread i reads a[i * 32]:
  thread:   0    1    2   ...
  address:  0  128  256   ...       32 separate transactions ✗
```

The strided version moves the same 32 useful floats but pays for 32 chunks — it
fetches 4096 bytes to use 128. **Up to 32× the traffic for the same data.**

This is why the library insists on row-major contiguous layout, and why the
PyTorch binding *rejects* a non-contiguous tensor rather than quietly copying it
(`python/ccml/torch_api.py`): a transposed tensor would silently turn every read
into the bad case.

### 3.2 Shared memory

Shared memory is a small, fast, programmer-managed scratchpad that all threads
in a block can read and write. It exists so a block can load a piece of data
from global memory **once** and then use it many times.

That is the entire idea behind tiling (section 8), and the reason our tiled
matmul is faster than the naive one.

---

## 4. What actually makes a kernel fast

### 4.1 The two ceilings

Every kernel is limited by one of two things:

* **Compute bound** — it is doing as much arithmetic as the chip can.
* **Memory bound** — it is waiting for data.

Which one applies depends on **arithmetic intensity**: how many floating-point
operations you perform per byte you load.

```
arithmetic intensity = FLOPs performed / bytes moved from global memory
```

For the T4:

```
ridge point = peak compute / peak bandwidth
            = 8.1e12 FLOP/s / 320.1e9 byte/s
            ≈ 25 FLOP per byte
```

**If your kernel does fewer than ~25 FLOPs per byte, you are memory bound, and
no amount of arithmetic cleverness will help.** The only thing that helps is
moving fewer bytes.

### 4.2 Worked example: why our matmul is slow

This is not hypothetical — it explains a number we measured.

**Naive matmul.** Each output element `C[i][j]` needs a row of A and a column of
B: `K` multiply-adds (2K FLOPs) reading `2K` floats (8K bytes).

```
intensity = 2K / 8K = 0.25 FLOP per byte
```

0.25 against a ridge of 25. **A hundred times too low.** The kernel spends
essentially all its time waiting for memory.

**Tiled matmul**, our `kMatmulTileDim = 16` version. A block handles a 16×16
patch of C. For each step along K it loads a 16×16 tile of A and one of B —
512 floats, 2048 bytes — and with them performs 16×16×16 multiply-adds, which is
8192 FLOPs.

```
intensity = 8192 / 2048 = 4 FLOP per byte
```

Sixteen times better than naive — and still **6× below the ridge**. So even a
perfect implementation of this tiling can only reach about `4/25 ≈ 16%` of peak
compute.

Now compare with what we actually measured
([phase 6](results/phase-6-case-study.md)):

| | Measured | % of fp32 peak |
|---|---|---|
| Our naive | 486 GFLOP/s | 6.0% |
| Our tiled | 700 GFLOP/s | 8.6% |
| cuBLAS | 5727 GFLOP/s | 70.7% |

We measured 8.6% against a model ceiling of ~16%. **The model predicted the
measurement.** Our kernel is not mysteriously slow; it is doing exactly what an
arithmetic intensity of 4 allows, and the tiling choice is what caps it.

And it tells us what to do next. If each thread computed a 4×4 patch of C
instead of one element (**register tiling**), a block of 256 threads would cover
64×64 outputs. Per K-step it would load 64×16 + 16×64 floats = 8192 bytes and
perform 64×64×16×2 = 131072 FLOPs:

```
intensity = 131072 / 8192 = 16 FLOP per byte
```

Four times higher, and close to the ridge. That is the reasoning — not a guess —
behind "register tiling is the next optimization target". It is a prediction
from the model, not a promise; it will be believed when it is measured.

### 4.3 Occupancy

**Occupancy** is the fraction of the thread slots on an SM that are in use. It
matters because it is how a GPU hides latency: when one warp stalls waiting for
memory, the SM instantly switches to another that is ready. More resident warps
means more chances to find work.

Occupancy is limited by the resources each block consumes — registers and shared
memory. A block using a lot of shared memory means fewer blocks fit per SM.

This is a real trade-off, not free: bigger tiles give better arithmetic
intensity but use more shared memory and lower occupancy. Finding the balance is
what autotuning is for. The roadmap sweeps it in phase 15, and enhancement S4 turns the result into a measured dispatch table.

---

## 5. Cooperation: barriers, races, reductions

### 5.1 The race condition

Threads run at unpredictable times relative to each other. If one thread writes
something another reads, without arranging an order, the result depends on
timing. That is a **data race**, and it usually produces *plausible but wrong*
answers — the worst kind of bug.

```cuda
shared[threadIdx.x] = input[i];       // every thread writes
float x = shared[threadIdx.x + 1];    // and reads its neighbour's
// BUG: the neighbour may not have written yet
```

### 5.2 `__syncthreads()`

The fix is a **barrier**: no thread passes it until every thread in the block
has arrived.

```cuda
shared[threadIdx.x] = input[i];
__syncthreads();                      // everyone has written by here
float x = shared[threadIdx.x + 1];    // now safe
```

**Every thread in the block must reach the barrier.** If some threads return
early and others wait, the block hangs forever. This is why our tiled matmul
zero-fills out-of-range elements instead of letting those threads return:

```cuda
tile_a[ly][lx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
__syncthreads();    // every thread arrives, including the ones with no data
```

Zeros contribute nothing to a sum, so the answer is unaffected — and the block
does not deadlock. See `src/matmul/matmul_kernels.cuh`.

### 5.3 Parallel reduction

A **reduction** turns many values into one: a sum, a maximum. It is the heart of
both layernorm and softmax.

Done sequentially, adding 8 numbers takes 7 steps. Done as a tree, it takes 3:

```
start:   3   1   4   1   5   9   2   6
step 1:  4       5      14       8        (pairs added, half the threads idle)
step 2:  9              22
step 3:  31
```

In general, `log2(n)` steps instead of `n`. For a 1024-element row: 10 steps
instead of 1023.

### 5.4 Warp shuffles

Within a warp there is something better than shared memory. Because the 32
threads run in lockstep, they can pass values **directly between registers**:

```cuda
value += __shfl_down_sync(0xffffffff, value, 16);  // add thread i+16's value
value += __shfl_down_sync(0xffffffff, value, 8);
value += __shfl_down_sync(0xffffffff, value, 4);
value += __shfl_down_sync(0xffffffff, value, 2);
value += __shfl_down_sync(0xffffffff, value, 1);
// thread 0 now holds the sum of all 32
```

Five steps, no shared memory, no barrier.

Our block reduction (`src/common/reduce.cuh`) uses two stages: shuffles within
each warp, then one warp combines the per-warp results through shared memory.
**Two barriers in total**, against one per level for a pure shared-memory tree.

```
256 threads = 8 warps
  stage 1: each warp shuffle-reduces its 32 values   -> 8 partial results
  stage 2: warp 0 reduces those 8                    -> 1 result
```

### 5.5 Why this is the scariest code in the library

A missing barrier does not crash. It produces a number that looks fine. Both
row-wise operators reuse **one** shared scratch buffer across **two** reductions
and depend entirely on the `__syncthreads()` between them.

The tool that catches this is `compute-sanitizer --tool racecheck`. **It has not
been run yet**, and that is recorded as the largest outstanding gap in
[correctness.md](correctness.md). Reading this section should tell you why that
matters more than it sounds.

---

## 6. Floating point, and why it bites

### 6.1 What a float32 is

32 bits: 1 sign, 8 exponent, 23 stored mantissa bits. About **7 decimal digits**
of precision — and that is *relative* precision, the same whether the number is
0.001 or 1000000.

The key constant is **machine epsilon**: the gap between 1.0 and the next
representable number.

```
eps_f32 = 1.1920928955078125e-07
```

So `1.0f + 1e-8f == 1.0f`. The addition is silently discarded.

### 6.2 ULP: the gap between neighbours

**ULP** = "unit in the last place" — the spacing between adjacent floats near
some value. It grows with magnitude:

| Near | Gap to next float |
|---|---|
| 1.0 | 0.00000012 |
| 1000.0 | 0.000061 |
| 10000.0 | 0.00098 |
| 16777216.0 | 2.0 |

That last row is worth pausing on: above 2²⁴, consecutive integers are not all
representable. `16777216.0f + 1.0f == 16777216.0f`.

### 6.3 Accumulated error

Adding `n` numbers in sequence, each addition rounds. Errors accumulate roughly
like a random walk: **∝ √n**.

That √n is exactly why our tolerance policy is
`8 × eps × √terms` ([correctness.md](correctness.md)). It is not a magic number;
it is the shape of the error, with a factor of 8 for margin.

### 6.4 Catastrophic cancellation

This is the one that actually broke something in this project.

Subtracting two nearly equal numbers destroys precision. The leading digits
cancel, and what is left is mostly the rounding error you were already carrying.

**Worked example — variance.** Textbook algebra says:

```
Var(x) = E[x²] - E[x]²
```

Take values around 1000 with a spread of 1:

```
E[x²]  ≈ 1000001
E[x]²  ≈ 1000000
difference = 1          <- the answer we want
```

Both terms are ~10⁶, and float32 holds ~7 digits, so each is known to about
±0.1. The difference of 1 therefore carries **10% error**. Larger means and the
answer becomes pure noise.

The fix is the two-pass method: find the mean first, then sum the squared
deviations about it. Nothing large ever gets subtracted from anything large.

```
mean = average(x)                      pass 1
var  = average((x - mean)²)            pass 2
```

Our layernorm uses exactly this, and
`layernorm.large_offset_small_variance` in `tests/test_layernorm.cpp` exists
solely to catch anyone who "optimizes" it back to one pass.

### 6.5 When even a correct mean is not enough

Layernorm divides by the standard deviation, so **an error in the mean is
amplified by 1/σ**.

Inputs around 10000 with σ = 0.01. A float32 mean is accurate to about half an
ULP at that magnitude: ±0.0005. Divide by σ:

```
0.0005 / 0.01 = 5% error in every output element
```

And that is with a *perfectly rounded* float32 mean. No summation order fixes
it. The only fix is to hold the mean in higher precision — which is why our
layernorm accumulates in **double**, documented in
`src/layernorm/layernorm_kernels.cuh`.

Matmul does **not** need this: a matrix product has no such amplification.
Different operator, different answer.

### 6.6 Overflow and underflow

`exp(x)` in float32 **overflows to infinity above x ≈ 88.7**. A softmax over
values around 1000 would produce `inf` for every large element and then
`inf/inf = NaN` for every output.

The fix is to subtract the row maximum first. The largest shifted value is
exactly 0, so the largest exponential is exactly 1 and nothing can overflow:

```
softmax(x) = exp(x - max) / Σ exp(x - max)
```

Mathematically identical — the max cancels top and bottom. Numerically the
difference between working and not working.

It also guarantees the denominator is never zero: the maximum element always
contributes `exp(0) = 1`, so the sum is at least 1. **There is no
divide-by-zero case to guard.**

Underflow works the other way: `exp(x)` in float32 becomes exactly 0 below
x ≈ −103.3, while float64 keeps going to −744.4. We measured the consequence in
[phase 3](results/phase-3-softmax.md): a relative error of exactly **1.0** for
elements where our float32 kernel correctly returns 0 and the float64 oracle
returns ~1e-45. The kernel was right; the *metric* was the wrong one, which
leads to the next section.

---

## 7. The three operators

### 7.1 Matrix multiplication

```
C[i][j] = Σ over k of A[i][k] × B[k][j]
```

Each output element is a dot product of a row of A with a column of B.

```
A (2×3)        B (3×2)        C (2×2)
[1 2 3]        [7  8]         [1·7+2·9+3·11   1·8+2·10+3·12]   [58  64]
[4 5 6]        [9 10]    =    [4·7+5·9+6·11   4·8+5·10+6·12] = [139 154]
               [11 12]
```

Cost: `M×N×K` multiply-adds. For a 1024³ product that is over a billion, and it
dominates everything in a Transformer. Our case study found a *single*
projection matmul taking longer than PyTorch's entire pipeline.

### 7.2 Layer normalization

For each row independently: centre it, scale it to unit variance, then apply a
learned scale and shift.

```
mean = average(row)
var  = average((row - mean)²)
out  = (row - mean) / sqrt(var + eps) × gamma + beta
```

`eps` is a small constant inside the square root. It does two jobs: it stops
division by zero when a row is constant, and it keeps the result stable when the
variance is tiny.

Worked example, row `[1, 2, 3, 4]`:

```
mean = 2.5
deviations = [-1.5, -0.5, 0.5, 1.5]
var  = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25
sd   = 1.118
out  = [-1.342, -0.447, 0.447, 1.342]
```

Mean 0, variance 1. You can verify the real kernel does this: our
`examples/usage.cpp` normalizes `0..7` and prints `-1.5275 ... 1.5275`, which is
`-3.5 / 2.2913` — checkable by hand.

### 7.3 Softmax

Turn a row of arbitrary numbers into a probability distribution.

```
softmax(x)[i] = exp(x[i]) / Σ exp(x[j])
```

Everything positive, everything summing to 1, larger inputs getting larger
shares.

```
input   [1, 2, 3]
exp     [2.718, 7.389, 20.086]
sum     30.193
output  [0.090, 0.245, 0.665]       sums to 1.000 ✓
```

In practice always computed with the max subtracted (section 6.6). Our kernel
makes three passes over the row: find the max, sum the exponentials, write the
normalized values.

---

## 8. Tiling, and why it is the whole game

Tiling is the single most important optimization technique in GPU computing, so
it is worth understanding properly.

**The problem.** In naive matmul, to compute one row of C you read the whole of
B. The thread next to you, computing the next element of the same row, reads
**the same row of A all over again**. Across a 16×16 block, every element of A is
fetched from global memory 16 times.

**The idea.** Load a tile into shared memory once. Let all 256 threads in the
block use it. Move to the next tile.

```
for each tile along K:
    every thread loads one element of A's tile and one of B's tile   (cooperative)
    __syncthreads()                                                  (all loaded)
    every thread accumulates this tile's contribution from shared    (reuse)
    __syncthreads()                                                  (all done reading)
```

**The payoff.** Each element loaded from global memory is used 16 times instead
of once. Arithmetic intensity goes from 0.25 to 4 FLOP/byte — the calculation in
section 4.2.

**The measured result.** 1.38×–1.55× faster than naive, across shapes
([phase 6](results/phase-6-case-study.md)). That is the only speedup claim this
project makes, and it is between two of its own kernels.

**Why only 1.5×,** when the intensity improved 16×? Because the naive kernel was
not getting zero reuse — the L2 cache was already catching much of it. Tiling
makes the reuse explicit and reliable rather than accidental. And because at 4
FLOP/byte we are still memory bound, so we are still on the wrong side of the
ridge point. Getting past that needs register tiling.

**The correctness catch.** When a dimension is not a multiple of the tile size,
the edge tiles are partly outside the matrix. Those threads must still
participate in the barriers (section 5.2), so they load zeros rather than
returning. This is the part that is easy to get wrong and easy to not notice —
hence `matmul.ragged_dimensions`, with M=67, N=53, K=41.

---

## 9. Asynchrony, streams, and honest timing

### 9.1 Kernel launches are asynchronous

`kernel<<<...>>>()` returns to the CPU almost immediately, before the GPU has
done anything. The launch is queued.

This has two consequences that matter constantly:

**Reading results too early is a race.** You must synchronise before touching
the output:

```cuda
ccml_softmax_f32(y, x, rows, cols, CCML_SOFTMAX_AUTO, stream);
ccml_stream_synchronize(stream);      // now y is valid
```

**Errors arrive late.** `cudaGetLastError()` after a launch only tells you
whether the launch was *accepted* — bad block size, too much shared memory. A
fault *during* execution (an out-of-bounds write) surfaces at the next
synchronisation and gets blamed on whatever call happens to synchronise.

That is why this library has a `CCML_SYNC_LAUNCH_CHECKS` build option, on by
default in Debug: it synchronises after every launch so faults are attributed to
the kernel that caused them. It is off in Release because synchronising
everywhere would serialise the whole program.

### 9.2 Streams

A **stream** is a queue of GPU work. Operations in one stream run in order;
operations in different streams may overlap.

Our library takes a stream and never creates one. That is deliberate: a library
that makes its own streams behind your back cannot be composed with one that
does not.

The PyTorch binding uses `torch.cuda.current_stream()`, which is what makes the
custom kernels compose with PyTorch's own work — ordered behind whatever
produced their input, ahead of whatever consumes their output, with no
synchronisation needed. `python/ccml/torch_api.py`.

### 9.3 Why a host clock lies

```cpp
auto start = std::chrono::now();
kernel<<<...>>>();                 // returns immediately!
auto end = std::chrono::now();     // measures the launch, not the work
```

This measures how long it took to *queue* the kernel. On Windows it is worse
still: the WDDM driver batches submissions, so the host clock can report
essentially anything.

**CUDA events** are timestamps recorded *in the stream*, between the surrounding
work, so they measure what the device actually did:

```cpp
start.record(stream);
... work ...
stop.record(stream);
stop.synchronize();
float ms = stop.elapsed_ms_since(start);
```

`support/cuda_event.h`.

### 9.4 Warmup is not a formality

The first launch of a kernel pays module loading and possibly JIT compilation,
and the GPU's clocks are still ramping up from idle. Including those samples can
report a kernel as several times slower than it is.

Every benchmark here discards warmup iterations and reports the spread — min,
max, standard deviation — not just a mean. A mean with no spread hides thermal
throttling, which a 70 W laptop-class GPU does under sustained load.

---

## 10. How we know the answers are right

### 10.1 An oracle

You cannot check a kernel against itself. You need an independent
implementation, written for obviousness rather than speed: straightforward
loops, double accumulation, no cleverness. `tests/support/reference.cpp`.

We use **two** independent references — our own CPU oracle and PyTorch. One
reference agreeing proves only that both share an assumption; two independent
ones disagreeing is what catches a wrong assumption.

### 10.2 Tolerance: why "equal" is the wrong question

Floating point results will not match exactly, so the question is always "close
enough — compared to what?"

**Absolute tolerance** `|got - ref| < a` fails where values are large: an error
of 0.001 is nothing if the answer is 1000000.

**Relative tolerance** `|got - ref| / |ref| < r` fails where values are near
zero: dividing by almost nothing makes any error look enormous.

So we use both, element-wise:

```
|got - ref| ≤ abs_tol + rel_tol × |ref|
```

Both failure modes showed up for real in this project:

* **Softmax** measured a *relative* error of exactly 1.0 for values that
  underflowed to zero in float32. A pure relative bound would have failed a
  correct kernel. The absolute term passed it, at 0.1% of budget.
* **Matmul** measured a relative error of **0.23** on an output element that had
  cancelled to near zero. Here even the combined form was wrong, because the
  per-element scale is meaningless for a dot product — an element can cancel to
  nothing while the terms that produced it were of size 1. Matmul therefore uses
  a tolerance scaled by the **RMS of the whole result**, the size of the quantity
  actually being accumulated.

The lesson worth carrying: **a failing test is not automatically a broken
kernel.** Both of these were correct kernels and wrong measuring sticks. The
evidence that distinguished them: the naive and tiled kernels agreed to the last
bit, a 1×1×1 product was exact, and the errors matched accumulation theory.

### 10.3 Property tests

Some things are true of a correct answer regardless of any reference:

* softmax outputs are finite, lie in [0,1], and each row sums to 1
* a constant row normalizes to zero
* a one-element softmax row is exactly 1.0

These are checked separately, because a kernel can match a wrong reference
perfectly.

### 10.4 Reproducibility

Every test uses a seeded generator, and we avoid `std::uniform_real_distribution`
because the C++ standard does not specify its output sequence — two standard
libraries would give different "random" numbers from the same seed. Our RNG
(`support/random.h`) derives values by explicit arithmetic, so a run is
reproducible across toolchains.

### 10.5 The tools not yet used

`compute-sanitizer --tool memcheck` finds out-of-bounds and misaligned accesses.
`--tool racecheck` finds missing barriers. **Neither has been run on any kernel
in this project**, and given section 5.5, that is the most important honest
caveat in the whole repository.

---

## 11. The software engineering around it

### 11.1 Compilation: what nvcc actually does

`nvcc` splits your `.cu` file. Host code goes to your normal C++ compiler; device
code becomes:

* **PTX** — a portable virtual assembly, forward-compatible
* **SASS** — actual machine code for a specific GPU

**Compute capability** identifies the GPU generation: 7.5 for Turing (the T4),
8.9 for Ada. `-DCCML_CUDA_ARCHITECTURES=75` compiles SASS for that architecture.
Code built only for 7.5 will not run on a different generation unless PTX is
embedded for it to JIT-compile at load.

This is why the whole project detects the architecture at runtime rather than
assuming — and why every result is labelled with the GPU it came from.

### 11.2 A flat C ABI

The public API is plain C: no classes, no templates, no exceptions, every
function returning a status code.

```c
ccml_status ccml_softmax_f32(float* y, const float* x, int rows, int cols,
                             ccml_softmax_algo algo, cudaStream_t stream);
```

C++ has no standard ABI — two compilers disagree about name mangling and object
layout. C has one that everything speaks. This is what lets a Python binding
work through `ctypes` with **no compiler at all**.

Since a status code alone is not much to debug with,
`ccml_last_error_string()` carries the detail, including CUDA's own message.

### 11.3 RAII

A GPU allocation that leaks is invisible until you run out of memory. In C++ the
cure is to tie the resource to an object's lifetime:

```cpp
{
    DeviceBuffer buffer(1024);   // allocates
    if (something) return;       // still freed
}                                // freed here
```

`support/device_buffer.h`, `support/cuda_stream.h`, `support/cuda_event.h`.

### 11.4 Separating interface from implementation

```
include/ccml/      public: what a user may depend on
src/<op>/<op>.cu   host wrapper: validation, launch configuration, dispatch
src/<op>/*_kernels.cuh   device code only: the optimization surface
src/common/        shared internals, not installed
```

The split means the whole of a performance rewrite happens inside one
`*_kernels.cuh`, with no consumer noticing. It is what makes "optimize later"
credible rather than aspirational.

### 11.5 The algorithm enum

Each operator takes an `algo` parameter. `AUTO` picks the current best; named
values select a specific kernel.

This is how variants are compared honestly — the naive-versus-tiled measurement
in the case study is exactly this mechanism in use. And an unbuilt variant
returns `NOT_IMPLEMENTED` rather than silently falling back, so a benchmark can
never accidentally measure a different kernel than it named.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Arithmetic intensity** | FLOPs per byte moved from global memory. Decides compute- vs memory-bound |
| **Barrier** | `__syncthreads()`; no thread in a block proceeds until all arrive |
| **Block** | Group of threads on one SM that can share memory and synchronise |
| **Coalescing** | Consecutive threads reading consecutive addresses, so one transaction serves a warp |
| **Compute capability** | GPU generation number, e.g. 7.5 Turing, 8.9 Ada |
| **Divergence** | Threads in one warp taking different branches; both paths execute |
| **Global memory** | Large, slow, GPU-wide DRAM |
| **Grid** | All the blocks in one kernel launch |
| **Kernel** | A `__global__` function: runs on the GPU, called from the CPU |
| **Machine epsilon** | Gap between 1.0 and the next float. 1.19e-7 for float32 |
| **Occupancy** | Fraction of an SM's thread slots in use; how latency gets hidden |
| **Reduction** | Combining many values into one (sum, max) |
| **Ridge point** | Arithmetic intensity where a chip stops being memory bound. ~25 FLOP/byte on a T4 |
| **SM** | Streaming multiprocessor; the unit that runs blocks. 40 on a T4 |
| **Shared memory** | Fast scratchpad shared by a block, managed by you |
| **Stream** | An ordered queue of GPU work; different streams may overlap |
| **Tiling** | Staging a block of data in shared memory so it is loaded once and reused many times |
| **ULP** | Unit in the last place: the gap between adjacent floats near a value |
| **Warp** | 32 threads executing in lockstep |
| **Warp shuffle** | Direct register-to-register exchange within a warp |

---

## Where to go next

* [design.md](design.md) — the specific choices this library made, and why
* [correctness.md](correctness.md) — the verification method and measured errors
* [results/](results/README.md) — what every phase actually measured
* [results/phase-6-case-study.md](results/phase-6-case-study.md) — the
  performance analysis that section 4 is the theory for
* [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) and [ROADMAP.md](ROADMAP.md) — what this becomes when it is finished
