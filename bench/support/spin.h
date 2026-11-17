#ifndef CCML_BENCH_SUPPORT_SPIN_H
#define CCML_BENCH_SUPPORT_SPIN_H

#include <cuda_runtime.h>

namespace ccml_bench {

/*
 * Launch-gap guard.
 *
 * Each timed iteration is [start event][kernel][stop event]. If the GPU is idle
 * when the start event is recorded, it executes the event at once and then
 * WAITS while the CPU is still submitting the kernel. That wait lands inside
 * the timed window. For kernels shorter than their own submission cost
 * (a few us from C++, 10-20 us from Python), it can double the measured time:
 * the first Kaggle run showed Python medians at 2.1x the batched mean below
 * 30 us.
 *
 * Enqueuing a short busy-wait kernel before the start event keeps the GPU
 * occupied while the CPU submits start, kernel and stop, so they then run
 * back to back. The spin touches no memory, so the cache state (warm or
 * flushed) is unchanged.
 */
void launch_spin(cudaStream_t stream, double microseconds);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_SPIN_H */
