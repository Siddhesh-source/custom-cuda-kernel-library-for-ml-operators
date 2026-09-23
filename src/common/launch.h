#ifndef CCML_SRC_COMMON_LAUNCH_H
#define CCML_SRC_COMMON_LAUNCH_H

/* Kernel launch configuration, validation and error checking.
 *
 * Header-only and inline so that tests can include it directly. Not installed;
 * this is not part of the public API.
 *
 * What launch checking can and cannot see
 * ---------------------------------------
 * A CUDA kernel launch is asynchronous. `cudaGetLastError()` straight after a
 * launch reports only whether the launch was ACCEPTED -- a bad grid or block
 * shape, too much shared memory, too many registers. Faults raised by the
 * kernel while it executes (an out-of-bounds write, a misaligned access) are
 * not visible until the next synchronisation, and will then be reported
 * against whatever call happens to synchronise next.
 *
 * That is why this header does two separate things: validate the configuration
 * BEFORE the launch, where a clear message can still be produced, and check for
 * a launch rejection after it. Build with CCML_SYNC_LAUNCH_CHECKS to also
 * synchronise after every launch, which attributes execution faults to the
 * kernel that actually caused them. It is on by default in Debug builds and off
 * in Release, because synchronising after every launch is ruinous for
 * performance and would quietly serialise everything.
 */

#include <cstdio>

#include <cuda_runtime.h>

#include "ccml/status.h"
#include "common/check.h"
#include "common/util.h"

namespace ccml {
namespace detail {

struct LaunchConfig {
  dim3   grid{1, 1, 1};
  dim3   block{1, 1, 1};
  size_t shared_bytes{0};
};

struct DeviceLimits {
  int    max_threads_per_block = 0;
  int    max_block_dim[3]      = {0, 0, 0};
  int    max_grid_dim[3]       = {0, 0, 0};
  size_t max_shared_per_block  = 0;
  int    warp_size             = 0;
};

/* Limits for the currently selected device.
 *
 * cudaGetDeviceProperties costs on the order of a millisecond, which is longer
 * than many of the kernels here will take to run, so the result is cached. The
 * cache is thread-local rather than shared: that needs no lock and cannot race,
 * at the cost of one extra query per thread that ever launches a kernel.
 */
inline ccml_status device_limits(DeviceLimits* out) {
  int device = 0;
  CCML_CUDA_TRY(cudaGetDevice(&device));

  static thread_local int          cached_device = -1;
  static thread_local DeviceLimits cached;

  if (cached_device != device) {
    cudaDeviceProp prop;
    CCML_CUDA_TRY(cudaGetDeviceProperties(&prop, device));

    cached.max_threads_per_block = prop.maxThreadsPerBlock;
    for (int axis = 0; axis < 3; ++axis) {
      cached.max_block_dim[axis] = prop.maxThreadsDim[axis];
      cached.max_grid_dim[axis]  = prop.maxGridSize[axis];
    }
    cached.max_shared_per_block = prop.sharedMemPerBlock;
    cached.warp_size            = prop.warpSize;
    cached_device               = device;
  }

  *out = cached;
  return CCML_SUCCESS;
}

/* Check a launch configuration against the device before launching.
 *
 * Without this, an oversized block shows up as a bare `invalid configuration
 * argument` from whichever call synchronises next, with nothing naming the
 * kernel or the offending dimension.
 */
inline ccml_status validate_launch(const LaunchConfig& config, const char* where) {
  DeviceLimits limits;
  const ccml_status status = device_limits(&limits);
  if (status != CCML_SUCCESS) {
    return status;
  }

  char message[320];

  const unsigned long long threads =
      static_cast<unsigned long long>(config.block.x) * config.block.y * config.block.z;

  if (threads == 0) {
    std::snprintf(message, sizeof(message),
                  "%s: block is empty (%ux%ux%u)", where,
                  config.block.x, config.block.y, config.block.z);
    set_last_error(message);
    return CCML_ERROR_INVALID_ARGUMENT;
  }

  if (threads > static_cast<unsigned long long>(limits.max_threads_per_block)) {
    std::snprintf(message, sizeof(message),
                  "%s: block of %ux%ux%u is %llu threads, above the device "
                  "limit of %d per block",
                  where, config.block.x, config.block.y, config.block.z,
                  threads, limits.max_threads_per_block);
    set_last_error(message);
    return CCML_ERROR_INVALID_ARGUMENT;
  }

  const unsigned block_dims[3] = {config.block.x, config.block.y, config.block.z};
  const unsigned grid_dims[3]  = {config.grid.x, config.grid.y, config.grid.z};
  const char*    axis_name[3]  = {"x", "y", "z"};

  for (int axis = 0; axis < 3; ++axis) {
    if (block_dims[axis] > static_cast<unsigned>(limits.max_block_dim[axis])) {
      std::snprintf(message, sizeof(message),
                    "%s: block.%s is %u, above the device limit of %d",
                    where, axis_name[axis], block_dims[axis],
                    limits.max_block_dim[axis]);
      set_last_error(message);
      return CCML_ERROR_INVALID_ARGUMENT;
    }

    if (grid_dims[axis] == 0) {
      std::snprintf(message, sizeof(message),
                    "%s: grid.%s is zero, so no blocks would run",
                    where, axis_name[axis]);
      set_last_error(message);
      return CCML_ERROR_INVALID_ARGUMENT;
    }

    if (grid_dims[axis] > static_cast<unsigned>(limits.max_grid_dim[axis])) {
      std::snprintf(message, sizeof(message),
                    "%s: grid.%s is %u, above the device limit of %d",
                    where, axis_name[axis], grid_dims[axis],
                    limits.max_grid_dim[axis]);
      set_last_error(message);
      return CCML_ERROR_INVALID_ARGUMENT;
    }
  }

  if (config.shared_bytes > limits.max_shared_per_block) {
    std::snprintf(message, sizeof(message),
                  "%s: requested %zu bytes of shared memory, above the device "
                  "limit of %zu per block",
                  where, config.shared_bytes, limits.max_shared_per_block);
    set_last_error(message);
    return CCML_ERROR_UNSUPPORTED_SHAPE;
  }

  return CCML_SUCCESS;
}

/* Grid sizing helpers. One block per `block` elements, rounding up. */
inline dim3 grid_for_1d(int elements, int block) {
  return dim3(static_cast<unsigned>(div_up(elements, block)), 1, 1);
}

inline dim3 grid_for_2d(int width, int height, int block_x, int block_y) {
  return dim3(static_cast<unsigned>(div_up(width, block_x)),
              static_cast<unsigned>(div_up(height, block_y)), 1);
}

}  // namespace detail
}  // namespace ccml

/* Synchronise after every launch so that execution faults are attributed to the
   kernel that caused them. Debug only -- see the note at the top of this file. */
#if defined(CCML_SYNC_LAUNCH_CHECKS)
#  define CCML_LAUNCH_SYNC(stream) CCML_CUDA_TRY(cudaStreamSynchronize(stream))
#else
#  define CCML_LAUNCH_SYNC(stream) ((void)0)
#endif

/*
 * Validate, launch, and check that the launch was accepted.
 *
 *   CCML_LAUNCH(my_kernel, config, stream, arg0, arg1);
 *
 * Returns a ccml_status from the enclosing function on any failure, so it may
 * only be used inside a function returning ccml_status.
 */
#define CCML_LAUNCH(kernel_name, config, stream, ...)                          \
  do {                                                                         \
    const ::ccml::detail::LaunchConfig& ccml_cfg_ = (config);                  \
    const ccml_status ccml_valid_ =                                            \
        ::ccml::detail::validate_launch(ccml_cfg_, #kernel_name);              \
    if (ccml_valid_ != CCML_SUCCESS) {                                         \
      return ccml_valid_;                                                      \
    }                                                                          \
    kernel_name<<<ccml_cfg_.grid, ccml_cfg_.block, ccml_cfg_.shared_bytes,     \
                  (stream)>>>(__VA_ARGS__);                                    \
    /* Catches a rejected launch. Execution faults surface at the next sync. */ \
    CCML_CUDA_TRY(cudaGetLastError());                                         \
    CCML_LAUNCH_SYNC(stream);                                                  \
  } while (0)

#endif /* CCML_SRC_COMMON_LAUNCH_H */
