#ifndef CCML_SRC_COMMON_ROWWISE_H
#define CCML_SRC_COMMON_ROWWISE_H

/*
 * Launch configuration shared by the row-wise operators.
 *
 * Layer normalization and softmax have the same shape of problem: a [rows,
 * cols] tensor reduced along the last dimension, one block per row, threads
 * striding across the row. They had identical copies of this sizing logic until
 * it was lifted here.
 *
 * Internal. Not installed, not part of the public API.
 */

#include <cuda_runtime.h>

#include "common/launch.h"
#include "common/util.h"

namespace ccml {
namespace detail {

/*
 * One block per row; threads per block sized to the row.
 *
 * Rounded to a whole number of warps because block_reduce reduces warp by warp,
 * and capped at the operator's block size. A narrow row gets a smaller block
 * rather than 256 threads of which all but a handful sit idle through every
 * barrier.
 *
 * This is sizing to the work, not tuning. No block size in this library has
 * been chosen by measurement.
 *
 * Requires rows >= 1 and cols >= 1; callers validate before reaching here.
 * A row count beyond the device's grid limit is not rejected here -- it is
 * caught by validate_launch, which can report the actual limit.
 */
inline LaunchConfig rowwise_launch_config(int rows, int cols, int max_block_size) {
  constexpr int kWarp = 32;

  /* round_up(cols, 32) is at least 32 for any cols >= 1, so no floor is needed. */
  int threads = round_up(cols, kWarp);
  if (threads > max_block_size) {
    threads = max_block_size;
  }

  LaunchConfig config;
  config.block = dim3(static_cast<unsigned>(threads), 1, 1);
  config.grid  = dim3(static_cast<unsigned>(rows), 1, 1);
  return config;
}

}  // namespace detail
}  // namespace ccml

#endif /* CCML_SRC_COMMON_ROWWISE_H */
