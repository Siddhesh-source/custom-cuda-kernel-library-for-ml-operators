#include "ccml/softmax.h"

#include "common/check.h"
#include "common/launch.h"
#include "common/rowwise.h"
#include "softmax/softmax_kernels.cuh"

namespace {

ccml_status validate(const float* y, const float* x, int rows, int cols) {
  CCML_REQUIRE(y != nullptr, CCML_ERROR_INVALID_ARGUMENT, "ccml_softmax_f32: y is null");
  CCML_REQUIRE(x != nullptr, CCML_ERROR_INVALID_ARGUMENT, "ccml_softmax_f32: x is null");
  CCML_REQUIRE(rows > 0 && cols > 0, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_softmax_f32: rows and cols must be positive");
  return CCML_SUCCESS;
}

ccml_softmax_algo resolve(ccml_softmax_algo algo) {
  return (algo == CCML_SOFTMAX_AUTO) ? CCML_SOFTMAX_BLOCK_PER_ROW : algo;
}

}  // namespace

extern "C" {

const char* ccml_softmax_algo_name(ccml_softmax_algo algo) {
  switch (algo) {
    case CCML_SOFTMAX_AUTO:          return "auto";
    case CCML_SOFTMAX_BLOCK_PER_ROW: return "block_per_row";
    case CCML_SOFTMAX_WARP_PER_ROW:  return "warp_per_row";
  }
  return "unknown";
}

ccml_status ccml_softmax_f32(float* y, const float* x, int rows, int cols,
                             ccml_softmax_algo algo, cudaStream_t stream) {
  ccml::detail::clear_last_error();

  const ccml_status valid = validate(y, x, rows, cols);
  if (valid != CCML_SUCCESS) {
    return valid;
  }

  switch (resolve(algo)) {
    case CCML_SOFTMAX_BLOCK_PER_ROW: {
      /* One block per row. A row count above the device's grid limit is caught
         by validate_launch inside CCML_LAUNCH and reported as such, rather than
         silently computing only part of the tensor. */
      const ccml::detail::LaunchConfig config = ccml::detail::rowwise_launch_config(
          rows, cols, ccml::kernels::kSoftmaxBlockSize);
      CCML_LAUNCH(ccml::kernels::softmax_block_per_row_f32, config, stream,
                  y, x, rows, cols);
      return CCML_SUCCESS;
    }

    case CCML_SOFTMAX_WARP_PER_ROW:
      ccml::detail::set_last_error(
          "ccml_softmax_f32: the warp-per-row algorithm is reserved but not built");
      return CCML_ERROR_NOT_IMPLEMENTED;

    default:
      ccml::detail::set_last_error("ccml_softmax_f32: unknown algorithm");
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

}  // extern "C"
