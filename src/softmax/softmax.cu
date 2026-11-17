#include "ccml/softmax.h"

#include <string>

#include "common/check.h"
#include "common/dtype.cuh"
#include "common/launch.h"
#include "common/rowwise.h"
#include "softmax/softmax_kernels.cuh"

namespace {

template <typename Public>
std::string fn_name() {
  return std::string("ccml_softmax_") + ccml::detail::dtype_suffix<Public>();
}

template <typename Public>
ccml_status validate(const Public* y, const Public* x, int rows, int cols) {
  const std::string fn = fn_name<Public>();
  CCML_REQUIRE(y != nullptr, CCML_ERROR_INVALID_ARGUMENT, (fn + ": y is null").c_str());
  CCML_REQUIRE(x != nullptr, CCML_ERROR_INVALID_ARGUMENT, (fn + ": x is null").c_str());
  CCML_REQUIRE(rows > 0 && cols > 0, CCML_ERROR_INVALID_ARGUMENT,
               (fn + ": rows and cols must be positive").c_str());
  return CCML_SUCCESS;
}

ccml_softmax_algo resolve(ccml_softmax_algo algo) {
  return (algo == CCML_SOFTMAX_AUTO) ? CCML_SOFTMAX_BLOCK_PER_ROW : algo;
}

/* One implementation for every storage type; the public entry points below are
   thin instantiations of it. */
template <typename Public>
ccml_status softmax(Public* y_pub, const Public* x_pub, int rows, int cols,
                    ccml_softmax_algo algo, cudaStream_t stream) {
  using T = ccml::detail::device_t<Public>;
  ccml::detail::clear_last_error();

  const ccml_status valid = validate(y_pub, x_pub, rows, cols);
  if (valid != CCML_SUCCESS) {
    return valid;
  }

  T* y = ccml::detail::device_ptr(y_pub);
  const T* x = ccml::detail::device_ptr(x_pub);

  switch (resolve(algo)) {
    case CCML_SOFTMAX_BLOCK_PER_ROW: {
      /* One block per row. A row count above the device's grid limit is caught
         by validate_launch inside CCML_LAUNCH and reported as such, rather than
         silently computing only part of the tensor. */
      const ccml::detail::LaunchConfig config = ccml::detail::rowwise_launch_config(
          rows, cols, ccml::kernels::kSoftmaxBlockSize);
      CCML_LAUNCH(ccml::kernels::softmax_block_per_row<T>, config, stream,
                  y, x, rows, cols);
      return CCML_SUCCESS;
    }

    case CCML_SOFTMAX_WARP_PER_ROW:
      ccml::detail::set_last_error(
          (fn_name<Public>() + ": the warp-per-row algorithm is reserved but not built")
              .c_str());
      return CCML_ERROR_NOT_IMPLEMENTED;

    default:
      ccml::detail::set_last_error((fn_name<Public>() + ": unknown algorithm").c_str());
      return CCML_ERROR_INVALID_ARGUMENT;
  }
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
  return softmax(y, x, rows, cols, algo, stream);
}

ccml_status ccml_softmax_f16(ccml_f16* y, const ccml_f16* x, int rows, int cols,
                             ccml_softmax_algo algo, cudaStream_t stream) {
  return softmax(y, x, rows, cols, algo, stream);
}

ccml_status ccml_softmax_bf16(ccml_bf16* y, const ccml_bf16* x, int rows, int cols,
                              ccml_softmax_algo algo, cudaStream_t stream) {
  return softmax(y, x, rows, cols, algo, stream);
}

}  // extern "C"
