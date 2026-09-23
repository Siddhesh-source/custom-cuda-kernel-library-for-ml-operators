#include "ccml/layernorm.h"

#include "common/check.h"
#include "common/launch.h"
#include "common/rowwise.h"
#include "layernorm/layernorm_kernels.cuh"

namespace {

ccml_status validate(const float* y, const float* x, int rows, int cols, float eps) {
  CCML_REQUIRE(y != nullptr, CCML_ERROR_INVALID_ARGUMENT, "ccml_layernorm_f32: y is null");
  CCML_REQUIRE(x != nullptr, CCML_ERROR_INVALID_ARGUMENT, "ccml_layernorm_f32: x is null");
  CCML_REQUIRE(rows > 0 && cols > 0, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_layernorm_f32: rows and cols must be positive");
  CCML_REQUIRE(eps > 0.0f, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_layernorm_f32: eps must be positive");
  return CCML_SUCCESS;
}

ccml_layernorm_algo resolve(ccml_layernorm_algo algo) {
  return (algo == CCML_LAYERNORM_AUTO) ? CCML_LAYERNORM_TWO_PASS : algo;
}

}  // namespace

extern "C" {

const char* ccml_layernorm_algo_name(ccml_layernorm_algo algo) {
  switch (algo) {
    case CCML_LAYERNORM_AUTO:     return "auto";
    case CCML_LAYERNORM_TWO_PASS: return "two_pass";
    case CCML_LAYERNORM_WELFORD:  return "welford";
  }
  return "unknown";
}

ccml_status ccml_layernorm_f32(float* y, const float* x,
                               const float* gamma, const float* beta,
                               int rows, int cols, float eps,
                               ccml_layernorm_algo algo, cudaStream_t stream) {
  ccml::detail::clear_last_error();

  const ccml_status valid = validate(y, x, rows, cols, eps);
  if (valid != CCML_SUCCESS) {
    return valid;
  }

  switch (resolve(algo)) {
    case CCML_LAYERNORM_TWO_PASS: {
      /* One block per row. A row count above the device's grid limit is caught
         by validate_launch inside CCML_LAUNCH and reported as such, rather than
         silently computing only part of the tensor. */
      const ccml::detail::LaunchConfig config = ccml::detail::rowwise_launch_config(
          rows, cols, ccml::kernels::kLayerNormBlockSize);
      CCML_LAUNCH(ccml::kernels::layernorm_two_pass_f32, config, stream,
                  y, x, gamma, beta, rows, cols, eps);
      return CCML_SUCCESS;
    }

    case CCML_LAYERNORM_WELFORD:
      ccml::detail::set_last_error(
          "ccml_layernorm_f32: the Welford algorithm is reserved but not built");
      return CCML_ERROR_NOT_IMPLEMENTED;

    default:
      ccml::detail::set_last_error("ccml_layernorm_f32: unknown algorithm");
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

}  // extern "C"
