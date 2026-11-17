#include "ccml/layernorm.h"

#include <string>

#include "common/check.h"
#include "common/dtype.cuh"
#include "common/launch.h"
#include "common/rowwise.h"
#include "layernorm/layernorm_kernels.cuh"

namespace {

template <typename Public>
std::string fn_name() {
  return std::string("ccml_layernorm_") + ccml::detail::dtype_suffix<Public>();
}

template <typename Public>
ccml_status validate(const Public* y, const Public* x, int rows, int cols, float eps) {
  const std::string fn = fn_name<Public>();
  CCML_REQUIRE(y != nullptr, CCML_ERROR_INVALID_ARGUMENT, (fn + ": y is null").c_str());
  CCML_REQUIRE(x != nullptr, CCML_ERROR_INVALID_ARGUMENT, (fn + ": x is null").c_str());
  CCML_REQUIRE(rows > 0 && cols > 0, CCML_ERROR_INVALID_ARGUMENT,
               (fn + ": rows and cols must be positive").c_str());
  CCML_REQUIRE(eps > 0.0f, CCML_ERROR_INVALID_ARGUMENT,
               (fn + ": eps must be positive").c_str());
  return CCML_SUCCESS;
}

ccml_layernorm_algo resolve(ccml_layernorm_algo algo) {
  return (algo == CCML_LAYERNORM_AUTO) ? CCML_LAYERNORM_TWO_PASS : algo;
}

/* One implementation for every storage type; the public entry points below are
   thin instantiations of it. */
template <typename Public>
ccml_status layernorm(Public* y_pub, const Public* x_pub, const Public* gamma_pub,
                      const Public* beta_pub, int rows, int cols, float eps,
                      ccml_layernorm_algo algo, cudaStream_t stream) {
  using T = ccml::detail::device_t<Public>;
  ccml::detail::clear_last_error();

  const ccml_status valid = validate(y_pub, x_pub, rows, cols, eps);
  if (valid != CCML_SUCCESS) {
    return valid;
  }

  T* y = ccml::detail::device_ptr(y_pub);
  const T* x = ccml::detail::device_ptr(x_pub);
  const T* gamma = ccml::detail::device_ptr(gamma_pub);
  const T* beta = ccml::detail::device_ptr(beta_pub);

  switch (resolve(algo)) {
    case CCML_LAYERNORM_TWO_PASS: {
      /* One block per row. A row count above the device's grid limit is caught
         by validate_launch inside CCML_LAUNCH and reported as such, rather than
         silently computing only part of the tensor. */
      const ccml::detail::LaunchConfig config = ccml::detail::rowwise_launch_config(
          rows, cols, ccml::kernels::kLayerNormBlockSize);
      CCML_LAUNCH(ccml::kernels::layernorm_two_pass<T>, config, stream,
                  y, x, gamma, beta, rows, cols, eps);
      return CCML_SUCCESS;
    }

    case CCML_LAYERNORM_WELFORD: {
      const ccml::detail::LaunchConfig config = ccml::detail::rowwise_launch_config(
          rows, cols, ccml::kernels::kLayerNormBlockSize);
      CCML_LAUNCH(ccml::kernels::layernorm_welford<T>, config, stream,
                  y, x, gamma, beta, rows, cols, eps);
      return CCML_SUCCESS;
    }

    default:
      ccml::detail::set_last_error((fn_name<Public>() + ": unknown algorithm").c_str());
      return CCML_ERROR_INVALID_ARGUMENT;
  }
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
  return layernorm(y, x, gamma, beta, rows, cols, eps, algo, stream);
}

ccml_status ccml_layernorm_f16(ccml_f16* y, const ccml_f16* x,
                               const ccml_f16* gamma, const ccml_f16* beta,
                               int rows, int cols, float eps,
                               ccml_layernorm_algo algo, cudaStream_t stream) {
  return layernorm(y, x, gamma, beta, rows, cols, eps, algo, stream);
}

ccml_status ccml_layernorm_bf16(ccml_bf16* y, const ccml_bf16* x,
                                const ccml_bf16* gamma, const ccml_bf16* beta,
                                int rows, int cols, float eps,
                                ccml_layernorm_algo algo, cudaStream_t stream) {
  return layernorm(y, x, gamma, beta, rows, cols, eps, algo, stream);
}

}  // extern "C"
