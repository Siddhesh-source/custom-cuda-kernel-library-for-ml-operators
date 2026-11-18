#include "ccml/layernorm.h"

#include <cstdint>
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

/* AUTO (enhancement S4), from the Kaggle T4 run at commit 29e38c6
   (results/sm_75/layernorm_variants/, docs/results/phase-12-layernorm.md):
   the best variant per column bucket, identical in FP32, FP16 and BF16, was
   WARP_PER_ROW for cols <= 256, VECTORIZED for 257-1024 (14 of 14 shapes)
   and WELFORD above 1024 (18 of 18). */
ccml_layernorm_algo resolve(ccml_layernorm_algo algo, int cols) {
  if (algo != CCML_LAYERNORM_AUTO) return algo;
  if (cols <= 256) return CCML_LAYERNORM_WARP_PER_ROW;
  return cols <= 1024 ? CCML_LAYERNORM_VECTORIZED : CCML_LAYERNORM_WELFORD;
}

/* Warp per row (entry 10): the per-lane register count is the smallest bucket
   that holds the row, so short rows do not carry 32 registers. */
template <typename T, int kPer>
ccml_status launch_wpr(T* y, const T* x, const T* gamma, const T* beta, int rows, int cols,
                       float eps, cudaStream_t stream) {
  constexpr int kRows = ccml::kernels::kRowsPerBlock;
  ccml::detail::LaunchConfig config;
  config.block = dim3(32 * kRows, 1, 1);
  config.grid = dim3(static_cast<unsigned>((rows + kRows - 1) / kRows), 1, 1);
  /* Through a pointer: the template argument list's comma would split the macro. */
  void (*const layernorm_warp_per_row)(T*, const T*, const T*, const T*, int, int, float) =
      &ccml::kernels::layernorm_warp_per_row<T, kPer>;
  CCML_LAUNCH(layernorm_warp_per_row, config, stream, y, x, gamma, beta, rows, cols, eps);
  return CCML_SUCCESS;
}

template <typename T>
ccml_status launch_warp_per_row(T* y, const T* x, const T* gamma, const T* beta, int rows,
                                int cols, float eps, cudaStream_t stream,
                                const std::string& fn) {
  const int per = (cols + 31) / 32;
  if (per <= 1) return launch_wpr<T, 1>(y, x, gamma, beta, rows, cols, eps, stream);
  if (per <= 2) return launch_wpr<T, 2>(y, x, gamma, beta, rows, cols, eps, stream);
  if (per <= 4) return launch_wpr<T, 4>(y, x, gamma, beta, rows, cols, eps, stream);
  if (per <= 8) return launch_wpr<T, 8>(y, x, gamma, beta, rows, cols, eps, stream);
  if (per <= 16) return launch_wpr<T, 16>(y, x, gamma, beta, rows, cols, eps, stream);
  if (per <= 24) return launch_wpr<T, 24>(y, x, gamma, beta, rows, cols, eps, stream);
  if (per <= 32) return launch_wpr<T, 32>(y, x, gamma, beta, rows, cols, eps, stream);
  ccml::detail::set_last_error(
      (fn + ": WARP_PER_ROW holds a row in one warp's registers, so cols <= 1024").c_str());
  return CCML_ERROR_UNSUPPORTED_SHAPE;
}

/* Vectorized (entry 11): kChunks 128-bit chunks per lane. */
template <typename T, int kChunks>
ccml_status launch_vec(T* y, const T* x, const T* gamma, const T* beta, int rows, int cols,
                       float eps, cudaStream_t stream) {
  constexpr int kRows = ccml::kernels::kRowsPerBlock;
  ccml::detail::LaunchConfig config;
  config.block = dim3(32 * kRows, 1, 1);
  config.grid = dim3(static_cast<unsigned>((rows + kRows - 1) / kRows), 1, 1);
  void (*const layernorm_vectorized)(T*, const T*, const T*, const T*, int, int, float) =
      &ccml::kernels::layernorm_vectorized<T, kChunks>;
  CCML_LAUNCH(layernorm_vectorized, config, stream, y, x, gamma, beta, rows, cols, eps);
  return CCML_SUCCESS;
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

/* Needs cols % (16 / sizeof(T)) == 0 and 16-byte-aligned pointers (a null
   gamma or beta is fine); otherwise WARP_PER_ROW's scalar kernel computes the
   identical result. */
template <typename T>
ccml_status launch_vectorized(T* y, const T* x, const T* gamma, const T* beta, int rows,
                              int cols, float eps, cudaStream_t stream, const std::string& fn) {
  constexpr int kV = static_cast<int>(16 / sizeof(T));
  const bool vec = cols % kV == 0 && aligned16(y) && aligned16(x) &&
                   (gamma == nullptr || aligned16(gamma)) && (beta == nullptr || aligned16(beta));
  if (!vec || cols > 1024) {
    return launch_warp_per_row(y, x, gamma, beta, rows, cols, eps, stream, fn);
  }
  const int chunks = (cols / kV + 31) / 32;
  if (chunks <= 1) return launch_vec<T, 1>(y, x, gamma, beta, rows, cols, eps, stream);
  if (chunks <= 2) return launch_vec<T, 2>(y, x, gamma, beta, rows, cols, eps, stream);
  if (chunks <= 4) return launch_vec<T, 4>(y, x, gamma, beta, rows, cols, eps, stream);
  return launch_vec<T, 8>(y, x, gamma, beta, rows, cols, eps, stream);
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

  switch (resolve(algo, cols)) {
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

    case CCML_LAYERNORM_WARP_PER_ROW:
      return launch_warp_per_row(y, x, gamma, beta, rows, cols, eps, stream,
                                 fn_name<Public>());

    case CCML_LAYERNORM_VECTORIZED:
      return launch_vectorized(y, x, gamma, beta, rows, cols, eps, stream, fn_name<Public>());

    default:
      ccml::detail::set_last_error((fn_name<Public>() + ": unknown algorithm").c_str());
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

/* ------------------------------------------ fused residual + bias --- */

template <typename T, int kChunks>
ccml_status launch_res_vec(T* y, T* s, const T* x, const T* r, const T* b, const T* g,
                           const T* be, int rows, int cols, float eps, cudaStream_t stream) {
  constexpr int kRows = ccml::kernels::kRowsPerBlock;
  ccml::detail::LaunchConfig config;
  config.block = dim3(32 * kRows, 1, 1);
  config.grid = dim3(static_cast<unsigned>((rows + kRows - 1) / kRows), 1, 1);
  void (*const layernorm_residual_vectorized)(T*, T*, const T*, const T*, const T*, const T*,
                                              const T*, int, int, float) =
      &ccml::kernels::layernorm_residual_vectorized<T, kChunks>;
  CCML_LAUNCH(layernorm_residual_vectorized, config, stream, y, s, x, r, b, g, be, rows, cols,
              eps);
  return CCML_SUCCESS;
}

template <typename T>
ccml_status launch_res_welford(T* y, T* s, const T* x, const T* r, const T* b, const T* g,
                               const T* be, int rows, int cols, float eps,
                               cudaStream_t stream) {
  const ccml::detail::LaunchConfig config = ccml::detail::rowwise_launch_config(
      rows, cols, ccml::kernels::kLayerNormBlockSize);
  CCML_LAUNCH(ccml::kernels::layernorm_residual_welford<T>, config, stream, y, s, x, r, b, g, be,
              rows, cols, eps);
  return CCML_SUCCESS;
}

template <typename Public>
ccml_status layernorm_residual(Public* y_pub, Public* s_pub, const Public* x_pub,
                               const Public* r_pub, const Public* b_pub, const Public* g_pub,
                               const Public* be_pub, int rows, int cols, float eps,
                               ccml_layernorm_algo algo, cudaStream_t stream) {
  using T = ccml::detail::device_t<Public>;
  ccml::detail::clear_last_error();
  const ccml_status valid = validate(y_pub, x_pub, rows, cols, eps);
  if (valid != CCML_SUCCESS) return valid;
  const std::string fn =
      std::string("ccml_layernorm_residual_") + ccml::detail::dtype_suffix<Public>();
  CCML_REQUIRE(r_pub != nullptr, CCML_ERROR_INVALID_ARGUMENT, (fn + ": residual is null").c_str());
  CCML_REQUIRE(s_pub == nullptr || (static_cast<const void*>(s_pub) != x_pub &&
                                    static_cast<const void*>(s_pub) != r_pub),
               CCML_ERROR_INVALID_ARGUMENT,
               (fn + ": sum_out must not alias x or residual").c_str());
  T* y = ccml::detail::device_ptr(y_pub);
  T* s = ccml::detail::device_ptr(s_pub);
  const T* x = ccml::detail::device_ptr(x_pub);
  const T* r = ccml::detail::device_ptr(r_pub);
  const T* b = ccml::detail::device_ptr(b_pub);
  const T* g = ccml::detail::device_ptr(g_pub);
  const T* be = ccml::detail::device_ptr(be_pub);
  /* Confirmed on the T4 (commit 29e38c6): fused VECTORIZED 1.19-2.54x faster
     than fused WELFORD on 16-bit rows of 257-1024 columns, equal elsewhere. */
  if (algo == CCML_LAYERNORM_AUTO) {
    algo = cols <= 1024 ? CCML_LAYERNORM_VECTORIZED : CCML_LAYERNORM_WELFORD;
  }
  switch (algo) {
    case CCML_LAYERNORM_VECTORIZED: {
      constexpr int kV = static_cast<int>(16 / sizeof(T));
      const bool vec = cols <= 1024 && cols % kV == 0 && aligned16(y) && aligned16(x) &&
                       aligned16(r) && (s == nullptr || aligned16(s)) &&
                       (b == nullptr || aligned16(b)) && (g == nullptr || aligned16(g)) &&
                       (be == nullptr || aligned16(be));
      /* Otherwise the block-per-row kernel computes the identical result. */
      if (!vec) return launch_res_welford(y, s, x, r, b, g, be, rows, cols, eps, stream);
      const int chunks = (cols / kV + 31) / 32;
      if (chunks <= 1) return launch_res_vec<T, 1>(y, s, x, r, b, g, be, rows, cols, eps, stream);
      if (chunks <= 2) return launch_res_vec<T, 2>(y, s, x, r, b, g, be, rows, cols, eps, stream);
      if (chunks <= 4) return launch_res_vec<T, 4>(y, s, x, r, b, g, be, rows, cols, eps, stream);
      return launch_res_vec<T, 8>(y, s, x, r, b, g, be, rows, cols, eps, stream);
    }
    case CCML_LAYERNORM_WELFORD:
      return launch_res_welford(y, s, x, r, b, g, be, rows, cols, eps, stream);
    default:
      ccml::detail::set_last_error(
          (fn + ": the fused residual exists for VECTORIZED and WELFORD only").c_str());
      return CCML_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace

extern "C" {

const char* ccml_layernorm_algo_name(ccml_layernorm_algo algo) {
  switch (algo) {
    case CCML_LAYERNORM_AUTO:     return "auto";
    case CCML_LAYERNORM_TWO_PASS: return "two_pass";
    case CCML_LAYERNORM_WELFORD:  return "welford";
    case CCML_LAYERNORM_WARP_PER_ROW: return "warp_per_row";
    case CCML_LAYERNORM_VECTORIZED:   return "vectorized";
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

ccml_status ccml_layernorm_residual_f32(float* y, float* sum_out, const float* x,
                                        const float* residual, const float* bias,
                                        const float* gamma, const float* beta, int rows, int cols,
                                        float eps, ccml_layernorm_algo algo,
                                        cudaStream_t stream) {
  return layernorm_residual(y, sum_out, x, residual, bias, gamma, beta, rows, cols, eps, algo,
                            stream);
}

ccml_status ccml_layernorm_residual_f16(ccml_f16* y, ccml_f16* sum_out, const ccml_f16* x,
                                        const ccml_f16* residual, const ccml_f16* bias,
                                        const ccml_f16* gamma, const ccml_f16* beta, int rows,
                                        int cols, float eps, ccml_layernorm_algo algo,
                                        cudaStream_t stream) {
  return layernorm_residual(y, sum_out, x, residual, bias, gamma, beta, rows, cols, eps, algo,
                            stream);
}

ccml_status ccml_layernorm_residual_bf16(ccml_bf16* y, ccml_bf16* sum_out, const ccml_bf16* x,
                                         const ccml_bf16* residual, const ccml_bf16* bias,
                                         const ccml_bf16* gamma, const ccml_bf16* beta, int rows,
                                         int cols, float eps, ccml_layernorm_algo algo,
                                         cudaStream_t stream) {
  return layernorm_residual(y, sum_out, x, residual, bias, gamma, beta, rows, cols, eps, algo,
                            stream);
}

}  // extern "C"
