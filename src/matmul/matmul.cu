#include "ccml/matmul.h"
#include "ccml/tuning.h"

#include <string>

#include "common/check.h"
#include "common/dtype.cuh"
#include "common/launch.h"
#include "common/util.h"
#include "matmul/matmul_kernels.cuh"

namespace {

std::string fn_name(const char* suffix) { return std::string("ccml_matmul_") + suffix; }

template <typename T>
ccml_status validate(const T* C, const T* A, const T* B, int M, int N, int K) {
  const std::string fn = fn_name(ccml::detail::dtype_suffix<T>());
  CCML_REQUIRE(C != nullptr, CCML_ERROR_INVALID_ARGUMENT, (fn + ": C is null").c_str());
  CCML_REQUIRE(A != nullptr, CCML_ERROR_INVALID_ARGUMENT, (fn + ": A is null").c_str());
  CCML_REQUIRE(B != nullptr, CCML_ERROR_INVALID_ARGUMENT, (fn + ": B is null").c_str());
  CCML_REQUIRE(M > 0 && N > 0 && K > 0, CCML_ERROR_INVALID_ARGUMENT,
               (fn + ": M, N and K must all be positive").c_str());
  return CCML_SUCCESS;
}

ccml_matmul_algo resolve(ccml_matmul_algo algo) {
  return (algo == CCML_MATMUL_AUTO) ? CCML_MATMUL_TILED : algo;
}

/* Both variants use the same square block and the same output decomposition:
   one thread per element of C, one block per tile of C. */
ccml::detail::LaunchConfig matmul_launch_config(int M, int N) {
  constexpr int kTile = ccml::kernels::kMatmulTileDim;

  ccml::detail::LaunchConfig config;
  config.block = dim3(kTile, kTile, 1);
  config.grid  = ccml::detail::grid_for_2d(N, M, kTile, kTile);
  return config;
}

/* Launches REGISTER_TILED with tile configuration I (compile-time). The kernel
   is taken through a function pointer because its template argument list
   contains commas, which the CCML_LAUNCH macro would split. */
template <typename T, int I>
ccml_status launch_register_tiled(T* C, const T* A, const T* B, int M, int N, int K,
                                  cudaStream_t stream) {
  constexpr ccml::kernels::MatmulTileConfig cfg = ccml::kernels::kMatmulTileConfigs[I];
  ccml::detail::LaunchConfig config;
  config.block = dim3((cfg.bm / cfg.tm) * (cfg.bn / cfg.tn), 1, 1);
  config.grid = ccml::detail::grid_for_2d(N, M, cfg.bn, cfg.bm);
  void (*const matmul_register_tiled)(T*, const T*, const T*, int, int, int) =
      &ccml::kernels::matmul_register_tiled<T, cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn>;
  CCML_LAUNCH(matmul_register_tiled, config, stream, C, A, B, M, N, K);
  return CCML_SUCCESS;
}

template <typename T>
ccml_status register_tiled(T* C, const T* A, const T* B, int M, int N, int K, int index,
                           cudaStream_t stream) {
  static_assert(ccml::kernels::kMatmulTileConfigCount == 6,
                "add a case below for every tile configuration");
  switch (index) {
    case 0: return launch_register_tiled<T, 0>(C, A, B, M, N, K, stream);
    case 1: return launch_register_tiled<T, 1>(C, A, B, M, N, K, stream);
    case 2: return launch_register_tiled<T, 2>(C, A, B, M, N, K, stream);
    case 3: return launch_register_tiled<T, 3>(C, A, B, M, N, K, stream);
    case 4: return launch_register_tiled<T, 4>(C, A, B, M, N, K, stream);
    case 5: return launch_register_tiled<T, 5>(C, A, B, M, N, K, stream);
    default:
      ccml::detail::set_last_error("ccml_matmul_register_tiled: tile config index out of range");
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

/* One implementation for every storage type; the public entry points below are
   thin instantiations of it. */
template <typename Public>
ccml_status matmul(Public* C_pub, const Public* A_pub, const Public* B_pub, int M,
                   int N, int K, ccml_matmul_algo algo, cudaStream_t stream) {
  using T = ccml::detail::device_t<Public>;
  ccml::detail::clear_last_error();

  const ccml_status valid = validate(C_pub, A_pub, B_pub, M, N, K);
  if (valid != CCML_SUCCESS) {
    return valid;
  }

  T* C = ccml::detail::device_ptr(C_pub);
  const T* A = ccml::detail::device_ptr(A_pub);
  const T* B = ccml::detail::device_ptr(B_pub);
  const ccml::detail::LaunchConfig config = matmul_launch_config(M, N);

  switch (resolve(algo)) {
    case CCML_MATMUL_NAIVE:
      CCML_LAUNCH(ccml::kernels::matmul_naive<T>, config, stream, C, A, B, M, N, K);
      return CCML_SUCCESS;

    case CCML_MATMUL_TILED:
      CCML_LAUNCH(ccml::kernels::matmul_tiled<T>, config, stream, C, A, B, M, N, K);
      return CCML_SUCCESS;

    case CCML_MATMUL_REGISTER_TILED:
      return register_tiled(C, A, B, M, N, K, /*index=*/0, stream);

    default:
      ccml::detail::set_last_error(
          (fn_name(ccml::detail::dtype_suffix<Public>()) + ": unknown algorithm").c_str());
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

}  // namespace

extern "C" {

const char* ccml_matmul_algo_name(ccml_matmul_algo algo) {
  switch (algo) {
    case CCML_MATMUL_AUTO:  return "auto";
    case CCML_MATMUL_NAIVE: return "naive";
    case CCML_MATMUL_TILED: return "tiled";
    case CCML_MATMUL_REGISTER_TILED: return "register_tiled";
  }
  return "unknown";
}

ccml_status ccml_matmul_f32(float* C, const float* A, const float* B,
                            int M, int N, int K,
                            ccml_matmul_algo algo, cudaStream_t stream) {
  return matmul(C, A, B, M, N, K, algo, stream);
}

ccml_status ccml_matmul_f16(ccml_f16* C, const ccml_f16* A, const ccml_f16* B,
                            int M, int N, int K,
                            ccml_matmul_algo algo, cudaStream_t stream) {
  return matmul(C, A, B, M, N, K, algo, stream);
}

ccml_status ccml_matmul_bf16(ccml_bf16* C, const ccml_bf16* A, const ccml_bf16* B,
                             int M, int N, int K,
                             ccml_matmul_algo algo, cudaStream_t stream) {
  return matmul(C, A, B, M, N, K, algo, stream);
}

/* ------------------------------------------------------------- tuning -- */

int ccml_matmul_tile_config_count(void) { return ccml::kernels::kMatmulTileConfigCount; }

ccml_status ccml_matmul_tile_config_get(int index, ccml_matmul_tile_config* out) {
  ccml::detail::clear_last_error();
  CCML_REQUIRE(out != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_matmul_tile_config_get: out is null");
  CCML_REQUIRE(index >= 0 && index < ccml::kernels::kMatmulTileConfigCount,
               CCML_ERROR_INVALID_ARGUMENT,
               "ccml_matmul_tile_config_get: index out of range");
  const ccml::kernels::MatmulTileConfig& c = ccml::kernels::kMatmulTileConfigs[index];
  out->bm = c.bm;
  out->bn = c.bn;
  out->bk = c.bk;
  out->tm = c.tm;
  out->tn = c.tn;
  out->threads = (c.bm / c.tm) * (c.bn / c.tn);
  return CCML_SUCCESS;
}

}  // extern "C"

namespace {

template <typename Public>
ccml_status register_tiled_entry(Public* C, const Public* A, const Public* B, int M, int N,
                                 int K, int index, cudaStream_t stream) {
  ccml::detail::clear_last_error();
  const ccml_status valid = validate(C, A, B, M, N, K);
  if (valid != CCML_SUCCESS) {
    return valid;
  }
  return register_tiled(ccml::detail::device_ptr(C), ccml::detail::device_ptr(A),
                        ccml::detail::device_ptr(B), M, N, K, index, stream);
}

}  // namespace

extern "C" {

ccml_status ccml_matmul_register_tiled_f32(float* C, const float* A, const float* B, int M,
                                           int N, int K, int index, cudaStream_t stream) {
  return register_tiled_entry(C, A, B, M, N, K, index, stream);
}

ccml_status ccml_matmul_register_tiled_f16(ccml_f16* C, const ccml_f16* A, const ccml_f16* B,
                                           int M, int N, int K, int index,
                                           cudaStream_t stream) {
  return register_tiled_entry(C, A, B, M, N, K, index, stream);
}

ccml_status ccml_matmul_register_tiled_bf16(ccml_bf16* C, const ccml_bf16* A,
                                            const ccml_bf16* B, int M, int N, int K,
                                            int index, cudaStream_t stream) {
  return register_tiled_entry(C, A, B, M, N, K, index, stream);
}

}  // extern "C"
