#include "ccml/matmul.h"
#include "ccml/tuning.h"

#include <cstdint>
#include <string>

#include "common/check.h"
#include "common/dtype.cuh"
#include "common/launch.h"
#include "common/util.h"
#include "matmul/matmul_kernels.cuh"

namespace {

using ccml::kernels::kMatmulTileConfigCount;
using ccml::kernels::kMatmulTileConfigs;
using ccml::kernels::MatmulTileConfig;

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

/* NAIVE and TILED use the same square block and the same output decomposition:
   one thread per element of C, one block per tile of C. */
ccml::detail::LaunchConfig matmul_launch_config(int M, int N) {
  constexpr int kTile = ccml::kernels::kMatmulTileDim;

  ccml::detail::LaunchConfig config;
  config.block = dim3(kTile, kTile, 1);
  config.grid  = ccml::detail::grid_for_2d(N, M, kTile, kTile);
  return config;
}

ccml::detail::LaunchConfig tile_launch_config(const MatmulTileConfig& cfg, int M, int N) {
  ccml::detail::LaunchConfig config;
  config.block = dim3((cfg.bm / cfg.tm) * (cfg.bn / cfg.tn), 1, 1);
  config.grid = ccml::detail::grid_for_2d(N, M, cfg.bn, cfg.bm);
  return config;
}

bool aligned(const void* p, std::size_t bytes) {
  return reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
}

/* --------------------------------------------- the tiled kernel family --
 *
 * REGISTER_TILED and VECTORIZED share the tile configurations in
 * kMatmulTileConfigs. Each kernel is taken through a function pointer because
 * its template argument list contains commas, which CCML_LAUNCH would split. */

template <typename T, int I>
ccml_status launch_register_tiled(T* C, const T* A, const T* B, int M, int N, int K,
                                  cudaStream_t stream) {
  constexpr MatmulTileConfig cfg = kMatmulTileConfigs[I];
  void (*const matmul_register_tiled)(T*, const T*, const T*, int, int, int) =
      &ccml::kernels::matmul_register_tiled<T, cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn>;
  CCML_LAUNCH(matmul_register_tiled, tile_launch_config(cfg, M, N), stream, C, A, B, M, N, K);
  return CCML_SUCCESS;
}

template <typename T, int I>
ccml_status launch_vectorized(T* C, const T* A, const T* B, int M, int N, int K,
                              cudaStream_t stream) {
  constexpr MatmulTileConfig cfg = kMatmulTileConfigs[I];
  if constexpr (cfg.tm % 4 != 0 || cfg.tn % 4 != 0 || cfg.bk % 4 != 0) {
    ccml::detail::set_last_error(
        "ccml_matmul: VECTORIZED needs a tile configuration whose TM, TN and BK are "
        "multiples of 4");
    return CCML_ERROR_INVALID_ARGUMENT;
  } else {
    /* Vector global access needs K and N to be multiples of 4 (so a row of A
       or B never splits a vector) and 4-element-aligned pointers. Otherwise
       the same kernel runs with scalar global access. */
    const std::size_t vec_bytes = 4 * sizeof(T);
    const bool vec = K % 4 == 0 && N % 4 == 0 && aligned(A, vec_bytes) &&
                     aligned(B, vec_bytes) && aligned(C, vec_bytes);
    const ccml::detail::LaunchConfig config = tile_launch_config(cfg, M, N);
    if (vec) {
      void (*const matmul_vectorized)(T*, const T*, const T*, int, int, int) =
          &ccml::kernels::matmul_vectorized<T, cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn, true>;
      CCML_LAUNCH(matmul_vectorized, config, stream, C, A, B, M, N, K);
    } else {
      void (*const matmul_vectorized_scalar_global)(T*, const T*, const T*, int, int, int) =
          &ccml::kernels::matmul_vectorized<T, cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn, false>;
      CCML_LAUNCH(matmul_vectorized_scalar_global, config, stream, C, A, B, M, N, K);
    }
    return CCML_SUCCESS;
  }
}

template <typename T, int I>
ccml_status launch_double_buffered(T* C, const T* A, const T* B, int M, int N, int K,
                                   cudaStream_t stream) {
  constexpr MatmulTileConfig cfg = kMatmulTileConfigs[I];
  if constexpr (cfg.tm % 4 != 0 || cfg.tn % 4 != 0 || cfg.bk % 4 != 0) {
    ccml::detail::set_last_error(
        "ccml_matmul: DOUBLE_BUFFERED needs a tile configuration whose TM, TN and BK are "
        "multiples of 4");
    return CCML_ERROR_INVALID_ARGUMENT;
  } else {
    /* Same vector-path preconditions as VECTORIZED. */
    const std::size_t vec_bytes = 4 * sizeof(T);
    const bool vec = K % 4 == 0 && N % 4 == 0 && aligned(A, vec_bytes) &&
                     aligned(B, vec_bytes) && aligned(C, vec_bytes);
    const ccml::detail::LaunchConfig config = tile_launch_config(cfg, M, N);
    if (vec) {
      void (*const matmul_double_buffered)(T*, const T*, const T*, int, int, int) =
          &ccml::kernels::matmul_double_buffered<T, cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn,
                                                 true>;
      CCML_LAUNCH(matmul_double_buffered, config, stream, C, A, B, M, N, K);
    } else {
      void (*const matmul_double_buffered_scalar_global)(T*, const T*, const T*, int, int,
                                                         int) =
          &ccml::kernels::matmul_double_buffered<T, cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn,
                                                 false>;
      CCML_LAUNCH(matmul_double_buffered_scalar_global, config, stream, C, A, B, M, N, K);
    }
    return CCML_SUCCESS;
  }
}

template <typename T, int I>
ccml_status launch_tiled_family(ccml_matmul_algo algo, T* C, const T* A, const T* B, int M,
                                int N, int K, cudaStream_t stream) {
  switch (algo) {
    case CCML_MATMUL_REGISTER_TILED:
      return launch_register_tiled<T, I>(C, A, B, M, N, K, stream);
    case CCML_MATMUL_VECTORIZED:
      return launch_vectorized<T, I>(C, A, B, M, N, K, stream);
    case CCML_MATMUL_DOUBLE_BUFFERED:
      return launch_double_buffered<T, I>(C, A, B, M, N, K, stream);
    default:
      ccml::detail::set_last_error(
          "ccml_matmul: this algorithm has no tile configurations "
          "(only REGISTER_TILED, VECTORIZED and DOUBLE_BUFFERED do)");
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

template <typename T>
ccml_status tiled_family(ccml_matmul_algo algo, T* C, const T* A, const T* B, int M, int N,
                         int K, int index, cudaStream_t stream) {
  static_assert(kMatmulTileConfigCount == 6, "add a case below for every tile configuration");
  switch (index) {
    case 0: return launch_tiled_family<T, 0>(algo, C, A, B, M, N, K, stream);
    case 1: return launch_tiled_family<T, 1>(algo, C, A, B, M, N, K, stream);
    case 2: return launch_tiled_family<T, 2>(algo, C, A, B, M, N, K, stream);
    case 3: return launch_tiled_family<T, 3>(algo, C, A, B, M, N, K, stream);
    case 4: return launch_tiled_family<T, 4>(algo, C, A, B, M, N, K, stream);
    case 5: return launch_tiled_family<T, 5>(algo, C, A, B, M, N, K, stream);
    default:
      ccml::detail::set_last_error("ccml_matmul: tile config index out of range");
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
    case CCML_MATMUL_VECTORIZED:
    case CCML_MATMUL_DOUBLE_BUFFERED:
      /* Configuration 0 until the tile sweep decides per shape (S4). */
      return tiled_family(resolve(algo), C, A, B, M, N, K, /*index=*/0, stream);

    default:
      ccml::detail::set_last_error(
          (fn_name(ccml::detail::dtype_suffix<Public>()) + ": unknown algorithm").c_str());
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

template <typename Public>
ccml_status tuned(Public* C, const Public* A, const Public* B, int M, int N, int K,
                  ccml_matmul_algo algo, int index, cudaStream_t stream) {
  ccml::detail::clear_last_error();
  const ccml_status valid = validate(C, A, B, M, N, K);
  if (valid != CCML_SUCCESS) {
    return valid;
  }
  return tiled_family(algo, ccml::detail::device_ptr(C), ccml::detail::device_ptr(A),
                      ccml::detail::device_ptr(B), M, N, K, index, stream);
}

}  // namespace

extern "C" {

const char* ccml_matmul_algo_name(ccml_matmul_algo algo) {
  switch (algo) {
    case CCML_MATMUL_AUTO:           return "auto";
    case CCML_MATMUL_NAIVE:          return "naive";
    case CCML_MATMUL_TILED:          return "tiled";
    case CCML_MATMUL_REGISTER_TILED: return "register_tiled";
    case CCML_MATMUL_VECTORIZED:     return "vectorized";
    case CCML_MATMUL_DOUBLE_BUFFERED: return "double_buffered";
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

int ccml_matmul_tile_config_count(void) { return kMatmulTileConfigCount; }

ccml_status ccml_matmul_tile_config_get(int index, ccml_matmul_tile_config* out) {
  ccml::detail::clear_last_error();
  CCML_REQUIRE(out != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_matmul_tile_config_get: out is null");
  CCML_REQUIRE(index >= 0 && index < kMatmulTileConfigCount, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_matmul_tile_config_get: index out of range");
  const MatmulTileConfig& c = kMatmulTileConfigs[index];
  out->bm = c.bm;
  out->bn = c.bn;
  out->bk = c.bk;
  out->tm = c.tm;
  out->tn = c.tn;
  out->threads = (c.bm / c.tm) * (c.bn / c.tn);
  return CCML_SUCCESS;
}

ccml_status ccml_matmul_tuned_f32(float* C, const float* A, const float* B, int M, int N,
                                  int K, ccml_matmul_algo algo, int index,
                                  cudaStream_t stream) {
  return tuned(C, A, B, M, N, K, algo, index, stream);
}

ccml_status ccml_matmul_tuned_f16(ccml_f16* C, const ccml_f16* A, const ccml_f16* B, int M,
                                  int N, int K, ccml_matmul_algo algo, int index,
                                  cudaStream_t stream) {
  return tuned(C, A, B, M, N, K, algo, index, stream);
}

ccml_status ccml_matmul_tuned_bf16(ccml_bf16* C, const ccml_bf16* A, const ccml_bf16* B, int M,
                                   int N, int K, ccml_matmul_algo algo, int index,
                                   cudaStream_t stream) {
  return tuned(C, A, B, M, N, K, algo, index, stream);
}

}  // extern "C"
