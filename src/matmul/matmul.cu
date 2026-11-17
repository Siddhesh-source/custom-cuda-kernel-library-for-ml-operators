#include "ccml/matmul.h"
#include "ccml/tuning.h"

#include <cstdint>
#include <string>
#include <type_traits>

#include "common/check.h"
#include "common/dtype.cuh"
#include "common/launch.h"
#include "common/util.h"
#include "matmul/matmul_kernels.cuh"

namespace {

using ccml::kernels::kMatmulTileConfigCount;
using ccml::kernels::kMatmulTileConfigs;
using ccml::kernels::MatmulTileConfig;
using ccml::kernels::kMatmulTcConfigs;
using ccml::kernels::MatmulTcConfig;

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

/* -------------------------------------------- the Tensor Core family --
 *
 * 16-bit input only. Whether a kernel body was compiled for an architecture
 * with Tensor Cores for its input type is read from the binary actually
 * loaded for this device (cudaFuncAttributes::binaryVersion), not guessed
 * from the device: a build for sm_75 run on an sm_89 GPU through PTX JIT
 * still has no BF16 body. */

template <typename Kernel>
int compiled_arch(Kernel kernel) {
  cudaFuncAttributes attr{};
  if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess) {
    (void)cudaGetLastError(); /* not sticky; clear it */
    return 0;
  }
  return attr.binaryVersion;
}

/* Lowest architecture with the Tensor Core instruction each variant uses:
   WMMA FP16 sm_70, mma.sync m16n8k8 FP16 sm_75, BF16 (either) sm_80. */
template <typename T>
constexpr int tc_min_arch(ccml_matmul_algo algo) {
  return std::is_same<T, __nv_bfloat16>::value ? 80 : (algo == CCML_MATMUL_WMMA ? 70 : 75);
}

/* `arch`: compiled_arch() of the kernel about to run, cached by the caller in
   a function-local static (one per instantiation, so per device binary). */
template <typename T>
ccml_status require_tc(int arch, ccml_matmul_algo algo) {
  const int need = tc_min_arch<T>(algo);
  if (arch < need) {
    ccml::detail::set_last_error(
        (std::string("ccml_matmul: ") + ccml_matmul_algo_name(algo) + " " +
         (std::is_same<T, __nv_bfloat16>::value ? "BF16" : "FP16") + " needs sm_" +
         std::to_string(need) + "+; this binary runs sm_" + std::to_string(arch) +
         " code on this device")
            .c_str());
    return CCML_ERROR_NOT_IMPLEMENTED;
  }
  return CCML_SUCCESS;
}

template <typename T, int I>
ccml_status launch_wmma(T* C, const T* A, const T* B, int M, int N, int K,
                        cudaStream_t stream) {
  if constexpr (std::is_same<T, float>::value) {
    ccml::detail::set_last_error(
        "ccml_matmul: Tensor Core variants take 16-bit input (f16, or bf16 on sm_80+); "
        "sm_75 has no FP32 Tensor Core path");
    return CCML_ERROR_NOT_IMPLEMENTED;
  } else {
    constexpr MatmulTcConfig cfg = kMatmulTcConfigs[I];
    /* 128-bit access needs rows of A, B and C to split into whole 8-element
       vectors, and 16-byte-aligned pointers. */
    const bool vec = K % 8 == 0 && N % 8 == 0 && aligned(A, 16) && aligned(B, 16) &&
                     aligned(C, 16);
    ccml::detail::LaunchConfig config;
    config.block = dim3((cfg.bm / cfg.wm) * (cfg.bn / cfg.wn) * 32, 1, 1);
    config.grid = ccml::detail::grid_for_2d(N, M, cfg.bn, cfg.bm);
    void (*const matmul_wmma)(T*, const T*, const T*, int, int, int) =
        &ccml::kernels::matmul_wmma<T, cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, true>;
    void (*const matmul_wmma_scalar_global)(T*, const T*, const T*, int, int, int) =
        &ccml::kernels::matmul_wmma<T, cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, false>;
    static const int arch = compiled_arch(matmul_wmma);
    const ccml_status ok = require_tc<T>(arch, CCML_MATMUL_WMMA);
    if (ok != CCML_SUCCESS) return ok;
    if (vec) {
      CCML_LAUNCH(matmul_wmma, config, stream, C, A, B, M, N, K);
    } else {
      CCML_LAUNCH(matmul_wmma_scalar_global, config, stream, C, A, B, M, N, K);
    }
    return CCML_SUCCESS;
  }
}

template <typename T, int I>
ccml_status launch_mma_sync(T* C, const T* A, const T* B, int M, int N, int K,
                            cudaStream_t stream) {
  if constexpr (std::is_same<T, float>::value) {
    ccml::detail::set_last_error(
        "ccml_matmul: Tensor Core variants take 16-bit input (f16, or bf16 on sm_80+); "
        "sm_75 has no FP32 Tensor Core path");
    return CCML_ERROR_NOT_IMPLEMENTED;
  } else {
    constexpr MatmulTcConfig cfg = kMatmulTcConfigs[I];
    const bool vec = K % 8 == 0 && N % 8 == 0 && aligned(A, 16) && aligned(B, 16) &&
                     aligned(C, 16);
    ccml::detail::LaunchConfig config;
    config.block = dim3((cfg.bm / cfg.wm) * (cfg.bn / cfg.wn) * 32, 1, 1);
    config.grid = ccml::detail::grid_for_2d(N, M, cfg.bn, cfg.bm);
    void (*const matmul_mma_sync)(T*, const T*, const T*, int, int, int) =
        &ccml::kernels::matmul_mma_sync<T, cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, true>;
    void (*const matmul_mma_sync_scalar_global)(T*, const T*, const T*, int, int, int) =
        &ccml::kernels::matmul_mma_sync<T, cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, false>;
    static const int arch = compiled_arch(matmul_mma_sync);
    const ccml_status ok = require_tc<T>(arch, CCML_MATMUL_MMA_SYNC);
    if (ok != CCML_SUCCESS) return ok;
    if (vec) {
      CCML_LAUNCH(matmul_mma_sync, config, stream, C, A, B, M, N, K);
    } else {
      CCML_LAUNCH(matmul_mma_sync_scalar_global, config, stream, C, A, B, M, N, K);
    }
    return CCML_SUCCESS;
  }
}

/* Swizzled MMA_SYNC (entry 7, S1): group of 8 block-rows, 1-D grid. */
constexpr int kSwizzleGroup = 8;

template <typename T, int I>
ccml_status launch_mma_sync_swizzled(T* C, const T* A, const T* B, int M, int N, int K,
                                     cudaStream_t stream) {
  if constexpr (std::is_same<T, float>::value) {
    ccml::detail::set_last_error(
        "ccml_matmul: Tensor Core variants take 16-bit input (f16, or bf16 on sm_80+); "
        "sm_75 has no FP32 Tensor Core path");
    return CCML_ERROR_NOT_IMPLEMENTED;
  } else {
    constexpr MatmulTcConfig cfg = kMatmulTcConfigs[I];
    const bool vec = K % 8 == 0 && N % 8 == 0 && aligned(A, 16) && aligned(B, 16) &&
                     aligned(C, 16);
    const dim3 tiles = ccml::detail::grid_for_2d(N, M, cfg.bn, cfg.bm);
    ccml::detail::LaunchConfig config;
    config.block = dim3((cfg.bm / cfg.wm) * (cfg.bn / cfg.wn) * 32, 1, 1);
    config.grid = dim3(tiles.x * tiles.y, 1, 1);
    void (*const swizzled)(T*, const T*, const T*, int, int, int) =
        &ccml::kernels::matmul_mma_sync_swizzled<T, cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn,
                                                 true, kSwizzleGroup>;
    void (*const swizzled_scalar_global)(T*, const T*, const T*, int, int, int) =
        &ccml::kernels::matmul_mma_sync_swizzled<T, cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn,
                                                 false, kSwizzleGroup>;
    static const int arch = compiled_arch(swizzled);
    const ccml_status ok = require_tc<T>(arch, CCML_MATMUL_MMA_SYNC_SWIZZLED);
    if (ok != CCML_SUCCESS) return ok;
    if (vec) {
      CCML_LAUNCH(swizzled, config, stream, C, A, B, M, N, K);
    } else {
      CCML_LAUNCH(swizzled_scalar_global, config, stream, C, A, B, M, N, K);
    }
    return CCML_SUCCESS;
  }
}

/* ------------------------------------------------------------ split-K --
 *
 * Optimization log entry 6 (S2). Number of K slices: enough CTAs for two per
 * SM, at least 4 k-steps per slice. S = 1 runs MMA_SYNC directly (no
 * workspace, no second kernel). The FP32 workspace comes from the
 * stream-ordered allocator on the caller's stream, so concurrent calls on
 * different streams never share it. */

int sm_count() {
  static const int count = [] {
    int device = 0, n = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
      (void)cudaGetLastError();
      return 1;
    }
    return n;
  }();
  return count;
}

/* Slice length in elements of K (a multiple of bk) for a grid of `tiles`. */
int splitk_slice(int tiles, int K, int bk) {
  const int ksteps = (K + bk - 1) / bk;
  const int max_slices = ksteps / 4 > 1 ? ksteps / 4 : 1;
  int slices = (2 * sm_count() + tiles - 1) / tiles;
  if (slices > max_slices) slices = max_slices;
  if (slices < 1) slices = 1;
  return ((ksteps + slices - 1) / slices) * bk;
}

template <typename T, int I, bool kAtomic>
ccml_status launch_split_k(T* C, const T* A, const T* B, int M, int N, int K,
                           cudaStream_t stream) {
  if constexpr (std::is_same<T, float>::value) {
    ccml::detail::set_last_error(
        "ccml_matmul: Tensor Core variants take 16-bit input (f16, or bf16 on sm_80+); "
        "sm_75 has no FP32 Tensor Core path");
    return CCML_ERROR_NOT_IMPLEMENTED;
  } else {
    constexpr MatmulTcConfig cfg = kMatmulTcConfigs[I];
    const bool vec = K % 8 == 0 && N % 8 == 0 && aligned(A, 16) && aligned(B, 16) &&
                     aligned(C, 16);
    const dim3 tiles = ccml::detail::grid_for_2d(N, M, cfg.bn, cfg.bm);
    const int slice = splitk_slice(static_cast<int>(tiles.x * tiles.y), K, cfg.bk);
    const int slices = (K + slice - 1) / slice;
    if (slices == 1) return launch_mma_sync<T, I>(C, A, B, M, N, K, stream);

    void (*const splitk_vec)(float*, const T*, const T*, int, int, int, int) =
        &ccml::kernels::matmul_mma_sync_splitk<T, cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, true,
                                               kAtomic>;
    void (*const splitk_scalar_global)(float*, const T*, const T*, int, int, int, int) =
        &ccml::kernels::matmul_mma_sync_splitk<T, cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn,
                                               false, kAtomic>;
    static const int arch = compiled_arch(splitk_vec);
    const ccml_status ok =
        require_tc<T>(arch, kAtomic ? CCML_MATMUL_SPLIT_K_ATOMIC : CCML_MATMUL_SPLIT_K);
    if (ok != CCML_SUCCESS) return ok;

    const long long count = static_cast<long long>(M) * N;
    const std::size_t bytes = static_cast<std::size_t>(kAtomic ? 1 : slices) * count * 4;
    void* workspace = nullptr;
    CCML_CUDA_TRY(cudaMallocAsync(&workspace, bytes, stream));
    float* const W = static_cast<float*>(workspace);
    /* Every launch below returns through this lambda, so the workspace is
       always freed. */
    const ccml_status run = [&]() -> ccml_status {
      if (kAtomic) CCML_CUDA_TRY(cudaMemsetAsync(W, 0, bytes, stream));
      ccml::detail::LaunchConfig config;
      config.block = dim3((cfg.bm / cfg.wm) * (cfg.bn / cfg.wn) * 32, 1, 1);
      config.grid = dim3(tiles.x, tiles.y, static_cast<unsigned>(slices));
      if (vec) {
        CCML_LAUNCH(splitk_vec, config, stream, W, A, B, M, N, K, slice);
      } else {
        CCML_LAUNCH(splitk_scalar_global, config, stream, W, A, B, M, N, K, slice);
      }
      ccml::detail::LaunchConfig reduce;
      reduce.block = dim3(256, 1, 1);
      const long long blocks = (count + 255) / 256;
      reduce.grid = dim3(static_cast<unsigned>(blocks < 4096 ? blocks : 4096), 1, 1);
      CCML_LAUNCH(ccml::kernels::splitk_reduce<T>, reduce, stream, C, W, count,
                  kAtomic ? 1 : slices);
      return CCML_SUCCESS;
    }();
    CCML_CUDA_TRY(cudaFreeAsync(workspace, stream));
    return run;
  }
}

/* The configuration each Tensor Core algo enum runs. MMA_SYNC uses 1
   (64x64x32, 4 warps): configuration 0 spills at its 128-register occupancy
   floor, and 1 was fastest in the sm_89 sweep (optimization log entry 5); the
   T4 sweep re-checks the choice. */
int tc_default_config(ccml_matmul_algo algo) {
  return (algo == CCML_MATMUL_MMA_SYNC || algo == CCML_MATMUL_MMA_SYNC_SWIZZLED) ? 1 : 0;
}

/* Split-K's configuration for the enum: the 16-row tile for decode sizes
   (M <= 16), MMA_SYNC's default otherwise. */
int splitk_default_config(int M) { return M <= 16 ? 2 : 1; }

bool is_tc_family(ccml_matmul_algo algo) {
  return algo == CCML_MATMUL_WMMA || algo == CCML_MATMUL_MMA_SYNC ||
         algo == CCML_MATMUL_SPLIT_K || algo == CCML_MATMUL_SPLIT_K_ATOMIC ||
         algo == CCML_MATMUL_MMA_SYNC_SWIZZLED;
}

template <typename T, int I>
ccml_status launch_tc_family(ccml_matmul_algo algo, T* C, const T* A, const T* B, int M, int N,
                             int K, cudaStream_t stream) {
  switch (algo) {
    case CCML_MATMUL_WMMA:
      return launch_wmma<T, I>(C, A, B, M, N, K, stream);
    case CCML_MATMUL_MMA_SYNC:
      return launch_mma_sync<T, I>(C, A, B, M, N, K, stream);
    case CCML_MATMUL_SPLIT_K:
      return launch_split_k<T, I, false>(C, A, B, M, N, K, stream);
    case CCML_MATMUL_SPLIT_K_ATOMIC:
      return launch_split_k<T, I, true>(C, A, B, M, N, K, stream);
    case CCML_MATMUL_MMA_SYNC_SWIZZLED:
      return launch_mma_sync_swizzled<T, I>(C, A, B, M, N, K, stream);
    default:
      ccml::detail::set_last_error("ccml_matmul: not a Tensor Core algorithm");
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

template <typename T>
ccml_status tc_family(ccml_matmul_algo algo, T* C, const T* A, const T* B, int M, int N, int K,
                      int index, cudaStream_t stream) {
  static_assert(ccml::kernels::kMatmulTcConfigCount == 3,
                "add a case below for every Tensor Core configuration");
  switch (index) {
    case 0: return launch_tc_family<T, 0>(algo, C, A, B, M, N, K, stream);
    case 1: return launch_tc_family<T, 1>(algo, C, A, B, M, N, K, stream);
    case 2: return launch_tc_family<T, 2>(algo, C, A, B, M, N, K, stream);
    default:
      ccml::detail::set_last_error("ccml_matmul: Tensor Core config index out of range");
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

    case CCML_MATMUL_WMMA:
    case CCML_MATMUL_MMA_SYNC:
    case CCML_MATMUL_MMA_SYNC_SWIZZLED:
      return tc_family(algo, C, A, B, M, N, K, tc_default_config(algo), stream);

    case CCML_MATMUL_SPLIT_K:
    case CCML_MATMUL_SPLIT_K_ATOMIC:
      return tc_family(algo, C, A, B, M, N, K, splitk_default_config(M), stream);

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
  if (is_tc_family(algo)) {
    return tc_family(algo, ccml::detail::device_ptr(C), ccml::detail::device_ptr(A),
                     ccml::detail::device_ptr(B), M, N, K, index, stream);
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
    case CCML_MATMUL_WMMA:           return "wmma";
    case CCML_MATMUL_MMA_SYNC:       return "mma_sync";
    case CCML_MATMUL_SPLIT_K:        return "split_k";
    case CCML_MATMUL_SPLIT_K_ATOMIC: return "split_k_atomic";
    case CCML_MATMUL_MMA_SYNC_SWIZZLED: return "mma_sync_swizzled";
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

int ccml_matmul_tc_config_count(void) { return ccml::kernels::kMatmulTcConfigCount; }

ccml_status ccml_matmul_tc_config_get(int index, ccml_matmul_tc_config* out) {
  ccml::detail::clear_last_error();
  CCML_REQUIRE(out != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_matmul_tc_config_get: out is null");
  CCML_REQUIRE(index >= 0 && index < ccml::kernels::kMatmulTcConfigCount,
               CCML_ERROR_INVALID_ARGUMENT, "ccml_matmul_tc_config_get: index out of range");
  const ccml::kernels::MatmulTcConfig& c = ccml::kernels::kMatmulTcConfigs[index];
  out->bm = c.bm;
  out->bn = c.bn;
  out->bk = c.bk;
  out->wm = c.wm;
  out->wn = c.wn;
  out->warps = (c.bm / c.wm) * (c.bn / c.wn);
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
