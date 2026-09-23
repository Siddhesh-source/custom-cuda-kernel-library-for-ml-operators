#include "ccml/matmul.h"

#include "common/check.h"
#include "common/launch.h"
#include "common/util.h"
#include "matmul/matmul_kernels.cuh"

namespace {

ccml_status validate(const float* C, const float* A, const float* B,
                     int M, int N, int K) {
  CCML_REQUIRE(C != nullptr, CCML_ERROR_INVALID_ARGUMENT, "ccml_matmul_f32: C is null");
  CCML_REQUIRE(A != nullptr, CCML_ERROR_INVALID_ARGUMENT, "ccml_matmul_f32: A is null");
  CCML_REQUIRE(B != nullptr, CCML_ERROR_INVALID_ARGUMENT, "ccml_matmul_f32: B is null");
  CCML_REQUIRE(M > 0 && N > 0 && K > 0, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_matmul_f32: M, N and K must all be positive");
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

}  // namespace

extern "C" {

const char* ccml_matmul_algo_name(ccml_matmul_algo algo) {
  switch (algo) {
    case CCML_MATMUL_AUTO:  return "auto";
    case CCML_MATMUL_NAIVE: return "naive";
    case CCML_MATMUL_TILED: return "tiled";
  }
  return "unknown";
}

ccml_status ccml_matmul_f32(float* C, const float* A, const float* B,
                            int M, int N, int K,
                            ccml_matmul_algo algo, cudaStream_t stream) {
  ccml::detail::clear_last_error();

  const ccml_status valid = validate(C, A, B, M, N, K);
  if (valid != CCML_SUCCESS) {
    return valid;
  }

  const ccml::detail::LaunchConfig config = matmul_launch_config(M, N);

  switch (resolve(algo)) {
    case CCML_MATMUL_NAIVE:
      CCML_LAUNCH(ccml::kernels::matmul_naive_f32, config, stream,
                  C, A, B, M, N, K);
      return CCML_SUCCESS;

    case CCML_MATMUL_TILED:
      CCML_LAUNCH(ccml::kernels::matmul_tiled_f32, config, stream,
                  C, A, B, M, N, K);
      return CCML_SUCCESS;

    default:
      ccml::detail::set_last_error("ccml_matmul_f32: unknown algorithm");
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

}  // extern "C"
