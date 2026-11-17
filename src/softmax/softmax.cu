#include "ccml/softmax.h"

#include <cstdint>
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

/* Warp per row (entry 14). */
template <typename T, int kV, int kChunks>
ccml_status launch_wpr(T* y, const T* x, int rows, int cols, cudaStream_t stream) {
  constexpr int kRows = ccml::kernels::kSoftmaxRowsPerBlock;
  ccml::detail::LaunchConfig config;
  config.block = dim3(32 * kRows, 1, 1);
  config.grid = dim3(static_cast<unsigned>((rows + kRows - 1) / kRows), 1, 1);
  /* Through a pointer: the template argument list's commas would split the macro. */
  void (*const softmax_warp_per_row)(T*, const T*, int, int) =
      &ccml::kernels::softmax_warp_per_row<T, kV, kChunks>;
  CCML_LAUNCH(softmax_warp_per_row, config, stream, y, x, rows, cols);
  return CCML_SUCCESS;
}

template <typename T, int kV>
ccml_status launch_wpr_chunks(T* y, const T* x, int rows, int cols, cudaStream_t stream) {
  /* cols <= 1024, so at most 1024 / (32 kV) chunks per lane: larger buckets
     are never instantiated (they would only be dead, spilling code). */
  constexpr int kMax = 1024 / (32 * kV);
  const int chunks = (cols / kV + 31) / 32;
  if (chunks <= 1) return launch_wpr<T, kV, 1>(y, x, rows, cols, stream);
  if constexpr (kMax >= 2) if (chunks <= 2) return launch_wpr<T, kV, 2>(y, x, rows, cols, stream);
  if constexpr (kMax >= 4) if (chunks <= 4) return launch_wpr<T, kV, 4>(y, x, rows, cols, stream);
  if constexpr (kMax >= 8) if (chunks <= 8) return launch_wpr<T, kV, 8>(y, x, rows, cols, stream);
  if constexpr (kMax >= 16) if (chunks <= 16) return launch_wpr<T, kV, 16>(y, x, rows, cols, stream);
  if constexpr (kMax >= 32) return launch_wpr<T, kV, 32>(y, x, rows, cols, stream);
  return CCML_ERROR_UNSUPPORTED_SHAPE; /* unreachable: cols <= 1024 is checked first */
}

template <typename T>
ccml_status launch_warp_per_row(T* y, const T* x, int rows, int cols, cudaStream_t stream,
                                const std::string& fn) {
  if (cols > 1024) {
    ccml::detail::set_last_error(
        (fn + ": WARP_PER_ROW holds a row in one warp's registers, so cols <= 1024").c_str());
    return CCML_ERROR_UNSUPPORTED_SHAPE;
  }
  constexpr int kV = static_cast<int>(16 / sizeof(T));
  const bool vec = cols % kV == 0 && reinterpret_cast<std::uintptr_t>(x) % 16 == 0 &&
                   reinterpret_cast<std::uintptr_t>(y) % 16 == 0;
  return vec ? launch_wpr_chunks<T, kV>(y, x, rows, cols, stream)
             : launch_wpr_chunks<T, 1>(y, x, rows, cols, stream);
}

/* Split row (entry 15): S segments per row so that rows x S >= 2 x SMs, each
   segment at least kMinSegment columns. S = 1 runs ONLINE (no workspace). */
constexpr int kMinSegment = 2048;

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

template <typename T>
ccml_status launch_split_row(T* y, const T* x, int rows, int cols, cudaStream_t stream) {
  int segments = (2 * sm_count() + rows - 1) / rows;
  const int max_segments = cols / kMinSegment > 1 ? cols / kMinSegment : 1;
  if (segments > max_segments) segments = max_segments;
  if (segments <= 1) {
    const ccml::detail::LaunchConfig config = ccml::detail::rowwise_launch_config(
        rows, cols, ccml::kernels::kSoftmaxBlockSize);
    CCML_LAUNCH(ccml::kernels::softmax_online<T>, config, stream, y, x, rows, cols);
    return CCML_SUCCESS;
  }
  const int seg = (cols + segments - 1) / segments;
  segments = (cols + seg - 1) / seg;
  const std::size_t bytes =
      static_cast<std::size_t>(rows) * segments * sizeof(ccml::kernels::OnlineMaxSum);
  void* workspace = nullptr;
  CCML_CUDA_TRY(cudaMallocAsync(&workspace, bytes, stream));
  auto* partial = static_cast<ccml::kernels::OnlineMaxSum*>(workspace);
  /* Every launch returns through this lambda, so the workspace is always freed. */
  const ccml_status run = [&]() -> ccml_status {
    ccml::detail::LaunchConfig config;
    config.block = dim3(static_cast<unsigned>(ccml::kernels::kSoftmaxBlockSize), 1, 1);
    config.grid = dim3(static_cast<unsigned>(segments), static_cast<unsigned>(rows), 1);
    CCML_LAUNCH(ccml::kernels::softmax_split_stats<T>, config, stream, partial, x, rows, cols,
                seg);
    CCML_LAUNCH(ccml::kernels::softmax_split_write<T>, config, stream, y, partial, x, rows,
                cols, seg);
    return CCML_SUCCESS;
  }();
  CCML_CUDA_TRY(cudaFreeAsync(workspace, stream));
  return run;
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

    case CCML_SOFTMAX_ONLINE: {
      const ccml::detail::LaunchConfig config = ccml::detail::rowwise_launch_config(
          rows, cols, ccml::kernels::kSoftmaxBlockSize);
      CCML_LAUNCH(ccml::kernels::softmax_online<T>, config, stream, y, x, rows, cols);
      return CCML_SUCCESS;
    }

    case CCML_SOFTMAX_WARP_PER_ROW:
      return launch_warp_per_row(y, x, rows, cols, stream, fn_name<Public>());

    case CCML_SOFTMAX_SPLIT_ROW:
      return launch_split_row(y, x, rows, cols, stream);

    default:
      ccml::detail::set_last_error((fn_name<Public>() + ": unknown algorithm").c_str());
      return CCML_ERROR_INVALID_ARGUMENT;
  }
}

/* --------------------------------------------- fused scale + mask --- */

template <typename T, int kV, int kChunks>
ccml_status launch_mwpr(T* y, const T* x, int rows, int cols,
                        const ccml::kernels::SoftmaxMask& m, cudaStream_t stream) {
  constexpr int kRows = ccml::kernels::kSoftmaxRowsPerBlock;
  ccml::detail::LaunchConfig config;
  config.block = dim3(32 * kRows, 1, 1);
  config.grid = dim3(static_cast<unsigned>((rows + kRows - 1) / kRows), 1, 1);
  void (*const softmax_masked_warp_per_row)(T*, const T*, int, int, ccml::kernels::SoftmaxMask) =
      &ccml::kernels::softmax_masked_warp_per_row<T, kV, kChunks>;
  CCML_LAUNCH(softmax_masked_warp_per_row, config, stream, y, x, rows, cols, m);
  return CCML_SUCCESS;
}

template <typename T, int kV>
ccml_status launch_mwpr_chunks(T* y, const T* x, int rows, int cols,
                               const ccml::kernels::SoftmaxMask& m, cudaStream_t stream) {
  /* cols <= 1024, so at most 1024 / (32 kV) chunks per lane: larger buckets
     are never instantiated (they would only be dead, spilling code). */
  constexpr int kMax = 1024 / (32 * kV);
  const int chunks = (cols / kV + 31) / 32;
  if (chunks <= 1) return launch_mwpr<T, kV, 1>(y, x, rows, cols, m, stream);
  if constexpr (kMax >= 2) if (chunks <= 2) return launch_mwpr<T, kV, 2>(y, x, rows, cols, m, stream);
  if constexpr (kMax >= 4) if (chunks <= 4) return launch_mwpr<T, kV, 4>(y, x, rows, cols, m, stream);
  if constexpr (kMax >= 8) if (chunks <= 8) return launch_mwpr<T, kV, 8>(y, x, rows, cols, m, stream);
  if constexpr (kMax >= 16) if (chunks <= 16) return launch_mwpr<T, kV, 16>(y, x, rows, cols, m, stream);
  if constexpr (kMax >= 32) return launch_mwpr<T, kV, 32>(y, x, rows, cols, m, stream);
  return CCML_ERROR_UNSUPPORTED_SHAPE; /* unreachable: cols <= 1024 is checked first */
}

template <typename Public>
ccml_status softmax_masked(Public* y_pub, const Public* x_pub, const Public* mask_pub,
                           int mask_rows, int rows, int cols, float scale, int causal_q,
                           ccml_softmax_algo algo, cudaStream_t stream) {
  using T = ccml::detail::device_t<Public>;
  ccml::detail::clear_last_error();
  const ccml_status valid = validate(y_pub, x_pub, rows, cols);
  if (valid != CCML_SUCCESS) return valid;
  const std::string fn = std::string("ccml_softmax_masked_") + ccml::detail::dtype_suffix<Public>();
  CCML_REQUIRE(mask_pub == nullptr || mask_rows > 0, CCML_ERROR_INVALID_ARGUMENT,
               (fn + ": mask_rows must be positive when a mask is given").c_str());
  CCML_REQUIRE(causal_q >= 0, CCML_ERROR_INVALID_ARGUMENT,
               (fn + ": causal_q must be >= 0 (0 = no causal mask)").c_str());
  T* y = ccml::detail::device_ptr(y_pub);
  const T* x = ccml::detail::device_ptr(x_pub);
  ccml::kernels::SoftmaxMask m{ccml::detail::device_ptr(mask_pub), mask_pub ? mask_rows : 1,
                               scale, causal_q};
  if (algo == CCML_SOFTMAX_AUTO) algo = cols <= 1024 ? CCML_SOFTMAX_WARP_PER_ROW : CCML_SOFTMAX_ONLINE;
  switch (algo) {
    case CCML_SOFTMAX_WARP_PER_ROW: {
      if (cols > 1024) {
        ccml::detail::set_last_error(
            (fn + ": WARP_PER_ROW holds a row in one warp's registers, so cols <= 1024").c_str());
        return CCML_ERROR_UNSUPPORTED_SHAPE;
      }
      constexpr int kV = static_cast<int>(16 / sizeof(T));
      const bool vec = cols % kV == 0 && reinterpret_cast<std::uintptr_t>(x) % 16 == 0 &&
                       reinterpret_cast<std::uintptr_t>(y) % 16 == 0;
      return vec ? launch_mwpr_chunks<T, kV>(y, x, rows, cols, m, stream)
                 : launch_mwpr_chunks<T, 1>(y, x, rows, cols, m, stream);
    }
    case CCML_SOFTMAX_ONLINE: {
      const ccml::detail::LaunchConfig config = ccml::detail::rowwise_launch_config(
          rows, cols, ccml::kernels::kSoftmaxBlockSize);
      CCML_LAUNCH(ccml::kernels::softmax_masked_online<T>, config, stream, y, x, rows, cols, m);
      return CCML_SUCCESS;
    }
    default:
      ccml::detail::set_last_error(
          (fn + ": the fused mask exists for WARP_PER_ROW and ONLINE only").c_str());
      return CCML_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace

extern "C" {

const char* ccml_softmax_algo_name(ccml_softmax_algo algo) {
  switch (algo) {
    case CCML_SOFTMAX_AUTO:          return "auto";
    case CCML_SOFTMAX_BLOCK_PER_ROW: return "block_per_row";
    case CCML_SOFTMAX_WARP_PER_ROW:  return "warp_per_row";
    case CCML_SOFTMAX_ONLINE:        return "online";
    case CCML_SOFTMAX_SPLIT_ROW:     return "split_row";
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

ccml_status ccml_softmax_masked_f32(float* y, const float* x, const float* mask, int mask_rows,
                                    int rows, int cols, float scale, int causal_q,
                                    ccml_softmax_algo algo, cudaStream_t stream) {
  return softmax_masked(y, x, mask, mask_rows, rows, cols, scale, causal_q, algo, stream);
}

ccml_status ccml_softmax_masked_f16(ccml_f16* y, const ccml_f16* x, const ccml_f16* mask,
                                    int mask_rows, int rows, int cols, float scale, int causal_q,
                                    ccml_softmax_algo algo, cudaStream_t stream) {
  return softmax_masked(y, x, mask, mask_rows, rows, cols, scale, causal_q, algo, stream);
}

ccml_status ccml_softmax_masked_bf16(ccml_bf16* y, const ccml_bf16* x, const ccml_bf16* mask,
                                     int mask_rows, int rows, int cols, float scale, int causal_q,
                                     ccml_softmax_algo algo, cudaStream_t stream) {
  return softmax_masked(y, x, mask, mask_rows, rows, cols, scale, causal_q, algo, stream);
}

}  // extern "C"
