#include "ccml/memory.h"

#include <cuda_runtime.h>

#include "common/check.h"

extern "C" {

ccml_status ccml_device_malloc(void** ptr, size_t bytes) {
  CCML_REQUIRE(ptr != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_device_malloc: ptr is null");
  CCML_REQUIRE(bytes > 0, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_device_malloc: bytes is zero");
  ccml::detail::clear_last_error();

  *ptr = nullptr;
  cudaError_t err = cudaMalloc(ptr, bytes);

  /* Out of memory is separated from other CUDA failures because callers can
     act on it -- retry with a smaller shape -- where they cannot act on, say,
     a corrupt context. */
  if (err == cudaErrorMemoryAllocation) {
    ccml::detail::set_last_cuda_error("cudaMalloc", err);
    return CCML_ERROR_OUT_OF_MEMORY;
  }
  if (err != cudaSuccess) {
    ccml::detail::set_last_cuda_error("cudaMalloc", err);
    return CCML_ERROR_CUDA;
  }
  return CCML_SUCCESS;
}

ccml_status ccml_device_free(void* ptr) {
  ccml::detail::clear_last_error();
  if (ptr == nullptr) {
    return CCML_SUCCESS;  /* freeing null is a no-op, as with free() */
  }
  CCML_CUDA_TRY(cudaFree(ptr));
  return CCML_SUCCESS;
}

ccml_status ccml_memcpy_h2d(void* dst, const void* src, size_t bytes) {
  CCML_REQUIRE(dst != nullptr && src != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_memcpy_h2d: null pointer");
  ccml::detail::clear_last_error();
  CCML_CUDA_TRY(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
  return CCML_SUCCESS;
}

ccml_status ccml_memcpy_d2h(void* dst, const void* src, size_t bytes) {
  CCML_REQUIRE(dst != nullptr && src != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_memcpy_d2h: null pointer");
  ccml::detail::clear_last_error();
  CCML_CUDA_TRY(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
  return CCML_SUCCESS;
}

ccml_status ccml_memcpy_h2d_async(void* dst, const void* src, size_t bytes,
                                  cudaStream_t stream) {
  CCML_REQUIRE(dst != nullptr && src != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_memcpy_h2d_async: null pointer");
  ccml::detail::clear_last_error();
  CCML_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
  return CCML_SUCCESS;
}

ccml_status ccml_memcpy_d2h_async(void* dst, const void* src, size_t bytes,
                                  cudaStream_t stream) {
  CCML_REQUIRE(dst != nullptr && src != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_memcpy_d2h_async: null pointer");
  ccml::detail::clear_last_error();
  CCML_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
  return CCML_SUCCESS;
}

ccml_status ccml_device_memset_async(void* ptr, int value, size_t bytes,
                                     cudaStream_t stream) {
  CCML_REQUIRE(ptr != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_device_memset_async: ptr is null");
  ccml::detail::clear_last_error();
  CCML_CUDA_TRY(cudaMemsetAsync(ptr, value, bytes, stream));
  return CCML_SUCCESS;
}

ccml_status ccml_stream_synchronize(cudaStream_t stream) {
  ccml::detail::clear_last_error();
  CCML_CUDA_TRY(cudaStreamSynchronize(stream));
  return CCML_SUCCESS;
}

ccml_status ccml_device_synchronize(void) {
  ccml::detail::clear_last_error();
  CCML_CUDA_TRY(cudaDeviceSynchronize());
  return CCML_SUCCESS;
}

}  // extern "C"
