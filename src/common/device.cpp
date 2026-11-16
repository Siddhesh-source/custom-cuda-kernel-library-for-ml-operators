#include "ccml/device.h"

#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "common/check.h"

extern "C" {

ccml_status ccml_device_count(int* count) {
  CCML_REQUIRE(count != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_device_count: count is null");
  ccml::detail::clear_last_error();
  CCML_CUDA_TRY(cudaGetDeviceCount(count));
  return CCML_SUCCESS;
}

ccml_status ccml_get_device_info(int device, ccml_device_info* info) {
  CCML_REQUIRE(info != nullptr, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_get_device_info: info is null");
  CCML_REQUIRE(device >= 0, CCML_ERROR_INVALID_ARGUMENT,
               "ccml_get_device_info: device index is negative");
  ccml::detail::clear_last_error();

  cudaDeviceProp prop;
  CCML_CUDA_TRY(cudaGetDeviceProperties(&prop, device));

  std::memset(info, 0, sizeof(*info));
  info->device = device;
  std::snprintf(info->name, sizeof(info->name), "%s", prop.name);
  info->compute_major                  = prop.major;
  info->compute_minor                  = prop.minor;
  info->multiprocessor_count           = prop.multiProcessorCount;
  info->max_threads_per_block          = prop.maxThreadsPerBlock;
  info->max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
  info->warp_size                      = prop.warpSize;
  info->shared_mem_per_block           = prop.sharedMemPerBlock;
  info->shared_mem_per_multiprocessor  = prop.sharedMemPerMultiprocessor;
  info->regs_per_block                 = prop.regsPerBlock;
  info->total_global_mem               = prop.totalGlobalMem;
  info->l2_cache_size                  = prop.l2CacheSize;
  info->memory_bus_width               = prop.memoryBusWidth;
  /* cudaDeviceProp::clockRate and ::memoryClockRate were removed in CUDA 13;
     the attribute query works on every toolkit from 12.0 on. Both are kHz. */
  int memory_clock_khz = 0;
  int clock_khz        = 0;
  CCML_CUDA_TRY(cudaDeviceGetAttribute(&memory_clock_khz,
                                       cudaDevAttrMemoryClockRate, device));
  CCML_CUDA_TRY(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
  info->memory_clock_khz               = memory_clock_khz;
  info->clock_khz                      = clock_khz;
  return CCML_SUCCESS;
}

}  // extern "C"
