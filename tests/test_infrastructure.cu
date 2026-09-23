/*
 * Infrastructure self-test.
 *
 * This is the only test that launches a kernel of its own, and it is
 * deliberately white-box: it includes src/common/launch.h to exercise launch
 * validation directly, rather than only through an operator. The kernels here
 * are trivial on purpose -- they exist to prove the plumbing works, not to
 * compute anything interesting.
 *
 * If this file fails, no operator result can be trusted, so it is worth running
 * first.
 *
 * Note on error detail: helpers included from src/common/check.h write to a
 * buffer belonging to THIS binary, read with ccml::detail::last_error(). Calls
 * into the shared library write to the library's buffer, read with the exported
 * ccml_last_error_string(). Both appear below, and the distinction is real.
 */

#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "ccml/device.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "common/launch.h"
#include "support/cuda_event.h"
#include "support/cuda_stream.h"
#include "support/device_buffer.h"
#include "tests/support/harness.h"

using ccml::detail::LaunchConfig;
using ccml_support::CudaStream;
using ccml_support::DeviceBuffer;
using ccml_support::EventTimer;

namespace {

constexpr int kBlockSize = 256;

__global__ void fill_iota_kernel(float* out, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    out[index] = static_cast<float>(index);
  }
}

__global__ void scale_kernel(float* out, const float* in, float factor, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    out[index] = in[index] * factor;
  }
}

/* Deliberately spends time on the device, so the event timer has something
   longer than its own resolution to measure. */
__global__ void busy_kernel(float* out, int count, int iterations) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  float value = out[index];
  for (int i = 0; i < iterations; ++i) {
    value = value * 1.0000001f + 1.0f;
  }
  out[index] = value;
}

/* These wrappers are the pattern every operator will follow: build a config,
   hand it to CCML_LAUNCH, return a status. */
ccml_status launch_fill_iota(float* out, int count, cudaStream_t stream) {
  LaunchConfig config;
  config.block = dim3(kBlockSize, 1, 1);
  config.grid  = ccml::detail::grid_for_1d(count, kBlockSize);
  CCML_LAUNCH(fill_iota_kernel, config, stream, out, count);
  return CCML_SUCCESS;
}

ccml_status launch_scale(float* out, const float* in, float factor, int count,
                         cudaStream_t stream) {
  LaunchConfig config;
  config.block = dim3(kBlockSize, 1, 1);
  config.grid  = ccml::detail::grid_for_1d(count, kBlockSize);
  CCML_LAUNCH(scale_kernel, config, stream, out, in, factor, count);
  return CCML_SUCCESS;
}

ccml_status launch_busy(float* out, int count, int iterations, cudaStream_t stream) {
  LaunchConfig config;
  config.block = dim3(kBlockSize, 1, 1);
  config.grid  = ccml::detail::grid_for_1d(count, kBlockSize);
  CCML_LAUNCH(busy_kernel, config, stream, out, count, iterations);
  return CCML_SUCCESS;
}

}  // namespace

/* ---------------------------------------------------------------- device -- */

CCML_TEST(infrastructure, device_query_is_sane) {
  int count = 0;
  CCML_ASSERT_OK(ccml_device_count(&count));
  ctx.check(count > 0, "no CUDA device available");

  ccml_device_info info;
  CCML_ASSERT_OK(ccml_get_device_info(0, &info));

  ctx.note(std::string(info.name) + " sm_" + std::to_string(info.compute_major)
           + std::to_string(info.compute_minor) + ", "
           + std::to_string(info.multiprocessor_count) + " SMs");

  /* Properties that every tiling decision depends on. A zero here would make
     later occupancy reasoning silently meaningless. */
  ctx.check(info.compute_major > 0, "compute capability major is zero");
  ctx.check(info.multiprocessor_count > 0, "SM count is zero");
  ctx.check(info.max_threads_per_block >= 256, "max threads per block below 256");
  ctx.check(info.warp_size == 32, "warp size is not 32");
  ctx.check(info.shared_mem_per_block >= 16384, "shared memory per block below 16 KiB");
  ctx.check(info.total_global_mem > 0, "global memory is zero");
}

/* ---------------------------------------------------------------- memory -- */

CCML_TEST(infrastructure, memory_round_trip) {
  const std::size_t count = 4096;
  std::vector<float> host_in(count);
  for (std::size_t i = 0; i < count; ++i) {
    host_in[i] = static_cast<float>(i) * 0.5f;
  }

  DeviceBuffer buffer(count);
  buffer.upload(host_in.data());

  std::vector<float> host_out(count, -1.0f);
  buffer.download(host_out.data());

  for (std::size_t i = 0; i < count; ++i) {
    if (host_out[i] != host_in[i]) {
      ctx.fail("round trip differed at index " + std::to_string(i));
    }
  }
}

CCML_TEST(infrastructure, memset_zeroes_device_memory) {
  const std::size_t count = 1024;
  DeviceBuffer buffer(count);

  std::vector<float> host(count, 7.0f);
  buffer.upload(host.data());

  CCML_ASSERT_OK(ccml_device_memset_async(buffer.data(), 0, buffer.bytes(), nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  buffer.download(host.data());
  for (std::size_t i = 0; i < count; ++i) {
    if (host[i] != 0.0f) {
      ctx.fail("memset left " + std::to_string(host[i]) + " at index "
               + std::to_string(i));
    }
  }
}

/* ------------------------------------------------------- kernel execution -- */

/* The one that matters: does a kernel actually run on the GPU and produce the
   values it should. */
CCML_TEST(infrastructure, kernel_launch_executes) {
  const int count = 10000;  /* not a multiple of the block size, on purpose */
  DeviceBuffer buffer(static_cast<std::size_t>(count));

  /* Poison first, so a kernel that fails to write every element is caught
     rather than reading back a plausible-looking zero. */
  CCML_ASSERT_OK(ccml_device_memset_async(buffer.data(), 0xFF, buffer.bytes(),
                                          nullptr));
  CCML_ASSERT_OK(launch_fill_iota(buffer.data(), count, nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  std::vector<float> host(static_cast<std::size_t>(count));
  buffer.download(host.data());

  for (int i = 0; i < count; ++i) {
    if (host[static_cast<std::size_t>(i)] != static_cast<float>(i)) {
      ctx.fail("fill_iota wrote " + std::to_string(host[static_cast<std::size_t>(i)])
               + " at index " + std::to_string(i) + ", expected "
               + std::to_string(i));
    }
  }
  ctx.note("launched and verified " + std::to_string(count) + " elements");
}

CCML_TEST(infrastructure, two_kernels_chain_correctly) {
  const int count = 5000;
  DeviceBuffer source(static_cast<std::size_t>(count));
  DeviceBuffer target(static_cast<std::size_t>(count));

  CCML_ASSERT_OK(launch_fill_iota(source.data(), count, nullptr));
  CCML_ASSERT_OK(launch_scale(target.data(), source.data(), 3.0f, count, nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  std::vector<float> host(static_cast<std::size_t>(count));
  target.download(host.data());

  for (int i = 0; i < count; ++i) {
    const float expected = static_cast<float>(i) * 3.0f;
    if (host[static_cast<std::size_t>(i)] != expected) {
      ctx.fail("chained kernels gave "
               + std::to_string(host[static_cast<std::size_t>(i)])
               + " at index " + std::to_string(i));
    }
  }
}

/* --------------------------------------------------------------- streams -- */

CCML_TEST(infrastructure, work_runs_on_an_explicit_stream) {
  const int count = 8192;
  CudaStream stream = CudaStream::create();
  ctx.check(stream.get() != nullptr, "created stream handle is null");

  DeviceBuffer buffer(static_cast<std::size_t>(count));

  std::vector<float> host(static_cast<std::size_t>(count), -1.0f);
  CCML_ASSERT_OK(ccml_memcpy_h2d_async(buffer.data(), host.data(), buffer.bytes(),
                                       stream.get()));
  CCML_ASSERT_OK(launch_fill_iota(buffer.data(), count, stream.get()));
  CCML_ASSERT_OK(ccml_memcpy_d2h_async(host.data(), buffer.data(), buffer.bytes(),
                                       stream.get()));

  /* Waiting on the stream, not the whole device. Reading host before this point
     would be a race, not a result. */
  CCML_ASSERT_OK(ccml_stream_synchronize(stream.get()));

  for (int i = 0; i < count; ++i) {
    if (host[static_cast<std::size_t>(i)] != static_cast<float>(i)) {
      ctx.fail("stream-ordered work gave the wrong value at index "
               + std::to_string(i));
    }
  }
}

CCML_TEST(infrastructure, default_constructed_stream_is_the_default_stream) {
  CudaStream stream;
  ctx.check(stream.get() == nullptr,
            "a default-constructed CudaStream should be the null (default) stream");

  DeviceBuffer buffer(256);
  CCML_ASSERT_OK(launch_fill_iota(buffer.data(), 256, stream.get()));
  CCML_ASSERT_OK(ccml_device_synchronize());
}

/* ----------------------------------------------------- launch validation -- */

CCML_TEST(infrastructure, launch_validation_accepts_a_reasonable_config) {
  LaunchConfig config;
  config.block = dim3(256, 1, 1);
  config.grid  = dim3(64, 1, 1);

  const ccml_status status =
      ccml::detail::validate_launch(config, "reasonable");
  ctx.check(status == CCML_SUCCESS,
            std::string("rejected a valid config: ") + ccml::detail::last_error());
}

CCML_TEST(infrastructure, launch_validation_rejects_oversized_block) {
  ccml_device_info info;
  CCML_ASSERT_OK(ccml_get_device_info(0, &info));

  LaunchConfig config;
  config.block = dim3(static_cast<unsigned>(info.max_threads_per_block) + 1, 1, 1);
  config.grid  = dim3(1, 1, 1);

  const ccml_status status =
      ccml::detail::validate_launch(config, "oversized_block");
  ctx.check(status == CCML_ERROR_INVALID_ARGUMENT,
            std::string("expected CCML_ERROR_INVALID_ARGUMENT, got ")
                + ccml_status_string(status));

  const std::string message = ccml::detail::last_error();
  ctx.note("message: " + message);
  ctx.check(message.find("oversized_block") != std::string::npos,
            "the message should name the launch site");
}

CCML_TEST(infrastructure, launch_validation_rejects_empty_block) {
  LaunchConfig config;
  config.block = dim3(0, 1, 1);
  config.grid  = dim3(1, 1, 1);

  const ccml_status status = ccml::detail::validate_launch(config, "empty_block");
  ctx.check(status == CCML_ERROR_INVALID_ARGUMENT,
            std::string("expected CCML_ERROR_INVALID_ARGUMENT, got ")
                + ccml_status_string(status));
}

CCML_TEST(infrastructure, launch_validation_rejects_zero_grid) {
  LaunchConfig config;
  config.block = dim3(128, 1, 1);
  config.grid  = dim3(0, 1, 1);

  const ccml_status status = ccml::detail::validate_launch(config, "zero_grid");
  ctx.check(status == CCML_ERROR_INVALID_ARGUMENT,
            std::string("expected CCML_ERROR_INVALID_ARGUMENT, got ")
                + ccml_status_string(status));
  ctx.note("message: " + std::string(ccml::detail::last_error()));
}

CCML_TEST(infrastructure, launch_validation_rejects_excess_shared_memory) {
  ccml_device_info info;
  CCML_ASSERT_OK(ccml_get_device_info(0, &info));

  LaunchConfig config;
  config.block         = dim3(128, 1, 1);
  config.grid          = dim3(1, 1, 1);
  config.shared_bytes  = info.shared_mem_per_block + 1024;

  const ccml_status status = ccml::detail::validate_launch(config, "excess_shared");
  ctx.check(status == CCML_ERROR_UNSUPPORTED_SHAPE,
            std::string("expected CCML_ERROR_UNSUPPORTED_SHAPE, got ")
                + ccml_status_string(status));
  ctx.note("message: " + std::string(ccml::detail::last_error()));
}

/* An invalid configuration must be refused before it reaches the driver, and
   must leave the device in a usable state afterwards. */
CCML_TEST(infrastructure, rejected_launch_leaves_the_device_usable) {
  DeviceBuffer buffer(1024);

  LaunchConfig bad;
  bad.block = dim3(100000, 1, 1);
  bad.grid  = dim3(1, 1, 1);
  const ccml_status status = ccml::detail::validate_launch(bad, "bad");
  ctx.check(status != CCML_SUCCESS, "an absurd block size was accepted");

  CCML_ASSERT_OK(launch_fill_iota(buffer.data(), 1024, nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  std::vector<float> host(1024);
  buffer.download(host.data());
  ctx.check(host[1023] == 1023.0f,
            "the device did not work normally after a rejected launch");
}

/* ---------------------------------------------------------------- timing -- */

CCML_TEST(infrastructure, event_timer_measures_a_kernel) {
  const int count = 1 << 16;
  DeviceBuffer buffer(static_cast<std::size_t>(count));
  CCML_ASSERT_OK(ccml_device_memset_async(buffer.data(), 0, buffer.bytes(), nullptr));

  CudaStream stream = CudaStream::create();

  /* Warm up first: the first launch pays module load, which would dominate. */
  CCML_ASSERT_OK(launch_busy(buffer.data(), count, 2000, stream.get()));
  CCML_ASSERT_OK(ccml_stream_synchronize(stream.get()));

  EventTimer timer;
  timer.start(stream.get());
  CCML_ASSERT_OK(launch_busy(buffer.data(), count, 20000, stream.get()));
  const double elapsed_ms = timer.stop_ms(stream.get());

  ctx.note("measured " + std::to_string(elapsed_ms) + " ms");
  ctx.check(elapsed_ms > 0.0, "elapsed time was not positive");
  ctx.check(elapsed_ms < 10000.0, "elapsed time was implausibly large");
}

/* -------------------------------------------------------- error handling -- */

CCML_TEST(infrastructure, failures_report_detail_not_just_a_code) {
  const ccml_status status = ccml_device_malloc(nullptr, 128);
  ctx.check(status == CCML_ERROR_INVALID_ARGUMENT,
            std::string("expected CCML_ERROR_INVALID_ARGUMENT, got ")
                + ccml_status_string(status));

  const std::string detail = ccml_last_error_string();
  ctx.note("detail: " + detail);
  ctx.check(!detail.empty(),
            "a failure left no detail behind, so the caller cannot diagnose it");
}

CCML_TEST(infrastructure, cuda_failures_carry_the_runtime_message) {
  /* An allocation far larger than any device has. The point is the message: the
     C ABI must not reduce a CUDA failure to a bare enum. */
  void* ptr = nullptr;
  const ccml_status status =
      ccml_device_malloc(&ptr, static_cast<size_t>(1) << 62);

  if (status == CCML_SUCCESS) {
    ccml_device_free(ptr);
    ctx.skip("this device accepted a 4 EiB allocation, so there is nothing to check");
  }

  const std::string detail = ccml_last_error_string();
  ctx.note(std::string(ccml_status_string(status)) + " -> " + detail);
  ctx.check(status == CCML_ERROR_OUT_OF_MEMORY || status == CCML_ERROR_CUDA,
            "an impossible allocation should fail as out-of-memory or CUDA error");
  ctx.check(detail.find("cudaMalloc") != std::string::npos,
            "the detail should name the call that failed");
}
