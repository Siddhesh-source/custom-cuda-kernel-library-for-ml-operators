/*
 * Prints the properties every tiling and block-size decision in this library
 * must be derived from.
 *
 * This is not a formality. Tile dimensions, block sizes and occupancy targets
 * are only defensible if they are traced back to measured numbers from the
 * actual device -- so run this and paste the output into docs/design.md rather
 * than quoting a specification sheet.
 */

#include <cstdio>

#include "ccml/device.h"
#include "ccml/status.h"

namespace {

void print_device(const ccml_device_info& info) {
  std::printf("device %d: %s\n", info.device, info.name);
  std::printf("  compute capability      : %d.%d (build with sm_%d%d)\n",
              info.compute_major, info.compute_minor,
              info.compute_major, info.compute_minor);
  std::printf("  multiprocessors         : %d\n", info.multiprocessor_count);
  std::printf("  max threads / block     : %d\n", info.max_threads_per_block);
  std::printf("  max threads / SM        : %d\n",
              info.max_threads_per_multiprocessor);
  std::printf("  warp size               : %d\n", info.warp_size);
  std::printf("  shared memory / block   : %zu bytes\n", info.shared_mem_per_block);
  std::printf("  shared memory / SM      : %zu bytes\n",
              info.shared_mem_per_multiprocessor);
  std::printf("  registers / block       : %d\n", info.regs_per_block);
  std::printf("  global memory           : %.2f GiB\n",
              static_cast<double>(info.total_global_mem) / (1024.0 * 1024.0 * 1024.0));
  std::printf("  L2 cache                : %d KiB\n", info.l2_cache_size / 1024);
  std::printf("  memory bus width        : %d bits\n", info.memory_bus_width);
  std::printf("  memory clock            : %d kHz\n", info.memory_clock_khz);
  std::printf("  SM clock                : %d kHz\n", info.clock_khz);

  /* Theoretical peak from the clock and bus width. It is an upper bound the
     hardware will not reach, useful only as a ceiling to measure against --
     never quote it as achieved bandwidth. */
  const double peak_gb_s = 2.0 * static_cast<double>(info.memory_clock_khz) *
                           (static_cast<double>(info.memory_bus_width) / 8.0) / 1.0e6;
  std::printf("  theoretical peak BW     : %.1f GB/s (ceiling, not achieved)\n",
              peak_gb_s);
}

}  // namespace

int main() {
  std::printf("ccml %s\n\n", ccml_version_string());

  int count = 0;
  const ccml_status status = ccml_device_count(&count);
  if (status != CCML_SUCCESS) {
    std::printf("ccml_device_count failed: %s (%s)\n",
                ccml_status_string(status), ccml_last_error_string());
    return 1;
  }

  if (count == 0) {
    std::printf("no CUDA devices found\n");
    return 1;
  }

  for (int device = 0; device < count; ++device) {
    ccml_device_info info;
    const ccml_status info_status = ccml_get_device_info(device, &info);
    if (info_status != CCML_SUCCESS) {
      std::printf("device %d: query failed: %s (%s)\n", device,
                  ccml_status_string(info_status), ccml_last_error_string());
      continue;
    }
    print_device(info);
    std::printf("\n");
  }
  return 0;
}
