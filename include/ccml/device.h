#ifndef CCML_DEVICE_H
#define CCML_DEVICE_H

#include <stddef.h>

#include "ccml/export.h"
#include "ccml/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Hardware facts that tiling and block-size decisions must be derived from,
 * rather than assumed. examples/device_query.cpp prints these so the measured
 * values can be recorded in docs/design.md. */
typedef struct ccml_device_info {
  int    device;
  char   name[256];
  int    compute_major;
  int    compute_minor;
  int    multiprocessor_count;
  int    max_threads_per_block;
  int    max_threads_per_multiprocessor;
  int    warp_size;
  size_t shared_mem_per_block;
  size_t shared_mem_per_multiprocessor;
  int    regs_per_block;
  size_t total_global_mem;
  int    l2_cache_size;
  int    memory_bus_width;
  int    memory_clock_khz;
  int    clock_khz;
} ccml_device_info;

CCML_API ccml_status ccml_device_count(int* count);
CCML_API ccml_status ccml_get_device_info(int device, ccml_device_info* info);

/* Version of the library that is actually loaded, e.g. "0.1.0". */
CCML_API const char* ccml_version_string(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_DEVICE_H */
