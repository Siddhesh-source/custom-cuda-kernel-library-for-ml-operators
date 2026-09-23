#ifndef CCML_MEMORY_H
#define CCML_MEMORY_H

#include <stddef.h>

#include "ccml/export.h"
#include "ccml/status.h"
#include "ccml/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Thin, honest wrappers over the CUDA runtime's allocation and copy calls.
 *
 * These exist so a binding (see python/) can drive the library without
 * depending on PyTorch, CuPy, or any other CUDA-aware framework. C++ callers
 * are free to ignore them and use the CUDA runtime directly -- they add no
 * bookkeeping, no pool, and no tracking, only error translation.
 *
 * The blocking variants are named plainly; the stream-ordered ones carry the
 * _async suffix and, like every asynchronous CUDA call, require the caller to
 * synchronise before touching the result. */

CCML_API ccml_status ccml_device_malloc(void** ptr, size_t bytes);
CCML_API ccml_status ccml_device_free(void* ptr);

/* Blocking with respect to the host. */
CCML_API ccml_status ccml_memcpy_h2d(void* dst, const void* src, size_t bytes);
CCML_API ccml_status ccml_memcpy_d2h(void* dst, const void* src, size_t bytes);

/* Stream-ordered. A null stream means the default stream.
 *
 * For a host-to-device copy the source should be page-locked for the copy to
 * be genuinely asynchronous; from ordinary pageable memory the runtime may
 * still stage it synchronously. That is CUDA's behaviour, not something these
 * wrappers hide. */
CCML_API ccml_status ccml_memcpy_h2d_async(void* dst, const void* src,
                                           size_t bytes, cudaStream_t stream);
CCML_API ccml_status ccml_memcpy_d2h_async(void* dst, const void* src,
                                           size_t bytes, cudaStream_t stream);

/* Sets each byte, not each element -- the value is an int for consistency with
 * memset, but only its low byte is used. Zeroing works; filling with 1.0f does
 * not. Useful in tests for poisoning a buffer so that a kernel which fails to
 * write every output is caught rather than reading back a plausible zero. */
CCML_API ccml_status ccml_device_memset_async(void* ptr, int value,
                                              size_t bytes, cudaStream_t stream);

/* Waits for one stream. Prefer this to ccml_device_synchronize when only one
 * stream's work matters: synchronising the whole device also waits on every
 * other stream, which turns a targeted wait into a global barrier. */
CCML_API ccml_status ccml_stream_synchronize(cudaStream_t stream);

/* Waits for every stream on the current device. */
CCML_API ccml_status ccml_device_synchronize(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_MEMORY_H */
