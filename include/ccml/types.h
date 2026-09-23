#ifndef CCML_TYPES_H
#define CCML_TYPES_H

/* Single place the public API pulls in the CUDA runtime, for cudaStream_t.
   ccml deliberately does not wrap CUDA types in its own aliases: an alias for
   cudaStream_t would hide nothing and cost every reader a lookup. */
#include <cuda_runtime.h>

/*
 * Conventions every operator in this library obeys
 * ================================================
 *
 * Layout
 *   Row-major and fully contiguous. No strides, no transpose flags, no
 *   broadcasting. A caller that needs those does the transform itself, where it
 *   can be fused with whatever else it is doing.
 *
 * Precision
 *   float32 storage throughout. Reductions accumulate in float32 unless an
 *   operator's own header says otherwise. Reference implementations used in
 *   tests accumulate in double -- that is what makes them a reference.
 *
 * Pointers
 *   Every tensor pointer is a DEVICE pointer. The library never allocates the
 *   caller's tensors, never copies to the host, and never takes ownership.
 *   Passing a host pointer is undefined and will usually fault on first access.
 *
 * Aliasing
 *   Input and output buffers must not overlap. None of the operators is written
 *   to be safe in place.
 *
 * Shapes
 *   All dimensions must be strictly positive. Sizes that are not a multiple of
 *   any internal tile or block size are valid and handled; that is a correctness
 *   requirement, not a special case.
 *
 * Streams
 *   Every operator takes a cudaStream_t. A null handle means the default
 *   stream. The library does not create, destroy, or own streams -- a library
 *   that made its own streams behind the caller's back could not be composed
 *   with one that did not.
 *
 * Asynchrony
 *   Operators are asynchronous with respect to the host. The result is not
 *   readable until the caller synchronises: ccml_stream_synchronize for one
 *   stream, ccml_device_synchronize for all of them. Reading the output buffer
 *   before that is a race, not a result.
 *
 * Errors
 *   Every entry point returns a ccml_status. Nothing throws, nothing aborts,
 *   and nothing prints. Because a launch is asynchronous, a success return
 *   means the work was validated and SUBMITTED -- a fault raised while the
 *   kernel executes surfaces at the next synchronisation, reported against
 *   whichever call synchronises. ccml_last_error_string() carries the detail
 *   that the status code cannot.
 *
 * Algorithm selection
 *   Each operator takes an algo enum whose AUTO value resolves to the current
 *   best default. Named values exist so that variants can be compared directly
 *   in tests and benchmarks; callers with no opinion should pass AUTO.
 *
 * Thread safety
 *   Entry points are safe to call concurrently from multiple host threads on
 *   distinct data. The error detail buffer is thread-local, so a failure on one
 *   thread never overwrites another's.
 */

#endif /* CCML_TYPES_H */
