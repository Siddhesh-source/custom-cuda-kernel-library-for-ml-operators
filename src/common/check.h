#ifndef CCML_SRC_COMMON_CHECK_H
#define CCML_SRC_COMMON_CHECK_H

/* Internal error-handling plumbing. Not installed, not part of the public API.
 *
 * Header-only on purpose. These used to be defined in status.cpp, but the
 * library is built with hidden visibility, so anything including this header
 * from outside the shared library -- the infrastructure test, for one -- could
 * not link against them. Making them inline removes the link dependency
 * entirely and costs nothing.
 *
 * One consequence worth knowing: the detail buffer is per-binary. Code compiled
 * into libccml writes to the library's buffer, which is what the exported
 * ccml_last_error_string() reads. A test binary that includes this header and
 * calls these helpers directly gets its own buffer, readable through
 * ccml::detail::last_error(). Only white-box tests ever see both.
 */

#include <cstdio>

#include <cuda_runtime.h>

#include "ccml/status.h"

namespace ccml {
namespace detail {

inline constexpr int kErrorBufferSize = 512;

/* Fixed size and allocation-free, so it is safe to fill in from an error path
   that may itself be an out-of-memory condition. */
inline char* error_buffer() {
  static thread_local char buffer[kErrorBufferSize] = {0};
  return buffer;
}

inline const char* last_error() { return error_buffer(); }

inline void clear_last_error() { error_buffer()[0] = '\0'; }

inline void set_last_error(const char* message) {
  char* buffer = error_buffer();
  if (message == nullptr) {
    buffer[0] = '\0';
    return;
  }
  std::snprintf(buffer, kErrorBufferSize, "%s", message);
}

inline void set_last_cuda_error(const char* expression, cudaError_t err) {
  std::snprintf(error_buffer(), kErrorBufferSize, "%s failed: %s (%s)",
                expression, cudaGetErrorString(err), cudaGetErrorName(err));
}

}  // namespace detail
}  // namespace ccml

/* Run a CUDA runtime call; on failure record it and return CCML_ERROR_CUDA. */
#define CCML_CUDA_TRY(expr)                                     \
  do {                                                          \
    cudaError_t ccml_err_ = (expr);                             \
    if (ccml_err_ != cudaSuccess) {                             \
      ::ccml::detail::set_last_cuda_error(#expr, ccml_err_);    \
      return CCML_ERROR_CUDA;                                   \
    }                                                           \
  } while (0)

/* Check for a launch rejection immediately after a <<<>>> launch. */
#define CCML_CUDA_CHECK_LAUNCH() CCML_CUDA_TRY(cudaGetLastError())

/* Argument validation. */
#define CCML_REQUIRE(cond, status, message)         \
  do {                                              \
    if (!(cond)) {                                  \
      ::ccml::detail::set_last_error(message);      \
      return (status);                              \
    }                                               \
  } while (0)

#endif /* CCML_SRC_COMMON_CHECK_H */
