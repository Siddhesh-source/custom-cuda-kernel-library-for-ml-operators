#ifndef CCML_STATUS_H
#define CCML_STATUS_H

#include "ccml/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Every ccml entry point returns one of these. Nothing throws, nothing aborts;
   this keeps the ABI usable from C and from ctypes. */
typedef enum ccml_status {
  CCML_SUCCESS                 = 0,
  CCML_ERROR_INVALID_ARGUMENT  = 1, /* null pointer, bad enum, non-positive dim */
  CCML_ERROR_UNSUPPORTED_SHAPE = 2, /* shape is valid but this algo cannot run it */
  CCML_ERROR_NOT_IMPLEMENTED   = 3, /* declared but no kernel behind it yet */
  CCML_ERROR_CUDA              = 4, /* a CUDA API call or kernel launch failed */
  CCML_ERROR_OUT_OF_MEMORY     = 5
} ccml_status;

/* Stable, allocation-free description of a status code. */
CCML_API const char* ccml_status_string(ccml_status status);

/* Detail for the most recent failure on the calling thread, or "" if none.
   The C ABI loses CUDA's error text; this gives it back. */
CCML_API const char* ccml_last_error_string(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_STATUS_H */
