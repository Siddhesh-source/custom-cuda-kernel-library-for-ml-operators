#include "ccml/status.h"

#include "ccml/device.h"
#include "ccml/version.h"
#include "common/check.h"

extern "C" {

const char* ccml_status_string(ccml_status status) {
  switch (status) {
    case CCML_SUCCESS:                 return "success";
    case CCML_ERROR_INVALID_ARGUMENT:  return "invalid argument";
    case CCML_ERROR_UNSUPPORTED_SHAPE: return "unsupported shape";
    case CCML_ERROR_NOT_IMPLEMENTED:   return "not implemented";
    case CCML_ERROR_CUDA:              return "CUDA error";
    case CCML_ERROR_OUT_OF_MEMORY:     return "out of memory";
  }
  return "unknown status";
}

/* The detail behind the most recent failure on this thread, inside this
   library. Never null; empty when nothing has failed. */
const char* ccml_last_error_string(void) { return ccml::detail::last_error(); }

const char* ccml_version_string(void) { return CCML_VERSION_STRING; }

}  // extern "C"
