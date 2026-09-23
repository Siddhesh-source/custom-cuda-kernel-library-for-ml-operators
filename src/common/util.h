#ifndef CCML_SRC_COMMON_UTIL_H
#define CCML_SRC_COMMON_UTIL_H

/* Small arithmetic helpers, usable from host and device code.
 *
 * The annotation is conditional because this header is included from plain .cpp
 * translation units as well as from .cu ones, and __host__ __device__ is only
 * meaningful to nvcc. Without it, calling div_up from a __global__ function is
 * rejected: "calling a constexpr __host__ function from a __global__ function
 * is not allowed". constexpr alone is not enough -- that makes it usable at
 * compile time, not on the device. */
#if defined(__CUDACC__)
#  define CCML_HOST_DEVICE __host__ __device__
#else
#  define CCML_HOST_DEVICE
#endif

namespace ccml {
namespace detail {

/* Ceiling division, for grid sizing. Both arguments must be positive. */
CCML_HOST_DEVICE constexpr int div_up(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

/* Round up to the next multiple. */
CCML_HOST_DEVICE constexpr int round_up(int value, int multiple) {
  return div_up(value, multiple) * multiple;
}

}  // namespace detail
}  // namespace ccml

#define CCML_UNUSED(x) ((void)(x))

#endif /* CCML_SRC_COMMON_UTIL_H */
