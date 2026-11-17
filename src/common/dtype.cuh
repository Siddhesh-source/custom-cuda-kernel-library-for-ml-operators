#ifndef CCML_SRC_COMMON_DTYPE_CUH
#define CCML_SRC_COMMON_DTYPE_CUH

/*
 * Storage-type plumbing shared by every operator: one templated kernel per
 * variant, instantiated for float, __half and __nv_bfloat16 (reference.md §3:
 * FP32, FP16, BF16).
 *
 * The rule is "widen on load, round once on store": every load converts to
 * float, all arithmetic is float (or double where an operator needs it), and
 * the only rounding to the storage type is the final store. A kernel never
 * computes in half precision, so FP16/BF16 results differ from FP32 ones only
 * by input and output rounding -- which is what the per-dtype tolerances in
 * docs/correctness.md are derived from.
 *
 * BF16 conversions are plain bit operations and compile for every
 * architecture, so BF16 storage works on sm_75 (Kaggle T4). Only BF16 *Tensor
 * Core* instructions need sm_80+ (docs/local.md L1).
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ccml/types.h"

namespace ccml {
namespace detail {

static_assert(sizeof(ccml_f16) == sizeof(__half), "ccml_f16 must match __half");
static_assert(sizeof(ccml_bf16) == sizeof(__nv_bfloat16),
              "ccml_bf16 must match __nv_bfloat16");

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

/* Direct double -> storage rounding, for kernels that compute in double. Going
   through float first would round twice, which is occasionally off by one
   unit in the last place of the 16-bit result. */
template <typename T>
__device__ __forceinline__ T from_double(double v);

template <>
__device__ __forceinline__ float from_double<float>(double v) {
  return static_cast<float>(v);
}
template <>
__device__ __forceinline__ __half from_double<__half>(double v) {
  return __double2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_double<__nv_bfloat16>(double v) {
  return __double2bfloat16(v);
}

/* Maps the public storage types to the CUDA ones the kernels use. */
template <typename Public>
struct DeviceType;
template <>
struct DeviceType<float> { using type = float; };
template <>
struct DeviceType<ccml_f16> { using type = __half; };
template <>
struct DeviceType<ccml_bf16> { using type = __nv_bfloat16; };

template <typename Public>
using device_t = typename DeviceType<Public>::type;

/* Pointer casts from the public ABI types; layout-identical by the
   static_asserts above. */
template <typename Public>
inline device_t<Public>* device_ptr(Public* p) {
  return reinterpret_cast<device_t<Public>*>(p);
}
template <typename Public>
inline const device_t<Public>* device_ptr(const Public* p) {
  return reinterpret_cast<const device_t<Public>*>(p);
}

/* Suffix used in error messages, e.g. "ccml_softmax_f16". */
template <typename Public>
constexpr const char* dtype_suffix();
template <>
constexpr const char* dtype_suffix<float>() { return "f32"; }
template <>
constexpr const char* dtype_suffix<ccml_f16>() { return "f16"; }
template <>
constexpr const char* dtype_suffix<ccml_bf16>() { return "bf16"; }

}  // namespace detail
}  // namespace ccml

#endif /* CCML_SRC_COMMON_DTYPE_CUH */
