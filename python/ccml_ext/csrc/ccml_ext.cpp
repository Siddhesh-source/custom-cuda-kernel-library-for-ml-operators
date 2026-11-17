// PyTorch bindings for the ccml kernels (reference.md section 3: a
// torch.utils.cpp_extension extension usable inside a model; ROADMAP Phase 14).
//
// Every op takes and returns torch tensors, runs on PyTorch's current CUDA
// stream, dispatches on dtype to ccml_<op>_f32 / _f16 / _bf16, and REJECTS
// (TORCH_CHECK) what it cannot handle without a hidden copy: tensors not on the
// GPU, non-contiguous tensors, mixed dtypes, unsupported dtypes. Row-wise ops
// normalize over the last dimension of an N-d tensor. Failures from the C ABI
// surface as Python exceptions carrying ccml_last_error_string().
//
// The Python layer (ccml_ext/ops.py) registers these as torch.library custom
// ops with fake implementations, so torch.compile can trace through them.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>
#include <string>
#include <tuple>

#include "ccml/layernorm.h"
#include "ccml/matmul.h"
#include "ccml/softmax.h"
#include "ccml/status.h"

namespace {

using torch::Tensor;

void ok(ccml_status s, const char* what) {
  TORCH_CHECK(s == CCML_SUCCESS, what, " failed: ", ccml_status_string(s), " (",
              ccml_last_error_string(), ")");
}

void check_tensor(const Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), "ccml_ext: ", name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), "ccml_ext: ", name,
              " must be contiguous (call .contiguous() explicitly if a copy is intended)");
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf ||
                  t.scalar_type() == at::kBFloat16,
              "ccml_ext: ", name, " must be float32, float16 or bfloat16, got ",
              t.scalar_type());
}

void check_same(const Tensor& a, const Tensor& b, const char* name) {
  check_tensor(b, name);
  TORCH_CHECK(b.scalar_type() == a.scalar_type(), "ccml_ext: ", name, " has dtype ",
              b.scalar_type(), " but the input is ", a.scalar_type());
  TORCH_CHECK(b.device() == a.device(), "ccml_ext: ", name, " is on a different device");
}

int as_int(int64_t v, const char* name) {
  TORCH_CHECK(v >= 0 && v <= INT32_MAX, "ccml_ext: ", name, " = ", v, " does not fit in int");
  return static_cast<int>(v);
}

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

template <typename T>
T* p(const Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}
template <typename T>
const T* pc(const std::optional<Tensor>& t) {
  return t.has_value() ? reinterpret_cast<const T*>(t->data_ptr()) : nullptr;
}

/* ------------------------------------------------------------- matmul -- */

/* C[M, N] = A[M, K] B[K, N]. */
Tensor matmul(const Tensor& a, const Tensor& b, int64_t algo) {
  check_tensor(a, "a");
  check_same(a, b, "b");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
              "ccml_ext.matmul: expected a [M, K] and b [K, N], got ", a.sizes(), " and ",
              b.sizes());
  const at::cuda::CUDAGuard guard(a.device());
  const int M = as_int(a.size(0), "M"), K = as_int(a.size(1), "K"), N = as_int(b.size(1), "N");
  Tensor c = torch::empty({M, N}, a.options());
  const auto al = static_cast<ccml_matmul_algo>(algo);
  switch (a.scalar_type()) {
    case at::kFloat:
      ok(ccml_matmul_f32(p<float>(c), p<float>(a), p<float>(b), M, N, K, al, stream()),
         "ccml_matmul_f32");
      break;
    case at::kHalf:
      ok(ccml_matmul_f16(p<ccml_f16>(c), p<ccml_f16>(a), p<ccml_f16>(b), M, N, K, al, stream()),
         "ccml_matmul_f16");
      break;
    default:
      ok(ccml_matmul_bf16(p<ccml_bf16>(c), p<ccml_bf16>(a), p<ccml_bf16>(b), M, N, K, al,
                          stream()),
         "ccml_matmul_bf16");
  }
  return c;
}

/* y[..., N] = act(x[..., K] W[K, N] + bias) + residual, 16-bit only (the fused
   epilogue, log entry 8). act: 0 none, 1 ReLU, 2 GELU (tanh). */
Tensor linear(const Tensor& x, const Tensor& w, const std::optional<Tensor>& bias,
              const std::optional<Tensor>& residual, int64_t act, int64_t algo) {
  check_tensor(x, "x");
  check_same(x, w, "weight");
  TORCH_CHECK(x.scalar_type() != at::kFloat,
              "ccml_ext.linear: the fused epilogue needs float16 or bfloat16 (FP32 has no "
              "Tensor Core path on sm_75); use matmul for float32");
  TORCH_CHECK(w.dim() == 2 && x.size(-1) == w.size(0), "ccml_ext.linear: x [..., K] and weight "
              "[K, N] expected, got ", x.sizes(), " and ", w.sizes());
  const int K = as_int(w.size(0), "K"), N = as_int(w.size(1), "N");
  const int M = as_int(x.numel() / K, "M");
  if (bias.has_value()) {
    check_same(x, *bias, "bias");
    TORCH_CHECK(bias->numel() == N, "ccml_ext.linear: bias must have N = ", N, " elements");
  }
  std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
  out_shape.back() = N;
  if (residual.has_value()) {
    check_same(x, *residual, "residual");
    TORCH_CHECK(residual->sizes() == at::IntArrayRef(out_shape),
                "ccml_ext.linear: residual must have the output's shape");
  }
  const at::cuda::CUDAGuard guard(x.device());
  Tensor y = torch::empty(out_shape, x.options());
  const auto a = static_cast<ccml_activation>(act);
  const auto al = static_cast<ccml_matmul_algo>(algo);
  if (x.scalar_type() == at::kHalf) {
    ok(ccml_matmul_epilogue_f16(p<ccml_f16>(y), p<ccml_f16>(x), p<ccml_f16>(w),
                                pc<ccml_f16>(bias), pc<ccml_f16>(residual), M, N, K, a, al,
                                stream()),
       "ccml_matmul_epilogue_f16");
  } else {
    ok(ccml_matmul_epilogue_bf16(p<ccml_bf16>(y), p<ccml_bf16>(x), p<ccml_bf16>(w),
                                 pc<ccml_bf16>(bias), pc<ccml_bf16>(residual), M, N, K, a, al,
                                 stream()),
       "ccml_matmul_epilogue_bf16");
  }
  return y;
}

/* ---------------------------------------------------------- layernorm -- */

void check_affine(const Tensor& x, const std::optional<Tensor>& t, const char* name, int cols) {
  if (!t.has_value()) return;
  check_same(x, *t, name);
  TORCH_CHECK(t->numel() == cols, "ccml_ext: ", name, " must have ", cols, " elements");
}

Tensor layernorm(const Tensor& x, const std::optional<Tensor>& gamma,
                 const std::optional<Tensor>& beta, double eps, int64_t algo) {
  check_tensor(x, "x");
  TORCH_CHECK(x.dim() >= 1 && x.numel() > 0, "ccml_ext.layernorm: empty input");
  const int cols = as_int(x.size(-1), "cols"), rows = as_int(x.numel() / cols, "rows");
  check_affine(x, gamma, "weight", cols);
  check_affine(x, beta, "bias", cols);
  const at::cuda::CUDAGuard guard(x.device());
  Tensor y = torch::empty_like(x);
  const auto al = static_cast<ccml_layernorm_algo>(algo);
  const float e = static_cast<float>(eps);
  switch (x.scalar_type()) {
    case at::kFloat:
      ok(ccml_layernorm_f32(p<float>(y), p<float>(x), pc<float>(gamma), pc<float>(beta), rows,
                            cols, e, al, stream()),
         "ccml_layernorm_f32");
      break;
    case at::kHalf:
      ok(ccml_layernorm_f16(p<ccml_f16>(y), p<ccml_f16>(x), pc<ccml_f16>(gamma),
                            pc<ccml_f16>(beta), rows, cols, e, al, stream()),
         "ccml_layernorm_f16");
      break;
    default:
      ok(ccml_layernorm_bf16(p<ccml_bf16>(y), p<ccml_bf16>(x), pc<ccml_bf16>(gamma),
                             pc<ccml_bf16>(beta), rows, cols, e, al, stream()),
         "ccml_layernorm_bf16");
  }
  return y;
}

/* (y, s): s = x + residual + bias (rounded), y = LN(s) (log entry 12). */
std::tuple<Tensor, Tensor> layernorm_residual(const Tensor& x, const Tensor& residual,
                                              const std::optional<Tensor>& bias,
                                              const std::optional<Tensor>& gamma,
                                              const std::optional<Tensor>& beta, double eps,
                                              int64_t algo) {
  check_tensor(x, "x");
  check_same(x, residual, "residual");
  TORCH_CHECK(residual.sizes() == x.sizes(), "ccml_ext.layernorm_residual: shape mismatch");
  const int cols = as_int(x.size(-1), "cols"), rows = as_int(x.numel() / cols, "rows");
  check_affine(x, bias, "bias", cols);
  check_affine(x, gamma, "weight", cols);
  check_affine(x, beta, "ln_bias", cols);
  const at::cuda::CUDAGuard guard(x.device());
  Tensor y = torch::empty_like(x), s = torch::empty_like(x);
  const auto al = static_cast<ccml_layernorm_algo>(algo);
  const float e = static_cast<float>(eps);
  switch (x.scalar_type()) {
    case at::kFloat:
      ok(ccml_layernorm_residual_f32(p<float>(y), p<float>(s), p<float>(x), p<float>(residual),
                                     pc<float>(bias), pc<float>(gamma), pc<float>(beta), rows,
                                     cols, e, al, stream()),
         "ccml_layernorm_residual_f32");
      break;
    case at::kHalf:
      ok(ccml_layernorm_residual_f16(p<ccml_f16>(y), p<ccml_f16>(s), p<ccml_f16>(x),
                                     p<ccml_f16>(residual), pc<ccml_f16>(bias),
                                     pc<ccml_f16>(gamma), pc<ccml_f16>(beta), rows, cols, e, al,
                                     stream()),
         "ccml_layernorm_residual_f16");
      break;
    default:
      ok(ccml_layernorm_residual_bf16(p<ccml_bf16>(y), p<ccml_bf16>(s), p<ccml_bf16>(x),
                                      p<ccml_bf16>(residual), pc<ccml_bf16>(bias),
                                      pc<ccml_bf16>(gamma), pc<ccml_bf16>(beta), rows, cols, e,
                                      al, stream()),
         "ccml_layernorm_residual_bf16");
  }
  return {y, s};
}

/* ------------------------------------------------------------ softmax -- */

Tensor softmax(const Tensor& x, int64_t algo) {
  check_tensor(x, "x");
  TORCH_CHECK(x.numel() > 0, "ccml_ext.softmax: empty input");
  const int cols = as_int(x.size(-1), "cols"), rows = as_int(x.numel() / cols, "rows");
  const at::cuda::CUDAGuard guard(x.device());
  Tensor y = torch::empty_like(x);
  const auto al = static_cast<ccml_softmax_algo>(algo);
  switch (x.scalar_type()) {
    case at::kFloat:
      ok(ccml_softmax_f32(p<float>(y), p<float>(x), rows, cols, al, stream()), "ccml_softmax_f32");
      break;
    case at::kHalf:
      ok(ccml_softmax_f16(p<ccml_f16>(y), p<ccml_f16>(x), rows, cols, al, stream()),
         "ccml_softmax_f16");
      break;
    default:
      ok(ccml_softmax_bf16(p<ccml_bf16>(y), p<ccml_bf16>(x), rows, cols, al, stream()),
         "ccml_softmax_bf16");
  }
  return y;
}

/* softmax(x * scale + mask) with optional bottom-right causal masking (log
   entry 16). mask: [mask_rows, cols] (any leading shape), broadcast over rows
   as row % mask_rows. Fully masked rows give zeros. */
Tensor softmax_masked(const Tensor& x, const std::optional<Tensor>& mask, double scale,
                      int64_t causal_q, int64_t algo) {
  check_tensor(x, "x");
  const int cols = as_int(x.size(-1), "cols"), rows = as_int(x.numel() / cols, "rows");
  int mask_rows = 0;
  if (mask.has_value()) {
    check_same(x, *mask, "mask");
    TORCH_CHECK(mask->size(-1) == cols && mask->numel() % cols == 0,
                "ccml_ext.softmax_masked: mask must be [..., cols]");
    mask_rows = as_int(mask->numel() / cols, "mask_rows");
  }
  const at::cuda::CUDAGuard guard(x.device());
  Tensor y = torch::empty_like(x);
  const auto al = static_cast<ccml_softmax_algo>(algo);
  const float s = static_cast<float>(scale);
  const int cq = as_int(causal_q, "causal_q");
  switch (x.scalar_type()) {
    case at::kFloat:
      ok(ccml_softmax_masked_f32(p<float>(y), p<float>(x), pc<float>(mask), mask_rows, rows, cols,
                                 s, cq, al, stream()),
         "ccml_softmax_masked_f32");
      break;
    case at::kHalf:
      ok(ccml_softmax_masked_f16(p<ccml_f16>(y), p<ccml_f16>(x), pc<ccml_f16>(mask), mask_rows,
                                 rows, cols, s, cq, al, stream()),
         "ccml_softmax_masked_f16");
      break;
    default:
      ok(ccml_softmax_masked_bf16(p<ccml_bf16>(y), p<ccml_bf16>(x), pc<ccml_bf16>(mask),
                                  mask_rows, rows, cols, s, cq, al, stream()),
         "ccml_softmax_masked_bf16");
  }
  return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ccml CUDA kernels for PyTorch (torch.utils.cpp_extension)";
  m.def("matmul", &matmul, "C = A B", py::arg("a"), py::arg("b"), py::arg("algo") = 0);
  m.def("linear", &linear, "act(x W + bias) + residual (16-bit, fused epilogue)", py::arg("x"),
        py::arg("weight"), py::arg("bias") = py::none(), py::arg("residual") = py::none(),
        py::arg("act") = 0, py::arg("algo") = 0);
  m.def("layernorm", &layernorm, "layer norm over the last dimension", py::arg("x"),
        py::arg("weight") = py::none(), py::arg("bias") = py::none(), py::arg("eps") = 1e-5,
        py::arg("algo") = 0);
  m.def("layernorm_residual", &layernorm_residual, "(LN(x + residual + bias), the sum)",
        py::arg("x"), py::arg("residual"), py::arg("bias") = py::none(),
        py::arg("weight") = py::none(), py::arg("ln_bias") = py::none(), py::arg("eps") = 1e-5,
        py::arg("algo") = 0);
  m.def("softmax", &softmax, "softmax over the last dimension", py::arg("x"), py::arg("algo") = 0);
  m.def("softmax_masked", &softmax_masked, "softmax(x * scale + mask), optional causal",
        py::arg("x"), py::arg("mask") = py::none(), py::arg("scale") = 1.0,
        py::arg("causal_q") = 0, py::arg("algo") = 0);
}
