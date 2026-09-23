/*
 * End-to-end use of the library through its public API only.
 *
 * Includes nothing from src/, which is the point: if this stops compiling, the
 * public headers are incomplete. It is the code the README's usage section
 * shows, kept here so it is compiled rather than left to rot in a document.
 */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <ccml/ccml.h>

namespace {

/* Every entry point returns a status and none of them throws, so a small helper
   keeps the calling code readable without hiding anything. */
bool ok(ccml_status status, const char* what) {
  if (status == CCML_SUCCESS) {
    return true;
  }
  std::printf("  %-28s %s (%s)\n", what, ccml_status_string(status),
              ccml_last_error_string());
  return false;
}

void print_row(const char* label, const std::vector<float>& values, int count) {
  std::printf("  %-10s", label);
  for (int i = 0; i < count; ++i) {
    std::printf(" %8.4f", static_cast<double>(values[i]));
  }
  std::printf("\n");
}

}  // namespace

int main() {
  std::printf("ccml %s\n\n", ccml_version_string());

  const int rows = 4;
  const int cols = 8;
  const std::size_t elements = static_cast<std::size_t>(rows) * cols;
  const std::size_t bytes = elements * sizeof(float);

  /* Host input: each row a simple ramp, offset per row. */
  std::vector<float> host_x(elements);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      host_x[static_cast<std::size_t>(r) * cols + c] =
          static_cast<float>(c) - 0.5f * static_cast<float>(r);
    }
  }

  /* The library never allocates the caller's tensors. These helpers exist so a
     binding with no other CUDA access can drive it; C++ callers may equally use
     cudaMalloc directly. */
  void* device_x = nullptr;
  void* device_y = nullptr;
  if (!ok(ccml_device_malloc(&device_x, bytes), "allocate x") ||
      !ok(ccml_device_malloc(&device_y, bytes), "allocate y")) {
    return 1;
  }
  if (!ok(ccml_memcpy_h2d(device_x, host_x.data(), bytes), "upload x")) {
    return 1;
  }

  std::vector<float> host_y(elements);
  print_row("input", host_x, cols);

  /* ---- softmax -------------------------------------------------------- */
  if (ok(ccml_softmax_f32(static_cast<float*>(device_y),
                          static_cast<const float*>(device_x), rows, cols,
                          CCML_SOFTMAX_AUTO, /*stream=*/nullptr),
         "ccml_softmax_f32")) {
    /* Operators are asynchronous. The result is not readable until a
       synchronisation; reading before this would be a race, not a result. */
    ok(ccml_device_synchronize(), "synchronize");
    ok(ccml_memcpy_d2h(host_y.data(), device_y, bytes), "download");

    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum += static_cast<double>(host_y[c]);
    }
    print_row("softmax", host_y, cols);
    std::printf("  %-10s row sums to %.6f\n", "", sum);
  }

  /* ---- layer normalization -------------------------------------------- */
  std::vector<float> host_gamma(static_cast<std::size_t>(cols), 1.0f);
  std::vector<float> host_beta(static_cast<std::size_t>(cols), 0.0f);

  void* device_gamma = nullptr;
  void* device_beta = nullptr;
  const std::size_t affine_bytes = static_cast<std::size_t>(cols) * sizeof(float);
  ok(ccml_device_malloc(&device_gamma, affine_bytes), "allocate gamma");
  ok(ccml_device_malloc(&device_beta, affine_bytes), "allocate beta");
  ok(ccml_memcpy_h2d(device_gamma, host_gamma.data(), affine_bytes), "upload gamma");
  ok(ccml_memcpy_h2d(device_beta, host_beta.data(), affine_bytes), "upload beta");

  /* gamma and beta are independently optional; pass nullptr to skip either. */
  if (ok(ccml_layernorm_f32(static_cast<float*>(device_y),
                            static_cast<const float*>(device_x),
                            static_cast<const float*>(device_gamma),
                            static_cast<const float*>(device_beta),
                            rows, cols, 1e-5f, CCML_LAYERNORM_AUTO,
                            /*stream=*/nullptr),
         "ccml_layernorm_f32")) {
    ok(ccml_device_synchronize(), "synchronize");
    ok(ccml_memcpy_d2h(host_y.data(), device_y, bytes), "download");
    print_row("layernorm", host_y, cols);
  }

  /* ---- matrix multiply ------------------------------------------------- */
  /* Not implemented yet. Shown because this is what a caller sees when an
     operator is declared but has no kernel behind it -- a clear status, not a
     crash and not a silently wrong answer. */
  std::printf("\n  matmul:\n");
  ccml_matmul_f32(static_cast<float*>(device_y),
                  static_cast<const float*>(device_x),
                  static_cast<const float*>(device_x), rows, rows, cols,
                  CCML_MATMUL_AUTO, /*stream=*/nullptr);
  std::printf("  %-28s %s\n", "ccml_matmul_f32",
              ccml_last_error_string());

  ccml_device_free(device_beta);
  ccml_device_free(device_gamma);
  ccml_device_free(device_y);
  ccml_device_free(device_x);
  return 0;
}
