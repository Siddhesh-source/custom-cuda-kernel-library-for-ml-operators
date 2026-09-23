#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ccml/matmul.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "support/device_buffer.h"
#include "support/random.h"
#include "tests/support/compare.h"
#include "tests/support/harness.h"
#include "tests/support/reference.h"

using ccml_support::DeviceBuffer;
using ccml_support::Rng;

namespace {

/* One shape, one algorithm, one comparison. Kept in a helper so each planned
   test case below is a single call with explicit parameters. */
void run_case(const ccml_test::Context& ctx,
              int M, int N, int K,
              ccml_matmul_algo algo,
              std::uint64_t seed) {
  std::vector<float> host_a(static_cast<std::size_t>(M) * K);
  std::vector<float> host_b(static_cast<std::size_t>(K) * N);

  Rng rng(seed);
  rng.normal(host_a.data(), host_a.size(), 0.0f, 1.0f);
  rng.normal(host_b.data(), host_b.size(), 0.0f, 1.0f);

  std::vector<double> expected;
  if (!ccml_test::matmul_reference(expected, host_a, host_b, M, N, K)) {
    ctx.skip("matmul_reference is not implemented yet");
  }

  DeviceBuffer device_a(host_a.size());
  DeviceBuffer device_b(host_b.size());
  DeviceBuffer device_c(static_cast<std::size_t>(M) * N);
  device_a.upload(host_a.data());
  device_b.upload(host_b.data());

  CCML_ASSERT_OK_OR_SKIP(ccml_matmul_f32(device_c.data(), device_a.data(),
                                         device_b.data(), M, N, K, algo,
                                         /*stream=*/nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  std::vector<float> got(device_c.count());
  device_c.download(got.data());

  /* Scale-aware: a matrix product element can cancel to near zero, which makes
     a per-element relative bound meaningless. See compare.h. */
  const double scale = ccml_test::rms(expected);
  const ccml_test::Tolerance tolerance =
      ccml_test::tolerance_for_f32_dot_product(K, scale);
  const ccml_test::ErrorStats stats =
      ccml_test::compare(got, expected, tolerance);

  ctx.note(std::string(ccml_matmul_algo_name(algo)) + " M=" + std::to_string(M) +
           " N=" + std::to_string(N) + " K=" + std::to_string(K));
  ctx.note("rms(reference)=" + std::to_string(scale) + "  " + tolerance.to_string());
  ctx.note(stats.to_string());

  ctx.check(ccml_test::within(stats), "matmul exceeded tolerance");
}

}  // namespace

/* Exercises the common case on both algorithms. */
CCML_TEST(matmul, square_tiled) {
  run_case(ctx, 128, 128, 128, CCML_MATMUL_TILED, 1);
}

CCML_TEST(matmul, square_naive) {
  run_case(ctx, 128, 128, 128, CCML_MATMUL_NAIVE, 1);
}

/* The shape most likely to expose a boundary bug: none of M, N or K is a
   multiple of the tile dimension. */
CCML_TEST(matmul, ragged_dimensions) {
  run_case(ctx, 67, 53, 41, CCML_MATMUL_TILED, 2);
}

CCML_TEST(matmul, single_element) {
  run_case(ctx, 1, 1, 1, CCML_MATMUL_TILED, 3);
}

CCML_TEST(matmul, tall_skinny) {
  run_case(ctx, 512, 1, 64, CCML_MATMUL_TILED, 4);
}

CCML_TEST(matmul, long_k) {
  run_case(ctx, 32, 32, 2048, CCML_MATMUL_TILED, 5);
}

/* Argument validation must work even with no kernel behind it -- these are the
   only matmul tests that can pass in the skeleton. */
CCML_TEST(matmul, rejects_null_output) {
  float dummy = 0.0f;
  const ccml_status status =
      ccml_matmul_f32(nullptr, &dummy, &dummy, 1, 1, 1, CCML_MATMUL_AUTO, nullptr);
  ctx.check(status == CCML_ERROR_INVALID_ARGUMENT,
            std::string("expected CCML_ERROR_INVALID_ARGUMENT, got ") +
                ccml_status_string(status));
}

CCML_TEST(matmul, rejects_nonpositive_dimensions) {
  float dummy = 0.0f;
  const ccml_status status =
      ccml_matmul_f32(&dummy, &dummy, &dummy, 0, 1, 1, CCML_MATMUL_AUTO, nullptr);
  ctx.check(status == CCML_ERROR_INVALID_ARGUMENT,
            std::string("expected CCML_ERROR_INVALID_ARGUMENT, got ") +
                ccml_status_string(status));
}

/* --------------------------------------------------------- added in phase 6 -- */

/* The tiled kernel must agree with the naive baseline, not merely with the
   oracle. A shared misunderstanding of the layout would show up here. */
CCML_TEST(matmul, tiled_agrees_with_naive) {
  const int M = 131, N = 67, K = 97;  /* none a multiple of the tile */

  std::vector<float> host_a(static_cast<std::size_t>(M) * K);
  std::vector<float> host_b(static_cast<std::size_t>(K) * N);
  Rng rng(900);
  rng.normal(host_a.data(), host_a.size(), 0.0f, 1.0f);
  rng.normal(host_b.data(), host_b.size(), 0.0f, 1.0f);

  DeviceBuffer device_a(host_a.size());
  DeviceBuffer device_b(host_b.size());
  DeviceBuffer naive(static_cast<std::size_t>(M) * N);
  DeviceBuffer tiled(static_cast<std::size_t>(M) * N);
  device_a.upload(host_a.data());
  device_b.upload(host_b.data());

  CCML_ASSERT_OK_OR_SKIP(ccml_matmul_f32(naive.data(), device_a.data(),
                                         device_b.data(), M, N, K,
                                         CCML_MATMUL_NAIVE, nullptr));
  CCML_ASSERT_OK_OR_SKIP(ccml_matmul_f32(tiled.data(), device_a.data(),
                                         device_b.data(), M, N, K,
                                         CCML_MATMUL_TILED, nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  std::vector<float> naive_out(naive.count());
  std::vector<float> tiled_out(tiled.count());
  naive.download(naive_out.data());
  tiled.download(tiled_out.data());

  double worst = 0.0;
  for (std::size_t i = 0; i < naive_out.size(); ++i) {
    worst = std::fmax(worst, std::fabs(static_cast<double>(naive_out[i]) -
                                       static_cast<double>(tiled_out[i])));
  }
  ctx.note("max |naive - tiled| = " + std::to_string(worst));

  /* They sum in a different order, so exact equality is not expected. */
  ctx.check(worst < 1e-3, "the two algorithms disagree by more than rounding");
}

CCML_TEST(matmul, transformer_shapes) {
  /* The shapes the case study actually uses. */
  run_case(ctx, 128, 768, 768, CCML_MATMUL_TILED, 910);
  run_case(ctx, 512, 768, 768, CCML_MATMUL_TILED, 911);
  run_case(ctx, 128, 128, 768, CCML_MATMUL_TILED, 912);
  run_case(ctx, 128, 768, 128, CCML_MATMUL_TILED, 913);
}

CCML_TEST(matmul, wide_and_thin) {
  run_case(ctx, 1, 512, 512, CCML_MATMUL_TILED, 920);
  run_case(ctx, 512, 1, 512, CCML_MATMUL_TILED, 921);
  run_case(ctx, 512, 512, 1, CCML_MATMUL_TILED, 922);
}
