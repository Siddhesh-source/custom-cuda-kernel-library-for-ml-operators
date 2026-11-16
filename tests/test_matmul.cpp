#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ccml/matmul.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "ccml/tuning.h"
#include "support/device_buffer.h"
#include "support/random.h"
#include "support/compare.h"
#include "tests/support/harness.h"
#include "support/reference.h"

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

/* ------------------------------------------- added with register blocking -- */

/* Every tile configuration of REGISTER_TILED, on shapes chosen so that no
   dimension is a multiple of any block or thread tile, plus the degenerate
   ones (a single row, a single column, K = 1). A configuration that only
   works on friendly shapes is wrong. */
CCML_TEST(matmul, register_tiled_every_config_on_awkward_shapes) {
  const int shapes[][3] = {{1, 1, 1},     {1, 512, 512}, {512, 1, 512},  {512, 512, 1},
                           {17, 33, 65},  {131, 67, 97}, {128, 128, 128}, {255, 257, 129},
                           {3, 768, 768}, {1000, 1000, 33}, {2047, 129, 513}};
  const int configs = ccml_matmul_tile_config_count();
  ctx.check(configs >= 1, "at least one tile configuration");

  for (int index = 0; index < configs; ++index) {
    ccml_matmul_tile_config cfg;
    CCML_ASSERT_OK(ccml_matmul_tile_config_get(index, &cfg));
    double worst_normalized = 0.0;
    for (const auto& shape : shapes) {
      const int M = shape[0], N = shape[1], K = shape[2];
      std::vector<float> host_a(static_cast<std::size_t>(M) * K);
      std::vector<float> host_b(static_cast<std::size_t>(K) * N);
      Rng rng(1000 + static_cast<std::uint64_t>(index) * 97 + M + N + K);
      rng.normal(host_a.data(), host_a.size(), 0.0f, 1.0f);
      rng.normal(host_b.data(), host_b.size(), 0.0f, 1.0f);
      std::vector<double> expected;
      ccml_test::matmul_reference(expected, host_a, host_b, M, N, K);

      DeviceBuffer a(host_a.size()), b(host_b.size()), c(static_cast<std::size_t>(M) * N);
      a.upload(host_a.data());
      b.upload(host_b.data());
      /* Poison: an element the kernel fails to write cannot pass. */
      CCML_ASSERT_OK(ccml_device_memset_async(c.data(), 0xFF, c.bytes(), nullptr));
      CCML_ASSERT_OK(ccml_matmul_register_tiled_f32(c.data(), a.data(), b.data(), M, N, K,
                                                    index, nullptr));
      CCML_ASSERT_OK(ccml_device_synchronize());
      std::vector<float> got(c.count());
      c.download(got.data());

      const ccml_test::Tolerance tol =
          ccml_test::tolerance_for_f32_dot_product(K, ccml_test::rms(expected));
      const ccml_test::ErrorStats stats = ccml_test::compare(got, expected, tol);
      worst_normalized = std::fmax(worst_normalized, stats.max_normalized_error);
      if (!ccml_test::within(stats)) {
        ctx.fail("config " + std::to_string(index) + " (" + std::to_string(cfg.bm) + "x" +
                 std::to_string(cfg.bn) + "x" + std::to_string(cfg.bk) + ", " +
                 std::to_string(cfg.tm) + "x" + std::to_string(cfg.tn) + ") failed at M=" +
                 std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                 " -- " + stats.to_string());
      }
    }
    ctx.note("config " + std::to_string(index) + " " + std::to_string(cfg.bm) + "x" +
             std::to_string(cfg.bn) + "x" + std::to_string(cfg.bk) + " thread " +
             std::to_string(cfg.tm) + "x" + std::to_string(cfg.tn) + " (" +
             std::to_string(cfg.threads) + " threads): worst normalized error " +
             std::to_string(worst_normalized));
  }
}

CCML_TEST(matmul, register_tiled_through_the_algo_enum) {
  run_case(ctx, 128, 768, 768, CCML_MATMUL_REGISTER_TILED, 930);
  run_case(ctx, 131, 67, 97, CCML_MATMUL_REGISTER_TILED, 931);
  run_case(ctx, 1, 2304, 768, CCML_MATMUL_REGISTER_TILED, 932);
  run_case(ctx, 256, 96, 2048, CCML_MATMUL_REGISTER_TILED, 933);
}

CCML_TEST(matmul, tile_config_api_rejects_bad_input) {
  ccml_matmul_tile_config cfg;
  ctx.check(ccml_matmul_tile_config_get(-1, &cfg) == CCML_ERROR_INVALID_ARGUMENT, "index -1");
  ctx.check(ccml_matmul_tile_config_get(ccml_matmul_tile_config_count(), &cfg) ==
                CCML_ERROR_INVALID_ARGUMENT,
            "index == count");
  ctx.check(ccml_matmul_tile_config_get(0, nullptr) == CCML_ERROR_INVALID_ARGUMENT, "null out");
  DeviceBuffer d(4);
  ctx.check(ccml_matmul_register_tiled_f32(d.data(), d.data(), d.data(), 2, 2, 1,
                                           ccml_matmul_tile_config_count(), nullptr) ==
                CCML_ERROR_INVALID_ARGUMENT,
            "config index out of range is rejected, not clamped");
  for (int i = 0; i < ccml_matmul_tile_config_count(); ++i) {
    CCML_ASSERT_OK(ccml_matmul_tile_config_get(i, &cfg));
    ctx.check(cfg.threads == (cfg.bm / cfg.tm) * (cfg.bn / cfg.tn) && cfg.threads <= 1024,
              "thread count consistent and launchable");
  }
}
