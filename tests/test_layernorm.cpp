#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "ccml/layernorm.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "support/device_buffer.h"
#include "support/random.h"
#include "support/compare.h"
#include "tests/support/harness.h"
#include "support/reference.h"

using ccml_support::DeviceBuffer;
using ccml_support::Rng;
using ccml_test::ErrorStats;
using ccml_test::Tolerance;

namespace {

constexpr float kEps = 1e-5f;

struct Shape {
  int rows;
  int cols;
};

struct CaseOptions {
  bool  use_gamma    = true;
  bool  use_beta     = true;
  float input_mean   = 0.0f;
  float input_stddev = 1.0f;
  float eps          = kEps;
  ccml_layernorm_algo algo = CCML_LAYERNORM_TWO_PASS;
};

std::string describe(const Shape& shape, const CaseOptions& options) {
  return "rows=" + std::to_string(shape.rows) +
         " cols=" + std::to_string(shape.cols) +
         " gamma=" + (options.use_gamma ? "y" : "n") +
         " beta=" + (options.use_beta ? "y" : "n");
}

/* Runs one shape, checks it, and returns the measured error so that grouped
   tests can report the worst case across a sweep. */
ErrorStats run_case(const ccml_test::Context& ctx, const Shape& shape,
                    const CaseOptions& options, std::uint64_t seed) {
  const std::size_t elements =
      static_cast<std::size_t>(shape.rows) * shape.cols;

  std::vector<float> host_x(elements);
  std::vector<float> host_gamma(
      options.use_gamma ? static_cast<std::size_t>(shape.cols) : 0);
  std::vector<float> host_beta(
      options.use_beta ? static_cast<std::size_t>(shape.cols) : 0);

  Rng rng(seed);
  rng.normal(host_x.data(), host_x.size(), options.input_mean,
             options.input_stddev);
  if (options.use_gamma) {
    rng.uniform(host_gamma.data(), host_gamma.size(), 0.5f, 1.5f);
  }
  if (options.use_beta) {
    rng.uniform(host_beta.data(), host_beta.size(), -1.0f, 1.0f);
  }

  std::vector<double> expected;
  if (!ccml_test::layernorm_reference(expected, host_x, host_gamma, host_beta,
                                      shape.rows, shape.cols, options.eps)) {
    ctx.skip("layernorm_reference rejected " + describe(shape, options));
  }

  DeviceBuffer device_x(elements);
  DeviceBuffer device_y(elements);
  device_x.upload(host_x.data());

  /* Poison the output so a kernel that fails to write part of the tensor is
     caught, rather than reading back a plausible zero. */
  CCML_ASSERT_OK(ccml_device_memset_async(device_y.data(), 0xFF,
                                          device_y.bytes(), nullptr));

  DeviceBuffer device_gamma;
  DeviceBuffer device_beta;
  if (options.use_gamma) {
    device_gamma.allocate(host_gamma.size());
    device_gamma.upload(host_gamma.data());
  }
  if (options.use_beta) {
    device_beta.allocate(host_beta.size());
    device_beta.upload(host_beta.data());
  }

  CCML_ASSERT_OK_OR_SKIP(ccml_layernorm_f32(
      device_y.data(), device_x.data(),
      options.use_gamma ? device_gamma.data() : nullptr,
      options.use_beta ? device_beta.data() : nullptr,
      shape.rows, shape.cols, options.eps, options.algo, /*stream=*/nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  std::vector<float> got(elements);
  device_y.download(got.data());

  const Tolerance tolerance =
      ccml_test::tolerance_for_f32_reduction(shape.cols);
  const ErrorStats stats = ccml_test::compare(got, expected, tolerance);

  if (!ccml_test::within(stats)) {
    ctx.fail(describe(shape, options) + " exceeded tolerance -- " +
             tolerance.to_string() + " -- " + stats.to_string());
  }
  return stats;
}

/* Runs a sweep and notes the worst error observed across it. */
void run_sweep(const ccml_test::Context& ctx, const std::vector<Shape>& shapes,
               const CaseOptions& options, std::uint64_t seed) {
  double worst_normalized = 0.0;
  double worst_abs = 0.0;
  double worst_rel = 0.0;
  std::string worst_shape;

  for (const Shape& shape : shapes) {
    const ErrorStats stats = run_case(ctx, shape, options, seed++);
    if (stats.max_normalized_error >= worst_normalized) {
      worst_normalized = stats.max_normalized_error;
      worst_abs        = stats.max_abs_error;
      worst_rel        = stats.max_rel_error;
      worst_shape      = describe(shape, options);
    }
  }

  char summary[256];
  std::snprintf(summary, sizeof(summary),
                "%zu shapes; worst max_abs=%.3e max_rel=%.3e normalized=%.3f",
                shapes.size(), worst_abs, worst_rel, worst_normalized);
  ctx.note(summary);
  ctx.note("worst at: " + worst_shape);
}

}  // namespace

/* ------------------------------------------------------- small dimensions -- */

/* Column counts at and below one warp, where most of a block sits idle and the
   reduction has fewer participants than lanes. */
CCML_TEST(layernorm, small_dimensions) {
  run_sweep(ctx,
            {{8, 1}, {8, 2}, {8, 3}, {8, 7}, {8, 16}, {8, 31}, {8, 32}, {8, 33}},
            CaseOptions{}, 100);
}

CCML_TEST(layernorm, single_column_is_finite) {
  /* A one-element row has zero variance, so this is entirely a test of the eps
     guard: without it the result is 0/0. */
  const ErrorStats stats = run_case(ctx, Shape{128, 1}, CaseOptions{}, 110);
  ctx.note(stats.to_string());
  ctx.check(stats.nonfinite_count == 0, "a zero-variance row produced NaN or inf");
}

/* ------------------------------------------------------- large dimensions -- */

CCML_TEST(layernorm, large_dimensions) {
  run_sweep(ctx, {{64, 1024}, {64, 2048}, {64, 4096}, {32, 8192}, {16, 16384}},
            CaseOptions{}, 200);
}

/* ----------------------------------------------------- awkward dimensions -- */

/* None of these is a multiple of the warp size or the block size, so every one
   exercises the striding tail. */
CCML_TEST(layernorm, awkward_dimensions) {
  run_sweep(ctx,
            {{33, 517}, {17, 1023}, {17, 1025}, {7, 4095}, {7, 4097}, {5, 6143}},
            CaseOptions{}, 300);
}

/* --------------------------------------------------------- row counts ----- */

CCML_TEST(layernorm, row_counts) {
  run_sweep(ctx,
            {{1, 768}, {2, 768}, {7, 768}, {255, 768}, {1024, 768}, {4099, 768}},
            CaseOptions{}, 400);
}

CCML_TEST(layernorm, many_rows) {
  const ErrorStats stats = run_case(ctx, Shape{20000, 256}, CaseOptions{}, 410);
  ctx.note(stats.to_string());
}

/* ------------------------------------------------------ affine variants --- */

CCML_TEST(layernorm, affine_variants) {
  const Shape shape{64, 768};

  CaseOptions both;
  CaseOptions neither;  neither.use_gamma = false; neither.use_beta = false;
  CaseOptions gamma_only; gamma_only.use_beta = false;
  CaseOptions beta_only;  beta_only.use_gamma = false;

  run_case(ctx, shape, both, 500);
  run_case(ctx, shape, neither, 501);
  run_case(ctx, shape, gamma_only, 502);
  run_case(ctx, shape, beta_only, 503);
}

/* ------------------------------------------------------------ numerics ---- */

/* The case a naive E[x^2] - E[x]^2 variance gets visibly wrong. With a mean far
   from zero and a small spread, the two terms agree to more digits than float32
   holds, and the subtraction returns mostly rounding error. */
CCML_TEST(layernorm, large_offset_small_variance) {
  CaseOptions options;
  options.input_mean   = 10000.0f;
  options.input_stddev = 0.01f;

  const ErrorStats stats = run_case(ctx, Shape{32, 512}, options, 600);
  ctx.note("mean=10000 stddev=0.01: " + stats.to_string());
  ctx.check(stats.nonfinite_count == 0, "produced NaN or inf");
}

CCML_TEST(layernorm, moderate_offset) {
  CaseOptions options;
  options.input_mean   = 1000.0f;
  options.input_stddev = 1.0f;

  const ErrorStats stats = run_case(ctx, Shape{64, 1024}, options, 610);
  ctx.note("mean=1000 stddev=1: " + stats.to_string());
}

CCML_TEST(layernorm, constant_row_is_finite) {
  /* Every value identical means variance is exactly zero. Output must be the
     bias alone, not NaN. */
  const int rows = 4;
  const int cols = 128;
  const std::size_t elements = static_cast<std::size_t>(rows) * cols;

  std::vector<float> host_x(elements, 3.5f);
  std::vector<double> expected;
  ctx.check(ccml_test::layernorm_reference(expected, host_x, {}, {}, rows, cols,
                                           kEps),
            "reference rejected a constant input");

  DeviceBuffer device_x(elements);
  DeviceBuffer device_y(elements);
  device_x.upload(host_x.data());

  CCML_ASSERT_OK_OR_SKIP(ccml_layernorm_f32(device_y.data(), device_x.data(),
                                            nullptr, nullptr, rows, cols, kEps,
                                            CCML_LAYERNORM_AUTO, nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  std::vector<float> got(elements);
  device_y.download(got.data());

  for (std::size_t i = 0; i < elements; ++i) {
    if (!std::isfinite(got[i])) {
      ctx.fail("constant row produced a non-finite value at " + std::to_string(i));
    }
    if (std::fabs(got[i]) > 1e-3f) {
      ctx.fail("constant row should normalize to zero, got "
               + std::to_string(got[i]));
    }
  }
}

CCML_TEST(layernorm, epsilon_is_configurable) {
  /* A larger eps shrinks every output, because it only ever adds to the
     denominator. Checking the direction proves eps reaches the kernel rather
     than being ignored. */
  const Shape shape{16, 256};
  const std::size_t elements =
      static_cast<std::size_t>(shape.rows) * shape.cols;

  std::vector<float> host_x(elements);
  Rng(700).normal(host_x.data(), host_x.size(), 0.0f, 1.0f);

  DeviceBuffer device_x(elements);
  DeviceBuffer small_eps(elements);
  DeviceBuffer large_eps(elements);
  device_x.upload(host_x.data());

  CCML_ASSERT_OK_OR_SKIP(ccml_layernorm_f32(
      small_eps.data(), device_x.data(), nullptr, nullptr, shape.rows,
      shape.cols, 1e-5f, CCML_LAYERNORM_AUTO, nullptr));
  CCML_ASSERT_OK_OR_SKIP(ccml_layernorm_f32(
      large_eps.data(), device_x.data(), nullptr, nullptr, shape.rows,
      shape.cols, 10.0f, CCML_LAYERNORM_AUTO, nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  std::vector<float> small_out(elements);
  std::vector<float> large_out(elements);
  small_eps.download(small_out.data());
  large_eps.download(large_out.data());

  std::size_t shrunk = 0;
  for (std::size_t i = 0; i < elements; ++i) {
    if (std::fabs(large_out[i]) < std::fabs(small_out[i])) {
      ++shrunk;
    }
  }
  ctx.note(std::to_string(shrunk) + " of " + std::to_string(elements)
           + " magnitudes shrank when eps went from 1e-5 to 10");
  ctx.check(shrunk > elements * 9 / 10,
            "a much larger eps should shrink almost every output");
}

/* --------------------------------------------------------- validation ----- */

CCML_TEST(layernorm, rejects_nonpositive_eps) {
  float dummy = 0.0f;
  const ccml_status status =
      ccml_layernorm_f32(&dummy, &dummy, nullptr, nullptr, 1, 1, 0.0f,
                         CCML_LAYERNORM_AUTO, nullptr);
  ctx.check(status == CCML_ERROR_INVALID_ARGUMENT,
            std::string("expected CCML_ERROR_INVALID_ARGUMENT, got ") +
                ccml_status_string(status));
}

CCML_TEST(layernorm, rejects_null_and_nonpositive_dimensions) {
  float dummy = 0.0f;

  ctx.check(ccml_layernorm_f32(nullptr, &dummy, nullptr, nullptr, 1, 1, kEps,
                               CCML_LAYERNORM_AUTO, nullptr)
                == CCML_ERROR_INVALID_ARGUMENT,
            "a null output should be rejected");

  ctx.check(ccml_layernorm_f32(&dummy, nullptr, nullptr, nullptr, 1, 1, kEps,
                               CCML_LAYERNORM_AUTO, nullptr)
                == CCML_ERROR_INVALID_ARGUMENT,
            "a null input should be rejected");

  ctx.check(ccml_layernorm_f32(&dummy, &dummy, nullptr, nullptr, 0, 1, kEps,
                               CCML_LAYERNORM_AUTO, nullptr)
                == CCML_ERROR_INVALID_ARGUMENT,
            "zero rows should be rejected");

  ctx.check(ccml_layernorm_f32(&dummy, &dummy, nullptr, nullptr, 1, 0, kEps,
                               CCML_LAYERNORM_AUTO, nullptr)
                == CCML_ERROR_INVALID_ARGUMENT,
            "zero cols should be rejected");
}

CCML_TEST(layernorm, welford_is_reserved_not_silently_substituted) {
  float dummy = 0.0f;
  const ccml_status status =
      ccml_layernorm_f32(&dummy, &dummy, nullptr, nullptr, 1, 1, kEps,
                         CCML_LAYERNORM_WELFORD, nullptr);
  ctx.check(status == CCML_ERROR_NOT_IMPLEMENTED,
            "an unbuilt algorithm must report NOT_IMPLEMENTED, not fall back "
            "to a different one");
}
