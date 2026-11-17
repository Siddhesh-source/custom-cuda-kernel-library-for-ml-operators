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

/* ------------------------------------------- every algorithm, every case --- */

/* Each built algorithm must meet the same FP32 tolerance on every case above:
   the numerics cases (mean far from zero) most of all, because the FP32
   variants rely on the pivot shift for them (optimization log entry 9). */
const ccml_layernorm_algo kAlgos[] = {CCML_LAYERNORM_TWO_PASS, CCML_LAYERNORM_WELFORD,
                                      CCML_LAYERNORM_WARP_PER_ROW};

CCML_TEST(layernorm, every_algorithm_on_every_case) {
  struct Case {
    Shape shape;
    float mean, stddev;
  };
  const Case cases[] = {
      {{8, 1}, 0, 1},       {{8, 7}, 0, 1},       {{8, 33}, 0, 1},       {{128, 1}, 0, 1},
      {{33, 517}, 0, 1},    {{17, 1023}, 0, 1},   {{17, 1025}, 0, 1},    {{7, 4097}, 0, 1},
      {{5, 6143}, 0, 1},    {{16, 16384}, 0, 1},  {{4099, 768}, 0, 1},   {{64, 2048}, 0, 1},
      {{32, 512}, 10000.0f, 0.01f},  {{64, 1024}, 1000.0f, 1.0f},
      {{64, 768}, -5000.0f, 0.5f},   {{16, 4096}, 100.0f, 0.001f}};
  for (ccml_layernorm_algo algo : kAlgos) {
    double worst = 0.0;
    for (const Case& c : cases) {
      /* WARP_PER_ROW holds the row in one warp: cols <= 1024 only (the
         rejection of wider rows is its own test below). */
      if (algo == CCML_LAYERNORM_WARP_PER_ROW && c.shape.cols > 1024) continue;
      CaseOptions o;
      o.input_mean = c.mean;
      o.input_stddev = c.stddev;
      o.algo = algo;
      worst = std::fmax(worst, run_case(ctx, c.shape, o, 800 + c.shape.cols).max_normalized_error);
    }
    ctx.note(std::string(ccml_layernorm_algo_name(algo)) + ": worst normalized error " +
             std::to_string(worst));
  }
}

/* Every per-lane register bucket of WARP_PER_ROW (1, 2, 4, 8, 16, 24, 32
   values per lane), at both edges of each, and its refusal past 1024. */
CCML_TEST(layernorm, warp_per_row_every_bucket) {
  double worst = 0.0;
  for (int cols : {1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
                   767, 768, 769, 1000, 1023, 1024}) {
    CaseOptions o;
    o.algo = CCML_LAYERNORM_WARP_PER_ROW;
    o.input_mean = cols % 2 ? 300.0f : 0.0f; /* odd widths also carry an offset */
    worst = std::fmax(worst, run_case(ctx, Shape{37, cols}, o, 1000 + cols).max_normalized_error);
  }
  ctx.note("worst normalized error " + std::to_string(worst));
  DeviceBuffer d(2 * 1025);
  ctx.check(ccml_layernorm_f32(d.data(), d.data(), nullptr, nullptr, 2, 1025, kEps,
                               CCML_LAYERNORM_WARP_PER_ROW, nullptr) ==
                CCML_ERROR_UNSUPPORTED_SHAPE,
            "cols = 1025 is refused with UNSUPPORTED_SHAPE, never truncated");
}

/* A pivot that is an outlier of its row (first element 1000 standard
   deviations from the rest) must not break the FP32 algorithms: Welford's
   update is stable for any offset, the pivot only has to keep the numbers
   small. */
CCML_TEST(layernorm, outlier_pivot) {
  for (ccml_layernorm_algo algo : kAlgos) {
    const int rows = 8, cols = 1000; /* <= 1024: every algorithm runs it */
    std::vector<float> host_x(static_cast<std::size_t>(rows) * cols);
    Rng(900).normal(host_x.data(), host_x.size(), 50.0f, 0.1f);
    for (int r = 0; r < rows; ++r) host_x[static_cast<std::size_t>(r) * cols] = 150.0f;
    std::vector<double> expected;
    ctx.check(ccml_test::layernorm_reference(expected, host_x, {}, {}, rows, cols, kEps),
              "reference");
    DeviceBuffer dx(host_x.size()), dy(host_x.size());
    dx.upload(host_x.data());
    CCML_ASSERT_OK(ccml_device_memset_async(dy.data(), 0xFF, dy.bytes(), nullptr));
    CCML_ASSERT_OK_OR_SKIP(ccml_layernorm_f32(dy.data(), dx.data(), nullptr, nullptr, rows, cols,
                                              kEps, algo, nullptr));
    CCML_ASSERT_OK(ccml_device_synchronize());
    std::vector<float> got(host_x.size());
    dy.download(got.data());
    const ErrorStats st = ccml_test::compare(got, expected,
                                             ccml_test::tolerance_for_f32_reduction(cols));
    ctx.note(std::string(ccml_layernorm_algo_name(algo)) + ": " + st.to_string());
    ctx.check(ccml_test::within(st), "outlier pivot exceeded tolerance");
  }
}
