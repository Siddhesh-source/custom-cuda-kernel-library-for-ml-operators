#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "ccml/memory.h"
#include "ccml/softmax.h"
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

struct Shape {
  int rows;
  int cols;
};

std::string describe(const Shape& shape) {
  return "rows=" + std::to_string(shape.rows) + " cols=" + std::to_string(shape.cols);
}

/* Runs softmax on the given host input and returns the GPU result. */
std::vector<float> run_softmax(const ccml_test::Context& ctx,
                               const std::vector<float>& host_x,
                               const Shape& shape,
                               ccml_softmax_algo algo) {
  DeviceBuffer device_x(host_x.size());
  DeviceBuffer device_y(host_x.size());
  device_x.upload(host_x.data());

  /* Poison the output so a kernel that fails to write part of the tensor is
     caught, rather than reading back a plausible zero. */
  CCML_ASSERT_OK(ccml_device_memset_async(device_y.data(), 0xFF,
                                          device_y.bytes(), nullptr));

  CCML_ASSERT_OK_OR_SKIP(ccml_softmax_f32(device_y.data(), device_x.data(),
                                          shape.rows, shape.cols, algo,
                                          /*stream=*/nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());

  std::vector<float> got(host_x.size());
  device_y.download(got.data());
  return got;
}

/* Properties that hold for any correct softmax, checked without reference to
   the oracle. A kernel can match a reference elementwise and still be wrong if
   the reference is wrong, so these are kept independent. */
void check_properties(const ccml_test::Context& ctx,
                      const std::vector<float>& got, const Shape& shape) {
  double worst_sum_error = 0.0;

  for (int r = 0; r < shape.rows; ++r) {
    double sum = 0.0;
    for (int c = 0; c < shape.cols; ++c) {
      const double value =
          static_cast<double>(got[static_cast<std::size_t>(r) * shape.cols + c]);
      if (!std::isfinite(value)) {
        ctx.fail(describe(shape) + ": non-finite output at row " +
                 std::to_string(r) + " column " + std::to_string(c));
      }
      if (value < 0.0 || value > 1.0) {
        ctx.fail(describe(shape) + ": output " + std::to_string(value) +
                 " outside [0, 1] at row " + std::to_string(r));
      }
      sum += value;
    }
    worst_sum_error = std::fmax(worst_sum_error, std::fabs(sum - 1.0));
  }

  if (worst_sum_error > 1e-5) {
    ctx.fail(describe(shape) + ": a row summed to 1 +/- " +
             std::to_string(worst_sum_error));
  }
}

struct InputOptions {
  float mean   = 0.0f;
  float stddev = 1.0f;
};

ErrorStats run_case(const ccml_test::Context& ctx, const Shape& shape,
                    const InputOptions& options, std::uint64_t seed,
                    ccml_softmax_algo algo = CCML_SOFTMAX_BLOCK_PER_ROW) {
  const std::size_t elements =
      static_cast<std::size_t>(shape.rows) * shape.cols;

  std::vector<float> host_x(elements);
  Rng(seed).normal(host_x.data(), host_x.size(), options.mean, options.stddev);

  std::vector<double> expected;
  if (!ccml_test::softmax_reference(expected, host_x, shape.rows, shape.cols)) {
    ctx.skip("softmax_reference rejected " + describe(shape));
  }

  const std::vector<float> got = run_softmax(ctx, host_x, shape, algo);
  check_properties(ctx, got, shape);

  const Tolerance tolerance =
      ccml_test::tolerance_for_f32_reduction(shape.cols);
  const ErrorStats stats = ccml_test::compare(got, expected, tolerance);

  if (!ccml_test::within(stats)) {
    ctx.fail(describe(shape) + " exceeded tolerance -- " +
             tolerance.to_string() + " -- " + stats.to_string());
  }
  return stats;
}

void run_sweep(const ccml_test::Context& ctx, const std::vector<Shape>& shapes,
               const InputOptions& options, std::uint64_t seed) {
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
      worst_shape      = describe(shape);
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

/* ------------------------------------------------------------- small rows -- */

CCML_TEST(softmax, small_rows) {
  run_sweep(ctx,
            {{8, 1}, {8, 2}, {8, 3}, {8, 7}, {8, 16}, {8, 31}, {8, 32}, {8, 33}},
            InputOptions{}, 100);
}

CCML_TEST(softmax, single_column_is_exactly_one) {
  /* A one-element row must produce exactly 1.0: exp(0)/exp(0). */
  const Shape shape{64, 1};
  std::vector<float> host_x(64);
  Rng(110).normal(host_x.data(), host_x.size(), 0.0f, 100.0f);

  const std::vector<float> got =
      run_softmax(ctx, host_x, shape, CCML_SOFTMAX_AUTO);

  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != 1.0f) {
      ctx.fail("a single-column row gave " + std::to_string(got[i]) +
               ", expected exactly 1.0");
    }
  }
}

/* ------------------------------------------------------------- large rows -- */

CCML_TEST(softmax, large_rows) {
  run_sweep(ctx, {{64, 1024}, {64, 2048}, {64, 4096}, {32, 8192}, {16, 16384}},
            InputOptions{}, 200);
}

/* --------------------------------------------------- non-power-of-two rows -- */

CCML_TEST(softmax, non_power_of_two_row_sizes) {
  run_sweep(ctx,
            {{33, 517}, {17, 1023}, {17, 1025}, {7, 4095}, {7, 4097}, {5, 6143},
             {11, 100}, {11, 999}},
            InputOptions{}, 300);
}

/* ----------------------------------------------------------- row counts ---- */

CCML_TEST(softmax, multiple_rows) {
  run_sweep(ctx,
            {{1, 512}, {2, 512}, {7, 512}, {255, 512}, {1024, 512}, {4099, 512}},
            InputOptions{}, 400);
}

CCML_TEST(softmax, many_rows) {
  const ErrorStats stats = run_case(ctx, Shape{20000, 256}, InputOptions{}, 410);
  ctx.note(stats.to_string());
}

/* ------------------------------------------------------- extreme inputs ---- */

/* expf overflows above about x = 88.7. Without the max subtraction every large
   element becomes inf and every output becomes inf/inf = NaN. This is the test
   that proves the numerical guard is present rather than assumed. */
CCML_TEST(softmax, large_positive_inputs) {
  InputOptions options;
  options.mean   = 5000.0f;
  options.stddev = 10.0f;

  const ErrorStats stats = run_case(ctx, Shape{16, 256}, options, 500);
  ctx.note("mean=5000: " + stats.to_string());
  ctx.check(stats.nonfinite_count == 0, "produced NaN or inf");
}

CCML_TEST(softmax, large_negative_inputs) {
  InputOptions options;
  options.mean   = -5000.0f;
  options.stddev = 10.0f;

  const ErrorStats stats = run_case(ctx, Shape{16, 256}, options, 510);
  ctx.note("mean=-5000: " + stats.to_string());
  ctx.check(stats.nonfinite_count == 0, "produced NaN or inf");
}

/* A spread far wider than float32 exp can represent: most terms underflow to
   zero, which is the correct answer, and the result must still sum to 1. */
CCML_TEST(softmax, extreme_dynamic_range) {
  const Shape shape{8, 512};
  const std::size_t elements =
      static_cast<std::size_t>(shape.rows) * shape.cols;

  std::vector<float> host_x(elements);
  Rng(520).uniform(host_x.data(), host_x.size(), -10000.0f, 10000.0f);

  std::vector<double> expected;
  ctx.check(ccml_test::softmax_reference(expected, host_x, shape.rows,
                                         shape.cols),
            "reference rejected the input");

  const std::vector<float> got =
      run_softmax(ctx, host_x, shape, CCML_SOFTMAX_AUTO);
  check_properties(ctx, got, shape);

  const ErrorStats stats = ccml_test::compare(
      got, expected, ccml_test::tolerance_for_f32_reduction(shape.cols));
  ctx.note("range +/-10000: " + stats.to_string());
  ctx.check(ccml_test::within(stats), "exceeded tolerance");
}

/* One element far above the rest: the output should be essentially one-hot. */
CCML_TEST(softmax, dominant_element_gives_one_hot) {
  const int rows = 4;
  const int cols = 128;
  const std::size_t elements = static_cast<std::size_t>(rows) * cols;

  std::vector<float> host_x(elements, -500.0f);
  for (int r = 0; r < rows; ++r) {
    host_x[static_cast<std::size_t>(r) * cols + (r * 7 % cols)] = 500.0f;
  }

  const std::vector<float> got =
      run_softmax(ctx, host_x, Shape{rows, cols}, CCML_SOFTMAX_AUTO);
  check_properties(ctx, got, Shape{rows, cols});

  for (int r = 0; r < rows; ++r) {
    const std::size_t winner =
        static_cast<std::size_t>(r) * cols + (r * 7 % cols);
    if (got[winner] != 1.0f) {
      ctx.fail("the dominant element should take the whole mass, got " +
               std::to_string(got[winner]));
    }
  }
}

/* Every value identical: the result must be exactly uniform. */
CCML_TEST(softmax, constant_row_is_uniform) {
  const int rows = 4;
  const int cols = 100;
  const std::size_t elements = static_cast<std::size_t>(rows) * cols;

  std::vector<float> host_x(elements, 7.25f);
  const std::vector<float> got =
      run_softmax(ctx, host_x, Shape{rows, cols}, CCML_SOFTMAX_AUTO);
  check_properties(ctx, got, Shape{rows, cols});

  const float expected = 1.0f / static_cast<float>(cols);
  for (std::size_t i = 0; i < elements; ++i) {
    if (std::fabs(got[i] - expected) > 1e-7f) {
      ctx.fail("a constant row should be uniform: got " +
               std::to_string(got[i]) + ", expected " + std::to_string(expected));
    }
  }
}

/* --------------------------------------------------------- validation ----- */

CCML_TEST(softmax, rejects_null_and_nonpositive_dimensions) {
  float dummy = 0.0f;

  ctx.check(ccml_softmax_f32(nullptr, &dummy, 1, 1, CCML_SOFTMAX_AUTO, nullptr)
                == CCML_ERROR_INVALID_ARGUMENT,
            "a null output should be rejected");
  ctx.check(ccml_softmax_f32(&dummy, nullptr, 1, 1, CCML_SOFTMAX_AUTO, nullptr)
                == CCML_ERROR_INVALID_ARGUMENT,
            "a null input should be rejected");
  ctx.check(ccml_softmax_f32(&dummy, &dummy, 0, 1, CCML_SOFTMAX_AUTO, nullptr)
                == CCML_ERROR_INVALID_ARGUMENT,
            "zero rows should be rejected");
  ctx.check(ccml_softmax_f32(&dummy, &dummy, 1, 0, CCML_SOFTMAX_AUTO, nullptr)
                == CCML_ERROR_INVALID_ARGUMENT,
            "zero cols should be rejected");
}

CCML_TEST(softmax, warp_per_row_is_reserved_not_silently_substituted) {
  float dummy = 0.0f;
  const ccml_status status = ccml_softmax_f32(&dummy, &dummy, 1, 1,
                                              CCML_SOFTMAX_WARP_PER_ROW, nullptr);
  ctx.check(status == CCML_ERROR_NOT_IMPLEMENTED,
            "an unbuilt algorithm must report NOT_IMPLEMENTED, not fall back "
            "to a different one");
}
