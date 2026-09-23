/*
 * Tests for the test harness itself.
 *
 * This suite exists because everything else depends on it. If compare() is
 * wrong, every operator test is wrong in the same direction and all of them
 * still report PASS. If the RNG is not reproducible, the reproducibility claim
 * in docs/correctness.md is false. Neither failure is visible from the outside,
 * so both are checked here directly.
 *
 * White-box: includes src/common for the utility headers.
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "ccml/device.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "common/launch.h"
#include "common/util.h"
#include "support/cuda_stream.h"
#include "support/device_buffer.h"
#include "support/random.h"
#include "tests/support/compare.h"
#include "tests/support/harness.h"

using ccml_support::CudaStream;
using ccml_support::DeviceBuffer;
using ccml_support::Rng;
using ccml_test::compare;
using ccml_test::ErrorStats;
using ccml_test::Tolerance;
using ccml_test::within;

/* ------------------------------------------------------------- compare -- */

CCML_TEST(harness, compare_identical_inputs_report_zero_error) {
  const std::vector<float>  got{1.0f, -2.5f, 1e6f, 0.0f};
  const std::vector<double> reference{1.0, -2.5, 1e6, 0.0};

  const ErrorStats stats = compare(got, reference, Tolerance{1e-9, 1e-9});

  ctx.check(stats.max_abs_error == 0.0, "max_abs_error should be exactly zero");
  ctx.check(stats.max_rel_error == 0.0, "max_rel_error should be exactly zero");
  ctx.check(stats.nonfinite_count == 0, "nothing was non-finite");
  ctx.check(stats.count == 4, "count should equal the input length");
  ctx.check(within(stats), "identical inputs must be within any tolerance");
}

CCML_TEST(harness, compare_reports_the_largest_deviation) {
  const std::vector<float>  got{1.0f, 2.0f, 3.5f};
  const std::vector<double> reference{1.0, 2.0, 3.0};

  const ErrorStats stats = compare(got, reference, Tolerance{1.0, 0.0});

  ctx.note(stats.to_string());
  ctx.check(std::fabs(stats.max_abs_error - 0.5) < 1e-6,
            "max_abs_error should be 0.5");
  ctx.check(std::fabs(stats.max_rel_error - (0.5 / 3.0)) < 1e-6,
            "max_rel_error should be 0.5/3");
  ctx.check(stats.worst_index == 2, "the worst element is at index 2");
}

/* The pass/fail decision rides entirely on max_normalized_error, so it is
   checked on both sides of the boundary rather than just "roughly right". */
CCML_TEST(harness, compare_normalized_error_decides_the_verdict) {
  const Tolerance tolerance{0.0, 0.1};  /* bound = 0.1 * |reference| */
  const std::vector<double> reference{1.0};

  const ErrorStats inside = compare(std::vector<float>{1.05f}, reference, tolerance);
  ctx.note("inside:  " + inside.to_string());
  ctx.check(inside.max_normalized_error < 1.0,
            "a deviation of half the bound should normalize below 1");
  ctx.check(within(inside), "a deviation inside the bound must pass");

  const ErrorStats outside = compare(std::vector<float>{1.2f}, reference, tolerance);
  ctx.note("outside: " + outside.to_string());
  ctx.check(outside.max_normalized_error > 1.0,
            "a deviation of twice the bound should normalize above 1");
  ctx.check(!within(outside), "a deviation outside the bound must fail");
}

/* A pure relative bound is useless where the reference is zero -- which is
   exactly what softmax tails and normalized values look like. This is the case
   the combined abs+rel form exists for. */
CCML_TEST(harness, compare_uses_absolute_tolerance_near_zero) {
  const std::vector<double> reference{0.0};
  const Tolerance relative_only{0.0, 0.1};
  const Tolerance combined{1e-3, 0.1};

  const ErrorStats strict = compare(std::vector<float>{1e-6f}, reference,
                                    relative_only);
  ctx.check(!within(strict),
            "with no absolute tolerance, any deviation from zero must fail");

  const ErrorStats lenient = compare(std::vector<float>{1e-6f}, reference,
                                     combined);
  ctx.check(within(lenient),
            "an absolute tolerance must admit small deviations from zero");
}

CCML_TEST(harness, compare_flags_nonfinite_results) {
  const std::vector<double> reference{1.0, 1.0, 1.0};
  const std::vector<float>  got{1.0f,
                                std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity()};

  const ErrorStats stats = compare(got, reference, Tolerance{1e9, 1e9});

  ctx.note(stats.to_string());
  ctx.check(stats.nonfinite_count == 2, "should have counted one NaN and one inf");
  ctx.check(!within(stats),
            "a non-finite result must fail however wide the tolerance");
}

CCML_TEST(harness, compare_flags_a_size_mismatch) {
  const ErrorStats stats = compare(std::vector<float>{1.0f, 2.0f},
                                   std::vector<double>{1.0},
                                   Tolerance{1e9, 1e9});

  ctx.check(stats.size_mismatch, "a size mismatch should be flagged");
  ctx.check(!within(stats), "a size mismatch must never pass");
}

CCML_TEST(harness, tolerance_grows_with_reduction_length) {
  const Tolerance small = ccml_test::tolerance_for_f32_reduction(1);
  const Tolerance large = ccml_test::tolerance_for_f32_reduction(10000);

  ctx.note("terms=1:     " + small.to_string());
  ctx.note("terms=10000: " + large.to_string());

  ctx.check(large.rel_tol > small.rel_tol,
            "a longer reduction must get a looser tolerance");
  /* sqrt(10000) = 100, so the ratio should be about 100x. */
  const double ratio = large.rel_tol / small.rel_tol;
  ctx.check(ratio > 50.0 && ratio < 200.0,
            "the tolerance should grow with sqrt(terms), got a ratio of "
                + std::to_string(ratio));
}

/* ----------------------------------------------------------------- rng -- */

/* Reproducibility is a stated project goal; this is the test that backs it. */
CCML_TEST(harness, rng_is_reproducible_for_a_given_seed) {
  std::vector<float> first(1000);
  std::vector<float> second(1000);

  Rng(12345).normal(first.data(), first.size(), 0.0f, 1.0f);
  Rng(12345).normal(second.data(), second.size(), 0.0f, 1.0f);

  for (std::size_t i = 0; i < first.size(); ++i) {
    if (first[i] != second[i]) {
      ctx.fail("the same seed produced different values at index "
               + std::to_string(i));
    }
  }
}

CCML_TEST(harness, rng_differs_between_seeds) {
  std::vector<float> first(1000);
  std::vector<float> second(1000);

  Rng(1).normal(first.data(), first.size(), 0.0f, 1.0f);
  Rng(2).normal(second.data(), second.size(), 0.0f, 1.0f);

  std::size_t identical = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    if (first[i] == second[i]) {
      ++identical;
    }
  }
  ctx.check(identical < 10,
            "different seeds produced " + std::to_string(identical)
                + " identical values, which suggests the seed is ignored");
}

CCML_TEST(harness, rng_uniform_respects_its_bounds) {
  std::vector<float> values(10000);
  Rng(7).uniform(values.data(), values.size(), -2.0f, 5.0f);

  float lowest = values[0];
  float highest = values[0];
  for (float value : values) {
    if (value < -2.0f || value >= 5.0f) {
      ctx.fail("value " + std::to_string(value) + " is outside [-2, 5)");
    }
    lowest = std::fmin(lowest, value);
    highest = std::fmax(highest, value);
  }

  ctx.note("range covered: [" + std::to_string(lowest) + ", "
           + std::to_string(highest) + "]");
  /* A generator stuck near one end would still satisfy the bounds check. */
  ctx.check(lowest < -1.0f && highest > 4.0f,
            "the samples should span most of the requested range");
}

CCML_TEST(harness, rng_normal_has_the_requested_moments) {
  const std::size_t count = 100000;
  std::vector<float> values(count);
  Rng(99).normal(values.data(), count, 5.0f, 2.0f);

  double sum = 0.0;
  for (float value : values) {
    sum += value;
  }
  const double mean = sum / static_cast<double>(count);

  double variance = 0.0;
  for (float value : values) {
    variance += (value - mean) * (value - mean);
  }
  const double stddev = std::sqrt(variance / static_cast<double>(count));

  ctx.note("mean=" + std::to_string(mean) + " stddev=" + std::to_string(stddev));
  ctx.check(std::fabs(mean - 5.0) < 0.05, "mean should be close to 5");
  ctx.check(std::fabs(stddev - 2.0) < 0.05, "stddev should be close to 2");
}

/* ------------------------------------------------------------ utilities -- */

CCML_TEST(harness, div_up_rounds_up) {
  using ccml::detail::div_up;
  ctx.check(div_up(0, 16) == 0, "0/16 should be 0 blocks");
  ctx.check(div_up(1, 16) == 1, "1/16 should be 1 block");
  ctx.check(div_up(16, 16) == 1, "16/16 should be 1 block");
  ctx.check(div_up(17, 16) == 2, "17/16 should be 2 blocks");
  ctx.check(div_up(32, 16) == 2, "32/16 should be 2 blocks");
}

CCML_TEST(harness, round_up_reaches_the_next_multiple) {
  using ccml::detail::round_up;
  ctx.check(round_up(1, 16) == 16, "1 should round to 16");
  ctx.check(round_up(16, 16) == 16, "16 should stay 16");
  ctx.check(round_up(17, 16) == 32, "17 should round to 32");
}

CCML_TEST(harness, grid_helpers_cover_every_element) {
  const int elements = 1000;
  const int block = 256;
  const dim3 grid = ccml::detail::grid_for_1d(elements, block);

  ctx.check(grid.x * block >= static_cast<unsigned>(elements),
            "the grid must cover every element");
  ctx.check((grid.x - 1) * block < static_cast<unsigned>(elements),
            "the grid should not be larger than necessary");
}

/* -------------------------------------------------------------- status -- */

CCML_TEST(harness, every_status_has_a_distinct_message) {
  const ccml_status codes[] = {
      CCML_SUCCESS, CCML_ERROR_INVALID_ARGUMENT, CCML_ERROR_UNSUPPORTED_SHAPE,
      CCML_ERROR_NOT_IMPLEMENTED, CCML_ERROR_CUDA, CCML_ERROR_OUT_OF_MEMORY};

  std::vector<std::string> seen;
  for (ccml_status code : codes) {
    const std::string text = ccml_status_string(code);
    ctx.check(!text.empty(), "a status string was empty");
    for (const std::string& previous : seen) {
      if (previous == text) {
        ctx.fail("two status codes share the message \"" + text + "\"");
      }
    }
    seen.push_back(text);
  }
}

CCML_TEST(harness, version_string_is_populated) {
  const std::string version = ccml_version_string();
  ctx.note("version: " + version);
  ctx.check(!version.empty(), "version string is empty");
  ctx.check(version.find('.') != std::string::npos,
            "version should look like a dotted version number");
}

/* ---------------------------------------------------------------- RAII -- */

CCML_TEST(harness, freeing_a_null_pointer_is_a_no_op) {
  CCML_ASSERT_OK(ccml_device_free(nullptr));
}

CCML_TEST(harness, device_buffer_move_transfers_ownership) {
  DeviceBuffer source(1024);
  const float* original = source.data();
  ctx.check(original != nullptr, "allocation returned null");

  DeviceBuffer moved(std::move(source));

  ctx.check(moved.data() == original, "the moved-to buffer should hold the pointer");
  ctx.check(moved.count() == 1024, "the moved-to buffer should hold the count");
  ctx.check(source.data() == nullptr,
            "the moved-from buffer must be emptied, or it will double-free");
  ctx.check(source.count() == 0, "the moved-from buffer should report zero count");
}

CCML_TEST(harness, cuda_stream_move_transfers_ownership) {
  CudaStream source = CudaStream::create();
  const cudaStream_t original = source.get();
  ctx.check(original != nullptr, "created stream is null");

  CudaStream moved(std::move(source));

  ctx.check(moved.get() == original, "the moved-to stream should hold the handle");
  ctx.check(source.get() == nullptr,
            "the moved-from stream must be emptied, or it will double-destroy");

  /* Both destructors run at scope exit; a double destroy would show up here or
     in the next CUDA call. */
  CCML_ASSERT_OK(ccml_stream_synchronize(moved.get()));
}
