/* Tests for the benchmark harness's own arithmetic: statistics, the shape-file
   reader, FLOP/byte counts, peak formulas, throttle classification and the
   CSV gate rule. If any of these were wrong, every reported number would be
   wrong in the same direction while still looking plausible. No timing. */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench/support/metrics.h"
#include "bench/support/report.h"
#include "bench/support/shapes.h"
#include "bench/support/stats.h"
#include "bench/support/telemetry.h"
#include "bench/support/timer.h"
#include "tests/support/harness.h"

using namespace ccml_bench;

namespace {
bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol; }
}  // namespace

/* ---------------------------------------------------------------- stats -- */

CCML_TEST(bench_stats, percentile_matches_numpy_linear_definition) {
  const std::vector<double> v{1, 2, 3, 4};  /* sorted */
  /* numpy.percentile([1,2,3,4], [0,10,25,50,90,100]) */
  ctx.check(near(percentile_sorted(v, 0.0), 1.0), "p0");
  ctx.check(near(percentile_sorted(v, 0.10), 1.3), "p10 = 1.3");
  ctx.check(near(percentile_sorted(v, 0.25), 1.75), "p25 = 1.75");
  ctx.check(near(percentile_sorted(v, 0.50), 2.5), "p50 = 2.5");
  ctx.check(near(percentile_sorted(v, 0.90), 3.7), "p90 = 3.7");
  ctx.check(near(percentile_sorted(v, 1.0), 4.0), "p100");
}

CCML_TEST(bench_stats, median_odd_even_and_unsorted) {
  ctx.check(near(median_of({5, 1, 3}), 3.0), "odd count");
  ctx.check(near(median_of({4, 1, 3, 2}), 2.5), "even count");
  ctx.check(near(median_of({7}), 7.0), "single sample");
}

CCML_TEST(bench_stats, summary_fields_are_consistent) {
  std::vector<double> v;
  for (int i = 1; i <= 101; ++i) v.push_back(static_cast<double>(i));
  const Summary s = summarize(v);
  ctx.check(s.valid && s.n == 101, "valid, 101 samples");
  ctx.check(near(s.median, 51.0), "median 51");
  ctx.check(near(s.min, 1.0) && near(s.max, 101.0), "min/max");
  ctx.check(near(s.mean, 51.0), "mean 51");
  ctx.check(near(s.p10, 11.0) && near(s.p90, 91.0), "p10 11, p90 91");
  ctx.check(near(s.iqr, 50.0), "iqr = p75 - p25 = 76 - 26");
  ctx.check(s.ci_lo <= s.median && s.median <= s.ci_hi, "CI brackets the median");
}

CCML_TEST(bench_stats, bootstrap_is_deterministic_and_degenerate_safe) {
  std::vector<double> v;
  for (int i = 0; i < 200; ++i) v.push_back(std::sin(i) * 10.0 + 100.0);
  double lo1, hi1, lo2, hi2;
  bootstrap_median_ci(v, 1000, 42, &lo1, &hi1);
  bootstrap_median_ci(v, 1000, 42, &lo2, &hi2);
  ctx.check(lo1 == lo2 && hi1 == hi2, "same seed, same interval");
  ctx.check(lo1 < hi1, "non-degenerate data gives a non-empty interval");

  const std::vector<double> flat(150, 3.0);
  double lo, hi;
  bootstrap_median_ci(flat, 500, 1, &lo, &hi);
  ctx.check(lo == 3.0 && hi == 3.0, "constant data gives a zero-width interval");

  const Summary empty = summarize({});
  ctx.check(!empty.valid, "no samples is not a valid summary");
}

CCML_TEST(bench_stats, iteration_count_never_drops_below_spec_floor) {
  TimingConfig cfg;  /* min 100, max 200, budget 3000 ms */
  ctx.check(choose_iterations(0.001, cfg) == 200, "fast kernel: capped at max");
  ctx.check(choose_iterations(20.0, cfg) == 150, "3000/20 = 150");
  ctx.check(choose_iterations(500.0, cfg) == 100, "slow kernel: floored at 100");
  ctx.check(choose_iterations(0.0, cfg) == 200, "unknown estimate: max");
}

/* --------------------------------------------------------------- shapes -- */

CCML_TEST(bench_stats, shape_reader_parses_sections_and_skips_unknowns) {
  const std::string json = R"({
    "_source": "test", "extra": {"a": [1, 2, {"b": "c"}], "t": true},
    "matmul": [ {"tag": "x.m1", "M": 1, "N": 2304, "K": 768},
                {"tag": "odd", "M": 7, "N": 3, "K": 5, "note": "ignored"} ],
    "softmax": [ {"tag": "s", "rows": 12, "cols": 128} ]
  })";
  const auto mm = parse_shapes(json, "matmul");
  ctx.check(mm.size() == 2, "two matmul shapes");
  ctx.check(mm[0].tag == "x.m1" && mm[0].M == 1 && mm[0].N == 2304 && mm[0].K == 768,
            "first entry");
  ctx.check(mm[1].M == 7 && mm[1].rows == 0, "unset fields are zero");
  const auto sm = parse_shapes(json, "softmax");
  ctx.check(sm.size() == 1 && sm[0].rows == 12 && sm[0].cols == 128, "softmax entry");

  bool threw = false;
  try {
    parse_shapes(json, "layernorm");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  ctx.check(threw, "a missing section is an error, not an empty list");

  threw = false;
  try {
    parse_shapes("{\"matmul\": [ {\"M\": 1,, } ]}", "matmul");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  ctx.check(threw, "malformed JSON is an error");
}

CCML_TEST(bench_stats, committed_shape_file_is_complete) {
#ifdef CCML_DEFAULT_SHAPES
  for (const char* op : {"matmul", "layernorm", "softmax"}) {
    const auto shapes = load_shapes(CCML_DEFAULT_SHAPES, op);
    ctx.check(!shapes.empty(), std::string(op) + " has shapes");
    bool has_odd = false;
    for (const auto& s : shapes) {
      ctx.check(!s.tag.empty(), "every shape has a tag");
      if (s.tag.rfind("odd.", 0) == 0) has_odd = true;
      if (std::string(op) == "matmul") {
        ctx.check(s.M > 0 && s.N > 0 && s.K > 0, "matmul dims positive: " + s.tag);
      } else {
        ctx.check(s.rows > 0 && s.cols > 0, "row-wise dims positive: " + s.tag);
      }
    }
    ctx.check(has_odd, std::string(op) + " includes odd shapes (reference.md §5)");
  }
#else
  ctx.skip("CCML_DEFAULT_SHAPES not defined");
#endif
}

/* -------------------------------------------------------------- metrics -- */

CCML_TEST(bench_stats, flop_and_byte_counts) {
  ctx.check(near(matmul_flops(2, 3, 4), 48.0), "2*M*N*K");
  ctx.check(near(matmul_bytes(2, 3, 4, 4), (8 + 12 + 6) * 4.0), "A+B+C once each");
  ctx.check(near(layernorm_bytes(10, 768, 4, true), (2.0 * 7680 + 2 * 768) * 4),
            "x, y, gamma, beta");
  ctx.check(near(layernorm_bytes(10, 768, 2, false), 2.0 * 7680 * 2), "no affine, fp16");
  ctx.check(near(softmax_bytes(3, 5, 4), 120.0), "x and y");
}

CCML_TEST(bench_stats, theoretical_peaks_match_published_t4_figures) {
  ccml_device_info t4{};
  t4.compute_major = 7;
  t4.compute_minor = 5;
  t4.multiprocessor_count = 40;
  t4.clock_khz = 1590000;
  t4.memory_clock_khz = 5001000;
  t4.memory_bus_width = 256;
  const DevicePeaks p = theoretical_peaks(t4);
  /* T4 datasheet: 8.1 TFLOPS FP32, 320 GB/s. */
  ctx.note("fp32 " + std::to_string(p.fp32_tflops_theory) + " TFLOPS, bw " +
           std::to_string(p.bandwidth_gbps_theory) + " GB/s");
  ctx.check(std::fabs(p.fp32_tflops_theory - 8.1408) < 1e-3, "40*64*2*1.59 GHz");
  ctx.check(std::fabs(p.bandwidth_gbps_theory - 320.064) < 1e-3, "2*5.001 GHz*32 B");
  ctx.check(fp32_cores_per_sm(8, 9) == 128, "Ada: 128 lanes/SM");
  ctx.check(fp32_cores_per_sm(3, 0) == 0, "unknown arch: 0, not a guess");
}

CCML_TEST(bench_stats, throttle_classification) {
  ctx.check(!is_throttled(0x0), "no reasons");
  ctx.check(!is_throttled(0x1), "idle is not throttling");
  ctx.check(!is_throttled(0x4), "power cap is recorded but not flagged");
  ctx.check(is_throttled(0x20), "SW thermal slowdown");
  ctx.check(is_throttled(0x40), "HW thermal slowdown");
  ctx.check(is_throttled(0x08), "HW slowdown");
}

/* ------------------------------------------------------------------ CSV -- */

CCML_TEST(bench_stats, csv_never_reports_time_for_a_failed_gate) {
  RunContext rc;
  rc.gpu = "Test GPU";
  rc.cc = "7.5";
  rc.peaks.fp32_tflops_theory = 8.0;

  Row row;
  row.op = "matmul";
  row.impl = "ccml";
  row.algo = "naive";
  row.dtype = "f32";
  row.shape.tag = "t";
  row.shape.M = row.shape.N = row.shape.K = 64;
  row.flops = matmul_flops(64, 64, 64);
  row.bytes = matmul_bytes(64, 64, 64, 4);
  row.timing.valid = true;
  row.timing.summary.valid = true;
  row.timing.summary.n = 200;
  row.timing.summary.median = 10.0;

  row.gate = GateResult::Fail;
  const std::string failed = csv_row(row, rc);
  ctx.check(failed.find(",FAIL,") != std::string::npos, "gate column says FAIL");
  ctx.check(failed.find(",10,") == std::string::npos, "no median for a FAIL row");

  row.gate = GateResult::Pass;
  const std::string passed = csv_row(row, rc);
  ctx.note(passed);
  ctx.check(passed.find(",PASS,") != std::string::npos, "gate column says PASS");
  ctx.check(passed.find(",200,10,") != std::string::npos, "iters and median present");

  int header_cols = 1, row_cols = 1;
  for (const char* c = csv_header(); *c; ++c) header_cols += *c == ',';
  for (char c : passed) row_cols += c == ',';
  ctx.check(header_cols == row_cols, "row has as many columns as the header");
}
