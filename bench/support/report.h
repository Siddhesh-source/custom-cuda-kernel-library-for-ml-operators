#ifndef CCML_BENCH_SUPPORT_REPORT_H
#define CCML_BENCH_SUPPORT_REPORT_H

#include <cstdio>
#include <string>

#include "bench/support/metrics.h"
#include "bench/support/shapes.h"
#include "bench/support/telemetry.h"
#include "bench/support/timer.h"

namespace ccml_bench {

/* Conditions shared by every row of one run (reference.md §11: same GPU,
   dtype, clocks; every result traceable to the machine that produced it). */
struct RunContext {
  std::string gpu;
  std::string cc;        /* e.g. "7.5" */
  std::string driver;
  std::string cuda;      /* runtime version, e.g. "13.0" */
  std::string commit;    /* $CCML_COMMIT, or "unknown" */
  DevicePeaks peaks;
  std::size_t l2_bytes = 0;
};

/* Throws if no device is available. */
RunContext collect_context(const TelemetrySource& telemetry,
                           const std::string& peaks_path);

enum class GateResult { Pass, Fail, NotRun };
const char* gate_name(GateResult gate);

struct Row {
  std::string op, impl, algo, dtype;
  ShapeEntry  shape;
  bool        cold = false;
  GateResult  gate = GateResult::NotRun;
  double      max_abs_err = -1.0;   /* < 0: not computed */
  double      max_rel_err = -1.0;
  double      flops = 0.0;          /* 0 for row-wise ops */
  double      bytes = 0.0;
  Measured    timing;               /* ignored unless gate == Pass */
  Telemetry   telemetry;
};

/* Column list. Documented in docs/benchmarking.md; keep the two in sync. */
const char* csv_header();
std::string csv_row(const Row& row, const RunContext& ctx);

/* Prints the run conditions as '#' comment lines. */
void print_context(std::FILE* out, const RunContext& ctx,
                   const TelemetrySource& telemetry);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_REPORT_H */
