#ifndef CCML_BENCH_SUPPORT_DRIVER_H
#define CCML_BENCH_SUPPORT_DRIVER_H

#include <cstdio>
#include <string>
#include <vector>

#include "bench/support/report.h"
#include "bench/support/shapes.h"
#include "bench/support/telemetry.h"
#include "bench/support/timer.h"
#include "support/compare.h"

namespace ccml_bench {

/* Command line shared by every bench executable:
 *   --shapes PATH   default: bench/shapes.json next to the source tree
 *   --out PATH      CSV destination (default: stdout)
 *   --peaks PATH    peaks.json from ccml_bench_peaks (enables %peak measured)
 *   --cache warm|cold|both     (default both)
 *   --filter TEXT   only shapes whose tag contains TEXT
 *   --warmup N --max-iters N --min-iters N --budget-ms X --rounds N
 *   --impls a,b     only these implementations (ccml, cublas, cublaslt, cudnn)
 *   --naive-max-gflop X   matmul: skip the naive kernel above X GFLOP per call
 *                   (its rows are reported NOT_RUN)
 */
struct Options {
  std::string  shapes;
  std::string  out;
  std::string  peaks;
  std::string  filter;
  bool         warm = true;
  bool         cold = true;
  std::string  impls;              /* empty = all */
  double       naive_max_gflop = 0.0;  /* 0 = no limit */
  TimingConfig timing;

  bool wants(const std::string& impl) const;
};

Options parse_options(int argc, char** argv);

/* A candidate that has been run once and compared with the fp64 reference.
   Only PASS candidates are ever timed (reference.md §11). */
struct Gated {
  Candidate  candidate;
  GateResult gate = GateResult::NotRun;
  double     max_abs_err = -1.0;
  double     max_rel_err = -1.0;
};

/* Runs `candidate` once, downloads `count` floats from `output`, and compares
   them against `reference` with `tolerance`. */
Gated gate(const Candidate& candidate, const float* output, std::size_t count,
           const std::vector<double>& reference,
           const ccml_test::Tolerance& tolerance, cudaStream_t stream);

/* Writes all rows of one shape: every gated candidate, for each requested
   cache mode, timing only those that passed. */
class Emitter {
 public:
  Emitter(const Options& options, const std::string& op);
  ~Emitter();
  void emit(const std::string& dtype, const ShapeEntry& shape, double flops,
            double bytes, const std::vector<Gated>& gated, cudaStream_t stream);
  /* Row for a shape that could not be set up (e.g. not enough memory). */
  void skip(const std::string& dtype, const ShapeEntry& shape, const std::string& why);
  const RunContext& context() const { return ctx_; }

 private:
  Options         options_;
  std::string     op_;
  TelemetrySource telemetry_;
  RunContext      ctx_;
  L2Flusher*      flusher_ = nullptr;
  std::FILE*      out_ = nullptr;
  bool            owns_out_ = false;
};

/* True if `bytes` more device memory can be allocated with 10% headroom. */
bool fits_in_memory(double bytes);

bool tag_matches(const ShapeEntry& shape, const std::string& filter);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_DRIVER_H */
