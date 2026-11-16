#include "bench/support/driver.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ccml_bench {

namespace {

[[noreturn]] void usage(const char* why) {
  std::fprintf(stderr,
               "error: %s\n"
               "usage: [--shapes PATH] [--out PATH] [--peaks PATH] "
               "[--cache warm|cold|both] [--filter TEXT]\n"
               "       [--warmup N] [--max-iters N] [--min-iters N] "
               "[--budget-ms X] [--rounds N]\n"
               "       [--impls a,b] [--naive-max-gflop X]\n",
               why);
  std::exit(2);
}

}  // namespace

Options parse_options(int argc, char** argv) {
  Options o;
#ifdef CCML_DEFAULT_SHAPES
  o.shapes = CCML_DEFAULT_SHAPES;
#endif
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + arg).c_str());
      return argv[++i];
    };
    if (arg == "--shapes") o.shapes = value();
    else if (arg == "--out") o.out = value();
    else if (arg == "--peaks") o.peaks = value();
    else if (arg == "--filter") o.filter = value();
    else if (arg == "--cache") {
      const std::string v = value();
      o.warm = v == "warm" || v == "both";
      o.cold = v == "cold" || v == "both";
      if (!o.warm && !o.cold) usage("--cache must be warm, cold or both");
    } else if (arg == "--warmup") o.timing.warmup = std::atoi(value().c_str());
    else if (arg == "--max-iters") o.timing.max_iters = std::atoi(value().c_str());
    else if (arg == "--min-iters") o.timing.min_iters = std::atoi(value().c_str());
    else if (arg == "--budget-ms") o.timing.budget_ms = std::atof(value().c_str());
    else if (arg == "--rounds") o.timing.rounds = std::atoi(value().c_str());
    else if (arg == "--impls") o.impls = value();
    else if (arg == "--naive-max-gflop") o.naive_max_gflop = std::atof(value().c_str());
    else usage(("unknown argument " + arg).c_str());
  }
  if (o.timing.min_iters < 100) {
    usage("--min-iters below 100 violates the measurement method (reference.md)");
  }
  if (o.timing.max_iters < o.timing.min_iters) o.timing.max_iters = o.timing.min_iters;
  if (o.shapes.empty()) usage("--shapes is required");
  return o;
}

bool Options::wants(const std::string& impl) const {
  if (impls.empty() || impls == "all") return true;
  return ("," + impls + ",").find("," + impl + ",") != std::string::npos;
}

Gated gate(const Candidate& candidate, const float* output, std::size_t count,
           const std::vector<double>& reference,
           const ccml_test::Tolerance& tolerance, cudaStream_t stream) {
  Gated g;
  g.candidate = candidate;
  if (!candidate.launch() || cudaGetLastError() != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    g.gate = GateResult::NotRun;
    return g;
  }
  std::vector<float> got(count);
  if (cudaMemcpy(got.data(), output, count * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    g.gate = GateResult::NotRun;
    return g;
  }
  const ccml_test::ErrorStats stats = ccml_test::compare(got, reference, tolerance);
  g.max_abs_err = stats.max_abs_error;
  g.max_rel_err = stats.max_rel_error;
  g.gate = ccml_test::within(stats) ? GateResult::Pass : GateResult::Fail;
  return g;
}

bool fits_in_memory(double bytes) {
  std::size_t free_bytes = 0, total = 0;
  if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess) return false;
  return bytes < 0.9 * static_cast<double>(free_bytes);
}

bool tag_matches(const ShapeEntry& shape, const std::string& filter) {
  return filter.empty() || shape.tag.find(filter) != std::string::npos;
}

Emitter::Emitter(const Options& options, const std::string& op)
    : options_(options), op_(op) {
  ctx_ = collect_context(telemetry_, options_.peaks);
  flusher_ = new L2Flusher(ctx_.l2_bytes);
  if (options_.out.empty()) {
    out_ = stdout;
  } else {
    out_ = std::fopen(options_.out.c_str(), "w");
    if (out_ == nullptr) throw std::runtime_error("cannot write " + options_.out);
    owns_out_ = true;
  }
  print_context(out_, ctx_, telemetry_);
  std::fprintf(out_, "%s\n", csv_header());
  std::fflush(out_);
}

Emitter::~Emitter() {
  delete flusher_;
  if (owns_out_) std::fclose(out_);
}

void Emitter::emit(const std::string& dtype, const ShapeEntry& shape, double flops,
                   double bytes, const std::vector<Gated>& gated,
                   cudaStream_t stream) {
  std::vector<Candidate> passing;
  for (const Gated& g : gated) {
    if (g.gate == GateResult::Pass) passing.push_back(g.candidate);
  }

  for (int mode = 0; mode < 2; ++mode) {
    const bool cold = mode == 1;
    if ((cold && !options_.cold) || (!cold && !options_.warm)) continue;
    /* A working set over 4x L2 cannot be served from L2 anyway, so a cold run
       would only repeat the warm one at twice the cost. Cold rows are measured
       where the distinction can matter. */
    if (cold && options_.warm && bytes > 4.0 * static_cast<double>(ctx_.l2_bytes)) continue;

    TimingConfig cfg = options_.timing;
    cfg.cold_l2 = cold;
    const std::vector<Measured> measured =
        passing.empty() ? std::vector<Measured>{}
                        : time_interleaved(passing, cfg, stream, flusher_);
    const Telemetry t = telemetry_.sample();

    std::size_t k = 0;
    for (const Gated& g : gated) {
      Row row;
      row.op = op_;
      row.impl = g.candidate.impl;
      row.algo = g.candidate.algo;
      row.dtype = dtype;
      row.shape = shape;
      row.cold = cold;
      row.gate = g.gate;
      row.max_abs_err = g.max_abs_err;
      row.max_rel_err = g.max_rel_err;
      row.flops = flops;
      row.bytes = bytes;
      row.telemetry = t;
      if (g.gate == GateResult::Pass) row.timing = measured[k++];
      std::fprintf(out_, "%s\n", csv_row(row, ctx_).c_str());
      if (row.gate == GateResult::Pass && !row.timing.valid) {
        std::fprintf(stderr, "  timing invalid for %s/%s %s: %s\n", row.impl.c_str(),
                     row.algo.c_str(), shape.tag.c_str(), row.timing.error.c_str());
      }
    }
    std::fflush(out_);
  }
  std::fprintf(stderr, "done %s %s\n", op_.c_str(), shape.tag.c_str());
}

void Emitter::skip(const std::string& dtype, const ShapeEntry& shape,
                   const std::string& why) {
  Row row;
  row.op = op_;
  row.impl = "ccml";
  row.algo = "-";
  row.dtype = dtype;
  row.shape = shape;
  row.gate = GateResult::NotRun;
  std::fprintf(out_, "%s\n", csv_row(row, ctx_).c_str());
  std::fprintf(stderr, "skipped %s %s: %s\n", op_.c_str(), shape.tag.c_str(),
               why.c_str());
  std::fflush(out_);
}

}  // namespace ccml_bench
