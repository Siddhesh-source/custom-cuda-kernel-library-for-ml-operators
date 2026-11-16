#include "bench/support/report.h"

#include <cstdlib>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ccml/device.h"
#include "ccml/status.h"

namespace ccml_bench {

RunContext collect_context(const TelemetrySource& telemetry,
                           const std::string& peaks_path) {
  ccml_device_info info;
  if (ccml_get_device_info(0, &info) != CCML_SUCCESS) {
    throw std::runtime_error(std::string("no CUDA device: ") +
                             ccml_last_error_string());
  }
  RunContext ctx;
  ctx.gpu = info.name;
  ctx.cc  = std::to_string(info.compute_major) + "." + std::to_string(info.compute_minor);
  ctx.driver = telemetry.driver_version();
  int runtime = 0;
  cudaRuntimeGetVersion(&runtime);
  ctx.cuda = std::to_string(runtime / 1000) + "." + std::to_string((runtime % 1000) / 10);
  const char* commit = std::getenv("CCML_COMMIT");
  ctx.commit = commit != nullptr && *commit != '\0' ? commit : "unknown";
  ctx.peaks = theoretical_peaks(info, telemetry.max_sm_clock_mhz());
  ctx.l2_bytes = static_cast<std::size_t>(info.l2_cache_size);
  if (!peaks_path.empty()) {
    load_measured_peaks(peaks_path, &ctx.peaks);
  }
  return ctx;
}

const char* gate_name(GateResult gate) {
  switch (gate) {
    case GateResult::Pass: return "PASS";
    case GateResult::Fail: return "FAIL";
    default: return "NOT_RUN";
  }
}

const char* csv_header() {
  return "op,impl,algo,dtype,tag,M,N,K,rows,cols,cache,iters,median_us,p10_us,"
         "p90_us,iqr_us,min_us,ci95_lo_us,ci95_hi_us,batched_mean_us,tflops,gbps,"
         "pct_peak_theory,pct_peak_measured,max_abs_err,max_rel_err,gate,"
         "sm_clock_mhz,mem_clock_mhz,temp_c,clock_reasons,throttled,gpu,cc,"
         "driver,cuda,commit";
}

namespace {

std::string num(double v, int precision = 6) {
  std::ostringstream os;
  os.precision(precision);
  os << v;
  return os.str();
}

std::string opt(double v, bool present, int precision = 6) {
  return present ? num(v, precision) : std::string();
}

}  // namespace

std::string csv_row(const Row& row, const RunContext& ctx) {
  /* reference.md §11: never report speed for a kernel that fails its tests.
     A row whose gate did not PASS carries no timing at all. */
  const bool timed = row.gate == GateResult::Pass && row.timing.valid;
  const Summary& s = row.timing.summary;
  const double seconds = s.median * 1e-6;

  const bool is_matmul = row.flops > 0.0;
  const double tflops = timed && is_matmul ? row.flops / seconds / 1e12 : 0.0;
  const double gbps = timed ? row.bytes / seconds / 1e9 : 0.0;
  const double achieved = is_matmul ? tflops : gbps;
  const double theory = is_matmul ? ctx.peaks.fp32_tflops_theory
                                  : ctx.peaks.bandwidth_gbps_theory;
  const double measured = is_matmul ? ctx.peaks.fp32_tflops_measured
                                    : ctx.peaks.bandwidth_gbps_measured;
  /* Only FP32 has a CUDA-core peak to compare against; the Tensor Core peaks
     for FP16/BF16 are added with the Tensor Core kernels. */
  const bool peak_applies = timed && (!is_matmul || row.dtype == "f32");

  std::ostringstream os;
  os << row.op << ',' << row.impl << ',' << row.algo << ',' << row.dtype << ','
     << row.shape.tag << ',' << row.shape.M << ',' << row.shape.N << ','
     << row.shape.K << ',' << row.shape.rows << ',' << row.shape.cols << ','
     << (row.cold ? "cold" : "warm") << ','
     << (timed ? std::to_string(s.n) : std::string()) << ','
     << opt(s.median, timed) << ',' << opt(s.p10, timed) << ','
     << opt(s.p90, timed) << ',' << opt(s.iqr, timed) << ','
     << opt(s.min, timed) << ',' << opt(s.ci_lo, timed) << ','
     << opt(s.ci_hi, timed) << ','
     << opt(row.timing.batched_mean_us, timed && !row.cold) << ','
     << opt(tflops, timed && is_matmul, 5) << ',' << opt(gbps, timed, 5) << ','
     << opt(100.0 * achieved / theory, peak_applies && theory > 0.0, 4) << ','
     << opt(100.0 * achieved / measured, peak_applies && measured > 0.0, 4) << ','
     << opt(row.max_abs_err, row.max_abs_err >= 0.0, 4) << ','
     << opt(row.max_rel_err, row.max_rel_err >= 0.0, 4) << ','
     << gate_name(row.gate) << ','
     << (row.telemetry.available ? std::to_string(row.telemetry.sm_clock_mhz) : "") << ','
     << (row.telemetry.available ? std::to_string(row.telemetry.mem_clock_mhz) : "") << ','
     << (row.telemetry.available ? std::to_string(row.telemetry.temp_c) : "") << ','
     << (row.telemetry.available ? "0x" + [&] {
          std::ostringstream h;
          h << std::hex << row.telemetry.reasons;
          return h.str();
        }() : std::string()) << ','
     << (row.telemetry.available ? (row.telemetry.throttled ? "1" : "0") : "") << ','
     << '"' << ctx.gpu << '"' << ',' << ctx.cc << ',' << ctx.driver << ','
     << ctx.cuda << ',' << ctx.commit;
  return os.str();
}

void print_context(std::FILE* out, const RunContext& ctx,
                   const TelemetrySource& telemetry) {
  std::fprintf(out, "# gpu: %s (cc %s)  driver: %s  cuda runtime: %s  commit: %s\n",
               ctx.gpu.c_str(), ctx.cc.c_str(), ctx.driver.c_str(),
               ctx.cuda.c_str(), ctx.commit.c_str());
  std::fprintf(out, "# peaks: fp32 %.3f TFLOPS theoretical, %.3f measured; "
               "bandwidth %.1f GB/s theoretical, %.1f measured (0 = not measured)\n",
               ctx.peaks.fp32_tflops_theory, ctx.peaks.fp32_tflops_measured,
               ctx.peaks.bandwidth_gbps_theory, ctx.peaks.bandwidth_gbps_measured);
  std::fprintf(out, "# telemetry: %s; other compute processes on GPU: %d\n",
               telemetry.available() ? "NVML" : "unavailable",
               telemetry.other_processes());
}

}  // namespace ccml_bench
