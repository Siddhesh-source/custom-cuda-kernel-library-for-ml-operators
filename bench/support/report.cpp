#include "bench/support/report.h"

#include <cstdio>

#include "ccml/device.h"
#include "ccml/status.h"

namespace ccml_bench {

void print_run_header() {
  std::printf("# ccml benchmark run\n");
  std::printf("# library version: %s\n", ccml_version_string());

  ccml_device_info info;
  if (ccml_get_device_info(0, &info) == CCML_SUCCESS) {
    std::printf("# device: %s (sm_%d%d, %d SMs)\n", info.name,
                info.compute_major, info.compute_minor,
                info.multiprocessor_count);
  } else {
    std::printf("# device: unavailable (%s)\n", ccml_last_error_string());
  }

  std::printf("# built: %s %s\n", __DATE__, __TIME__);
  std::printf("#\n");
  std::printf("# PROVISIONAL. These timings are of this library against "
              "itself.\n");
  std::printf("# No comparison against cuBLAS or cuDNN has been made, and no "
              "claim\n");
  std::printf("# of relative performance is supported by this output.\n");
}

void print_csv_header() {
  std::printf("op,algo,shape,samples,mean_ms,min_ms,max_ms,stddev_ms\n");
}

void print_csv_row(const Record& record) {
  if (!record.timing.valid) {
    std::printf("%s,%s,%s,0,,,,\n", record.op.c_str(), record.algo.c_str(),
                record.shape.c_str());
    return;
  }
  std::printf("%s,%s,%s,%d,%.6f,%.6f,%.6f,%.6f\n",
              record.op.c_str(), record.algo.c_str(), record.shape.c_str(),
              record.timing.samples, record.timing.mean_ms,
              record.timing.min_ms, record.timing.max_ms,
              record.timing.stddev_ms);
}

}  // namespace ccml_bench
