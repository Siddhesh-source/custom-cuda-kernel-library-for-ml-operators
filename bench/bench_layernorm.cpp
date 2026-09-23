#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bench/support/report.h"
#include "bench/support/timer.h"
#include "ccml/layernorm.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "support/device_buffer.h"
#include "support/random.h"

using ccml_support::DeviceBuffer;
using ccml_support::Rng;

namespace {

struct Shape {
  int rows;
  int cols;
};

constexpr int   kWarmup  = 10;
constexpr int   kRepeats = 50;
constexpr float kEps     = 1e-5f;

/* Column counts chosen to span narrow, warp-sized and wide rows, since that is
   what decides the reduction strategy. Placeholder until target shapes land. */
const Shape kShapes[] = {
    {4096, 256},
    {4096, 768},
    {4096, 4096},
};

void bench_one(const Shape& shape) {
  const std::size_t elements = static_cast<std::size_t>(shape.rows) * shape.cols;

  std::vector<float> host_x(elements);
  std::vector<float> host_gamma(static_cast<std::size_t>(shape.cols));
  std::vector<float> host_beta(static_cast<std::size_t>(shape.cols));

  Rng rng(1234);
  rng.normal(host_x.data(), host_x.size(), 0.0f, 1.0f);
  rng.uniform(host_gamma.data(), host_gamma.size(), 0.5f, 1.5f);
  rng.uniform(host_beta.data(), host_beta.size(), -1.0f, 1.0f);

  DeviceBuffer x(elements);
  DeviceBuffer y(elements);
  DeviceBuffer gamma(host_gamma.size());
  DeviceBuffer beta(host_beta.size());
  x.upload(host_x.data());
  gamma.upload(host_gamma.data());
  beta.upload(host_beta.data());

  const auto launch = [&]() -> bool {
    return ccml_layernorm_f32(y.data(), x.data(), gamma.data(), beta.data(),
                              shape.rows, shape.cols, kEps,
                              CCML_LAYERNORM_TWO_PASS,
                              /*stream=*/nullptr) == CCML_SUCCESS;
  };

  ccml_bench::Record record;
  record.op    = "layernorm";
  record.algo  = ccml_layernorm_algo_name(CCML_LAYERNORM_TWO_PASS);
  record.shape = "rows=" + std::to_string(shape.rows) +
                 ",cols=" + std::to_string(shape.cols);
  record.timing =
      ccml_bench::time_launch(launch, kWarmup, kRepeats, /*stream=*/nullptr);

  ccml_bench::print_csv_row(record);
}

}  // namespace

int main() {
  ccml_bench::print_run_header();
  ccml_bench::print_csv_header();

  for (const Shape& shape : kShapes) {
    bench_one(shape);
  }
  return 0;
}
