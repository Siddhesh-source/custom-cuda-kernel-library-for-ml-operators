#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bench/support/report.h"
#include "bench/support/timer.h"
#include "ccml/memory.h"
#include "ccml/softmax.h"
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

constexpr int kWarmup  = 10;
constexpr int kRepeats = 50;

const Shape kShapes[] = {
    {4096, 256},
    {4096, 1024},
    {4096, 4096},
};

void bench_one(const Shape& shape) {
  const std::size_t elements = static_cast<std::size_t>(shape.rows) * shape.cols;

  std::vector<float> host_x(elements);
  Rng rng(1234);
  rng.normal(host_x.data(), host_x.size(), 0.0f, 1.0f);

  DeviceBuffer x(elements);
  DeviceBuffer y(elements);
  x.upload(host_x.data());

  const auto launch = [&]() -> bool {
    return ccml_softmax_f32(y.data(), x.data(), shape.rows, shape.cols,
                            CCML_SOFTMAX_BLOCK_PER_ROW,
                            /*stream=*/nullptr) == CCML_SUCCESS;
  };

  ccml_bench::Record record;
  record.op    = "softmax";
  record.algo  = ccml_softmax_algo_name(CCML_SOFTMAX_BLOCK_PER_ROW);
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
