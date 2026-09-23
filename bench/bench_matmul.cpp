#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bench/support/report.h"
#include "bench/support/timer.h"
#include "ccml/matmul.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "support/device_buffer.h"
#include "support/random.h"

using ccml_support::DeviceBuffer;
using ccml_support::Rng;

namespace {

/* Kept small on purpose: the development GPU has 4 GB total with roughly 2.9 GB
   free. Shapes here must be revisited once the sponsor's target shapes are
   known -- see docs/benchmarking.md. */
struct Shape {
  int M;
  int N;
  int K;
};

constexpr int kWarmup  = 10;
constexpr int kRepeats = 50;

const Shape kShapes[] = {
    {256, 256, 256},
    {512, 512, 512},
    {1024, 1024, 1024},
};

const ccml_matmul_algo kAlgos[] = {CCML_MATMUL_NAIVE, CCML_MATMUL_TILED};

void bench_one(const Shape& shape, ccml_matmul_algo algo) {
  std::vector<float> host_a(static_cast<std::size_t>(shape.M) * shape.K);
  std::vector<float> host_b(static_cast<std::size_t>(shape.K) * shape.N);

  Rng rng(1234);
  rng.normal(host_a.data(), host_a.size(), 0.0f, 1.0f);
  rng.normal(host_b.data(), host_b.size(), 0.0f, 1.0f);

  DeviceBuffer a(host_a.size());
  DeviceBuffer b(host_b.size());
  DeviceBuffer c(static_cast<std::size_t>(shape.M) * shape.N);
  a.upload(host_a.data());
  b.upload(host_b.data());

  const auto launch = [&]() -> bool {
    return ccml_matmul_f32(c.data(), a.data(), b.data(), shape.M, shape.N,
                           shape.K, algo, /*stream=*/nullptr) == CCML_SUCCESS;
  };

  ccml_bench::Record record;
  record.op    = "matmul";
  record.algo  = ccml_matmul_algo_name(algo);
  record.shape = "M=" + std::to_string(shape.M) + ",N=" + std::to_string(shape.N) +
                 ",K=" + std::to_string(shape.K);
  record.timing =
      ccml_bench::time_launch(launch, kWarmup, kRepeats, /*stream=*/nullptr);

  ccml_bench::print_csv_row(record);
}

}  // namespace

int main() {
  ccml_bench::print_run_header();
  ccml_bench::print_csv_header();

  for (const Shape& shape : kShapes) {
    for (ccml_matmul_algo algo : kAlgos) {
      bench_one(shape, algo);
    }
  }
  return 0;
}
