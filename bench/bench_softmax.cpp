/* Softmax benchmark: every ccml variant against cuDNN (when the build found
   it) on every shape in bench/shapes.json. Variants not yet built report
   NOT_RUN, never a time. */

#include <cstdio>
#include <exception>
#include <vector>

#include "bench/baselines/cudnn_softmax.h"
#include "bench/support/driver.h"
#include "bench/support/metrics.h"
#include "ccml/softmax.h"
#include "ccml/status.h"
#include "support/compare.h"
#include "support/device_buffer.h"
#include "support/random.h"
#include "support/reference.h"

using namespace ccml_bench;
using ccml_support::DeviceBuffer;

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Emitter emitter(options, "softmax");
    cudaStream_t stream = nullptr;
    CudnnSoftmax cudnn(stream);

    for (const ShapeEntry& shape : load_shapes(options.shapes, "softmax")) {
      if (!tag_matches(shape, options.filter)) continue;
      const int rows = static_cast<int>(shape.rows), cols = static_cast<int>(shape.cols);
      const std::size_t elements = static_cast<std::size_t>(rows) * cols;
      if (!fits_in_memory(8.0 * static_cast<double>(elements))) {
        emitter.skip("f32", shape, "insufficient device memory");
        continue;
      }

      std::vector<float> host_x(elements);
      ccml_support::Rng rng(1234);
      rng.normal(host_x.data(), host_x.size(), 0.0f, 1.0f);

      std::vector<double> reference;
      ccml_test::softmax_reference(reference, host_x, rows, cols);
      const ccml_test::Tolerance tolerance = ccml_test::tolerance_for_softmax_f32(cols);

      DeviceBuffer x(elements), y(elements);
      x.upload(host_x.data());

      std::vector<Candidate> candidates;
      for (ccml_softmax_algo algo :
           {CCML_SOFTMAX_BLOCK_PER_ROW, CCML_SOFTMAX_WARP_PER_ROW}) {
        if (!options.wants("ccml")) break;
        candidates.push_back({"ccml", ccml_softmax_algo_name(algo), [&, algo] {
                                return ccml_softmax_f32(y.data(), x.data(), rows, cols,
                                                        algo, stream) == CCML_SUCCESS;
                              }});
      }
      if (CudnnSoftmax::available() && options.wants("cudnn")) {
        const bool ready = cudnn.prepare(rows, cols);
        candidates.push_back({"cudnn", "softmax_accurate_instance", [&, ready] {
                                return ready && cudnn.run(y.data(), x.data());
                              }});
      }

      std::vector<Gated> gated;
      for (const Candidate& candidate : candidates) {
        gated.push_back(gate(candidate, y.data(), y.count(), reference, tolerance, stream));
      }
      emitter.emit("f32", shape, 0.0, softmax_bytes(rows, cols, 4), gated, stream);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ccml_bench_softmax: %s\n", e.what());
    return 1;
  }
  return 0;
}
