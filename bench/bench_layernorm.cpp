/* Layernorm benchmark: every ccml variant on every shape in bench/shapes.json.
   The vendor baseline for layernorm is PyTorch eager (scripts/bench_torch.py):
   cuDNN has no legacy-API layernorm (see bench/baselines/cudnn_softmax.h).
   Variants not yet built report NOT_RUN, never a time. */

#include <cstdio>
#include <exception>
#include <vector>

#include "bench/support/driver.h"
#include "bench/support/metrics.h"
#include "ccml/layernorm.h"
#include "ccml/status.h"
#include "support/compare.h"
#include "support/device_buffer.h"
#include "support/random.h"
#include "support/reference.h"

using namespace ccml_bench;
using ccml_support::DeviceBuffer;

int main(int argc, char** argv) {
  constexpr float kEps = 1e-5f;
  try {
    const Options options = parse_options(argc, argv);
    Emitter emitter(options, "layernorm");
    cudaStream_t stream = nullptr;

    for (const ShapeEntry& shape : load_shapes(options.shapes, "layernorm")) {
      if (!tag_matches(shape, options.filter)) continue;
      const int rows = static_cast<int>(shape.rows), cols = static_cast<int>(shape.cols);
      const std::size_t elements = static_cast<std::size_t>(rows) * cols;
      if (!fits_in_memory(4.0 * (2.0 * static_cast<double>(elements) + 2.0 * cols))) {
        emitter.skip("f32", shape, "insufficient device memory");
        continue;
      }

      std::vector<float> host_x(elements), host_gamma(static_cast<std::size_t>(cols)),
          host_beta(static_cast<std::size_t>(cols));
      ccml_support::Rng rng(1234);
      rng.normal(host_x.data(), host_x.size(), 0.0f, 1.0f);
      rng.uniform(host_gamma.data(), host_gamma.size(), 0.5f, 1.5f);
      rng.uniform(host_beta.data(), host_beta.size(), -1.0f, 1.0f);

      std::vector<double> reference;
      ccml_test::layernorm_reference(reference, host_x, host_gamma, host_beta, rows,
                                     cols, kEps);
      const ccml_test::Tolerance tolerance = ccml_test::tolerance_for_f32_reduction(cols);

      DeviceBuffer x(elements), y(elements), gamma(host_gamma.size()),
          beta(host_beta.size());
      x.upload(host_x.data());
      gamma.upload(host_gamma.data());
      beta.upload(host_beta.data());

      std::vector<Gated> gated;
      for (ccml_layernorm_algo algo : {CCML_LAYERNORM_TWO_PASS, CCML_LAYERNORM_WELFORD}) {
        const Candidate candidate{"ccml", ccml_layernorm_algo_name(algo), [&, algo] {
                                    return ccml_layernorm_f32(y.data(), x.data(),
                                                              gamma.data(), beta.data(),
                                                              rows, cols, kEps, algo,
                                                              stream) == CCML_SUCCESS;
                                  }};
        gated.push_back(gate(candidate, y.data(), y.count(), reference, tolerance, stream));
      }
      emitter.emit("f32", shape, 0.0, layernorm_bytes(rows, cols, 4, true), gated, stream);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ccml_bench_layernorm: %s\n", e.what());
    return 1;
  }
  return 0;
}
