/* Matmul benchmark: every ccml variant against cuBLAS and cuBLASLt, on every
   shape in bench/shapes.json (reference.md §3, §4, §6). Timing only; the
   in-run correctness gate decides what may be timed (reference.md §11). */

#include <cstdio>
#include <exception>
#include <vector>

#include "bench/baselines/cublas.h"
#include "bench/support/driver.h"
#include "bench/support/metrics.h"
#include "ccml/matmul.h"
#include "ccml/status.h"
#include "support/compare.h"
#include "support/device_buffer.h"
#include "support/random.h"

using namespace ccml_bench;
using ccml_support::DeviceBuffer;

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Emitter emitter(options, "matmul");
    cudaStream_t stream = nullptr;
    CublasBaseline cublas(stream);

    for (const ShapeEntry& shape : load_shapes(options.shapes, "matmul")) {
      if (!tag_matches(shape, options.filter)) continue;
      const int M = static_cast<int>(shape.M), N = static_cast<int>(shape.N),
                K = static_cast<int>(shape.K);
      const double elems = static_cast<double>(shape.M * shape.K + shape.K * shape.N +
                                               shape.M * shape.N);
      /* fp32 operands + fp64 copies for the DGEMM gate. */
      if (!fits_in_memory(elems * (4.0 + 8.0))) {
        emitter.skip("f32", shape, "insufficient device memory");
        continue;
      }

      std::vector<float> host_a(static_cast<std::size_t>(M) * K);
      std::vector<float> host_b(static_cast<std::size_t>(K) * N);
      ccml_support::Rng rng(1234);
      rng.normal(host_a.data(), host_a.size(), 0.0f, 1.0f);
      rng.normal(host_b.data(), host_b.size(), 0.0f, 1.0f);

      std::vector<double> reference;
      if (!cublas.dgemm_reference(reference, host_a, host_b, M, N, K)) {
        emitter.skip("f32", shape, "fp64 reference failed");
        continue;
      }
      const ccml_test::Tolerance tolerance =
          ccml_test::tolerance_for_f32_dot_product(K, ccml_test::rms(reference));

      DeviceBuffer a(host_a.size()), b(host_b.size()),
          c(static_cast<std::size_t>(M) * N);
      a.upload(host_a.data());
      b.upload(host_b.data());

      std::vector<Candidate> candidates;
      const double gflop = matmul_flops(M, N, K) / 1e9;
      if (options.wants("ccml")) {
        for (ccml_matmul_algo algo : {CCML_MATMUL_NAIVE, CCML_MATMUL_TILED}) {
          /* Over the cap the naive kernel is reported NOT_RUN, never timed. */
          const bool skip = algo == CCML_MATMUL_NAIVE && options.naive_max_gflop > 0.0 &&
                            gflop > options.naive_max_gflop;
          candidates.push_back({"ccml", ccml_matmul_algo_name(algo), [&, algo, skip] {
                                  return !skip &&
                                         ccml_matmul_f32(c.data(), a.data(), b.data(), M,
                                                         N, K, algo, stream) == CCML_SUCCESS;
                                }});
        }
      }
      if (options.wants("cublas")) {
        candidates.push_back({"cublas", "sgemm", [&] {
                                return cublas.sgemm(c.data(), a.data(), b.data(), M, N, K);
                              }});
      }
      if (options.wants("cublaslt")) {
        const bool lt_ready = cublas.lt_prepare(M, N, K);
        candidates.push_back({"cublaslt", "heuristic_top1", [&, lt_ready] {
                                return lt_ready &&
                                       cublas.lt_matmul(c.data(), a.data(), b.data());
                              }});
      }

      std::vector<Gated> gated;
      for (const Candidate& candidate : candidates) {
        gated.push_back(gate(candidate, c.data(), c.count(), reference, tolerance, stream));
      }
      emitter.emit("f32", shape, matmul_flops(M, N, K), matmul_bytes(M, N, K, 4),
                   gated, stream);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ccml_bench_matmul: %s\n", e.what());
    return 1;
  }
  return 0;
}
