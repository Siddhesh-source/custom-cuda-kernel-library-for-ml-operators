/* Softmax benchmark: every ccml variant against cuDNN (when the build found
   it) on every shape in bench/shapes.json, for each requested storage type
   (--dtype f32,f16,bf16; reference.md section 5 dtype comparison). Variants
   not yet built report NOT_RUN, never a time.

   FP16/BF16: inputs are rounded to the storage type first, the fp64 reference
   uses the rounded values, and the tolerance is the FP32 one widened for the
   final rounding (as tests/test_dtypes.cpp). */

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench/baselines/cudnn_softmax.h"
#include "bench/support/driver.h"
#include "bench/support/metrics.h"
#include "bench/support/typed.h"
#include "ccml/softmax.h"
#include "ccml/status.h"
#include "support/compare.h"
#include "support/random.h"
#include "support/reference.h"

using namespace ccml_bench;

namespace {

/* Every variant the bench times. */
const ccml_softmax_algo kAlgos[] = {CCML_SOFTMAX_BLOCK_PER_ROW, CCML_SOFTMAX_ONLINE,
                                    CCML_SOFTMAX_WARP_PER_ROW, CCML_SOFTMAX_SPLIT_ROW,
                                    CCML_SOFTMAX_BLOCK_RESIDENT};

ccml_status run(float* y, const float* x, int r, int c, ccml_softmax_algo a, cudaStream_t s) {
  return ccml_softmax_f32(y, x, r, c, a, s);
}
ccml_status run(ccml_f16* y, const ccml_f16* x, int r, int c, ccml_softmax_algo a,
                cudaStream_t s) {
  return ccml_softmax_f16(y, x, r, c, a, s);
}
ccml_status run(ccml_bf16* y, const ccml_bf16* x, int r, int c, ccml_softmax_algo a,
                cudaStream_t s) {
  return ccml_softmax_bf16(y, x, r, c, a, s);
}

template <typename D>
void run_shape(const Options& options, Emitter& emitter, CudnnSoftmax& cudnn,
               const ShapeEntry& shape, cudaStream_t stream) {
  using Pub = typename D::Pub;
  const int rows = static_cast<int>(shape.rows), cols = static_cast<int>(shape.cols);
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  if (!fits_in_memory(2.0 * sizeof(Pub) * static_cast<double>(n))) {
    emitter.skip(D::name, shape, "insufficient device memory");
    return;
  }
  std::vector<float> xs(n);
  ccml_support::Rng rng(1234);
  rng.normal(xs.data(), xs.size(), 0.0f, 1.0f);
  std::vector<Pub> hx(n);
  for (std::size_t i = 0; i < n; ++i) {
    hx[i] = D::from_float(xs[i]);
    xs[i] = D::to_float(hx[i]);
  }
  std::vector<double> reference;
  ccml_test::softmax_reference(reference, xs, rows, cols);
  const ccml_test::Tolerance tolerance = D::widen(ccml_test::tolerance_for_softmax_f32(cols));

  DeviceArray<Pub> x(n), y(n);
  x.upload(hx);
  Pub* const py = y.data();
  const Pub* const px = x.data();

  std::vector<Candidate> candidates;
  if (options.wants("ccml")) {
    for (ccml_softmax_algo algo : kAlgos) {
      candidates.push_back({"ccml", ccml_softmax_algo_name(algo), [=] {
                              return run(py, px, rows, cols, algo, stream) == CCML_SUCCESS;
                            }});
    }
  }
  if (CudnnSoftmax::available() && options.wants("cudnn")) {
    const bool ready = cudnn.prepare(rows, cols, D::blas);
    candidates.push_back({"cudnn", "softmax_accurate_instance", [&cudnn, ready, py, px] {
                            return ready && cudnn.run(py, px);
                          }});
  }
  const auto fetch = [&y](std::vector<float>& got) {
    std::vector<Pub> raw;
    if (!y.download(raw)) return false;
    got.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) got[i] = D::to_float(raw[i]);
    return true;
  };
  std::vector<Gated> gated;
  for (const Candidate& candidate : candidates) {
    /* 0xFF bytes: NaN in all three types, so an unwritten element fails. */
    if (cudaMemset(y.data(), 0xFF, n * sizeof(Pub)) != cudaSuccess) {
      throw std::runtime_error("cudaMemset failed");
    }
    gated.push_back(gate(candidate, fetch, reference, tolerance, stream));
  }
  emitter.emit(D::name, shape, 0.0, softmax_bytes(rows, cols, static_cast<int>(sizeof(Pub))),
               gated, stream);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Emitter emitter(options, "softmax");
    cudaStream_t stream = nullptr;
    CudnnSoftmax cudnn(stream);
    for (const ShapeEntry& shape : load_shapes(options.shapes, "softmax")) {
      if (!tag_matches(shape, options.filter)) continue;
      for (const std::string& dtype : options.dtypes) {
        if (dtype == "f32") run_shape<F32>(options, emitter, cudnn, shape, stream);
        if (dtype == "f16") run_shape<F16>(options, emitter, cudnn, shape, stream);
        if (dtype == "bf16") run_shape<BF16>(options, emitter, cudnn, shape, stream);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ccml_bench_softmax: %s\n", e.what());
    return 1;
  }
  return 0;
}
