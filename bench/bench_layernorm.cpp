/* Layernorm benchmark: every ccml variant on every shape in bench/shapes.json,
   for each requested storage type (--dtype f32,f16,bf16; reference.md section 5
   dtype comparison). The vendor baseline for layernorm is PyTorch eager
   (scripts/bench_torch.py): cuDNN has no legacy-API layernorm (see
   bench/baselines/cudnn_softmax.h). Variants not yet built report NOT_RUN,
   never a time.

   FP16/BF16: inputs are rounded to the storage type first, the fp64 reference
   uses the rounded values, and the tolerance is the FP32 one widened for the
   final rounding (as tests/test_dtypes.cpp). */

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench/support/driver.h"
#include "bench/support/metrics.h"
#include "bench/support/typed.h"
#include "ccml/layernorm.h"
#include "ccml/status.h"
#include "support/compare.h"
#include "support/random.h"
#include "support/reference.h"

using namespace ccml_bench;

namespace {

constexpr float kEps = 1e-5f;

/* Every variant the bench times. */
const ccml_layernorm_algo kAlgos[] = {CCML_LAYERNORM_TWO_PASS, CCML_LAYERNORM_WELFORD};

ccml_status run(float* y, const float* x, const float* g, const float* b, int rows, int cols,
                ccml_layernorm_algo algo, cudaStream_t s) {
  return ccml_layernorm_f32(y, x, g, b, rows, cols, kEps, algo, s);
}
ccml_status run(ccml_f16* y, const ccml_f16* x, const ccml_f16* g, const ccml_f16* b, int rows,
                int cols, ccml_layernorm_algo algo, cudaStream_t s) {
  return ccml_layernorm_f16(y, x, g, b, rows, cols, kEps, algo, s);
}
ccml_status run(ccml_bf16* y, const ccml_bf16* x, const ccml_bf16* g, const ccml_bf16* b,
                int rows, int cols, ccml_layernorm_algo algo, cudaStream_t s) {
  return ccml_layernorm_bf16(y, x, g, b, rows, cols, kEps, algo, s);
}

template <typename D>
void run_shape(const Options& options, Emitter& emitter, const ShapeEntry& shape,
               cudaStream_t stream) {
  using Pub = typename D::Pub;
  const int rows = static_cast<int>(shape.rows), cols = static_cast<int>(shape.cols);
  const std::size_t elements = static_cast<std::size_t>(rows) * cols;
  if (!fits_in_memory(static_cast<double>(sizeof(Pub)) *
                      (2.0 * static_cast<double>(elements) + 2.0 * cols))) {
    emitter.skip(D::name, shape, "insufficient device memory");
    return;
  }
  auto make = [](std::size_t n, std::vector<Pub>* stored, std::vector<float>* exact) {
    stored->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      (*stored)[i] = D::from_float((*exact)[i]);
      (*exact)[i] = D::to_float((*stored)[i]);
    }
  };
  std::vector<float> xs(elements), gs(static_cast<std::size_t>(cols)),
      bs(static_cast<std::size_t>(cols));
  ccml_support::Rng rng(1234);
  rng.normal(xs.data(), xs.size(), 0.0f, 1.0f);
  rng.uniform(gs.data(), gs.size(), 0.5f, 1.5f);
  rng.uniform(bs.data(), bs.size(), -1.0f, 1.0f);
  std::vector<Pub> hx, hg, hb;
  make(xs.size(), &hx, &xs);
  make(gs.size(), &hg, &gs);
  make(bs.size(), &hb, &bs);

  std::vector<double> reference;
  ccml_test::layernorm_reference(reference, xs, gs, bs, rows, cols, kEps);
  const ccml_test::Tolerance tolerance = D::widen(ccml_test::tolerance_for_f32_reduction(cols));

  DeviceArray<Pub> x(elements), y(elements), gamma(hg.size()), beta(hb.size());
  x.upload(hx);
  gamma.upload(hg);
  beta.upload(hb);
  Pub* const py = y.data();
  const Pub* const px = x.data();
  const Pub* const pg = gamma.data();
  const Pub* const pb = beta.data();

  const auto fetch = [&y](std::vector<float>& got) {
    std::vector<Pub> raw;
    if (!y.download(raw)) return false;
    got.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) got[i] = D::to_float(raw[i]);
    return true;
  };
  std::vector<Gated> gated;
  if (options.wants("ccml")) {
    for (ccml_layernorm_algo algo : kAlgos) {
      const Candidate candidate{"ccml", ccml_layernorm_algo_name(algo), [=] {
                                  return run(py, px, pg, pb, rows, cols, algo, stream) ==
                                         CCML_SUCCESS;
                                }};
      /* 0xFF bytes: NaN in all three types, so an unwritten element fails. */
      if (cudaMemset(y.data(), 0xFF, y.count() * sizeof(Pub)) != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
      }
      gated.push_back(gate(candidate, fetch, reference, tolerance, stream));
    }
  }
  emitter.emit(D::name, shape, 0.0,
               layernorm_bytes(rows, cols, static_cast<int>(sizeof(Pub)), true), gated, stream);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Emitter emitter(options, "layernorm");
    cudaStream_t stream = nullptr;
    for (const ShapeEntry& shape : load_shapes(options.shapes, "layernorm")) {
      if (!tag_matches(shape, options.filter)) continue;
      for (const std::string& dtype : options.dtypes) {
        if (dtype == "f32") run_shape<F32>(options, emitter, shape, stream);
        if (dtype == "f16") run_shape<F16>(options, emitter, shape, stream);
        if (dtype == "bf16") run_shape<BF16>(options, emitter, shape, stream);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ccml_bench_layernorm: %s\n", e.what());
    return 1;
  }
  return 0;
}
