/* Fused residual + bias + layer norm vs separate kernels (docs/optimization-log.md
   entry 12; enhancements F2, F4; reference.md section 5 "fusion vs separate
   kernels"). Timing only; every candidate passes the fp64 gate first.

   Candidates (all write s and y):
     ccml fused_vectorized     ccml_layernorm_residual, VECTORIZED (cols <= 1024)
     ccml fused_welford        ccml_layernorm_residual, WELFORD
     ccml add+vectorized       residual_add kernel, then ccml_layernorm VECTORIZED on s
     ccml add+welford          residual_add kernel, then ccml_layernorm WELFORD on s
   PyTorch eager / torch.compile run the same op in scripts/bench_torch.py
   (op layernorm_residual).

   bytes = (x, r read; s, y written) 4 x rows x cols + (bias, gamma, beta) 3 x cols,
   times sizeof(T): the minimum a fused implementation moves. */

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench/baselines/residual_add.h"
#include "bench/support/driver.h"
#include "bench/support/metrics.h"
#include "bench/support/typed.h"
#include "ccml/layernorm.h"
#include "support/compare.h"
#include "support/random.h"
#include "support/reference.h"

using namespace ccml_bench;

namespace {

constexpr float kEps = 1e-5f;

ccml_status ln(float* y, const float* x, const float* g, const float* b, int rows, int cols,
               ccml_layernorm_algo a, cudaStream_t s) {
  return ccml_layernorm_f32(y, x, g, b, rows, cols, kEps, a, s);
}
ccml_status ln(ccml_f16* y, const ccml_f16* x, const ccml_f16* g, const ccml_f16* b, int rows,
               int cols, ccml_layernorm_algo a, cudaStream_t s) {
  return ccml_layernorm_f16(y, x, g, b, rows, cols, kEps, a, s);
}
ccml_status ln(ccml_bf16* y, const ccml_bf16* x, const ccml_bf16* g, const ccml_bf16* b,
               int rows, int cols, ccml_layernorm_algo a, cudaStream_t s) {
  return ccml_layernorm_bf16(y, x, g, b, rows, cols, kEps, a, s);
}
ccml_status fused(float* y, float* s, const float* x, const float* r, const float* b,
                  const float* g, const float* be, int rows, int cols, ccml_layernorm_algo a,
                  cudaStream_t st) {
  return ccml_layernorm_residual_f32(y, s, x, r, b, g, be, rows, cols, kEps, a, st);
}
ccml_status fused(ccml_f16* y, ccml_f16* s, const ccml_f16* x, const ccml_f16* r,
                  const ccml_f16* b, const ccml_f16* g, const ccml_f16* be, int rows, int cols,
                  ccml_layernorm_algo a, cudaStream_t st) {
  return ccml_layernorm_residual_f16(y, s, x, r, b, g, be, rows, cols, kEps, a, st);
}
ccml_status fused(ccml_bf16* y, ccml_bf16* s, const ccml_bf16* x, const ccml_bf16* r,
                  const ccml_bf16* b, const ccml_bf16* g, const ccml_bf16* be, int rows,
                  int cols, ccml_layernorm_algo a, cudaStream_t st) {
  return ccml_layernorm_residual_bf16(y, s, x, r, b, g, be, rows, cols, kEps, a, st);
}

template <typename D>
void run_shape(const Options& options, Emitter& emitter, const ShapeEntry& shape,
               cudaStream_t stream) {
  using Pub = typename D::Pub;
  const int rows = static_cast<int>(shape.rows), cols = static_cast<int>(shape.cols);
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  if (!fits_in_memory(static_cast<double>(sizeof(Pub)) * (4.0 * n + 3.0 * cols) + 8.0 * n)) {
    emitter.skip(D::name, shape, "insufficient device memory");
    return;
  }
  std::vector<float> fx(n), fr(n), fb(cols), fg(cols), fbe(cols);
  ccml_support::Rng rng(1234);
  rng.normal(fx.data(), n, 0.0f, 1.0f);
  rng.normal(fr.data(), n, 0.0f, 1.0f);
  rng.normal(fb.data(), fb.size(), 0.0f, 0.5f);
  rng.uniform(fg.data(), fg.size(), 0.5f, 1.5f);
  rng.uniform(fbe.data(), fbe.size(), -1.0f, 1.0f);
  auto store = [](std::vector<float>& f) {
    std::vector<Pub> p(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) {
      p[i] = D::from_float(f[i]);
      f[i] = D::to_float(p[i]);
    }
    return p;
  };
  const std::vector<Pub> hx = store(fx), hr = store(fr), hb = store(fb), hg = store(fg),
                         hbe = store(fbe);
  /* Reference: fp64 layer norm of the rounded s (the contract). */
  std::vector<float> s_exact(n);
  for (std::size_t i = 0; i < n; ++i) {
    s_exact[i] = D::to_float(D::from_float(fx[i] + fr[i] + fb[i % static_cast<std::size_t>(cols)]));
  }
  std::vector<double> reference;
  ccml_test::layernorm_reference(reference, s_exact, fg, fbe, rows, cols, kEps);
  const ccml_test::Tolerance tol = D::widen(ccml_test::tolerance_for_f32_reduction(cols));

  DeviceArray<Pub> x(n), r(n), b(hb.size()), g(hg.size()), be(hbe.size()), y(n), s(n);
  x.upload(hx);
  r.upload(hr);
  b.upload(hb);
  g.upload(hg);
  be.upload(hbe);
  Pub* const py = y.data();
  Pub* const ps = s.data();
  const Pub *const px = x.data(), *const pr = r.data(), *const pb = b.data(),
                   *const pg = g.data(), *const pbe = be.data();

  std::vector<Candidate> cands;
  if (options.wants("ccml")) {
    cands.push_back({"ccml", "fused_vectorized", [=] {
                       return fused(py, ps, px, pr, pb, pg, pbe, rows, cols,
                                    CCML_LAYERNORM_VECTORIZED, stream) == CCML_SUCCESS;
                     }});
    cands.push_back({"ccml", "fused_welford", [=] {
                       return fused(py, ps, px, pr, pb, pg, pbe, rows, cols,
                                    CCML_LAYERNORM_WELFORD, stream) == CCML_SUCCESS;
                     }});
    cands.push_back({"ccml", "add+vectorized", [=] {
                       return residual_add(ps, px, pr, pb, rows, cols, D::blas, stream) &&
                              ln(py, ps, pg, pbe, rows, cols, CCML_LAYERNORM_VECTORIZED,
                                 stream) == CCML_SUCCESS;
                     }});
    cands.push_back({"ccml", "add+welford", [=] {
                       return residual_add(ps, px, pr, pb, rows, cols, D::blas, stream) &&
                              ln(py, ps, pg, pbe, rows, cols, CCML_LAYERNORM_WELFORD, stream) ==
                                  CCML_SUCCESS;
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
  for (const Candidate& c : cands) {
    if (cudaMemset(y.data(), 0xFF, n * sizeof(Pub)) != cudaSuccess ||
        cudaMemset(s.data(), 0xFF, n * sizeof(Pub)) != cudaSuccess) {
      throw std::runtime_error("cudaMemset failed");
    }
    gated.push_back(gate(c, fetch, reference, tol, stream));
  }
  emitter.emit(D::name, shape, 0.0, (4.0 * n + 3.0 * cols) * sizeof(Pub), gated, stream);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Emitter emitter(options, "layernorm_residual");
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
    std::fprintf(stderr, "ccml_bench_layernorm_residual: %s\n", e.what());
    return 1;
  }
  return 0;
}
