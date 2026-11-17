/* Fused scale + mask softmax vs separate kernels (docs/optimization-log.md
   entry 16; enhancements F3, F4; reference.md section 5 "fusion vs separate
   kernels"). Timing only; every candidate passes the fp64 gate first.

   Attention shapes only (tags containing ".attn."): rows = batch x heads x
   seq, cols = seq. Two masks per shape:
     causal   causal_q = cols (self-attention), no additive mask
     padding  an additive [1, cols] mask, -inf on the last 1/8 of the keys
   Neither fully masks a row, so the PyTorch rows (which give NaN there) are
   comparable.

   Candidates:
     ccml fused_warp_per_row / fused_online     ccml_softmax_masked_*
     ccml mask+warp_per_row / mask+online       scale_mask kernel, then ccml_softmax_*
   PyTorch eager and torch.compile: scripts/bench_torch.py, op softmax_masked.

   bytes = 2 x rows x cols (+ the mask row) x sizeof(T): the fused minimum. */

#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench/baselines/scale_mask.h"
#include "bench/support/driver.h"
#include "bench/support/metrics.h"
#include "bench/support/typed.h"
#include "ccml/softmax.h"
#include "support/compare.h"
#include "support/random.h"

using namespace ccml_bench;

namespace {

constexpr float kScale = 0.125f;

ccml_status plain(float* y, const float* x, int r, int c, ccml_softmax_algo a, cudaStream_t s) {
  return ccml_softmax_f32(y, x, r, c, a, s);
}
ccml_status plain(ccml_f16* y, const ccml_f16* x, int r, int c, ccml_softmax_algo a,
                  cudaStream_t s) {
  return ccml_softmax_f16(y, x, r, c, a, s);
}
ccml_status plain(ccml_bf16* y, const ccml_bf16* x, int r, int c, ccml_softmax_algo a,
                  cudaStream_t s) {
  return ccml_softmax_bf16(y, x, r, c, a, s);
}
ccml_status fused(float* y, const float* x, const float* m, int mr, int r, int c, int cq,
                  ccml_softmax_algo a, cudaStream_t s) {
  return ccml_softmax_masked_f32(y, x, m, mr, r, c, kScale, cq, a, s);
}
ccml_status fused(ccml_f16* y, const ccml_f16* x, const ccml_f16* m, int mr, int r, int c,
                  int cq, ccml_softmax_algo a, cudaStream_t s) {
  return ccml_softmax_masked_f16(y, x, m, mr, r, c, kScale, cq, a, s);
}
ccml_status fused(ccml_bf16* y, const ccml_bf16* x, const ccml_bf16* m, int mr, int r, int c,
                  int cq, ccml_softmax_algo a, cudaStream_t s) {
  return ccml_softmax_masked_bf16(y, x, m, mr, r, c, kScale, cq, a, s);
}

template <typename D>
void run_shape(const Options& options, Emitter& emitter, const ShapeEntry& base, bool causal,
               cudaStream_t stream) {
  using Pub = typename D::Pub;
  ShapeEntry shape = base;
  shape.tag += causal ? ".causal" : ".padding";
  const int rows = static_cast<int>(shape.rows), cols = static_cast<int>(shape.cols);
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  if (!fits_in_memory(3.0 * sizeof(Pub) * static_cast<double>(n))) {
    emitter.skip(D::name, shape, "insufficient device memory");
    return;
  }
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> fx(n);
  ccml_support::Rng(1234).normal(fx.data(), n, 0.0f, 3.0f);
  std::vector<Pub> hx(n), hm(static_cast<std::size_t>(cols));
  for (std::size_t i = 0; i < n; ++i) {
    hx[i] = D::from_float(fx[i]);
    fx[i] = D::to_float(hx[i]);
  }
  std::vector<float> fm(static_cast<std::size_t>(cols), 0.0f);
  for (int c = cols - cols / 8; c < cols; ++c) fm[static_cast<std::size_t>(c)] = -inf;
  for (int c = 0; c < cols; ++c) hm[static_cast<std::size_t>(c)] = D::from_float(fm[static_cast<std::size_t>(c)]);
  const int causal_q = causal ? cols : 0;
  const int mask_rows = causal ? 0 : 1;

  std::vector<double> ref(n);
  for (int r = 0; r < rows; ++r) {
    const int limit = causal ? r % causal_q : cols;
    double m = -std::numeric_limits<double>::infinity(), s = 0.0;
    std::vector<double> v(static_cast<std::size_t>(cols));
    for (int c = 0; c < cols; ++c) {
      double t = static_cast<double>(fx[static_cast<std::size_t>(r) * cols + c]) * kScale;
      if (!causal) t += fm[static_cast<std::size_t>(c)];
      if (c > limit) t = -std::numeric_limits<double>::infinity();
      v[static_cast<std::size_t>(c)] = t;
      m = std::fmax(m, t);
    }
    for (double& t : v) {
      t = std::isinf(t) ? 0.0 : std::exp(t - m);
      s += t;
    }
    for (int c = 0; c < cols; ++c) ref[static_cast<std::size_t>(r) * cols + c] = v[static_cast<std::size_t>(c)] / s;
  }
  const ccml_test::Tolerance tol = D::widen(ccml_test::tolerance_for_softmax_f32(cols));

  DeviceArray<Pub> x(n), y(n), tmp(n), mask(static_cast<std::size_t>(cols));
  x.upload(hx);
  mask.upload(hm);
  Pub* const py = y.data();
  Pub* const pt = tmp.data();
  const Pub* const px = x.data();
  const Pub* const pm = causal ? nullptr : mask.data();

  std::vector<Candidate> cands;
  if (options.wants("ccml")) {
    if (cols <= 1024) {
      cands.push_back({"ccml", "fused_warp_per_row", [=] {
                         return fused(py, px, pm, mask_rows, rows, cols, causal_q,
                                      CCML_SOFTMAX_WARP_PER_ROW, stream) == CCML_SUCCESS;
                       }});
    }
    cands.push_back({"ccml", "fused_online", [=] {
                       return fused(py, px, pm, mask_rows, rows, cols, causal_q,
                                    CCML_SOFTMAX_ONLINE, stream) == CCML_SUCCESS;
                     }});
    if (cols <= 1024) {
      cands.push_back({"ccml", "mask+warp_per_row", [=] {
                         return scale_mask(pt, px, pm, mask_rows, rows, cols, kScale, causal_q,
                                           D::blas, stream) &&
                                plain(py, pt, rows, cols, CCML_SOFTMAX_WARP_PER_ROW, stream) ==
                                    CCML_SUCCESS;
                       }});
    }
    cands.push_back({"ccml", "mask+online", [=] {
                       return scale_mask(pt, px, pm, mask_rows, rows, cols, kScale, causal_q,
                                         D::blas, stream) &&
                              plain(py, pt, rows, cols, CCML_SOFTMAX_ONLINE, stream) ==
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
  /* The separate pipelines round v = x * scale + mask to T before the softmax:
     one extra rounding of the logits, unit * |v|, which moves each output by
     at most a relative unit * (|v_i| + max |v|). Allowed for the
     "mask+" candidates only, as for the unfused matmul epilogue. */
  /* y_i changes by a factor exp(v_i d_i - sum_j y_j v_j d_j), |d| <= unit, so
     the relative change is at most unit * (|v_i| + max_j |v_j|). */
  std::vector<double> extra(n);
  for (int r = 0; r < rows; ++r) {
    double vmax = 0.0;
    for (int c = 0; c < cols; ++c) {
      vmax = std::fmax(vmax, std::fabs(static_cast<double>(fx[static_cast<std::size_t>(r) * cols + c]) * kScale));
    }
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(r) * cols + c;
      const double v = std::fabs(static_cast<double>(fx[i]) * kScale);
      extra[i] = 1.2 * D::unit_roundoff * (v + vmax) * ref[i];
    }
  }
  std::vector<Gated> gated;
  for (const Candidate& c : cands) {
    if (cudaMemset(y.data(), 0xFF, n * sizeof(Pub)) != cudaSuccess) {
      throw std::runtime_error("cudaMemset failed");
    }
    const bool unfused = c.algo.rfind("mask+", 0) == 0;
    gated.push_back(gate(c, fetch, ref, tol, stream, unfused ? extra : std::vector<double>()));
  }
  emitter.emit(D::name, shape, 0.0, (2.0 * n + (causal ? 0 : cols)) * sizeof(Pub), gated, stream);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Emitter emitter(options, "softmax_masked");
    cudaStream_t stream = nullptr;
    for (const ShapeEntry& shape : load_shapes(options.shapes, "softmax")) {
      if (!tag_matches(shape, options.filter) || shape.tag.find(".attn.") == std::string::npos) continue;
      for (const std::string& dtype : options.dtypes) {
        for (bool causal : {true, false}) {
          if (dtype == "f32") run_shape<F32>(options, emitter, shape, causal, stream);
          if (dtype == "f16") run_shape<F16>(options, emitter, shape, causal, stream);
          if (dtype == "bf16") run_shape<BF16>(options, emitter, shape, causal, stream);
        }
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ccml_bench_softmax_masked: %s\n", e.what());
    return 1;
  }
  return 0;
}
