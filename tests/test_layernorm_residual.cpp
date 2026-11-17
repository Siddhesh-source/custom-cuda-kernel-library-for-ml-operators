/* Fused residual + bias + layer norm (docs/optimization-log.md entry 12;
   enhancement F2), FP32, FP16 and BF16.

   Contract: s = round_T(float(x) + float(residual) + float(bias)) and
   y = layernorm(s). So:
     - sum_out must equal that s BITWISE (the host computes it with the same
       FP32 additions in the same order and the same rounding);
     - y is compared with an fp64 layer norm of the rounded s, at the layer
       norm tolerance widened for the storage type.
   Outputs are NaN-poisoned before every call. */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ccml/layernorm.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "support/compare.h"
#include "support/random.h"
#include "support/reference.h"
#include "tests/support/harness.h"

namespace {

constexpr float kEps = 1e-5f;

struct F32 {
  using Pub = float;
  static constexpr const char* name = "f32";
  static Pub from_float(float f) { return f; }
  static float to_float(Pub p) { return p; }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) { return t; }
  static ccml_status run(Pub* y, Pub* s, const Pub* x, const Pub* r, const Pub* b, const Pub* g,
                         const Pub* be, int rows, int cols, ccml_layernorm_algo a) {
    return ccml_layernorm_residual_f32(y, s, x, r, b, g, be, rows, cols, kEps, a, nullptr);
  }
};

struct F16 {
  using Pub = ccml_f16;
  static constexpr const char* name = "f16";
  static Pub from_float(float f) {
    const __half h = __float2half_rn(f);
    Pub p;
    std::memcpy(&p, &h, sizeof(p));
    return p;
  }
  static float to_float(Pub p) {
    __half h;
    std::memcpy(&h, &p, sizeof(h));
    return __half2float(h);
  }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) {
    return ccml_test::widen_for_storage(t, ccml_test::kF16UnitRoundoff,
                                        ccml_test::kF16SubnormalStep);
  }
  static ccml_status run(Pub* y, Pub* s, const Pub* x, const Pub* r, const Pub* b, const Pub* g,
                         const Pub* be, int rows, int cols, ccml_layernorm_algo a) {
    return ccml_layernorm_residual_f16(y, s, x, r, b, g, be, rows, cols, kEps, a, nullptr);
  }
};

struct BF16 {
  using Pub = ccml_bf16;
  static constexpr const char* name = "bf16";
  static Pub from_float(float f) {
    const __nv_bfloat16 h = __float2bfloat16_rn(f);
    Pub p;
    std::memcpy(&p, &h, sizeof(p));
    return p;
  }
  static float to_float(Pub p) {
    __nv_bfloat16 h;
    std::memcpy(&h, &p, sizeof(h));
    return __bfloat162float(h);
  }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) {
    return ccml_test::widen_for_storage(t, ccml_test::kBF16UnitRoundoff, 1.1754943508222875e-38);
  }
  static ccml_status run(Pub* y, Pub* s, const Pub* x, const Pub* r, const Pub* b, const Pub* g,
                         const Pub* be, int rows, int cols, ccml_layernorm_algo a) {
    return ccml_layernorm_residual_bf16(y, s, x, r, b, g, be, rows, cols, kEps, a, nullptr);
  }
};

template <typename Pub>
struct Dev {
  explicit Dev(std::size_t n) : n_(n) {
    if (ccml_device_malloc(&p_, n * sizeof(Pub)) != CCML_SUCCESS) p_ = nullptr;
  }
  ~Dev() { ccml_device_free(p_); }
  Pub* get() { return static_cast<Pub*>(p_); }
  bool up(const std::vector<Pub>& h) {
    return ccml_memcpy_h2d(p_, h.data(), h.size() * sizeof(Pub)) == CCML_SUCCESS;
  }
  bool down(std::vector<Pub>& h) {
    h.resize(n_);
    return ccml_memcpy_d2h(h.data(), p_, n_ * sizeof(Pub)) == CCML_SUCCESS;
  }
  bool poison() {
    return ccml_device_memset_async(p_, 0xFF, n_ * sizeof(Pub), nullptr) == CCML_SUCCESS;
  }
  void* p_ = nullptr;
  std::size_t n_;
};

/* One case; `off` shifts every pointer that many elements into its buffer. */
template <typename D>
double check(const ccml_test::Context& ctx, int rows, int cols, bool bias, bool sum_out,
             ccml_layernorm_algo algo, float mean = 0.0f, int off = 0) {
  using Pub = typename D::Pub;
  const std::size_t n = static_cast<std::size_t>(rows) * cols, o = static_cast<std::size_t>(off);
  std::vector<float> fx(n), fr(n), fb(cols), fg(cols), fbe(cols);
  ccml_support::Rng rng(static_cast<std::uint64_t>(rows) * 31 + cols + off);
  rng.normal(fx.data(), n, mean, 1.0f);
  rng.normal(fr.data(), n, 0.0f, 1.0f);
  rng.normal(fb.data(), fb.size(), 0.0f, 0.5f);
  rng.uniform(fg.data(), fg.size(), 0.5f, 1.5f);
  rng.uniform(fbe.data(), fbe.size(), -1.0f, 1.0f);
  auto store = [&](const std::vector<float>& f) {
    std::vector<Pub> p(o, D::from_float(0.0f));
    for (float v : f) p.push_back(D::from_float(v));
    return p;
  };
  const std::vector<Pub> hx = store(fx), hr = store(fr), hb = store(fb), hg = store(fg),
                         hbe = store(fbe);
  /* Host model of the contract: the same FP32 additions, one rounding. */
  std::vector<Pub> s_expected(n);
  std::vector<float> s_exact(n), g_exact(cols), be_exact(cols);
  for (std::size_t i = 0; i < n; ++i) {
    const int c = static_cast<int>(i % static_cast<std::size_t>(cols));
    float s = D::to_float(hx[o + i]) + D::to_float(hr[o + i]);
    if (bias) s += D::to_float(hb[o + static_cast<std::size_t>(c)]);
    s_expected[i] = D::from_float(s);
    s_exact[i] = D::to_float(s_expected[i]);
  }
  for (int c = 0; c < cols; ++c) {
    g_exact[static_cast<std::size_t>(c)] = D::to_float(hg[o + static_cast<std::size_t>(c)]);
    be_exact[static_cast<std::size_t>(c)] = D::to_float(hbe[o + static_cast<std::size_t>(c)]);
  }
  std::vector<double> ref;
  ccml_test::layernorm_reference(ref, s_exact, g_exact, be_exact, rows, cols, kEps);

  Dev<Pub> dx(n + o), dr(n + o), db(cols + o), dg(cols + o), dbe(cols + o), dy(n + o), ds(n + o);
  ctx.check(dx.up(hx) && dr.up(hr) && db.up(hb) && dg.up(hg) && dbe.up(hbe) && dy.poison() &&
                ds.poison(),
            "device setup");
  const std::string what = std::string(D::name) + " " + ccml_layernorm_algo_name(algo) + " " +
                           std::to_string(rows) + "x" + std::to_string(cols) +
                           " bias=" + std::to_string(bias) + " sum_out=" + std::to_string(sum_out) +
                           " offset=" + std::to_string(off);
  CCML_ASSERT_OK(D::run(dy.get() + o, sum_out ? ds.get() + o : nullptr, dx.get() + o,
                        dr.get() + o, bias ? db.get() + o : nullptr, dg.get() + o, dbe.get() + o,
                        rows, cols, algo));
  CCML_ASSERT_OK(ccml_device_synchronize());
  std::vector<Pub> y_all, s_all;
  ctx.check(dy.down(y_all) && ds.down(s_all), "download");
  if (sum_out) {
    ctx.check(std::memcmp(s_all.data() + o, s_expected.data(), n * sizeof(Pub)) == 0,
              what + ": sum_out is not bitwise the rounded x + residual + bias");
  }
  std::vector<float> got(n);
  for (std::size_t i = 0; i < n; ++i) got[i] = D::to_float(y_all[o + i]);
  const ccml_test::Tolerance tol = D::widen(ccml_test::tolerance_for_f32_reduction(cols));
  const ccml_test::ErrorStats st = ccml_test::compare(got, ref, tol);
  if (!ccml_test::within(st)) ctx.fail(what + " -- " + tol.to_string() + " -- " + st.to_string());
  return st.max_normalized_error;
}

template <typename D>
void every_case(const ccml_test::Context& ctx) {
  const int shapes[][2] = {{1, 1},    {3, 7},     {37, 768}, {33, 1024}, {5, 1000},
                           {17, 132}, {9, 1025},  {4, 4096}, {2, 6143},  {300, 64}};
  for (ccml_layernorm_algo algo :
       {CCML_LAYERNORM_VECTORIZED, CCML_LAYERNORM_WELFORD, CCML_LAYERNORM_AUTO}) {
    double worst = 0.0;
    for (const auto& s : shapes) {
      for (int combo = 0; combo < 4; ++combo) {
        worst = std::fmax(worst, check<D>(ctx, s[0], s[1], combo & 1, combo & 2, algo));
      }
    }
    worst = std::fmax(worst, check<D>(ctx, 16, 768, true, true, algo, 100.0f)); /* offset */
    worst = std::fmax(worst, check<D>(ctx, 16, 768, true, true, algo, 0.0f, 1)); /* misaligned */
    ctx.note(std::string(D::name) + " " + ccml_layernorm_algo_name(algo) +
             ": worst normalized error " + std::to_string(worst));
  }
}

}  // namespace

CCML_TEST(layernorm_residual, F32_every_case) { every_case<F32>(ctx); }
CCML_TEST(layernorm_residual, F16_every_case) { every_case<F16>(ctx); }
CCML_TEST(layernorm_residual, BF16_every_case) { every_case<BF16>(ctx); }

CCML_TEST(layernorm_residual, rejects_bad_arguments) {
  Dev<float> d(64);
  float* p = d.get();
  ctx.check(ccml_layernorm_residual_f32(p, nullptr, p + 16, nullptr, nullptr, nullptr, nullptr, 2,
                                        4, kEps, CCML_LAYERNORM_AUTO,
                                        nullptr) == CCML_ERROR_INVALID_ARGUMENT,
            "null residual");
  ctx.check(ccml_layernorm_residual_f32(p, p + 16, p + 16, p + 32, nullptr, nullptr, nullptr, 2,
                                        4, kEps, CCML_LAYERNORM_AUTO,
                                        nullptr) == CCML_ERROR_INVALID_ARGUMENT,
            "sum_out aliasing x");
  ctx.check(ccml_layernorm_residual_f32(p, nullptr, p + 16, p + 32, nullptr, nullptr, nullptr, 2,
                                        4, kEps, CCML_LAYERNORM_TWO_PASS,
                                        nullptr) == CCML_ERROR_NOT_IMPLEMENTED,
            "no fused residual for TWO_PASS");
}
