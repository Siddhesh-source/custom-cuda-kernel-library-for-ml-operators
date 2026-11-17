/* Fused scale + mask softmax (docs/optimization-log.md entry 16; enhancement
   F3), FP32, FP16 and BF16, both fused paths (WARP_PER_ROW, ONLINE) and AUTO.

   Reference: v = x * scale + mask[r % mask_rows][c] in double from the
   rounded inputs, causal keys set to -inf, softmax in double; a fully masked
   row is all zeros (the documented policy). Outputs are NaN-poisoned before
   every call. */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ccml/memory.h"
#include "ccml/softmax.h"
#include "ccml/status.h"
#include "support/compare.h"
#include "support/random.h"
#include "tests/support/harness.h"

namespace {

struct F32 {
  using Pub = float;
  static constexpr const char* name = "f32";
  static Pub from_float(float f) { return f; }
  static float to_float(Pub p) { return p; }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) { return t; }
  static ccml_status run(Pub* y, const Pub* x, const Pub* m, int mr, int r, int c, float s,
                         int cq, ccml_softmax_algo a) {
    return ccml_softmax_masked_f32(y, x, m, mr, r, c, s, cq, a, nullptr);
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
  static ccml_status run(Pub* y, const Pub* x, const Pub* m, int mr, int r, int c, float s,
                         int cq, ccml_softmax_algo a) {
    return ccml_softmax_masked_f16(y, x, m, mr, r, c, s, cq, a, nullptr);
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
  static ccml_status run(Pub* y, const Pub* x, const Pub* m, int mr, int r, int c, float s,
                         int cq, ccml_softmax_algo a) {
    return ccml_softmax_masked_bf16(y, x, m, mr, r, c, s, cq, a, nullptr);
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
  bool poison() { return ccml_device_memset_async(p_, 0xFF, n_ * sizeof(Pub), nullptr) == CCML_SUCCESS; }
  void* p_ = nullptr;
  std::size_t n_;
};

struct Config {
  int rows, cols, mask_rows; /* mask_rows 0: no additive mask */
  int causal_q;
  float scale;
  bool mask_some_rows_fully; /* make every 5th mask row all -inf */
};

/* Returns the worst normalized error; fails on any mismatch, including a
   fully masked row that is not exactly zero. */
template <typename D>
double check(const ccml_test::Context& ctx, const Config& g, ccml_softmax_algo algo) {
  using Pub = typename D::Pub;
  const float inf = std::numeric_limits<float>::infinity();
  const std::size_t n = static_cast<std::size_t>(g.rows) * g.cols;
  std::vector<float> fx(n);
  ccml_support::Rng(static_cast<std::uint64_t>(g.rows) * 13 + g.cols + g.causal_q)
      .normal(fx.data(), n, 0.0f, 3.0f);
  std::vector<Pub> hx(n), hm;
  for (std::size_t i = 0; i < n; ++i) {
    hx[i] = D::from_float(fx[i]);
    fx[i] = D::to_float(hx[i]);
  }
  std::vector<float> fm;
  if (g.mask_rows > 0) {
    fm.assign(static_cast<std::size_t>(g.mask_rows) * g.cols, 0.0f);
    for (int r = 0; r < g.mask_rows; ++r) {
      for (int c = 0; c < g.cols; ++c) {
        float v = (c * 7 + r * 3) % 5 == 0 ? -inf : 0.5f * static_cast<float>((c + r) % 3);
        if (g.mask_some_rows_fully && r % 5 == 0) v = -inf;
        fm[static_cast<std::size_t>(r) * g.cols + c] = v;
      }
    }
    hm.resize(fm.size());
    for (std::size_t i = 0; i < fm.size(); ++i) {
      hm[i] = D::from_float(fm[i]);
      fm[i] = D::to_float(hm[i]);
    }
  }
  /* Reference and the set of fully masked rows. */
  std::vector<double> ref(n);
  std::vector<bool> all_masked(static_cast<std::size_t>(g.rows));
  for (int r = 0; r < g.rows; ++r) {
    const int limit = g.causal_q > 0 ? r % g.causal_q + (g.cols - g.causal_q) : g.cols;
    std::vector<double> v(static_cast<std::size_t>(g.cols));
    double m = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < g.cols; ++c) {
      double t = static_cast<double>(fx[static_cast<std::size_t>(r) * g.cols + c]) * g.scale;
      if (g.mask_rows > 0) t += fm[static_cast<std::size_t>(r % g.mask_rows) * g.cols + c];
      if (c > limit) t = -std::numeric_limits<double>::infinity();
      v[static_cast<std::size_t>(c)] = t;
      m = std::fmax(m, t);
    }
    all_masked[static_cast<std::size_t>(r)] = std::isinf(m) && m < 0;
    double s = 0.0;
    for (double& t : v) {
      t = std::isinf(t) ? 0.0 : std::exp(t - m);
      s += t;
    }
    for (int c = 0; c < g.cols; ++c) {
      ref[static_cast<std::size_t>(r) * g.cols + c] =
          all_masked[static_cast<std::size_t>(r)] ? 0.0 : v[static_cast<std::size_t>(c)] / s;
    }
  }
  Dev<Pub> dx(n), dy(n), dm(hm.empty() ? 1 : hm.size());
  ctx.check(dx.up(hx) && dy.poison() && (hm.empty() || dm.up(hm)), "device setup");
  const std::string what = std::string(D::name) + " " + ccml_softmax_algo_name(algo) + " " +
                           std::to_string(g.rows) + "x" + std::to_string(g.cols) +
                           " mask_rows=" + std::to_string(g.mask_rows) +
                           " causal_q=" + std::to_string(g.causal_q);
  CCML_ASSERT_OK(D::run(dy.get(), dx.get(), hm.empty() ? nullptr : dm.get(), g.mask_rows, g.rows,
                        g.cols, g.scale, g.causal_q, algo));
  CCML_ASSERT_OK(ccml_device_synchronize());
  std::vector<Pub> yh;
  ctx.check(dy.down(yh), "download");
  std::vector<float> got(n);
  for (std::size_t i = 0; i < n; ++i) got[i] = D::to_float(yh[i]);
  for (int r = 0; r < g.rows; ++r) {
    if (!all_masked[static_cast<std::size_t>(r)]) continue;
    for (int c = 0; c < g.cols; ++c) {
      if (got[static_cast<std::size_t>(r) * g.cols + c] != 0.0f) {
        ctx.fail(what + ": fully masked row " + std::to_string(r) + " is not exactly zero");
      }
    }
  }
  const ccml_test::Tolerance tol = D::widen(ccml_test::tolerance_for_softmax_f32(g.cols));
  const ccml_test::ErrorStats st = ccml_test::compare(got, ref, tol);
  if (!ccml_test::within(st)) ctx.fail(what + " -- " + tol.to_string() + " -- " + st.to_string());
  return st.max_normalized_error;
}

const Config kConfigs[] = {
    {64, 128, 0, 128, 0.125f, false},    /* causal self-attention */
    {96, 128, 0, 32, 0.125f, false},     /* causal, 3 blocks of 32 queries, cols > causal_q */
    {40, 77, 0, 77, 1.0f, false},        /* causal, odd width */
    {16, 100, 0, 120, 0.5f, false},      /* causal_q > cols: some rows fully masked */
    {48, 256, 1, 0, 0.125f, false},      /* padding mask broadcast to every row */
    {48, 256, 6, 0, 0.125f, true},       /* per-row-group mask, some groups fully masked */
    {24, 1000, 3, 1000, 0.125f, false},  /* mask + causal */
    {8, 1024, 2, 0, 2.0f, true},         /* widest warp-per-row */
    {4, 3000, 1, 3000, 0.125f, false},   /* beyond warp-per-row: ONLINE only */
    {3, 1, 0, 0, 1.0f, false},           /* single column */
};

template <typename D>
void every_config(const ccml_test::Context& ctx) {
  for (ccml_softmax_algo algo :
       {CCML_SOFTMAX_WARP_PER_ROW, CCML_SOFTMAX_ONLINE, CCML_SOFTMAX_AUTO}) {
    double worst = 0.0;
    for (const Config& g : kConfigs) {
      if (algo == CCML_SOFTMAX_WARP_PER_ROW && g.cols > 1024) continue;
      worst = std::fmax(worst, check<D>(ctx, g, algo));
    }
    ctx.note(std::string(D::name) + " " + ccml_softmax_algo_name(algo) +
             ": worst normalized error " + std::to_string(worst));
  }
}

}  // namespace

CCML_TEST(softmax_masked, F32_every_config) { every_config<F32>(ctx); }
CCML_TEST(softmax_masked, F16_every_config) { every_config<F16>(ctx); }
CCML_TEST(softmax_masked, BF16_every_config) { every_config<BF16>(ctx); }

/* A NaN in x makes its row NaN even when the row is otherwise fully masked. */
CCML_TEST(softmax_masked, nan_poisons_even_a_masked_row) {
  const float inf = std::numeric_limits<float>::infinity();
  for (ccml_softmax_algo algo : {CCML_SOFTMAX_WARP_PER_ROW, CCML_SOFTMAX_ONLINE}) {
    const int cols = 64;
    std::vector<float> hx(cols, 1.0f), hm(cols, -inf);
    hx[5] = std::numeric_limits<float>::quiet_NaN();
    Dev<float> dx(cols), dy(cols), dm(cols);
    ctx.check(dx.up(hx) && dm.up(hm) && dy.poison(), "setup");
    CCML_ASSERT_OK(ccml_softmax_masked_f32(dy.get(), dx.get(), dm.get(), 1, 1, cols, 1.0f, 0, algo,
                                           nullptr));
    CCML_ASSERT_OK(ccml_device_synchronize());
    std::vector<float> got;
    ctx.check(dy.down(got), "download");
    for (float v : got) ctx.check(std::isnan(v), std::string(ccml_softmax_algo_name(algo)) +
                                                     ": NaN must poison the masked row");
  }
}

CCML_TEST(softmax_masked, rejects_bad_arguments) {
  Dev<float> d(16);
  ctx.check(ccml_softmax_masked_f32(d.get(), d.get(), d.get(), 0, 2, 4, 1.0f, 0,
                                    CCML_SOFTMAX_AUTO, nullptr) == CCML_ERROR_INVALID_ARGUMENT,
            "mask_rows 0 with a mask");
  ctx.check(ccml_softmax_masked_f32(d.get(), d.get(), nullptr, 0, 2, 4, 1.0f, -1,
                                    CCML_SOFTMAX_AUTO, nullptr) == CCML_ERROR_INVALID_ARGUMENT,
            "negative causal_q");
  ctx.check(ccml_softmax_masked_f32(d.get(), d.get(), nullptr, 0, 2, 4, 1.0f, 0,
                                    CCML_SOFTMAX_BLOCK_PER_ROW, nullptr) ==
                CCML_ERROR_NOT_IMPLEMENTED,
            "no fused mask for BLOCK_PER_ROW");
  ctx.check(ccml_softmax_masked_f32(d.get(), d.get(), nullptr, 0, 2, 1025, 1.0f, 0,
                                    CCML_SOFTMAX_WARP_PER_ROW, nullptr) ==
                CCML_ERROR_UNSUPPORTED_SHAPE,
            "warp per row past 1024 columns");
}
