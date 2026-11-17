/* Fused linear epilogue, C = act(A B + bias) + residual (docs/optimization-log.md
   entry 8; enhancement F1), against an fp64 reference computed from the same
   rounded 16-bit inputs. MMA_SYNC fuses it into the store, SPLIT_K into the
   reduction. Outputs are NaN-poisoned before every call. BF16 SKIPs below
   sm_80 (docs/local.md L1). */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ccml/matmul.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "support/compare.h"
#include "support/random.h"
#include "support/reference.h"
#include "tests/support/harness.h"

namespace {

struct F16 {
  using Pub = ccml_f16;
  static constexpr const char* name = "f16";
  static constexpr double unit = ccml_test::kF16UnitRoundoff;
  static constexpr double floor = ccml_test::kF16SubnormalStep;
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
  static ccml_status run(Pub* c, const Pub* a, const Pub* b, const Pub* bias, const Pub* res,
                         int m, int n, int k, ccml_activation act, ccml_matmul_algo algo) {
    return ccml_matmul_epilogue_f16(c, a, b, bias, res, m, n, k, act, algo, nullptr);
  }
};

struct BF16 {
  using Pub = ccml_bf16;
  static constexpr const char* name = "bf16";
  static constexpr double unit = ccml_test::kBF16UnitRoundoff;
  static constexpr double floor = 1.1754943508222875e-38;
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
  static ccml_status run(Pub* c, const Pub* a, const Pub* b, const Pub* bias, const Pub* res,
                         int m, int n, int k, ccml_activation act, ccml_matmul_algo algo) {
    return ccml_matmul_epilogue_bf16(c, a, b, bias, res, m, n, k, act, algo, nullptr);
  }
};

double gelu_tanh(double x) {
  return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

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

template <typename D>
std::vector<typename D::Pub> rounded(std::size_t n, std::uint64_t seed, float stddev,
                                     std::vector<float>* exact) {
  std::vector<float> raw(n);
  ccml_support::Rng rng(seed);
  rng.normal(raw.data(), n, 0.0f, stddev);
  std::vector<typename D::Pub> out(n);
  exact->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = D::from_float(raw[i]);
    (*exact)[i] = D::to_float(out[i]);
  }
  return out;
}

/* One case. `offset` shifts every pointer that many elements into its buffer
   (a misaligned view). Returns the normalized error. */
template <typename D>
double check(const ccml_test::Context& ctx, int M, int N, int K, bool with_bias,
             bool with_residual, ccml_activation act, ccml_matmul_algo algo, int offset = 0) {
  using Pub = typename D::Pub;
  const std::uint64_t seed = static_cast<std::uint64_t>(M) * 7919 + N * 31 + K + act;
  std::vector<float> a_x, b_x, bias_x, res_x;
  std::vector<Pub> a = rounded<D>(static_cast<std::size_t>(M) * K, seed, 1.0f, &a_x);
  std::vector<Pub> b = rounded<D>(static_cast<std::size_t>(K) * N, seed + 1, 1.0f, &b_x);
  /* Bias and residual on the scale of the product (rms ~ sqrt(K)). */
  const float scale = std::sqrt(static_cast<float>(K));
  std::vector<Pub> bias = rounded<D>(static_cast<std::size_t>(N), seed + 2, scale, &bias_x);
  std::vector<Pub> res = rounded<D>(static_cast<std::size_t>(M) * N, seed + 3, scale, &res_x);

  std::vector<double> prod;
  ccml_test::matmul_reference(prod, a_x, b_x, M, N, K);
  std::vector<double> ref(prod.size());
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      const std::size_t i = static_cast<std::size_t>(r) * N + c;
      double v = prod[i] + (with_bias ? bias_x[static_cast<std::size_t>(c)] : 0.0);
      if (act == CCML_ACT_RELU) v = v > 0.0 ? v : 0.0;
      if (act == CCML_ACT_GELU_TANH) v = gelu_tanh(v);
      ref[i] = v + (with_residual ? res_x[i] : 0.0);
    }
  }
  /* The accumulation error is the matmul's (scale: rms of the product); GELU's
     slope is at most 1.13 and tanhf adds a few float ulps, covered by 2x. */
  ccml_test::Tolerance tol = ccml_test::tolerance_for_f32_dot_product(K, ccml_test::rms(prod));
  tol.abs_tol *= 2.0;
  tol = ccml_test::widen_for_storage(tol, D::unit, D::floor);

  const std::size_t off = static_cast<std::size_t>(offset);
  auto pad = [&](const std::vector<Pub>& v) {
    std::vector<Pub> p(off, D::from_float(0.0f));
    p.insert(p.end(), v.begin(), v.end());
    return p;
  };
  Dev<Pub> da(a.size() + off), db(b.size() + off), dbias(bias.size() + off),
      dres(res.size() + off), dc(static_cast<std::size_t>(M) * N + off);
  ctx.check(da.up(pad(a)) && db.up(pad(b)) && dbias.up(pad(bias)) && dres.up(pad(res)) &&
                dc.poison(),
            "device setup");
  const ccml_status s =
      D::run(dc.get() + off, da.get() + off, db.get() + off,
             with_bias ? dbias.get() + off : nullptr, with_residual ? dres.get() + off : nullptr,
             M, N, K, act, algo);
  const std::string what = std::string(D::name) + " " + ccml_matmul_algo_name(algo) + " " +
                           std::to_string(M) + "x" + std::to_string(N) + "x" +
                           std::to_string(K) + " bias=" + std::to_string(with_bias) +
                           " residual=" + std::to_string(with_residual) +
                           " act=" + std::to_string(act) + " offset=" + std::to_string(offset);
  if (s == CCML_ERROR_NOT_IMPLEMENTED) ctx.skip(what + ": " + ccml_last_error_string());
  if (s != CCML_SUCCESS) ctx.fail(what + " -> " + ccml_status_string(s));
  CCML_ASSERT_OK(ccml_device_synchronize());
  std::vector<Pub> c_all;
  ctx.check(dc.down(c_all), "download");
  std::vector<float> got(c_all.size() - off);
  for (std::size_t i = 0; i < got.size(); ++i) got[i] = D::to_float(c_all[i + off]);
  const ccml_test::ErrorStats st = ccml_test::compare(got, ref, tol);
  if (!ccml_test::within(st)) ctx.fail(what + " -- " + tol.to_string() + " -- " + st.to_string());
  return st.max_normalized_error;
}

const ccml_matmul_algo kAlgos[] = {CCML_MATMUL_MMA_SYNC, CCML_MATMUL_SPLIT_K};

/* Every epilogue combination on awkward shapes, decode shapes (several
   split-K slices) and a target shape, for both fused paths. */
template <typename D>
void every_combination(const ccml_test::Context& ctx) {
  const int shapes[][3] = {{1, 1, 1},     {17, 33, 65},   {131, 67, 97},  {1, 768, 3072},
                           {16, 768, 768}, {64, 136, 264}, {128, 768, 768}, {255, 257, 129}};
  struct Combo {
    bool bias, residual;
    ccml_activation act;
  };
  const Combo combos[] = {{true, false, CCML_ACT_GELU_TANH}, {true, true, CCML_ACT_NONE},
                          {false, true, CCML_ACT_NONE},      {true, false, CCML_ACT_RELU},
                          {false, false, CCML_ACT_GELU_TANH}, {false, false, CCML_ACT_NONE},
                          {true, true, CCML_ACT_GELU_TANH}};
  for (ccml_matmul_algo algo : kAlgos) {
    double worst = 0.0;
    for (const auto& s : shapes) {
      for (const Combo& c : combos) {
        worst = std::max(worst, check<D>(ctx, s[0], s[1], s[2], c.bias, c.residual, c.act, algo));
      }
    }
    ctx.note(std::string(D::name) + " " + ccml_matmul_algo_name(algo) +
             ": worst normalized error " + std::to_string(worst));
  }
}

template <typename D>
void misaligned(const ccml_test::Context& ctx) {
  for (ccml_matmul_algo algo : kAlgos) {
    for (int offset : {1, 3}) {
      check<D>(ctx, 16, 128, 96, true, true, CCML_ACT_GELU_TANH, algo, offset);
      check<D>(ctx, 70, 136, 72, true, true, CCML_ACT_NONE, algo, offset);
    }
  }
}

}  // namespace

CCML_TEST(matmul_epilogue, F16_every_combination) { every_combination<F16>(ctx); }
CCML_TEST(matmul_epilogue, F16_misaligned_views) { misaligned<F16>(ctx); }
CCML_TEST(matmul_epilogue, BF16_every_combination) { every_combination<BF16>(ctx); }
CCML_TEST(matmul_epilogue, BF16_misaligned_views) { misaligned<BF16>(ctx); }

CCML_TEST(matmul_epilogue, rejects_bad_arguments) {
  ccml_f16* d = nullptr;
  CCML_ASSERT_OK(ccml_device_malloc(reinterpret_cast<void**>(&d), 16 * sizeof(ccml_f16)));
  ctx.check(ccml_matmul_epilogue_f16(d, d, d, nullptr, nullptr, 2, 2, 2,
                                     static_cast<ccml_activation>(7), CCML_MATMUL_MMA_SYNC,
                                     nullptr) == CCML_ERROR_INVALID_ARGUMENT,
            "unknown activation");
  ctx.check(ccml_matmul_epilogue_f16(nullptr, d, d, nullptr, nullptr, 2, 2, 2, CCML_ACT_NONE,
                                     CCML_MATMUL_MMA_SYNC, nullptr) == CCML_ERROR_INVALID_ARGUMENT,
            "null C");
  ctx.check(ccml_matmul_epilogue_f16(d, d, d, nullptr, nullptr, 2, 2, 2, CCML_ACT_NONE,
                                     CCML_MATMUL_TILED, nullptr) == CCML_ERROR_NOT_IMPLEMENTED,
            "no fused epilogue for a CUDA-core algo");
  ccml_device_free(d);
}
