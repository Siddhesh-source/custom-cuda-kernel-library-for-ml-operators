/* FP16 and BF16 storage for all three operators (reference.md §3: correctness
   for FP32, FP16 and BF16 with explicit tolerances; §9: every kernel passes for
   each dtype).

   Method: inputs are generated in float, rounded to the storage type, and the
   fp64 reference is computed from the ROUNDED values. The only error left to
   bound is then the kernel's own float32 arithmetic plus one final rounding to
   the storage type, which is what support/compare.h's widen_for_storage
   allows. The f32 suites (test_matmul, test_layernorm, test_softmax) carry the
   full shape and edge-case coverage; this suite repeats the representative
   shapes per dtype and adds the cases where 16-bit storage behaves
   differently: overflow to inf, masked rows, NaN propagation, range limits.

   The PyTorch float64 comparison for the same dtypes is in
   python/tests/test_torch_api.py. */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ccml/layernorm.h"
#include "ccml/matmul.h"
#include "ccml/memory.h"
#include "ccml/softmax.h"
#include "ccml/status.h"
#include "support/compare.h"
#include "support/random.h"
#include "support/reference.h"
#include "tests/support/harness.h"

using ccml_test::Tolerance;

namespace {

/* ------------------------------------------------------- dtype traits -- */

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
  static ccml_status matmul(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                            ccml_matmul_algo algo) {
    return ccml_matmul_f16(c, a, b, m, n, k, algo, nullptr);
  }
  static ccml_status layernorm(Pub* y, const Pub* x, const Pub* g, const Pub* b,
                               int rows, int cols, float eps,
                               ccml_layernorm_algo algo = CCML_LAYERNORM_AUTO) {
    return ccml_layernorm_f16(y, x, g, b, rows, cols, eps, algo, nullptr);
  }
  static ccml_status softmax(Pub* y, const Pub* x, int rows, int cols) {
    return ccml_softmax_f16(y, x, rows, cols, CCML_SOFTMAX_AUTO, nullptr);
  }
};

struct BF16 {
  using Pub = ccml_bf16;
  static constexpr const char* name = "bf16";
  static constexpr double unit = ccml_test::kBF16UnitRoundoff;
  /* BF16 subnormals start near 1e-38, where float32 itself underflows. */
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
  static ccml_status matmul(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                            ccml_matmul_algo algo) {
    return ccml_matmul_bf16(c, a, b, m, n, k, algo, nullptr);
  }
  static ccml_status layernorm(Pub* y, const Pub* x, const Pub* g, const Pub* b,
                               int rows, int cols, float eps,
                               ccml_layernorm_algo algo = CCML_LAYERNORM_AUTO) {
    return ccml_layernorm_bf16(y, x, g, b, rows, cols, eps, algo, nullptr);
  }
  static ccml_status softmax(Pub* y, const Pub* x, int rows, int cols) {
    return ccml_softmax_bf16(y, x, rows, cols, CCML_SOFTMAX_AUTO, nullptr);
  }
};

/* ------------------------------------------------------------ helpers -- */

/* Owning device array of 16-bit elements. */
template <typename Pub>
class Device16 {
 public:
  explicit Device16(std::size_t n) : n_(n) {
    if (ccml_device_malloc(&ptr_, n * sizeof(Pub)) != CCML_SUCCESS) ptr_ = nullptr;
  }
  ~Device16() { ccml_device_free(ptr_); }
  Device16(const Device16&) = delete;
  Device16& operator=(const Device16&) = delete;
  Pub* get() { return static_cast<Pub*>(ptr_); }
  bool ok() const { return ptr_ != nullptr; }
  bool upload(const std::vector<Pub>& h) {
    return ccml_memcpy_h2d(ptr_, h.data(), n_ * sizeof(Pub)) == CCML_SUCCESS;
  }
  bool download(std::vector<Pub>& h) {
    h.resize(n_);
    return ccml_memcpy_d2h(h.data(), ptr_, n_ * sizeof(Pub)) == CCML_SUCCESS;
  }
  /* 0xFF bytes: NaN in both FP16 and BF16, so an unwritten element is caught. */
  bool poison() { return ccml_device_memset_async(ptr_, 0xFF, n_ * sizeof(Pub), nullptr) == CCML_SUCCESS; }

 private:
  void*       ptr_ = nullptr;
  std::size_t n_ = 0;
};

/* Rounds float values to the storage type. Returns both the stored values and
   the floats they represent exactly, which is what the reference is fed. */
template <typename D>
void quantize(const std::vector<float>& in, std::vector<typename D::Pub>* stored,
              std::vector<float>* exact) {
  stored->resize(in.size());
  exact->resize(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    (*stored)[i] = D::from_float(in[i]);
    (*exact)[i] = D::to_float((*stored)[i]);
  }
}

template <typename D>
std::vector<float> widen(const std::vector<typename D::Pub>& v) {
  std::vector<float> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = D::to_float(v[i]);
  return out;
}

template <typename D>
void report(const ccml_test::Context& ctx, const std::string& what,
            const ccml_test::ErrorStats& stats, const Tolerance& tol) {
  ctx.note(std::string(D::name) + " " + what + "  max_abs=" +
           std::to_string(stats.max_abs_error) + " max_rel=" +
           std::to_string(stats.max_rel_error) + "  " + tol.to_string());
  if (!ccml_test::within(stats)) {
    ctx.fail(std::string(D::name) + " " + what + " exceeded tolerance -- " +
             tol.to_string() + " -- " + stats.to_string());
  }
}

/* ------------------------------------------------------- operator runs -- */

template <typename D>
void check_matmul(const ccml_test::Context& ctx, int M, int N, int K,
                  ccml_matmul_algo algo) {
  using Pub = typename D::Pub;
  std::vector<float> a_raw(static_cast<std::size_t>(M) * K), b_raw(static_cast<std::size_t>(K) * N);
  ccml_support::Rng rng(static_cast<std::uint64_t>(M) * 131 + N * 17 + K);
  rng.normal(a_raw.data(), a_raw.size(), 0.0f, 1.0f);
  rng.normal(b_raw.data(), b_raw.size(), 0.0f, 1.0f);

  std::vector<Pub> a, b;
  std::vector<float> a_exact, b_exact;
  quantize<D>(a_raw, &a, &a_exact);
  quantize<D>(b_raw, &b, &b_exact);

  std::vector<double> ref;
  ccml_test::matmul_reference(ref, a_exact, b_exact, M, N, K);

  Device16<Pub> da(a.size()), db(b.size()), dc(static_cast<std::size_t>(M) * N);
  ctx.check(da.ok() && db.ok() && dc.ok() && da.upload(a) && db.upload(b) && dc.poison(),
            "device setup");
  CCML_ASSERT_OK(D::matmul(dc.get(), da.get(), db.get(), M, N, K, algo));
  CCML_ASSERT_OK(ccml_device_synchronize());
  std::vector<Pub> c;
  ctx.check(dc.download(c), "download");

  const Tolerance tol = ccml_test::widen_for_storage(
      ccml_test::tolerance_for_f32_dot_product(K, ccml_test::rms(ref)), D::unit, D::floor);
  report<D>(ctx,
            std::string("matmul ") + ccml_matmul_algo_name(algo) + " " + std::to_string(M) +
                "x" + std::to_string(N) + "x" + std::to_string(K),
            ccml_test::compare(widen<D>(c), ref, tol), tol);
}

template <typename D>
void check_layernorm(const ccml_test::Context& ctx, int rows, int cols, float offset,
                     ccml_layernorm_algo algo = CCML_LAYERNORM_AUTO) {
  using Pub = typename D::Pub;
  if (algo == CCML_LAYERNORM_WARP_PER_ROW && cols > 1024) return; /* refused by design */
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  std::vector<float> x_raw(n), g_raw(static_cast<std::size_t>(cols)), b_raw(static_cast<std::size_t>(cols));
  ccml_support::Rng rng(static_cast<std::uint64_t>(rows) * 7 + cols);
  rng.normal(x_raw.data(), n, offset, 1.0f);
  rng.uniform(g_raw.data(), g_raw.size(), 0.5f, 1.5f);
  rng.uniform(b_raw.data(), b_raw.size(), -1.0f, 1.0f);

  std::vector<Pub> x, g, b;
  std::vector<float> x_exact, g_exact, b_exact;
  quantize<D>(x_raw, &x, &x_exact);
  quantize<D>(g_raw, &g, &g_exact);
  quantize<D>(b_raw, &b, &b_exact);

  constexpr float kEps = 1e-5f;
  std::vector<double> ref;
  ccml_test::layernorm_reference(ref, x_exact, g_exact, b_exact, rows, cols, kEps);

  Device16<Pub> dx(n), dy(n), dg(g.size()), dbeta(b.size());
  ctx.check(dx.ok() && dy.ok() && dg.ok() && dbeta.ok() && dx.upload(x) && dg.upload(g) &&
                dbeta.upload(b) && dy.poison(),
            "device setup");
  CCML_ASSERT_OK(
      D::layernorm(dy.get(), dx.get(), dg.get(), dbeta.get(), rows, cols, kEps, algo));
  CCML_ASSERT_OK(ccml_device_synchronize());
  std::vector<Pub> y;
  ctx.check(dy.download(y), "download");

  const Tolerance tol = ccml_test::widen_for_storage(
      ccml_test::tolerance_for_f32_reduction(cols), D::unit, D::floor);
  report<D>(ctx,
            std::string("layernorm ") + ccml_layernorm_algo_name(algo) + " " +
                std::to_string(rows) + "x" + std::to_string(cols) +
                " offset=" + std::to_string(offset),
            ccml_test::compare(widen<D>(y), ref, tol), tol);
}

template <typename D>
std::vector<float> run_softmax(const ccml_test::Context& ctx, const std::vector<float>& x_raw,
                               int rows, int cols, std::vector<float>* x_exact) {
  using Pub = typename D::Pub;
  std::vector<Pub> x;
  quantize<D>(x_raw, &x, x_exact);
  Device16<Pub> dx(x.size()), dy(x.size());
  ctx.check(dx.ok() && dy.ok() && dx.upload(x) && dy.poison(), "device setup");
  CCML_ASSERT_OK(D::softmax(dy.get(), dx.get(), rows, cols));
  CCML_ASSERT_OK(ccml_device_synchronize());
  std::vector<Pub> y;
  ctx.check(dy.download(y), "download");
  return widen<D>(y);
}

template <typename D>
void check_softmax(const ccml_test::Context& ctx, int rows, int cols, float scale) {
  std::vector<float> x_raw(static_cast<std::size_t>(rows) * cols), x_exact;
  ccml_support::Rng rng(static_cast<std::uint64_t>(rows) * 3 + cols);
  rng.normal(x_raw.data(), x_raw.size(), 0.0f, scale);
  const std::vector<float> got = run_softmax<D>(ctx, x_raw, rows, cols, &x_exact);

  std::vector<double> ref;
  ccml_test::softmax_reference(ref, x_exact, rows, cols);
  const Tolerance tol = ccml_test::widen_for_storage(
      ccml_test::tolerance_for_softmax_f32(cols), D::unit, D::floor);
  report<D>(ctx,
            "softmax " + std::to_string(rows) + "x" + std::to_string(cols) +
                " scale=" + std::to_string(scale),
            ccml_test::compare(got, ref, tol), tol);
}

const int kMatmulShapes[][3] = {{1, 1, 1}, {17, 33, 65}, {128, 128, 128},
                                {64, 257, 1023}, {3, 768, 768}, {256, 96, 2048}};
const ccml_layernorm_algo kLayerNormAlgos[] = {CCML_LAYERNORM_TWO_PASS,
                                                CCML_LAYERNORM_WELFORD,
                                                CCML_LAYERNORM_WARP_PER_ROW};
const int kRowShapes[][2] = {{1, 1}, {7, 33}, {64, 768}, {128, 1025}, {16, 4096}, {3, 32000}};

}  // namespace

/* ------------------------------------------------- reference agreement -- */

#define CCML_DTYPE_TESTS(D)                                                      \
  CCML_TEST(dtypes, D##_matmul_all_variants) {                                \
    for (const auto& s : kMatmulShapes) {                                        \
      check_matmul<D>(ctx, s[0], s[1], s[2], CCML_MATMUL_NAIVE);                 \
      check_matmul<D>(ctx, s[0], s[1], s[2], CCML_MATMUL_TILED);                 \
      check_matmul<D>(ctx, s[0], s[1], s[2], CCML_MATMUL_REGISTER_TILED);        \
      check_matmul<D>(ctx, s[0], s[1], s[2], CCML_MATMUL_VECTORIZED);            \
      check_matmul<D>(ctx, s[0], s[1], s[2], CCML_MATMUL_DOUBLE_BUFFERED);       \
    }                                                                            \
  }                                                                              \
  CCML_TEST(dtypes, D##_layernorm) {                                             \
    for (ccml_layernorm_algo algo : kLayerNormAlgos) {                           \
      for (const auto& s : kRowShapes)                                           \
        check_layernorm<D>(ctx, s[0], s[1], 0.0f, algo);                         \
      /* Mean far from zero: what FP64 two-pass and the FP32 pivot exist */    \
      /* for. 100 is representable to within its rounding in both formats. */  \
      check_layernorm<D>(ctx, 64, 768, 100.0f, algo);                            \
    }                                                                            \
  }                                                                              \
  CCML_TEST(dtypes, D##_softmax) {                                               \
    for (const auto& s : kRowShapes) check_softmax<D>(ctx, s[0], s[1], 1.0f);    \
    check_softmax<D>(ctx, 32, 512, 30.0f); /* wide dynamic range */              \
  }

CCML_DTYPE_TESTS(F16)
CCML_DTYPE_TESTS(BF16)

/* ---------------------------------------- where 16-bit storage differs -- */

/* 300 * 300 = 90000 exceeds FP16's largest finite value (65504), so the f32
   result rounds to +inf on store -- IEEE behaviour, and what PyTorch's FP16
   matmul returns. BF16 has float32's exponent range, so the same product is
   finite there. */
CCML_TEST(dtypes, f16_overflow_rounds_to_inf_bf16_does_not) {
  {
    std::vector<ccml_f16> a{F16::from_float(300.0f)}, b{F16::from_float(300.0f)}, c;
    Device16<ccml_f16> da(1), db(1), dc(1);
    ctx.check(da.upload(a) && db.upload(b), "upload");
    CCML_ASSERT_OK(ccml_matmul_f16(dc.get(), da.get(), db.get(), 1, 1, 1,
                                   CCML_MATMUL_AUTO, nullptr));
    CCML_ASSERT_OK(ccml_device_synchronize());
    dc.download(c);
    ctx.check(std::isinf(F16::to_float(c[0])) && F16::to_float(c[0]) > 0,
              "f16 product above 65504 must be +inf");
  }
  {
    std::vector<ccml_bf16> a{BF16::from_float(300.0f)}, b{BF16::from_float(300.0f)}, c;
    Device16<ccml_bf16> da(1), db(1), dc(1);
    ctx.check(da.upload(a) && db.upload(b), "upload");
    CCML_ASSERT_OK(ccml_matmul_bf16(dc.get(), da.get(), db.get(), 1, 1, 1,
                                    CCML_MATMUL_AUTO, nullptr));
    CCML_ASSERT_OK(ccml_device_synchronize());
    dc.download(c);
    const float v = BF16::to_float(c[0]);
    ctx.check(std::isfinite(v) && std::fabs(v - 90000.0f) <= 90000.0f * 1.0f / 256.0f,
              "bf16 product is finite and within one rounding of 90000");
  }
}

/* Rows at FP16's extreme magnitudes: the max subtraction keeps softmax finite,
   and layernorm's double accumulation keeps the variance (~4e9) exact. */
CCML_TEST(dtypes, f16_extreme_magnitudes_stay_finite) {
  const int cols = 64;
  std::vector<float> x(cols);
  for (int c = 0; c < cols; ++c) x[static_cast<std::size_t>(c)] = (c % 2 == 0) ? 65504.0f : -65504.0f;
  std::vector<float> exact;
  const std::vector<float> s = run_softmax<F16>(ctx, x, 1, cols, &exact);
  double sum = 0.0;
  for (float v : s) {
    ctx.check(std::isfinite(v), "softmax output finite at +-65504");
    sum += v;
  }
  ctx.check(std::fabs(sum - 1.0) < 64 * ccml_test::kF16UnitRoundoff, "softmax row sums to 1");

  std::vector<ccml_f16> xs, ys;
  std::vector<float> xe;
  quantize<F16>(x, &xs, &xe);
  Device16<ccml_f16> dx(cols), dy(cols);
  ctx.check(dx.upload(xs), "upload");
  CCML_ASSERT_OK(ccml_layernorm_f16(dy.get(), dx.get(), nullptr, nullptr, 1, cols, 1e-5f,
                                    CCML_LAYERNORM_AUTO, nullptr));
  CCML_ASSERT_OK(ccml_device_synchronize());
  dy.download(ys);
  for (int c = 0; c < cols; ++c) {
    const float v = F16::to_float(ys[static_cast<std::size_t>(c)]);
    ctx.check(std::isfinite(v) && std::fabs(std::fabs(v) - 1.0f) < 2e-3f,
              "layernorm of +-65504 alternating is +-1");
  }
}

/* Masking policy, identical to torch.softmax:
     - a row with some -inf entries gives exactly 0 there, the rest sums to 1;
     - a row that is entirely -inf gives NaN everywhere (0/0), as in PyTorch;
     - a NaN anywhere in a row makes the whole row NaN.
   Checked for both 16-bit types and documented in docs/correctness.md. */
template <typename D>
void check_masking(const ccml_test::Context& ctx) {
  const float inf = std::numeric_limits<float>::infinity();
  const int cols = 8;
  std::vector<float> x{1.0f, -inf, 2.0f, -inf, 0.5f, -inf, -1.0f, 3.0f,   /* partial */
                       -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,    /* full */
                       0.0f, 1.0f, std::nanf(""), 2.0f, 0.0f, 0.0f, 0.0f, 0.0f};  /* NaN */
  std::vector<float> exact;
  const std::vector<float> y = run_softmax<D>(ctx, x, 3, cols, &exact);

  double sum = 0.0;
  for (int c = 0; c < cols; ++c) {
    const float v = y[static_cast<std::size_t>(c)];
    if (std::isinf(x[static_cast<std::size_t>(c)])) {
      ctx.check(v == 0.0f, std::string(D::name) + ": masked entry must be exactly 0");
    }
    sum += v;
  }
  ctx.check(std::fabs(sum - 1.0) < 8 * D::unit, std::string(D::name) + ": partial row sums to 1");
  for (int c = 0; c < cols; ++c) {
    ctx.check(std::isnan(y[static_cast<std::size_t>(cols + c)]),
              std::string(D::name) + ": fully masked row is NaN, as in PyTorch");
    ctx.check(std::isnan(y[static_cast<std::size_t>(2 * cols + c)]),
              std::string(D::name) + ": NaN input makes the whole row NaN");
  }
}

CCML_TEST(dtypes, softmax_masking_and_nan_policy_f16) { check_masking<F16>(ctx); }
CCML_TEST(dtypes, softmax_masking_and_nan_policy_bf16) { check_masking<BF16>(ctx); }

/* A constant row has zero variance: the output must be exactly beta (eps keeps
   it finite), for both 16-bit types. */
template <typename D>
void check_constant_row(const ccml_test::Context& ctx) {
  using Pub = typename D::Pub;
  const int cols = 300;
  std::vector<Pub> x(cols, D::from_float(7.25f)), beta(cols), y;
  std::vector<Pub> gamma(cols, D::from_float(2.0f));
  for (int c = 0; c < cols; ++c) beta[static_cast<std::size_t>(c)] = D::from_float(0.125f * (c % 9));
  Device16<Pub> dx(cols), dy(cols), dg(cols), db(cols);
  ctx.check(dx.upload(x) && dg.upload(gamma) && db.upload(beta), "upload");
  CCML_ASSERT_OK(D::layernorm(dy.get(), dx.get(), dg.get(), db.get(), 1, cols, 1e-5f));
  CCML_ASSERT_OK(ccml_device_synchronize());
  dy.download(y);
  for (int c = 0; c < cols; ++c) {
    ctx.check(D::to_float(y[static_cast<std::size_t>(c)]) ==
                  D::to_float(beta[static_cast<std::size_t>(c)]),
              std::string(D::name) + ": constant row must give exactly beta");
  }
}

CCML_TEST(dtypes, layernorm_constant_row_is_beta_f16) { check_constant_row<F16>(ctx); }
CCML_TEST(dtypes, layernorm_constant_row_is_beta_bf16) { check_constant_row<BF16>(ctx); }

/* The error detail names the entry point that failed, per dtype. */
CCML_TEST(dtypes, errors_name_the_dtype_entry_point) {
  ctx.check(ccml_softmax_f16(nullptr, nullptr, 1, 1, CCML_SOFTMAX_AUTO, nullptr) ==
                CCML_ERROR_INVALID_ARGUMENT,
            "null pointers rejected");
  ctx.check(std::string(ccml_last_error_string()).rfind("ccml_softmax_f16", 0) == 0,
            "detail starts with ccml_softmax_f16");
  ctx.check(ccml_matmul_bf16(nullptr, nullptr, nullptr, 1, 1, 1, CCML_MATMUL_AUTO, nullptr) ==
                CCML_ERROR_INVALID_ARGUMENT,
            "null pointers rejected");
  ctx.check(std::string(ccml_last_error_string()).rfind("ccml_matmul_bf16", 0) == 0,
            "detail starts with ccml_matmul_bf16");
  Device16<ccml_bf16> d(4);
  ctx.check(ccml_layernorm_bf16(d.get(), d.get(), nullptr, nullptr, 2, 2, 1e-5f,
                                static_cast<ccml_layernorm_algo>(99),
                                nullptr) == CCML_ERROR_INVALID_ARGUMENT,
            "unknown layernorm algorithm rejected");
  ctx.check(std::string(ccml_last_error_string()).rfind("ccml_layernorm_bf16", 0) == 0,
            "detail starts with ccml_layernorm_bf16");
}
