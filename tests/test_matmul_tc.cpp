/* Tensor Core matmul variants (reference.md §3 matmul ladder, Tensor Cores;
   docs/optimization-log.md entries 4 onward). 16-bit input, FP32 accumulate.

   Method as in tests/test_dtypes.cpp: inputs are rounded to the storage type
   and the fp64 reference is computed from the ROUNDED values, so the only
   error to bound is the kernel's FP32 accumulation plus one rounding on store
   (support/compare.h widen_for_storage; docs/correctness.md).

   Every output buffer is poisoned with 0xFF bytes (NaN in FP16 and BF16)
   before each call, so an element a kernel fails to write cannot pass.

   BF16 Tensor Cores need sm_80+. On the T4 every BF16 case returns
   CCML_ERROR_NOT_IMPLEMENTED and the test reports SKIP, never PASS
   (docs/local.md L1); the BF16 cases run on the local sm_89 GPU. */

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ccml/matmul.h"
#include "ccml/memory.h"
#include "ccml/status.h"
#include "ccml/tuning.h"
#include "support/compare.h"
#include "support/random.h"
#include "support/reference.h"
#include "tests/support/harness.h"

using ccml_test::Tolerance;

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
  static ccml_status run(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                         ccml_matmul_algo algo) {
    return ccml_matmul_f16(c, a, b, m, n, k, algo, nullptr);
  }
  static ccml_status tuned(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                           ccml_matmul_algo algo, int index) {
    return ccml_matmul_tuned_f16(c, a, b, m, n, k, algo, index, nullptr);
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
  static ccml_status run(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                         ccml_matmul_algo algo) {
    return ccml_matmul_bf16(c, a, b, m, n, k, algo, nullptr);
  }
  static ccml_status tuned(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                           ccml_matmul_algo algo, int index) {
    return ccml_matmul_tuned_bf16(c, a, b, m, n, k, algo, index, nullptr);
  }
};

/* Every Tensor Core variant; a new one is added here and inherits every test. */
const ccml_matmul_algo kTcAlgos[] = {CCML_MATMUL_WMMA, CCML_MATMUL_MMA_SYNC, CCML_MATMUL_SPLIT_K,
                                     CCML_MATMUL_SPLIT_K_ATOMIC};

/* Owning device array of 16-bit elements, with `extra` elements of slack so a
   view can start at an offset. */
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
  bool poison() {
    return ccml_device_memset_async(ptr_, 0xFF, n_ * sizeof(Pub), nullptr) == CCML_SUCCESS;
  }

 private:
  void*       ptr_ = nullptr;
  std::size_t n_ = 0;
};

/* One problem: rounded inputs, their fp64 product, and the tolerance. */
template <typename D>
struct Problem {
  std::vector<typename D::Pub> a, b;
  std::vector<double> ref;
  Tolerance tol;
};

template <typename D>
Problem<D> make_problem(int M, int N, int K, std::uint64_t seed) {
  Problem<D> p;
  std::vector<float> a_raw(static_cast<std::size_t>(M) * K), b_raw(static_cast<std::size_t>(K) * N);
  ccml_support::Rng rng(seed);
  rng.normal(a_raw.data(), a_raw.size(), 0.0f, 1.0f);
  rng.normal(b_raw.data(), b_raw.size(), 0.0f, 1.0f);
  p.a.resize(a_raw.size());
  p.b.resize(b_raw.size());
  for (std::size_t i = 0; i < a_raw.size(); ++i) {
    p.a[i] = D::from_float(a_raw[i]);
    a_raw[i] = D::to_float(p.a[i]);
  }
  for (std::size_t i = 0; i < b_raw.size(); ++i) {
    p.b[i] = D::from_float(b_raw[i]);
    b_raw[i] = D::to_float(p.b[i]);
  }
  ccml_test::matmul_reference(p.ref, a_raw, b_raw, M, N, K);
  p.tol = ccml_test::widen_for_storage(
      ccml_test::tolerance_for_f32_dot_product(K, ccml_test::rms(p.ref)), D::unit, D::floor);
  return p;
}

/* Runs `call(c, a, b)` on a poisoned output, starting every pointer `offset`
   elements into its buffer, and compares. Returns the normalized error; fails
   the test on a mismatch. A NOT_IMPLEMENTED status skips the whole test. */
template <typename D, typename Call>
double check(const ccml_test::Context& ctx, int M, int N, int K, int offset, std::uint64_t seed,
             const std::string& what, Call call) {
  using Pub = typename D::Pub;
  const Problem<D> p = make_problem<D>(M, N, K, seed);
  const std::size_t off = static_cast<std::size_t>(offset);
  Device16<Pub> da(p.a.size() + off), db(p.b.size() + off),
      dc(static_cast<std::size_t>(M) * N + off);
  std::vector<Pub> a_host(off, D::from_float(0.0f)), b_host(off, D::from_float(0.0f));
  a_host.insert(a_host.end(), p.a.begin(), p.a.end());
  b_host.insert(b_host.end(), p.b.begin(), p.b.end());
  ctx.check(da.ok() && db.ok() && dc.ok() && da.upload(a_host) && db.upload(b_host) &&
                dc.poison(),
            "device setup");
  const ccml_status s = call(dc.get() + off, da.get() + off, db.get() + off);
  if (s == CCML_ERROR_NOT_IMPLEMENTED) ctx.skip(what + ": " + ccml_last_error_string());
  if (s != CCML_SUCCESS) {
    ctx.fail(what + " -> " + ccml_status_string(s) + " (" + ccml_last_error_string() + ")");
  }
  CCML_ASSERT_OK(ccml_device_synchronize());
  std::vector<Pub> c_all;
  ctx.check(dc.download(c_all), "download");
  std::vector<float> got(c_all.size() - off);
  for (std::size_t i = 0; i < got.size(); ++i) got[i] = D::to_float(c_all[i + off]);
  const ccml_test::ErrorStats stats = ccml_test::compare(got, p.ref, p.tol);
  if (!ccml_test::within(stats)) {
    ctx.fail(std::string(D::name) + " " + what + " M=" + std::to_string(M) +
             " N=" + std::to_string(N) + " K=" + std::to_string(K) + " offset=" +
             std::to_string(offset) + " -- " + p.tol.to_string() + " -- " + stats.to_string());
  }
  return stats.max_normalized_error;
}

std::string tc_name(int index) {
  ccml_matmul_tc_config c;
  ccml_matmul_tc_config_get(index, &c);
  return std::to_string(c.bm) + "x" + std::to_string(c.bn) + "x" + std::to_string(c.bk) +
         " warp " + std::to_string(c.wm) + "x" + std::to_string(c.wn);
}

/* No dimension a multiple of any tile, the degenerate ones (a single row, a
   single column, K = 1), and shapes whose N and K are multiples of 8 so the
   128-bit path runs too. */
const int kAwkward[][3] = {{1, 1, 1},       {1, 512, 512},  {512, 1, 512},   {512, 512, 1},
                           {17, 33, 65},    {131, 67, 97},  {128, 128, 128}, {255, 257, 129},
                           {3, 768, 768},   {1000, 1000, 33}, {2047, 129, 513},
                           /* N, K multiples of 8: the vector path */
                           {131, 136, 136}, {1, 2304, 768}, {257, 520, 264}, {7, 8, 8},
                           {16, 128, 32},   {130, 264, 72}};

template <typename D>
void every_config_on_awkward_shapes(const ccml_test::Context& ctx) {
  for (ccml_matmul_algo algo : kTcAlgos) {
    for (int index = 0; index < ccml_matmul_tc_config_count(); ++index) {
      double worst = 0.0;
      for (const auto& s : kAwkward) {
        const std::string what = std::string(ccml_matmul_algo_name(algo)) + " config " +
                                 std::to_string(index) + " (" + tc_name(index) + ")";
        worst = std::max(worst, check<D>(ctx, s[0], s[1], s[2], 0,
                                         5000 + static_cast<std::uint64_t>(index) * 31 + s[0] +
                                             s[1] + s[2],
                                         what, [&](auto* c, const auto* a, const auto* b) {
                                           return D::tuned(c, a, b, s[0], s[1], s[2], algo,
                                                           index);
                                         }));
      }
      ctx.note(std::string(D::name) + " " + ccml_matmul_algo_name(algo) + " config " +
               std::to_string(index) + " (" + tc_name(index) +
               "): worst normalized error " + std::to_string(worst));
    }
  }
}

/* The pipeline alternates two buffers per BK = 32 step: one step, two, odd
   counts, a ragged last step, and K < 16 (a partial MMA k-slice). */
template <typename D>
void every_k_step_count(const ccml_test::Context& ctx) {
  for (ccml_matmul_algo algo : kTcAlgos) {
    for (int K : {1, 7, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 97, 1000}) {
      for (int N : {136, 135}) { /* vector path and scalar path */
        check<D>(ctx, 67, N, K, 0, 6000 + static_cast<std::uint64_t>(K) + N,
                 std::string(ccml_matmul_algo_name(algo)) + " k-steps",
                 [&](auto* c, const auto* a, const auto* b) {
                   return D::run(c, a, b, 67, N, K, algo);
                 });
      }
    }
  }
}

/* Pointers one element into their buffers (a sub-tensor view) are not 16-byte
   aligned: the scalar global path must run, and be right. */
template <typename D>
void misaligned_views(const ccml_test::Context& ctx) {
  for (ccml_matmul_algo algo : kTcAlgos) {
    for (int offset : {1, 2, 4}) {
      check<D>(ctx, 64, 128, 96, offset, 7000 + static_cast<std::uint64_t>(offset),
               std::string(ccml_matmul_algo_name(algo)) + " misaligned",
               [&](auto* c, const auto* a, const auto* b) {
                 return D::run(c, a, b, 64, 128, 96, algo);
               });
    }
  }
}

/* Target shapes through the algo enum (configuration 0). */
template <typename D>
void target_shapes(const ccml_test::Context& ctx) {
  const int shapes[][3] = {{128, 768, 768}, {1, 2304, 768}, {16, 3072, 768}, {256, 2304, 768},
                           {64, 768, 3072}};
  for (ccml_matmul_algo algo : kTcAlgos) {
    for (const auto& s : shapes) {
      const double e = check<D>(ctx, s[0], s[1], s[2], 0,
                                8000 + static_cast<std::uint64_t>(s[0]) + s[1],
                                std::string(ccml_matmul_algo_name(algo)) + " enum",
                                [&](auto* c, const auto* a, const auto* b) {
                                  return D::run(c, a, b, s[0], s[1], s[2], algo);
                                });
      ctx.note(std::string(D::name) + " " + ccml_matmul_algo_name(algo) + " " +
               std::to_string(s[0]) + "x" + std::to_string(s[1]) + "x" + std::to_string(s[2]) +
               ": normalized error " + std::to_string(e));
    }
  }
}

}  // namespace

#define CCML_TC_TESTS(D)                                                                     \
  CCML_TEST(matmul_tc, D##_every_config_on_awkward_shapes) {                                 \
    every_config_on_awkward_shapes<D>(ctx);                                                  \
  }                                                                                          \
  CCML_TEST(matmul_tc, D##_every_k_step_count) { every_k_step_count<D>(ctx); }               \
  CCML_TEST(matmul_tc, D##_misaligned_views) { misaligned_views<D>(ctx); }                   \
  CCML_TEST(matmul_tc, D##_target_shapes) { target_shapes<D>(ctx); }

CCML_TC_TESTS(F16)
CCML_TC_TESTS(BF16)

/* FP32 input has no Tensor Core path on sm_75: refused, never silently
   rounded to FP16. */
CCML_TEST(matmul_tc, f32_input_is_not_implemented) {
  float* d = nullptr;
  CCML_ASSERT_OK(ccml_device_malloc(reinterpret_cast<void**>(&d), 4 * sizeof(float)));
  for (ccml_matmul_algo algo : kTcAlgos) {
    const ccml_status s = ccml_matmul_f32(d, d, d, 2, 2, 1, algo, nullptr);
    ctx.check(s == CCML_ERROR_NOT_IMPLEMENTED,
              std::string(ccml_matmul_algo_name(algo)) + " f32 -> " + ccml_status_string(s));
  }
  ccml_device_free(d);
}

CCML_TEST(matmul_tc, config_api_rejects_bad_input) {
  ccml_matmul_tc_config cfg;
  ctx.check(ccml_matmul_tc_config_get(-1, &cfg) == CCML_ERROR_INVALID_ARGUMENT, "index -1");
  ctx.check(ccml_matmul_tc_config_get(ccml_matmul_tc_config_count(), &cfg) ==
                CCML_ERROR_INVALID_ARGUMENT,
            "index == count");
  ctx.check(ccml_matmul_tc_config_get(0, nullptr) == CCML_ERROR_INVALID_ARGUMENT, "null out");
  for (int i = 0; i < ccml_matmul_tc_config_count(); ++i) {
    CCML_ASSERT_OK(ccml_matmul_tc_config_get(i, &cfg));
    ctx.check(cfg.warps == (cfg.bm / cfg.wm) * (cfg.bn / cfg.wn) && cfg.warps * 32 <= 1024,
              "warp count consistent and launchable");
  }
  ccml_f16* d = nullptr;
  CCML_ASSERT_OK(ccml_device_malloc(reinterpret_cast<void**>(&d), 8 * sizeof(ccml_f16)));
  for (ccml_matmul_algo algo : kTcAlgos) {
    ctx.check(ccml_matmul_tuned_f16(d, d, d, 2, 2, 1, algo, ccml_matmul_tc_config_count(),
                                    nullptr) == CCML_ERROR_INVALID_ARGUMENT,
              "config index out of range is rejected, not clamped");
  }
  ccml_device_free(d);
}

/* Enhancement C3: SPLIT_K reduces in a fixed order, so the same input gives
   the same bits every run; SPLIT_K_ATOMIC does not promise that (atomicAdd
   order varies) and is only required to be within tolerance, which the shape
   tests above check. A decode shape with a long K, so several slices run. */
CCML_TEST(matmul_tc, split_k_is_deterministic_atomic_is_marked) {
  const int M = 16, N = 768, K = 3072;
  const Problem<F16> p = make_problem<F16>(M, N, K, 9100);
  Device16<ccml_f16> da(p.a.size()), db(p.b.size()), dc(static_cast<std::size_t>(M) * N);
  ctx.check(da.ok() && db.ok() && dc.ok() && da.upload(p.a) && db.upload(p.b), "device setup");
  for (ccml_matmul_algo algo : {CCML_MATMUL_SPLIT_K, CCML_MATMUL_SPLIT_K_ATOMIC}) {
    std::vector<ccml_f16> first, again;
    int differing_runs = 0;
    for (int run = 0; run < 20; ++run) {
      ctx.check(dc.poison(), "poison");
      CCML_ASSERT_OK_OR_SKIP(F16::run(dc.get(), da.get(), db.get(), M, N, K, algo));
      CCML_ASSERT_OK(ccml_device_synchronize());
      ctx.check(dc.download(run == 0 ? first : again), "download");
      if (run > 0 && std::memcmp(first.data(), again.data(), first.size() * sizeof(ccml_f16)) != 0) {
        ++differing_runs;
      }
    }
    ctx.note(std::string(ccml_matmul_algo_name(algo)) + ": " + std::to_string(differing_runs) +
             " of 19 repeat runs differed bitwise from the first");
    if (algo == CCML_MATMUL_SPLIT_K) {
      ctx.check(differing_runs == 0, "SPLIT_K must be bitwise deterministic");
    }
  }
}
