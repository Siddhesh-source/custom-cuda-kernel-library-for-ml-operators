/* Matmul benchmark: every ccml variant against cuBLAS and cuBLASLt, on every
   shape in bench/shapes.json, for each requested storage type (reference.md
   §3, §4, §5 dtype comparison, §6). Timing only; the in-run correctness gate
   decides what may be timed (reference.md §11).

   FP16/BF16: inputs are generated in float, rounded to the storage type, and
   the fp64 DGEMM reference is computed from the ROUNDED values, as in
   tests/test_dtypes.cpp; the tolerance is the f32 one widened for the final
   rounding (support/compare.h, docs/correctness.md). The vendor baseline is
   cublasGemmEx / cuBLASLt with 16-bit storage and FP32 accumulation, the same
   contract as ccml_matmul_f16/bf16. */

#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "bench/baselines/cublas.h"
#include "bench/support/driver.h"
#include "bench/support/metrics.h"
#include "ccml/matmul.h"
#include "ccml/status.h"
#include "ccml/tuning.h"
#include "support/compare.h"
#include "support/random.h"

using namespace ccml_bench;

namespace {

/* ------------------------------------------------------- dtype traits -- */

struct F32 {
  using Pub = float;
  static constexpr const char* name = "f32";
  static constexpr BlasType blas = BlasType::F32;
  static Pub from_float(float f) { return f; }
  static float to_float(Pub p) { return p; }
  static ccml_status run(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                         ccml_matmul_algo algo, cudaStream_t s) {
    return ccml_matmul_f32(c, a, b, m, n, k, algo, s);
  }
  static ccml_status tuned(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                           ccml_matmul_algo algo, int index, cudaStream_t s) {
    return ccml_matmul_tuned_f32(c, a, b, m, n, k, algo, index, s);
  }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) { return t; }
};

struct F16 {
  using Pub = ccml_f16;
  static constexpr const char* name = "f16";
  static constexpr BlasType blas = BlasType::F16;
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
                         ccml_matmul_algo algo, cudaStream_t s) {
    return ccml_matmul_f16(c, a, b, m, n, k, algo, s);
  }
  static ccml_status tuned(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                           ccml_matmul_algo algo, int index, cudaStream_t s) {
    return ccml_matmul_tuned_f16(c, a, b, m, n, k, algo, index, s);
  }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) {
    return ccml_test::widen_for_storage(t, ccml_test::kF16UnitRoundoff,
                                        ccml_test::kF16SubnormalStep);
  }
};

struct BF16 {
  using Pub = ccml_bf16;
  static constexpr const char* name = "bf16";
  static constexpr BlasType blas = BlasType::BF16;
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
                         ccml_matmul_algo algo, cudaStream_t s) {
    return ccml_matmul_bf16(c, a, b, m, n, k, algo, s);
  }
  static ccml_status tuned(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                           ccml_matmul_algo algo, int index, cudaStream_t s) {
    return ccml_matmul_tuned_bf16(c, a, b, m, n, k, algo, index, s);
  }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) {
    return ccml_test::widen_for_storage(t, ccml_test::kBF16UnitRoundoff,
                                        1.1754943508222875e-38 /* FLT_MIN */);
  }
};

/* Owning device array of any element type. */
template <typename T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t n) : n_(n) {
    if (cudaMalloc(&ptr_, n * sizeof(T)) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
  }
  ~DeviceArray() { cudaFree(ptr_); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  T* data() { return static_cast<T*>(ptr_); }
  std::size_t count() const { return n_; }
  void upload(const std::vector<T>& h) {
    if (cudaMemcpy(ptr_, h.data(), n_ * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("upload failed");
    }
  }
  bool download(std::vector<T>& h) const {
    h.resize(n_);
    return cudaMemcpy(h.data(), ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }

 private:
  void*       ptr_ = nullptr;
  std::size_t n_ = 0;
};

/* Every variant the bench times through the algo enum. */
const ccml_matmul_algo kAlgos[] = {CCML_MATMUL_NAIVE,          CCML_MATMUL_TILED,
                                   CCML_MATMUL_REGISTER_TILED, CCML_MATMUL_VECTORIZED,
                                   CCML_MATMUL_DOUBLE_BUFFERED, CCML_MATMUL_WMMA,
                                   CCML_MATMUL_MMA_SYNC,        CCML_MATMUL_SPLIT_K,
                                   CCML_MATMUL_SPLIT_K_ATOMIC, CCML_MATMUL_MMA_SYNC_SWIZZLED};

/* The Tensor Core family and its sweep-row prefix. f32 rows report NOT_RUN
   (no FP32 Tensor Core path on sm_75). */
const std::pair<ccml_matmul_algo, const char*> kTcFamily[] = {{CCML_MATMUL_WMMA, "wmma_"},
                                                              {CCML_MATMUL_MMA_SYNC, "mma_"},
                                                              {CCML_MATMUL_SPLIT_K, "splitk_"}};

template <typename D>
void run_shape(const Options& options, Emitter& emitter, CublasBaseline& cublas,
               const ShapeEntry& shape, cudaStream_t stream) {
  using Pub = typename D::Pub;
  const int M = static_cast<int>(shape.M), N = static_cast<int>(shape.N),
            K = static_cast<int>(shape.K);
  const double elems =
      static_cast<double>(shape.M * shape.K + shape.K * shape.N + shape.M * shape.N);
  /* Operands + fp64 copies for the DGEMM gate. */
  if (!fits_in_memory(elems * (sizeof(Pub) + 8.0))) {
    emitter.skip(D::name, shape, "insufficient device memory");
    return;
  }

  std::vector<float> raw_a(static_cast<std::size_t>(M) * K);
  std::vector<float> raw_b(static_cast<std::size_t>(K) * N);
  ccml_support::Rng rng(1234);
  rng.normal(raw_a.data(), raw_a.size(), 0.0f, 1.0f);
  rng.normal(raw_b.data(), raw_b.size(), 0.0f, 1.0f);
  /* Round to the storage type; the reference sees exactly the stored values. */
  std::vector<Pub> host_a(raw_a.size()), host_b(raw_b.size());
  for (std::size_t i = 0; i < raw_a.size(); ++i) {
    host_a[i] = D::from_float(raw_a[i]);
    raw_a[i] = D::to_float(host_a[i]);
  }
  for (std::size_t i = 0; i < raw_b.size(); ++i) {
    host_b[i] = D::from_float(raw_b[i]);
    raw_b[i] = D::to_float(host_b[i]);
  }

  std::vector<double> reference;
  if (!cublas.dgemm_reference(reference, raw_a, raw_b, M, N, K)) {
    emitter.skip(D::name, shape, "fp64 reference failed");
    return;
  }
  const ccml_test::Tolerance tolerance =
      D::widen(ccml_test::tolerance_for_f32_dot_product(K, ccml_test::rms(reference)));

  DeviceArray<Pub> a(host_a.size()), b(host_b.size()), c(static_cast<std::size_t>(M) * N);
  a.upload(host_a);
  b.upload(host_b);
  Pub* const pc = c.data();
  const Pub* const pa = a.data();
  const Pub* const pb = b.data();

  std::vector<Candidate> candidates;
  const double gflop = matmul_flops(M, N, K) / 1e9;
  if (options.wants("ccml")) {
    for (ccml_matmul_algo algo : kAlgos) {
      /* Over the cap the naive kernel is reported NOT_RUN, never timed. */
      const bool skip = algo == CCML_MATMUL_NAIVE && options.naive_max_gflop > 0.0 &&
                        gflop > options.naive_max_gflop;
      candidates.push_back({"ccml", ccml_matmul_algo_name(algo), [=] {
                              return !skip && D::run(pc, pa, pb, M, N, K, algo, stream) ==
                                                  CCML_SUCCESS;
                            }});
    }
  }
  /* Tile sweep: every configuration of every tiled variant as its own row,
     named <variant prefix>_<tile shape> ("rt_" register_tiled, "vec_"
     vectorized, "db_" double_buffered). Index 0 duplicates the enum path on
     purpose, so the two can be cross-checked in one table. An unsupported
     pairing reports NOT_RUN. */
  if (options.wants("ccml_tiles") && options.tile_sweep) {
    const std::pair<ccml_matmul_algo, const char*> family[] = {
        {CCML_MATMUL_REGISTER_TILED, "rt_"},
        {CCML_MATMUL_VECTORIZED, "vec_"},
        {CCML_MATMUL_DOUBLE_BUFFERED, "db_"}};
    for (const auto& [variant, prefix] : family) {
      for (int index = 0; index < ccml_matmul_tile_config_count(); ++index) {
        ccml_matmul_tile_config t;
        ccml_matmul_tile_config_get(index, &t);
        const std::string name = prefix + std::to_string(t.bm) + "x" + std::to_string(t.bn) +
                                 "x" + std::to_string(t.bk) + "_" + std::to_string(t.tm) + "x" +
                                 std::to_string(t.tn);
        const ccml_matmul_algo v = variant;
        candidates.push_back({"ccml_tiles", name, [=] {
                                return D::tuned(pc, pa, pb, M, N, K, v, index, stream) ==
                                       CCML_SUCCESS;
                              }});
      }
    }
    /* Tensor Core configurations: <prefix><block tile>_w<warp tile>. */
    if (D::blas != BlasType::F32) {
      for (const auto& [variant, prefix] : kTcFamily) {
        for (int index = 0; index < ccml_matmul_tc_config_count(); ++index) {
          ccml_matmul_tc_config t;
          ccml_matmul_tc_config_get(index, &t);
          const std::string name = prefix + std::to_string(t.bm) + "x" + std::to_string(t.bn) +
                                   "x" + std::to_string(t.bk) + "_w" + std::to_string(t.wm) +
                                   "x" + std::to_string(t.wn);
          const ccml_matmul_algo v = variant;
          candidates.push_back({"ccml_tiles", name, [=] {
                                  return D::tuned(pc, pa, pb, M, N, K, v, index, stream) ==
                                         CCML_SUCCESS;
                                }});
        }
      }
    }
  }
  if (options.wants("cublas")) {
    if (D::blas == BlasType::F32) {
      candidates.push_back({"cublas", "sgemm", [&cublas, pc, pa, pb, M, N, K] {
                              return cublas.sgemm(reinterpret_cast<float*>(pc),
                                                  reinterpret_cast<const float*>(pa),
                                                  reinterpret_cast<const float*>(pb), M, N, K);
                            }});
    } else {
      candidates.push_back({"cublas", "gemmex", [&cublas, pc, pa, pb, M, N, K] {
                              return cublas.gemm_ex(pc, pa, pb, M, N, K, D::blas);
                            }});
    }
  }
  if (options.wants("cublaslt")) {
    const bool lt_ready = cublas.lt_prepare(M, N, K, D::blas);
    candidates.push_back({"cublaslt", "heuristic_top1", [&cublas, lt_ready, pc, pa, pb] {
                            return lt_ready && cublas.lt_matmul(pc, pa, pb);
                          }});
  }

  const auto fetch = [&c](std::vector<float>& got) {
    std::vector<Pub> raw;
    if (!c.download(raw)) return false;
    got.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) got[i] = D::to_float(raw[i]);
    return true;
  };
  std::vector<Gated> gated;
  for (const Candidate& candidate : candidates) {
    /* Poison the output: 0xFF bytes are NaN in all three types, so an element
       a candidate fails to write cannot pass on a previous candidate's data. */
    if (cudaMemset(c.data(), 0xFF, c.count() * sizeof(Pub)) != cudaSuccess) {
      throw std::runtime_error("cudaMemset failed");
    }
    gated.push_back(gate(candidate, fetch, reference, tolerance, stream));
  }
  emitter.emit(D::name, shape, matmul_flops(M, N, K),
               matmul_bytes(M, N, K, static_cast<int>(sizeof(Pub))), gated, stream);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Emitter emitter(options, "matmul");
    cudaStream_t stream = nullptr;
    CublasBaseline cublas(stream);

    for (const ShapeEntry& shape : load_shapes(options.shapes, "matmul")) {
      if (!tag_matches(shape, options.filter)) continue;
      for (const std::string& dtype : options.dtypes) {
        if (dtype == "f32") run_shape<F32>(options, emitter, cublas, shape, stream);
        if (dtype == "f16") run_shape<F16>(options, emitter, cublas, shape, stream);
        if (dtype == "bf16") run_shape<BF16>(options, emitter, cublas, shape, stream);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ccml_bench_matmul: %s\n", e.what());
    return 1;
  }
  return 0;
}
