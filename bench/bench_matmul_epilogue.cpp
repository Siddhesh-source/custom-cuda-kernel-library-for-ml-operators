/* Fused linear epilogue benchmark (docs/optimization-log.md entry 8;
   enhancements F1, F4, B2; reference.md section 5 "fusion vs separate
   kernels"). Timing only; every candidate passes the fp64 gate first.

   Per target shape, the transformer epilogue that shape carries:
     *.ffn_up    C = gelu(A B + bias)
     *.attn_out, *.ffn_down    C = A B + bias + residual
     *.qkv       C = A B + bias
   (the LM head has no bias and is skipped).

   Candidates, all 16-bit in / FP32 accumulate:
     ccml  fused_mma_sync         ccml_matmul_epilogue, MMA_SYNC (fused in the store)
     ccml  fused_split_k          ccml_matmul_epilogue, SPLIT_K (fused in the reduction)
     ccml  mma_sync+separate      ccml_matmul MMA_SYNC, then the elementwise kernel
     cublas gemmex+separate       cublasGemmEx, then the same elementwise kernel
     cublaslt lt_epilogue         cuBLASLt with its own fused epilogue (B2)

   The CSV's flops are 2 M N K (the epilogue's FLOPs are not counted); bytes
   are A + B + C plus bias and residual, the minimum any implementation moves. */

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "bench/baselines/cublas.h"
#include "bench/baselines/epilogue.h"
#include "bench/support/driver.h"
#include "bench/support/metrics.h"
#include "bench/support/typed.h"
#include "ccml/matmul.h"
#include "support/compare.h"
#include "support/random.h"

using namespace ccml_bench;

namespace {

struct Epilogue {
  bool bias = false, residual = false;
  int act = 0; /* 0 none, 2 GELU */
};

bool epilogue_for(const std::string& tag, Epilogue* e) {
  if (tag.find(".ffn_up.") != std::string::npos) {
    e->bias = true;
    e->act = 2;
  } else if (tag.find(".attn_out.") != std::string::npos ||
             tag.find(".ffn_down.") != std::string::npos) {
    e->bias = true;
    e->residual = true;
  } else if (tag.find(".qkv.") != std::string::npos) {
    e->bias = true;
  } else {
    return false;
  }
  return true;
}

template <typename D>
ccml_status fused(typename D::Pub* c, const typename D::Pub* a, const typename D::Pub* b,
                  const typename D::Pub* bias, const typename D::Pub* res, int M, int N, int K,
                  int act, ccml_matmul_algo algo, cudaStream_t s) {
  if constexpr (D::blas == BlasType::F16) {
    return ccml_matmul_epilogue_f16(c, a, b, bias, res, M, N, K, static_cast<ccml_activation>(act),
                                    algo, s);
  } else {
    return ccml_matmul_epilogue_bf16(c, a, b, bias, res, M, N, K,
                                     static_cast<ccml_activation>(act), algo, s);
  }
}

template <typename D>
void run_shape(const Options& options, Emitter& emitter, CublasBaseline& cublas,
               const ShapeEntry& shape, const Epilogue& e, cudaStream_t stream) {
  using Pub = typename D::Pub;
  const int M = static_cast<int>(shape.M), N = static_cast<int>(shape.N),
            K = static_cast<int>(shape.K);
  const double elems = static_cast<double>(shape.M * shape.K + shape.K * shape.N +
                                           2 * shape.M * shape.N + shape.N);
  if (!fits_in_memory(elems * (sizeof(Pub) + 8.0))) {
    emitter.skip(D::name, shape, "insufficient device memory");
    return;
  }
  auto make = [](std::size_t n, std::uint64_t seed, float sd, std::vector<Pub>* stored,
                 std::vector<float>* exact) {
    exact->resize(n);
    ccml_support::Rng rng(seed);
    rng.normal(exact->data(), n, 0.0f, sd);
    stored->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      (*stored)[i] = D::from_float((*exact)[i]);
      (*exact)[i] = D::to_float((*stored)[i]);
    }
  };
  const float scale = std::sqrt(static_cast<float>(K));
  std::vector<Pub> ha, hb, hbias, hres;
  std::vector<float> xa, xb, xbias, xres;
  make(static_cast<std::size_t>(M) * K, 1234, 1.0f, &ha, &xa);
  make(static_cast<std::size_t>(K) * N, 1235, 1.0f, &hb, &xb);
  make(static_cast<std::size_t>(N), 1236, scale, &hbias, &xbias);
  make(static_cast<std::size_t>(M) * N, 1237, scale, &hres, &xres);

  std::vector<double> prod;
  if (!cublas.dgemm_reference(prod, xa, xb, M, N, K)) {
    emitter.skip(D::name, shape, "fp64 reference failed");
    return;
  }
  std::vector<double> ref(prod.size());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    double v = prod[i] + (e.bias ? xbias[i % static_cast<std::size_t>(N)] : 0.0);
    if (e.act == 2) v = 0.5 * v * (1.0 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
    ref[i] = v + (e.residual ? xres[i] : 0.0);
  }
  /* As tests/test_matmul_epilogue.cpp: the matmul's tolerance (scale: the
     product's rms), doubled for the GELU slope, widened for the rounding. */
  ccml_test::Tolerance tol = ccml_test::tolerance_for_f32_dot_product(K, ccml_test::rms(prod));
  tol.abs_tol *= 2.0;
  tol = D::widen(tol);

  DeviceArray<Pub> a(ha.size()), b(hb.size()), bias(hbias.size()), res(hres.size()),
      c(static_cast<std::size_t>(M) * N);
  a.upload(ha);
  b.upload(hb);
  bias.upload(hbias);
  res.upload(hres);
  Pub* const pc = c.data();
  const Pub* const pa = a.data();
  const Pub* const pb = b.data();
  const Pub* const pbias = e.bias ? bias.data() : nullptr;
  const Pub* const pres = e.residual ? res.data() : nullptr;
  const int act = e.act;

  std::vector<Candidate> cands;
  if (options.wants("ccml")) {
    cands.push_back({"ccml", "fused_mma_sync", [=] {
                       return fused<D>(pc, pa, pb, pbias, pres, M, N, K, act,
                                       CCML_MATMUL_MMA_SYNC, stream) == CCML_SUCCESS;
                     }});
    cands.push_back({"ccml", "fused_split_k", [=] {
                       return fused<D>(pc, pa, pb, pbias, pres, M, N, K, act, CCML_MATMUL_SPLIT_K,
                                       stream) == CCML_SUCCESS;
                     }});
    cands.push_back({"ccml", "mma_sync+separate", [=] {
                       return D::run(pc, pa, pb, M, N, K, CCML_MATMUL_MMA_SYNC, stream) ==
                                  CCML_SUCCESS &&
                              apply_epilogue(pc, pbias, pres, M, N, act, D::blas, stream);
                     }});
    cands.push_back({"ccml", "split_k+separate", [=] {
                       return D::run(pc, pa, pb, M, N, K, CCML_MATMUL_SPLIT_K, stream) ==
                                  CCML_SUCCESS &&
                              apply_epilogue(pc, pbias, pres, M, N, act, D::blas, stream);
                     }});
  }
  if (options.wants("cublas")) {
    cands.push_back({"cublas", "gemmex+separate", [&cublas, pc, pa, pb, pbias, pres, M, N, K, act, stream] {
                       return cublas.gemm_ex(pc, pa, pb, M, N, K, D::blas) &&
                              apply_epilogue(pc, pbias, pres, M, N, act, D::blas, stream);
                     }});
  }
  if (options.wants("cublaslt")) {
    const bool ready = cublas.lt_prepare_epilogue(M, N, K, D::blas, pbias, act == 2);
    cands.push_back({"cublaslt", "lt_epilogue", [&cublas, ready, pc, pa, pb, pres] {
                       return ready && cublas.lt_matmul_epilogue(pc, pa, pb, pres);
                     }});
  }

  const auto fetch = [&c](std::vector<float>& got) {
    std::vector<Pub> raw;
    if (!c.download(raw)) return false;
    got.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) got[i] = D::to_float(raw[i]);
    return true;
  };
  /* Unfused pipelines round the product to the storage type before the
     epilogue kernel: one extra rounding of the intermediate, at most
     unit * |A B| (round to nearest), carried through the activation (GELU
     slope <= 1.13). They are gated with that per-element allowance added.
     cuBLASLt's epilogue gets it too: measured on sm_89, its error equals the
     unfused pipelines' exactly (it rounds the product before the epilogue;
     docs/optimization-log.md entry 8). ccml's fused kernels are held to the
     strict tolerance. */
  std::vector<double> extra(prod.size());
  for (std::size_t i = 0; i < extra.size(); ++i) {
    extra[i] = 1.2 * D::unit_roundoff * std::fabs(prod[i]);
  }
  std::vector<Gated> gated;
  for (const Candidate& cand : cands) {
    if (cudaMemset(c.data(), 0xFF, c.count() * sizeof(Pub)) != cudaSuccess) {
      throw std::runtime_error("cudaMemset failed");
    }
    const bool unfused =
        cand.algo.find("+separate") != std::string::npos || cand.impl == "cublaslt";
    gated.push_back(
        gate(cand, fetch, ref, tol, stream, unfused ? extra : std::vector<double>()));
  }
  const double bytes = (static_cast<double>(M) * K + static_cast<double>(K) * N +
                        static_cast<double>(M) * N * (e.residual ? 2 : 1) + (e.bias ? N : 0)) *
                       sizeof(Pub);
  emitter.emit(D::name, shape, matmul_flops(M, N, K), bytes, gated, stream);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    /* f32 has no fused path; the default --dtype f32 means f16 here. */
    if (options.dtypes.size() == 1 && options.dtypes[0] == "f32") options.dtypes = {"f16"};
    Emitter emitter(options, "matmul_epilogue");
    cudaStream_t stream = nullptr;
    CublasBaseline cublas(stream);
    for (const ShapeEntry& shape : load_shapes(options.shapes, "matmul")) {
      Epilogue e;
      if (!tag_matches(shape, options.filter) || !epilogue_for(shape.tag, &e)) continue;
      for (const std::string& dtype : options.dtypes) {
        if (dtype == "f16") run_shape<F16>(options, emitter, cublas, shape, e, stream);
        if (dtype == "bf16") run_shape<BF16>(options, emitter, cublas, shape, e, stream);
      }
    }
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "ccml_bench_matmul_epilogue: %s\n", ex.what());
    return 1;
  }
  return 0;
}
