#include "bench/baselines/cublas.h"

#include <stdexcept>
#include <string>

#include <cublasLt.h>
#include <cublas_v2.h>

namespace ccml_bench {

namespace {

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

void check_blas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed with cublasStatus " +
                             std::to_string(static_cast<int>(status)));
  }
}

cublasHandle_t blas(void* h) { return static_cast<cublasHandle_t>(h); }
cublasLtHandle_t lt(void* h) { return static_cast<cublasLtHandle_t>(h); }

cudaDataType_t data_type(BlasType t) {
  switch (t) {
    case BlasType::F16: return CUDA_R_16F;
    case BlasType::BF16: return CUDA_R_16BF;
    default: return CUDA_R_32F;
  }
}

}  // namespace

struct CublasBaseline::LtPlan {
  cublasLtMatmulDesc_t   op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t   algo{};
};

CublasBaseline::CublasBaseline(cudaStream_t stream) : stream_(stream) {
  cublasHandle_t h = nullptr;
  check_blas(cublasCreate(&h), "cublasCreate");
  handle_ = h;
  check_blas(cublasSetStream(h, stream_), "cublasSetStream");
  /* Pinned to the default math mode: FP32 in, FP32 compute, no TF32, so the
     comparison against FP32 CUDA-core kernels is like for like. */
  check_blas(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");

  cublasLtHandle_t l = nullptr;
  check_blas(cublasLtCreate(&l), "cublasLtCreate");
  lt_handle_ = l;
  check_cuda(cudaMalloc(&workspace_, workspace_bytes_), "cudaMalloc(cublasLt workspace)");
}

CublasBaseline::~CublasBaseline() {
  release_lt();
  cudaFree(workspace_);
  if (lt_handle_ != nullptr) cublasLtDestroy(lt(lt_handle_));
  if (handle_ != nullptr) cublasDestroy(blas(handle_));
}

bool CublasBaseline::sgemm(float* C, const float* A, const float* B, int M, int N,
                           int K) {
  const float alpha = 1.0f, beta = 0.0f;
  return cublasSgemm(blas(handle_), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                     N, A, K, &beta, C, N) == CUBLAS_STATUS_SUCCESS;
}

bool CublasBaseline::gemm_ex(void* C, const void* A, const void* B, int M, int N, int K,
                             BlasType type) {
  const float alpha = 1.0f, beta = 0.0f;
  const cudaDataType_t t = data_type(type);
  return cublasGemmEx(blas(handle_), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, t, N, A, t,
                      K, &beta, C, t, N, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
}

void CublasBaseline::release_lt() {
  if (lt_ == nullptr) return;
  if (lt_->a) cublasLtMatrixLayoutDestroy(lt_->a);
  if (lt_->b) cublasLtMatrixLayoutDestroy(lt_->b);
  if (lt_->c) cublasLtMatrixLayoutDestroy(lt_->c);
  if (lt_->op) cublasLtMatmulDescDestroy(lt_->op);
  delete lt_;
  lt_ = nullptr;
}

bool CublasBaseline::lt_prepare(int M, int N, int K, BlasType type) {
  release_lt();
  lt_ = new LtPlan();
  const cudaDataType_t t = data_type(type);
  try {
    /* Scale type FP32 (alpha and beta are floats) with FP32 compute. */
    check_blas(cublasLtMatmulDescCreate(&lt_->op, CUBLAS_COMPUTE_32F, CUDA_R_32F),
               "cublasLtMatmulDescCreate");
    /* Column-major view: first operand B^T [N,K], second A^T [K,M], out [N,M]. */
    check_blas(cublasLtMatrixLayoutCreate(&lt_->b, t, N, K, N), "layout B");
    check_blas(cublasLtMatrixLayoutCreate(&lt_->a, t, K, M, K), "layout A");
    check_blas(cublasLtMatrixLayoutCreate(&lt_->c, t, N, M, N), "layout C");

    cublasLtMatmulPreference_t pref = nullptr;
    check_blas(cublasLtMatmulPreferenceCreate(&pref), "preference");
    check_blas(cublasLtMatmulPreferenceSetAttribute(
                   pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_,
                   sizeof(workspace_bytes_)),
               "preference workspace");
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        lt(lt_handle_), lt_->op, lt_->b, lt_->a, lt_->c, lt_->c, pref, 1, &result,
        &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (status != CUBLAS_STATUS_SUCCESS || found == 0) {
      release_lt();
      return false;
    }
    lt_->algo = result.algo;
    return true;
  } catch (const std::exception&) {
    release_lt();
    return false;
  }
}

bool CublasBaseline::lt_matmul(void* C, const void* A, const void* B) {
  if (lt_ == nullptr) return false;
  const float alpha = 1.0f, beta = 0.0f;
  return cublasLtMatmul(lt(lt_handle_), lt_->op, &alpha, B, lt_->b, A, lt_->a,
                        &beta, C, lt_->c, C, lt_->c, &lt_->algo, workspace_,
                        workspace_bytes_, stream_) == CUBLAS_STATUS_SUCCESS;
}

bool CublasBaseline::dgemm_reference(std::vector<double>& C,
                                     const std::vector<float>& A,
                                     const std::vector<float>& B, int M, int N,
                                     int K) {
  const std::vector<double> a(A.begin(), A.end());
  const std::vector<double> b(B.begin(), B.end());
  C.assign(static_cast<std::size_t>(M) * N, 0.0);
  double *da = nullptr, *db = nullptr, *dc = nullptr;
  bool ok = false;
  try {
    check_cuda(cudaMalloc(&da, a.size() * sizeof(double)), "cudaMalloc A64");
    check_cuda(cudaMalloc(&db, b.size() * sizeof(double)), "cudaMalloc B64");
    check_cuda(cudaMalloc(&dc, C.size() * sizeof(double)), "cudaMalloc C64");
    check_cuda(cudaMemcpy(da, a.data(), a.size() * sizeof(double),
                          cudaMemcpyHostToDevice), "upload A64");
    check_cuda(cudaMemcpy(db, b.data(), b.size() * sizeof(double),
                          cudaMemcpyHostToDevice), "upload B64");
    const double alpha = 1.0, beta = 0.0;
    check_blas(cublasDgemm(blas(handle_), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           db, N, da, K, &beta, dc, N), "cublasDgemm");
    check_cuda(cudaStreamSynchronize(stream_), "sync DGEMM");
    check_cuda(cudaMemcpy(C.data(), dc, C.size() * sizeof(double),
                          cudaMemcpyDeviceToHost), "download C64");
    ok = true;
  } catch (const std::exception&) {
    ok = false;
  }
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return ok;
}

}  // namespace ccml_bench
