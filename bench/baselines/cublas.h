#ifndef CCML_BENCH_BASELINES_CUBLAS_H
#define CCML_BENCH_BASELINES_CUBLAS_H

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

namespace ccml_bench {

/*
 * cuBLAS vendor baselines for row-major FP32 matmul (reference.md §4), and an
 * fp64 DGEMM used as the correctness reference for the benchmark gate on shapes
 * too large for the CPU oracle.
 *
 * Linked into bench/ only. The library itself depends on the CUDA runtime and
 * nothing else.
 *
 * Row-major C[M,N] = A[M,K] B[K,N] is computed as column-major
 * C^T[N,M] = B^T[N,K] A^T[K,M], so no transposes or copies are needed.
 */
class CublasBaseline {
 public:
  explicit CublasBaseline(cudaStream_t stream);
  ~CublasBaseline();
  CublasBaseline(const CublasBaseline&) = delete;
  CublasBaseline& operator=(const CublasBaseline&) = delete;

  /* cublasSgemm, default math mode (no TF32). */
  bool sgemm(float* C, const float* A, const float* B, int M, int N, int K);

  /* cublasLt with its top-1 heuristic algorithm for this shape. prepare()
     runs the heuristic once; lt_matmul() then launches with that algorithm. */
  bool lt_prepare(int M, int N, int K);
  bool lt_matmul(float* C, const float* A, const float* B);

  /* fp64 reference: host float inputs, host double output. */
  bool dgemm_reference(std::vector<double>& C, const std::vector<float>& A,
                       const std::vector<float>& B, int M, int N, int K);

 private:
  struct LtPlan;
  void release_lt();

  cudaStream_t stream_;
  void*        handle_ = nullptr;    /* cublasHandle_t */
  void*        lt_handle_ = nullptr; /* cublasLtHandle_t */
  LtPlan*      lt_ = nullptr;
  void*        workspace_ = nullptr;
  std::size_t  workspace_bytes_ = 32u << 20;
};

}  // namespace ccml_bench

#endif /* CCML_BENCH_BASELINES_CUBLAS_H */
