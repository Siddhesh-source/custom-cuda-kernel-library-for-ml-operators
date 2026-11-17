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
/* Storage type of A, B and C. Computation is FP32 in every case. */
enum class BlasType { F32, F16, BF16 };

class CublasBaseline {
 public:
  explicit CublasBaseline(cudaStream_t stream);
  ~CublasBaseline();
  CublasBaseline(const CublasBaseline&) = delete;
  CublasBaseline& operator=(const CublasBaseline&) = delete;

  /* cublasSgemm, default math mode (no TF32). */
  bool sgemm(float* C, const float* A, const float* B, int M, int N, int K);

  /* cublasGemmEx with 16-bit A, B and C and CUBLAS_COMPUTE_32F (FP32
     accumulation, the same contract as ccml_matmul_f16/bf16). cuBLAS picks its
     own kernel, Tensor Cores included, with CUBLAS_GEMM_DEFAULT. */
  bool gemm_ex(void* C, const void* A, const void* B, int M, int N, int K, BlasType type);

  /* cublasLt with its top-1 heuristic algorithm for this shape and type.
     prepare() runs the heuristic once; lt_matmul() then launches with that
     algorithm. */
  bool lt_prepare(int M, int N, int K, BlasType type = BlasType::F32);
  bool lt_matmul(void* C, const void* A, const void* B);

  /* cuBLASLt's own fused epilogue (enhancement B2), for comparison with
     ccml_matmul_epilogue: C = gelu(A B + bias) (`gelu`, CUBLASLT_EPILOGUE_GELU_BIAS)
     or C = A B + bias + residual (CUBLASLT_EPILOGUE_BIAS with beta = 1 and the
     residual as the C operand). bias: [N] device pointer of `type`. Top-1
     heuristic for that epilogue. */
  bool lt_prepare_epilogue(int M, int N, int K, BlasType type, const void* bias, bool gelu);
  bool lt_matmul_epilogue(void* C, const void* A, const void* B, const void* residual);

  /* fp64 reference: host float inputs, host double output. */
  bool dgemm_reference(std::vector<double>& C, const std::vector<float>& A,
                       const std::vector<float>& B, int M, int N, int K);

 private:
  struct LtPlan;
  void release_lt();
  bool lt_prepare_impl(int M, int N, int K, BlasType type, int epilogue, const void* bias);

  cudaStream_t stream_;
  void*        handle_ = nullptr;    /* cublasHandle_t */
  void*        lt_handle_ = nullptr; /* cublasLtHandle_t */
  LtPlan*      lt_ = nullptr;
  void*        workspace_ = nullptr;
  std::size_t  workspace_bytes_ = 32u << 20;
};

}  // namespace ccml_bench

#endif /* CCML_BENCH_BASELINES_CUBLAS_H */
