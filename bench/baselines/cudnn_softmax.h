#ifndef CCML_BENCH_BASELINES_CUDNN_SOFTMAX_H
#define CCML_BENCH_BASELINES_CUDNN_SOFTMAX_H

#include <cuda_runtime.h>

#include "bench/baselines/cublas.h"

namespace ccml_bench {

/*
 * cuDNN softmax vendor baseline (reference.md §4), CUDNN_SOFTMAX_ACCURATE
 * (max-subtracted, matching ours) over each row of a row-major [rows, cols]
 * tensor, expressed as NCHW = [rows, cols, 1, 1] with MODE_INSTANCE.
 *
 * cuDNN has no legacy-API layer normalization; its layernorm exists only in
 * the graph API. The layernorm vendor baseline is therefore PyTorch eager
 * (scripts/bench_torch.py), as documented in docs/benchmarking.md.
 *
 * Available only when CMake found cuDNN (CCML_BENCH_HAVE_CUDNN).
 */
class CudnnSoftmax {
 public:
  explicit CudnnSoftmax(cudaStream_t stream);
  ~CudnnSoftmax();
  CudnnSoftmax(const CudnnSoftmax&) = delete;
  CudnnSoftmax& operator=(const CudnnSoftmax&) = delete;

  /* type: storage of x and y (FP32, FP16 or BF16; cuDNN computes in FP32). */
  bool prepare(int rows, int cols, BlasType type = BlasType::F32);
  bool run(void* y, const void* x);
  static bool available();

 private:
  void* handle_ = nullptr;
  void* desc_ = nullptr;
};

}  // namespace ccml_bench

#endif /* CCML_BENCH_BASELINES_CUDNN_SOFTMAX_H */
