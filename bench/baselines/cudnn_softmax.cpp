#include "bench/baselines/cudnn_softmax.h"

#ifdef CCML_BENCH_HAVE_CUDNN
#include <cudnn.h>
#endif

namespace ccml_bench {

#ifdef CCML_BENCH_HAVE_CUDNN

namespace {
cudnnHandle_t h(void* p) { return static_cast<cudnnHandle_t>(p); }
cudnnTensorDescriptor_t d(void* p) { return static_cast<cudnnTensorDescriptor_t>(p); }
}  // namespace

bool CudnnSoftmax::available() { return true; }

CudnnSoftmax::CudnnSoftmax(cudaStream_t stream) {
  cudnnHandle_t handle = nullptr;
  if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) return;
  cudnnSetStream(handle, stream);
  handle_ = handle;
  cudnnTensorDescriptor_t desc = nullptr;
  if (cudnnCreateTensorDescriptor(&desc) == CUDNN_STATUS_SUCCESS) desc_ = desc;
}

CudnnSoftmax::~CudnnSoftmax() {
  if (desc_ != nullptr) cudnnDestroyTensorDescriptor(d(desc_));
  if (handle_ != nullptr) cudnnDestroy(h(handle_));
}

bool CudnnSoftmax::prepare(int rows, int cols, BlasType type) {
  if (handle_ == nullptr || desc_ == nullptr) return false;
  const cudnnDataType_t t = type == BlasType::F16    ? CUDNN_DATA_HALF
                            : type == BlasType::BF16 ? CUDNN_DATA_BFLOAT16
                                                     : CUDNN_DATA_FLOAT;
  return cudnnSetTensor4dDescriptor(d(desc_), CUDNN_TENSOR_NCHW, t, rows, cols, 1, 1) ==
         CUDNN_STATUS_SUCCESS;
}

bool CudnnSoftmax::run(void* y, const void* x) {
  const float alpha = 1.0f, beta = 0.0f;
  return cudnnSoftmaxForward(h(handle_), CUDNN_SOFTMAX_ACCURATE,
                             CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, d(desc_), x,
                             &beta, d(desc_), y) == CUDNN_STATUS_SUCCESS;
}

#else

bool CudnnSoftmax::available() { return false; }
CudnnSoftmax::CudnnSoftmax(cudaStream_t) {}
CudnnSoftmax::~CudnnSoftmax() = default;
bool CudnnSoftmax::prepare(int, int, BlasType) { return false; }
bool CudnnSoftmax::run(void*, const void*) { return false; }

#endif

}  // namespace ccml_bench
