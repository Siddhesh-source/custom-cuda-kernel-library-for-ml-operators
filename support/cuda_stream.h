#ifndef CCML_SUPPORT_CUDA_STREAM_H
#define CCML_SUPPORT_CUDA_STREAM_H

#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

namespace ccml_support {

/* Owning CUDA stream.
 *
 * The library's own entry points take a cudaStream_t and do not own it, which
 * is the right split: a library that creates streams behind the caller's back
 * cannot be composed with one that does not. This wrapper exists for tests and
 * benchmarks, which do need to own one, and which have several early-return
 * paths where a hand-managed cudaStreamDestroy would be missed.
 *
 * A default-constructed CudaStream is the default stream (a null handle) and
 * owns nothing, so it is safe to pass around and destroy.
 */
class CudaStream {
 public:
  /* The default stream. Creates nothing and destroys nothing. */
  CudaStream() = default;

  /* Creates a real stream. cudaStreamNonBlocking keeps it independent of the
     legacy default stream, so work on it is not implicitly serialised against
     everything else in the process. */
  static CudaStream create(unsigned flags = cudaStreamNonBlocking) {
    cudaStream_t handle = nullptr;
    const cudaError_t err = cudaStreamCreateWithFlags(&handle, flags);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaStreamCreateWithFlags: ")
                               + cudaGetErrorString(err));
    }
    return CudaStream(handle, /*owning=*/true);
  }

  ~CudaStream() { reset(); }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CudaStream(CudaStream&& other) noexcept
      : handle_(other.handle_), owning_(other.owning_) {
    other.handle_ = nullptr;
    other.owning_ = false;
  }

  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = other.handle_;
      owning_ = other.owning_;
      other.handle_ = nullptr;
      other.owning_ = false;
    }
    return *this;
  }

  cudaStream_t get() const { return handle_; }
  operator cudaStream_t() const { return handle_; }

  void synchronize() const {
    const cudaError_t err = cudaStreamSynchronize(handle_);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaStreamSynchronize: ")
                               + cudaGetErrorString(err));
    }
  }

 private:
  CudaStream(cudaStream_t handle, bool owning) : handle_(handle), owning_(owning) {}

  void reset() {
    if (owning_ && handle_ != nullptr) {
      /* Destructors do not throw. A failure here means the context is already
         gone, at which point there is nothing useful left to do. */
      cudaStreamDestroy(handle_);
    }
    handle_ = nullptr;
    owning_ = false;
  }

  cudaStream_t handle_ = nullptr;
  bool         owning_ = false;
};

}  // namespace ccml_support

#endif /* CCML_SUPPORT_CUDA_STREAM_H */
