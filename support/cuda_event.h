#ifndef CCML_SUPPORT_CUDA_EVENT_H
#define CCML_SUPPORT_CUDA_EVENT_H

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace ccml_support {

/* Owning CUDA event, and the elapsed-time measurement built from a pair.
 *
 * Why events rather than a host clock
 * -----------------------------------
 * A kernel launch returns to the host almost immediately, so wrapping one in a
 * host timer measures launch overhead, not execution. Worse on Windows, where
 * the WDDM driver model batches command submission: the host clock can report
 * a duration bearing no relation to when the work actually ran. A cudaEvent is
 * recorded in the stream itself, between the surrounding work, so it times what
 * the device did.
 *
 * Events are RAII here because the timing path has several places an exception
 * or early return can leave a handle leaked, and a leaked event is invisible
 * until the process runs out of them.
 */
class CudaEvent {
 public:
  CudaEvent() {
    const cudaError_t err = cudaEventCreate(&handle_);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaEventCreate: ")
                               + cudaGetErrorString(err));
    }
  }

  ~CudaEvent() {
    if (handle_ != nullptr) {
      /* Destructors do not throw; a failure here means the context is gone. */
      cudaEventDestroy(handle_);
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (handle_ != nullptr) {
        cudaEventDestroy(handle_);
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  cudaEvent_t get() const { return handle_; }

  void record(cudaStream_t stream) {
    check(cudaEventRecord(handle_, stream), "cudaEventRecord");
  }

  void synchronize() {
    check(cudaEventSynchronize(handle_), "cudaEventSynchronize");
  }

  /* Milliseconds from `start` to this event. Both must have been recorded and
     completed. Resolution is roughly half a microsecond. */
  float elapsed_ms_since(const CudaEvent& start) const {
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, start.handle_, handle_),
          "cudaEventElapsedTime");
    return milliseconds;
  }

 private:
  static void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
  }

  cudaEvent_t handle_ = nullptr;
};

/* One start/stop pair. Time a single stream region:
 *
 *     EventTimer timer;
 *     timer.start(stream);
 *     ... launch work on stream ...
 *     const double ms = timer.stop_ms(stream);
 *
 * stop_ms synchronises on the stop event, so it blocks until the timed work has
 * finished. That is the point -- an unsynchronised reading is meaningless.
 */
class EventTimer {
 public:
  void start(cudaStream_t stream) { start_.record(stream); }

  double stop_ms(cudaStream_t stream) {
    stop_.record(stream);
    stop_.synchronize();
    return static_cast<double>(stop_.elapsed_ms_since(start_));
  }

 private:
  CudaEvent start_;
  CudaEvent stop_;
};

}  // namespace ccml_support

#endif /* CCML_SUPPORT_CUDA_EVENT_H */
