#ifndef CCML_SUPPORT_DEVICE_BUFFER_H
#define CCML_SUPPORT_DEVICE_BUFFER_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "ccml/memory.h"
#include "ccml/status.h"

namespace ccml_support {

/* Owning float device buffer. Exists so tests and benchmarks cannot leak on an
   early return, and so they exercise the same public allocation API that the
   Python binding uses. */
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) { allocate(count); }

  ~DeviceBuffer() { ccml_device_free(ptr_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      ccml_device_free(ptr_);
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void allocate(std::size_t count) {
    ccml_device_free(ptr_);
    ptr_ = nullptr;
    count_ = 0;
    void* raw = nullptr;
    const ccml_status status = ccml_device_malloc(&raw, count * sizeof(float));
    throw_if_failed(status, "ccml_device_malloc");
    ptr_ = static_cast<float*>(raw);
    count_ = count;
  }

  void upload(const float* host) {
    throw_if_failed(ccml_memcpy_h2d(ptr_, host, bytes()), "ccml_memcpy_h2d");
  }

  void download(float* host) const {
    throw_if_failed(ccml_memcpy_d2h(host, ptr_, bytes()), "ccml_memcpy_d2h");
  }

  float* data() { return ptr_; }
  const float* data() const { return ptr_; }
  std::size_t count() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(float); }

 private:
  static void throw_if_failed(ccml_status status, const char* what) {
    if (status != CCML_SUCCESS) {
      throw std::runtime_error(std::string(what) + ": " +
                               ccml_status_string(status) + " (" +
                               ccml_last_error_string() + ")");
    }
  }

  float* ptr_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace ccml_support

#endif /* CCML_SUPPORT_DEVICE_BUFFER_H */
