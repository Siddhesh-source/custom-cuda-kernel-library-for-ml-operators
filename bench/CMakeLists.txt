# Timing only. None of these targets is registered with CTest: a timing must
# never be able to fail a build. Correctness of what is timed is enforced by the
# in-run gate (bench/support/driver.h), and the statistics code itself is
# unit-tested in tests/test_bench_stats.cpp.
#
# Vendor libraries (cuBLAS, cuBLASLt, NVML, optional cuDNN) are linked HERE
# only. The ccml library itself depends on the CUDA runtime and nothing else.

find_package(CUDAToolkit REQUIRED)

add_library(ccml_bench_support STATIC
  support/stats.cpp
  support/metrics.cpp
  support/shapes.cpp
  support/telemetry.cpp
  support/timer.cpp
  support/report.cpp
  support/driver.cpp)

target_include_directories(ccml_bench_support PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(ccml_bench_support PUBLIC ccml::ccml ccml_support CUDA::cudart)
target_compile_definitions(ccml_bench_support PUBLIC
  CCML_DEFAULT_SHAPES="${CMAKE_CURRENT_SOURCE_DIR}/shapes.json")

# NVML: clocks, temperature, throttle reasons per row. Optional; without it the
# telemetry columns are left empty rather than invented.
if(TARGET CUDA::nvml)
  target_link_libraries(ccml_bench_support PUBLIC CUDA::nvml)
  target_compile_definitions(ccml_bench_support PUBLIC CCML_BENCH_HAVE_NVML)
  message(STATUS "ccml bench: NVML telemetry enabled")
else()
  message(STATUS "ccml bench: NVML not found; telemetry columns will be empty")
endif()

# Vendor baselines.
add_library(ccml_bench_baselines STATIC
  baselines/cublas.cpp
  baselines/cudnn_softmax.cpp)
target_include_directories(ccml_bench_baselines PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(ccml_bench_baselines PUBLIC CUDA::cublas CUDA::cublasLt CUDA::cudart)

# cuDNN is optional: Kaggle ships it inside the PyTorch wheel (pass
# -DCUDNN_ROOT=<site-packages>/nvidia/cudnn); the local machine has none.
find_path(CCML_CUDNN_INCLUDE cudnn.h HINTS ${CUDNN_ROOT} ENV CUDNN_ROOT PATH_SUFFIXES include)
find_library(CCML_CUDNN_LIBRARY NAMES cudnn libcudnn.so.9 HINTS ${CUDNN_ROOT} ENV CUDNN_ROOT
             PATH_SUFFIXES lib lib64 lib/x64)
if(CCML_CUDNN_INCLUDE AND CCML_CUDNN_LIBRARY)
  target_include_directories(ccml_bench_baselines PRIVATE ${CCML_CUDNN_INCLUDE})
  target_link_libraries(ccml_bench_baselines PUBLIC ${CCML_CUDNN_LIBRARY})
  target_compile_definitions(ccml_bench_baselines PRIVATE CCML_BENCH_HAVE_CUDNN)
  # cudnnSoftmaxForward is part of cuDNN's legacy API, deprecated but supported.
  if(NOT MSVC)
    set_source_files_properties(baselines/cudnn_softmax.cpp PROPERTIES
      COMPILE_OPTIONS "-Wno-deprecated-declarations")
  endif()
  message(STATUS "ccml bench: cuDNN baseline enabled (${CCML_CUDNN_LIBRARY})")
else()
  message(STATUS "ccml bench: cuDNN not found; softmax cuDNN baseline disabled")
endif()

foreach(op IN ITEMS matmul layernorm softmax)
  add_executable(ccml_bench_${op} bench_${op}.cpp)
  target_link_libraries(ccml_bench_${op} PRIVATE ccml_bench_support ccml_bench_baselines)
endforeach()

add_executable(ccml_bench_peaks bench_peaks.cu)
target_link_libraries(ccml_bench_peaks PRIVATE ccml_bench_support)
set_target_properties(ccml_bench_peaks PROPERTIES
  CUDA_ARCHITECTURES "${CCML_CUDA_ARCHITECTURES}")
