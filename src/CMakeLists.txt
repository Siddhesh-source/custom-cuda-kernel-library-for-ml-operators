find_package(CUDAToolkit REQUIRED)

add_library(ccml
  common/status.cpp
  common/device.cpp
  common/memory.cpp
  matmul/matmul.cu
  layernorm/layernorm.cu
  softmax/softmax.cu)

add_library(ccml::ccml ALIAS ccml)

target_include_directories(ccml
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

# The CUDA runtime is the only dependency. Nothing else may be added here
# without a note in docs/design.md explaining why.
target_link_libraries(ccml PUBLIC CUDA::cudart)

target_compile_definitions(ccml PRIVATE CCML_BUILD_LIBRARY)
if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(ccml PUBLIC CCML_STATIC)
endif()

# See the option in the top-level CMakeLists. Debug always gets it.
target_compile_definitions(ccml PRIVATE
  $<$<OR:$<CONFIG:Debug>,$<BOOL:${CCML_SYNC_LAUNCH_CHECKS}>>:CCML_SYNC_LAUNCH_CHECKS>)

set_target_properties(ccml PROPERTIES
  CUDA_ARCHITECTURES        "${CCML_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_VISIBILITY_PRESET      hidden
  CUDA_VISIBILITY_PRESET     hidden
  VISIBILITY_INLINES_HIDDEN  ON
  VERSION                    ${PROJECT_VERSION}
  SOVERSION                  ${PROJECT_VERSION_MAJOR})

if(MSVC)
  target_compile_options(ccml PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
  if(CCML_WARNINGS_AS_ERRORS)
    target_compile_options(ccml PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/WX>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>)
  endif()
else()
  target_compile_options(ccml PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
  if(CCML_WARNINGS_AS_ERRORS)
    target_compile_options(ccml PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()
endif()

# Source-line attribution for Nsight Compute later on. Free at runtime, and
# incompatible with the -G that CMake adds for Debug, hence the config guard.
target_compile_options(ccml PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>)
