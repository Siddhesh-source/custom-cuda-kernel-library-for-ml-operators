cmake_minimum_required(VERSION 3.25)

project(ccml
        VERSION 0.1.0
        DESCRIPTION "Custom CUDA kernels for common ML operators"
        LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CCML_BUILD_TESTS        "Build correctness tests"            ON)
option(CCML_BUILD_BENCHMARKS   "Build benchmarks"                   ON)
option(CCML_BUILD_EXAMPLES     "Build examples"                     ON)
option(CCML_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"  OFF)

# Synchronise after every kernel launch so that an execution fault is reported
# against the kernel that caused it, rather than against whatever call happens
# to synchronise next. Always on in Debug. Ruinous for performance, so never
# enable it for a build you intend to time.
option(CCML_SYNC_LAUNCH_CHECKS
       "Synchronise after every kernel launch (forced on in Debug)" OFF)

# Default target is sm_89 (Ada; RTX 3050 A Laptop GPU on the dev machine).
# Override with -DCCML_CUDA_ARCHITECTURES=86 etc.
set(CCML_CUDA_ARCHITECTURES "89" CACHE STRING
    "CUDA architectures to compile device code for")

# The Python ctypes binding loads a shared library, so shared is the default.
if(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "Build ccml as a shared library" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Language settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep the .dll next to the .exe files on Windows so tests/benchmarks just run.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(GNUInstallDirs)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/ccml/version.h"
  @ONLY)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)

# Utilities shared by tests and benchmarks; built only when one of them is.
if(CCML_BUILD_TESTS OR CCML_BUILD_BENCHMARKS)
  add_subdirectory(support)
endif()

if(CCML_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(CCML_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CCML_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# ---------------------------------------------------------------------------
# Install / export -- lets a downstream project do find_package(ccml)
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS ccml
        EXPORT  ccmlTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/ccml/version.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ccml)

install(EXPORT ccmlTargets
        FILE      ccml-targets.cmake
        NAMESPACE ccml::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccml)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ccml-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ccml-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccml)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ccml-config-version.cmake"
  COMPATIBILITY SameMajorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ccml-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ccml-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccml)
