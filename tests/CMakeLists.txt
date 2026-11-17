# Correctness only. Nothing in this directory measures time, and nothing here
# may ever fail because of a performance threshold.

# Test registry and runner. The fp64 references and tolerance comparison live in
# support/, because the benchmark correctness gate uses them too.
add_library(ccml_test_support STATIC
  support/harness.cpp)

target_include_directories(ccml_test_support PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(ccml_test_support PUBLIC ccml::ccml ccml_support)

# --------------------------------------------------------------------------
# One executable per suite, all built the same way.
#
#   tests/test_<name>.cpp|cu  ->  target ccml_test_<name>  ->  ctest name <name>
#
# Every suite gets src/ on its include path. Most do not need it, but the
# infrastructure suite tests launch validation directly and the harness suite
# tests the utility headers, and a uniform rule is easier to follow than an
# exception. Nothing in include/ccml depends on this.
# --------------------------------------------------------------------------
function(ccml_add_test name source)
  set(target ccml_test_${name})

  add_executable(${target} ${source} support/test_main.cpp)
  target_link_libraries(${target} PRIVATE ccml_test_support)
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src)

  # Harmless on the suites with no device code.
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${CCML_CUDA_ARCHITECTURES}")

  # Must match the library, or a suite would exercise a launch path the library
  # does not actually use.
  target_compile_definitions(${target} PRIVATE
    $<$<OR:$<CONFIG:Debug>,$<BOOL:${CCML_SYNC_LAUNCH_CHECKS}>>:CCML_SYNC_LAUNCH_CHECKS>)

  add_test(NAME ${name} COMMAND ${target})
endfunction()

# Foundations. These run first because every operator result depends on them:
# if the launch path or the comparison machinery is broken, an operator suite
# can report PASS while being wrong.
ccml_add_test(infrastructure test_infrastructure.cu)  # launches its own kernels
ccml_add_test(harness        test_harness.cpp)        # tests the test harness

# Operators. Each is independently runnable: `ctest -R softmax` builds and runs
# softmax alone.
ccml_add_test(matmul    test_matmul.cpp)
ccml_add_test(layernorm test_layernorm.cpp)
ccml_add_test(softmax   test_softmax.cpp)

# FP16 and BF16 storage for all three operators (reference.md section 3).
ccml_add_test(dtypes    test_dtypes.cpp)

# Tensor Core matmul variants, FP16 and BF16 (BF16 SKIPs below sm_80).
ccml_add_test(matmul_tc test_matmul_tc.cpp)

# The benchmark harness's statistics, shape reader and report logic. Built only
# with the benchmarks, because it links their support library.
if(CCML_BUILD_BENCHMARKS)
  ccml_add_test(bench_stats test_bench_stats.cpp)
  target_link_libraries(ccml_test_bench_stats PRIVATE ccml_bench_support)
endif()
