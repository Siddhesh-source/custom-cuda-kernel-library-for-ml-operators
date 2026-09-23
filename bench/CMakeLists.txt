# Timing only. Nothing in this directory asserts correctness, and none of these
# targets is registered with CTest -- a timing measurement must never be able to
# fail a build.

add_library(ccml_bench_support STATIC
  support/timer.cpp
  support/report.cpp)

target_include_directories(ccml_bench_support PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(ccml_bench_support PUBLIC ccml::ccml ccml_support)

foreach(op IN ITEMS matmul layernorm softmax)
  add_executable(ccml_bench_${op} bench_${op}.cpp)
  target_link_libraries(ccml_bench_${op} PRIVATE ccml_bench_support)
endforeach()
