# Utilities shared by BOTH tests and benchmarks. Deliberately tiny: device
# buffer ownership, deterministic input generation, fp64 references and
# tolerance comparison (the benchmark correctness gate reuses the last two).
#
# This exists so that tests/ and bench/ stay independent of each other. Neither
# includes the other; both depend on this.

add_library(ccml_support STATIC
  random.cpp
  compare.cpp
  reference.cpp)

target_include_directories(ccml_support PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(ccml_support PUBLIC ccml::ccml)
