# Minimal consumers of the public API.
#
# These include only what is in include/ccml and add no include path into src/.
# If one of them stops compiling, the public headers are the problem -- which is
# the reason they are built as part of the default build rather than on demand.

foreach(example IN ITEMS device_query usage)
  add_executable(ccml_${example} ${example}.cpp)
  target_link_libraries(ccml_${example} PRIVATE ccml::ccml)
endforeach()
