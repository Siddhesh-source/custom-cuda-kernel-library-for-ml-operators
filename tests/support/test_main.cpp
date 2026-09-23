#include "tests/support/harness.h"

/* Linked directly into every test executable rather than into the support
   archive, so that main() is unconditionally present. */
int main(int argc, char** argv) { return ccml_test::run_all(argc, argv); }
