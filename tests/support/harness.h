#ifndef CCML_TESTS_SUPPORT_HARNESS_H
#define CCML_TESTS_SUPPORT_HARNESS_H

#include <stdexcept>
#include <string>
#include <vector>

/*
 * A ~150-line test runner. Deliberately not a framework.
 *
 * Three outcomes, and the distinction between them matters here:
 *   PASS - the kernel ran and matched the reference within tolerance
 *   FAIL - the kernel ran and did not match, or errored unexpectedly
 *   SKIP - the kernel (or its reference) is not implemented yet
 *
 * SKIP is what keeps the skeleton honest: an unimplemented operator must never
 * be able to report PASS.
 */

namespace ccml_test {

enum class Outcome { kPass, kFail, kSkip };

/* Thrown by Context::skip(). Caught by the runner. */
class SkipTest : public std::runtime_error {
 public:
  explicit SkipTest(const std::string& reason) : std::runtime_error(reason) {}
};

/* Thrown by Context::fail() and the check macros. */
class TestFailure : public std::runtime_error {
 public:
  explicit TestFailure(const std::string& reason) : std::runtime_error(reason) {}
};

class Context {
 public:
  [[noreturn]] void skip(const std::string& reason) const { throw SkipTest(reason); }
  [[noreturn]] void fail(const std::string& reason) const { throw TestFailure(reason); }

  void check(bool condition, const std::string& reason) const {
    if (!condition) {
      fail(reason);
    }
  }

  /* Record a non-fatal note printed alongside the result, for measured error
     magnitudes and similar evidence. */
  void note(const std::string& text) const;
};

using TestFn = void (*)(const Context&);

bool register_test(const char* suite, const char* name, TestFn fn);

/* Defined in harness.cpp; test executables get main() from it. */
int run_all(int argc, char** argv);

}  // namespace ccml_test

/*
 * Declares and registers a test in one statement.
 *
 *   CCML_TEST(matmul, small_square) { ... }
 */
#define CCML_TEST(suite, name)                                               \
  static void ccml_test_##suite##_##name(const ::ccml_test::Context& ctx);   \
  static const bool ccml_test_##suite##_##name##_registered =                \
      ::ccml_test::register_test(#suite, #name,                              \
                                 &ccml_test_##suite##_##name);               \
  static void ccml_test_##suite##_##name(const ::ccml_test::Context& ctx)

/* Fails the test unless the call returned CCML_SUCCESS. */
#define CCML_ASSERT_OK(expr)                                                  \
  do {                                                                        \
    const ccml_status ccml_s_ = (expr);                                       \
    if (ccml_s_ != CCML_SUCCESS) {                                            \
      ctx.fail(std::string(#expr) + " -> " + ccml_status_string(ccml_s_) +    \
               " (" + ccml_last_error_string() + ")");                        \
    }                                                                         \
  } while (0)

/* Skips the test if the operator is not built yet; fails on any other error. */
#define CCML_ASSERT_OK_OR_SKIP(expr)                                          \
  do {                                                                        \
    const ccml_status ccml_s_ = (expr);                                       \
    if (ccml_s_ == CCML_ERROR_NOT_IMPLEMENTED) {                              \
      ctx.skip(std::string(#expr) + ": " + ccml_last_error_string());         \
    }                                                                         \
    if (ccml_s_ != CCML_SUCCESS) {                                            \
      ctx.fail(std::string(#expr) + " -> " + ccml_status_string(ccml_s_) +    \
               " (" + ccml_last_error_string() + ")");                        \
    }                                                                         \
  } while (0)

#endif /* CCML_TESTS_SUPPORT_HARNESS_H */
