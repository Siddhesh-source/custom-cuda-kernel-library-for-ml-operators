#include "tests/support/harness.h"

#include <cstdio>
#include <string>
#include <vector>

namespace ccml_test {

namespace {

struct RegisteredTest {
  std::string suite;
  std::string name;
  TestFn fn;
};

/* Function-local static: avoids any dependency on initialization order between
   translation units, which matters because registration happens during static
   initialization. */
std::vector<RegisteredTest>& registry() {
  static std::vector<RegisteredTest> tests;
  return tests;
}

std::vector<std::string>& notes() {
  static std::vector<std::string> current;
  return current;
}

}  // namespace

void Context::note(const std::string& text) const { notes().push_back(text); }

bool register_test(const char* suite, const char* name, TestFn fn) {
  registry().push_back(RegisteredTest{suite, name, fn});
  return true;
}

int run_all(int argc, char** argv) {
  /* Optional single argument: run only tests whose name contains it. */
  const char* filter = (argc > 1) ? argv[1] : nullptr;

  int passed = 0;
  int failed = 0;
  int skipped = 0;

  const Context ctx;

  for (const RegisteredTest& test : registry()) {
    const std::string full = test.suite + "." + test.name;
    if (filter != nullptr && full.find(filter) == std::string::npos) {
      continue;
    }

    notes().clear();
    Outcome outcome = Outcome::kPass;
    std::string detail;

    try {
      test.fn(ctx);
    } catch (const SkipTest& e) {
      outcome = Outcome::kSkip;
      detail = e.what();
    } catch (const TestFailure& e) {
      outcome = Outcome::kFail;
      detail = e.what();
    } catch (const std::exception& e) {
      outcome = Outcome::kFail;
      detail = std::string("unexpected exception: ") + e.what();
    }

    const char* label = "PASS";
    if (outcome == Outcome::kFail) {
      label = "FAIL";
      ++failed;
    } else if (outcome == Outcome::kSkip) {
      label = "SKIP";
      ++skipped;
    } else {
      ++passed;
    }

    std::printf("[%s] %s\n", label, full.c_str());
    for (const std::string& n : notes()) {
      std::printf("       %s\n", n.c_str());
    }
    if (!detail.empty()) {
      std::printf("       %s\n", detail.c_str());
    }
  }

  std::printf("\n%d passed, %d failed, %d skipped\n", passed, failed, skipped);

  if (skipped > 0 && failed == 0) {
    std::printf("NOTE: skipped tests are not evidence of correctness.\n");
  }
  return (failed == 0) ? 0 : 1;
}

}  // namespace ccml_test
