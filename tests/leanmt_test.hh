// Minimal test framework for leanmt — header-only, no third-party deps.
// Tests register with TEST("name"); each test executable links a single
// test .cc file and ends with `int main() { return leanmt::test::run_all(); }`.
#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace leanmt::test {

struct Registry {
  struct Case {
    std::string name;
    std::function<void()> fn;
  };

  std::vector<Case> cases;
  int checks_failed = 0;
  int checks_total = 0;

  static Registry& instance() {
    static Registry registry;
    return registry;
  }
};

struct Registrar {
  Registrar(std::string name, std::function<void()> fn) {
    Registry::instance().cases.push_back({std::move(name), std::move(fn)});
  }
};

// Directory containing the downloaded array of test fixtures
// (ende.student.tiny11 model/vocab/shortlist). Set LEANMT_FIXTURES to
// override; falls back to the in-tree default baked in by CMake.
inline std::string fixture_dir() {
  const char* env = std::getenv("LEANMT_FIXTURES");
  if (env != nullptr && env[0] != '\0') {
    return env;
  }
  return LEANMT_FIXTURE_DEFAULT_DIR;
}

inline void check(bool ok, const char* expr, const char* file, int line,
                  const std::string& detail = "") {
  Registry& registry = Registry::instance();
  ++registry.checks_total;
  if (!ok) {
    ++registry.checks_failed;
    std::cerr << "    FAIL " << file << ":" << line << ": CHECK(" << expr
              << ")" << (detail.empty() ? "" : "  " + detail) << "\n";
  }
}

inline std::string str(const std::string& v) { return v; }
inline std::string str(const char* v) { return v ? v : "(null)"; }
inline std::string str(std::string_view v) { return std::string(v); }

template <class T>
inline std::string str(const T& value) {
  std::ostringstream out;
  out << value;
  return out.str();
}

template <class A, class B>
inline void check_eq(const A& a, const B& b, const char* ea, const char* eb,
                     const char* file, int line) {
  bool ok;
  if constexpr (std::is_integral_v<A> && std::is_integral_v<B>) {
    ok = std::cmp_equal(a, b);  // sign-safe for mixed int widths
  } else {
    ok = (a == b);
  }
  std::ostringstream detail;
  detail << str(a) << " vs " << str(b);
  check(ok, (std::string(ea) + " == " + std::string(eb)).c_str(), file, line,
        detail.str());
}

inline void check_close(double a, double b, double eps, const char* ea,
                        const char* eb, const char* file, int line) {
  std::ostringstream detail;
  detail << str(a) << " vs " << str(b) << " (eps " << str(eps) << ")";
  check(std::fabs(a - b) <= eps,
        (std::string("fabs(") + ea + " - " + eb + ") <= eps").c_str(), file,
        line, detail.str());
}

inline void check_throws(const std::function<void()>& fn,
                         const std::string& expect_substr, const char* expr,
                         const char* file, int line) {
  try {
    fn();
  } catch (const std::exception& e) {
    std::string what = e.what();
    if (what.find(expect_substr) != std::string::npos) {
      check(true, expr, file, line);
    } else {
      check(false, expr, file, line,
            "wrong exception text, expected substring \"" + expect_substr +
                "\", got: " + what);
    }
    return;
  } catch (...) {
    check(false, expr, file, line, "threw unknown (non-std::exception) type");
    return;
  }
  check(false, expr, file, line, "expected throw, but nothing was thrown");
}

inline int run_all() {
  Registry& registry = Registry::instance();
  int index = 0;
  for (const auto& test_case : registry.cases) {
    ++index;
    int failed_before = registry.checks_failed;
    try {
      test_case.fn();
    } catch (const std::exception& e) {
      check(false, "no uncaught exception", __FILE__, __LINE__,
            std::string("uncaught: ") + e.what());
    } catch (...) {
      check(false, "no uncaught exception", __FILE__, __LINE__, "uncaught");
    }
    bool ok = registry.checks_failed == failed_before;
    std::cout << (ok ? "ok " : "not ok ") << index << " - " << test_case.name
              << "\n";
  }
  std::cout << (registry.checks_failed == 0 ? "# PASS" : "# FAIL") << " "
            << (registry.checks_total - registry.checks_failed) << "/"
            << registry.checks_total << " checks, " << registry.cases.size()
            << " cases\n";
  return registry.checks_failed == 0 ? 0 : 1;
}

}  // namespace leanmt::test

#define LEANMT_TEST_CAT2(a, b) a##b
#define LEANMT_TEST_CAT(a, b) LEANMT_TEST_CAT2(a, b)

#define TEST(name)                                                \
  static void LEANMT_TEST_CAT(leanmt_test_fn_, __LINE__)();      \
  static ::leanmt::test::Registrar LEANMT_TEST_CAT(              \
      leanmt_test_reg_, __LINE__)(name,                          \
                                  LEANMT_TEST_CAT(leanmt_test_fn_, __LINE__)); \
  static void LEANMT_TEST_CAT(leanmt_test_fn_, __LINE__)()

#define CHECK(cond) ::leanmt::test::check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
// Like CHECK, but aborts the test case on failure. Use when subsequent checks
// would dereference nullptr / read out of bounds otherwise.
#define REQUIRE(cond)                                              \
  do {                                                             \
    bool ok = static_cast<bool>(cond);                             \
    ::leanmt::test::check(ok, #cond, __FILE__, __LINE__);         \
    if (!ok) return;                                               \
  } while (0)
#define CHECK_EQ(a, b) ::leanmt::test::check_eq((a), (b), #a, #b, __FILE__, __LINE__)
#define CHECK_CLOSE(a, b, eps) \
  ::leanmt::test::check_close((a), (b), (eps), #a, #b, __FILE__, __LINE__)
#define CHECK_THROWS(expr, substr)                                      \
  ::leanmt::test::check_throws([&] { (void)(expr); }, substr, #expr,    \
                               __FILE__, __LINE__)

#ifndef LEANMT_FIXTURE_DEFAULT_DIR
#define LEANMT_FIXTURE_DEFAULT_DIR "tests/fixtures/models/ende.student.tiny11"
#endif
