// fastpoly - minimal header-only test harness (no external dependencies).
#ifndef FASTPOLY_FP_TEST_HPP
#define FASTPOLY_FP_TEST_HPP

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fptest {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}
inline long& failures() {
  static long f = 0;
  return f;
}
inline bool& verbose() {
  static bool v = false;
  return v;
}

struct Reg {
  Reg(const char* n, void (*f)()) { registry().push_back({n, f}); }
};

inline int run_all() {
  for (const auto& c : registry()) {
    const long before = failures();
    std::printf("[ RUN  ] %s\n", c.name);
    std::fflush(stdout);
    c.fn();
    const bool ok = failures() == before;
    std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", c.name);
    if (!ok) std::fflush(stdout);
  }
  if (failures() == 0) {
    std::printf("\n== %zu tests passed ==\n", registry().size());
    return 0;
  }
  std::printf("\n== %ld check(s) FAILED ==\n", failures());
  return 1;
}

}  // namespace fptest

#define FP_TEST(name)                                          \
  static void fp_test_##name();                                \
  static ::fptest::Reg fp_reg_##name(#name, fp_test_##name);   \
  static void fp_test_##name()

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("  CHECK(%s) failed at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++::fptest::failures();                                                \
    }                                                                        \
  } while (0)

#define CHECK_EQ(a, b)                                                          \
  do {                                                                          \
    auto fp_va = (a);                                                           \
    auto fp_vb = (b);                                                           \
    if (!(fp_va == fp_vb)) {                                                    \
      std::printf("  CHECK_EQ(%s, %s) failed at %s:%d\n", #a, #b, __FILE__,     \
                  __LINE__);                                                    \
      ++::fptest::failures();                                                   \
    }                                                                           \
  } while (0)

#define CHECK_MSG(cond, fmt, ...)                                        \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::printf("  " fmt " (at %s:%d)\n", __VA_ARGS__, __FILE__, __LINE__); \
      ++::fptest::failures();                                            \
    }                                                                    \
  } while (0)

#define FP_TEST_MAIN() \
  int main() { return ::fptest::run_all(); }

#endif  // FASTPOLY_FP_TEST_HPP
