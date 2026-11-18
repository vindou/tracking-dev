// Minimal test framework (only the C++ standard library is permitted).
//   TEST(name) { CHECK(cond); CHECK_EQ(a, b); CHECK_LE(a, b); CHECK_GE(a, b); }
// trk_tests runs all tests, or those whose name contains argv[1]. Exit code = number of failed tests.
#pragma once
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace tf {
struct Test { const char* name; std::function<void()> fn; };
inline std::vector<Test>& registry() { static std::vector<Test> r; return r; }
inline int& failures() { static int f = 0; return f; }
std::string& self_path();
struct Register { Register(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };
template <typename A, typename B>
void check_cmp(bool ok, const A& a, const B& b, const char* expr, const char* file, int line) {
    if (ok) return;
    std::ostringstream os;
    os << a << " vs " << b;
    std::fprintf(stderr, "  CHECK FAILED %s:%d: %s  (%s)\n", file, line, expr, os.str().c_str());
    failures()++;
}
}  // namespace tf

#define TF_CAT2(a, b) a##b
#define TF_CAT(a, b) TF_CAT2(a, b)
#define TEST(name) static void test_##name(); static tf::Register TF_CAT(reg_, name)(#name, test_##name); static void test_##name()
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "  CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); tf::failures()++; } } while (0)
// Arguments are evaluated once.
#define TF_CMP(a, b, op) do { const auto& tf_a = (a); const auto& tf_b = (b); tf::check_cmp(tf_a op tf_b, tf_a, tf_b, #a " " #op " " #b, __FILE__, __LINE__); } while (0)
#define CHECK_EQ(a, b) TF_CMP(a, b, ==)
#define CHECK_LE(a, b) TF_CMP(a, b, <=)
#define CHECK_GE(a, b) TF_CMP(a, b, >=)
