// fastpoly - micro benchmarks for the NTT and the polynomial operations.
//
// Build:  cmake --build build --target fastpoly_bench
// Run:    ./build/fastpoly_bench [size]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "fastpoly/fastpoly.hpp"

using M = fpx::mod998244353;
using fpx::Poly;
using vec = fpx::poly::vec<M>;

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

template <class F>
double best_of(int reps, F&& f) {
  double best = 1e18;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = clk::now();
    f();
    best = std::min(best, ms_since(t0));
  }
  return best;
}

std::mt19937_64 rng(1);

vec rnd(size_t n) {
  vec a(n);
  for (auto& x : a) x = M::from_int(rng() % M::mod);
  return a;
}

void bench_ntt(size_t n) {
  auto plan = fpx::NttPlan<M>::get(static_cast<uint32_t>(n));
  std::vector<uint32_t> a(n);
  for (auto& x : a) x = static_cast<uint32_t>(rng() % M::mod);
  const double f = best_of(5, [&] { plan->forward(a.data()); });
  const double g = best_of(5, [&] { plan->inverse(a.data()); });
  const double per = 1e6 * f / (double(n) * (double(std::log2(double(n)))));
  std::printf("  NTT n=%-9zu forward %8.3f ms  inverse %8.3f ms  %5.2f ns/(elem*log n)\n", n, f, g, per);
}

void bench_conv(size_t n) {
  const vec a = rnd(n), b = rnd(n);
  const double t = best_of(3, [&] { volatile auto r = fpx::poly::conv(a, b); (void)r; });
  const double per = 1e6 * t / (double(n) * std::log2(double(n)));
  std::printf("  conv  n=%-9zu %8.3f ms  %5.2f ns/(elem*log n)\n", n, t, per);
}

void bench_series(size_t n) {
  vec a = rnd(n);
  a[0] = M::from_int(1);
  vec b = a;
  b[0] = M();
  const double ti = best_of(3, [&] { volatile auto r = fpx::poly::inv(a, n); (void)r; });
  const double te = best_of(3, [&] { volatile auto r = fpx::poly::exp(b, n); (void)r; });
  const double tl = best_of(3, [&] { volatile auto r = fpx::poly::log(a, n); (void)r; });
  vec sq = fpx::poly::conv(a, a, n);
  sq[0] = sq[0];  // make it a square
  const double ts = best_of(3, [&] { volatile auto r = fpx::poly::sqrt(sq, n); (void)r; });
  const double tp = best_of(3, [&] { volatile auto r = fpx::poly::pow(a, 1000003, n); (void)r; });
  std::printf("  series n=%-8zu inv %7.3f  log %7.3f  exp %7.3f  sqrt %7.3f  pow %7.3f ms\n",
              n, ti, tl, te, ts, tp);
}

}  // namespace

int main(int argc, char** argv) {
  std::printf("fastpoly benchmark | simd backend: %s (%d x uint32 lanes) | mod = %u\n",
              fpx::simd_backend(), fpx::simd::lane, M::mod);
  size_t max_n = 1u << 20;
  if (argc > 1) max_n = std::strtoul(argv[1], nullptr, 10);
  std::printf("\n-- NTT --\n");
  for (size_t n = 1024; n <= max_n; n <<= 2) bench_ntt(n);
  std::printf("\n-- convolution --\n");
  for (size_t n = 1024; n <= max_n; n <<= 2) bench_conv(n);
  std::printf("\n-- series operations --\n");
  for (size_t n = 1024; n <= max_n; n <<= 2) bench_series(n);
  return 0;
}
