// fastpoly - repeatable micro benchmarks for NTT and polynomial operations.
// Build: cmake --build build --target fastpoly_bench
// Run:   ./build/fastpoly_bench [max-size] [--size N] [--op OP] [--reps R]
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fastpoly/fastpoly.hpp"
#if __has_include("fastpoly/memory.hpp")
#define FASTPOLY_BENCH_HAS_SCRATCH_POOL 1
#endif

#ifndef FASTPOLY_BENCH_MOD
#define FASTPOLY_BENCH_MOD 998244353u
#endif
#ifndef FASTPOLY_BENCH_ROOT
#define FASTPOLY_BENCH_ROOT 3u
#endif
using M = fpx::Mont<FASTPOLY_BENCH_MOD, FASTPOLY_BENCH_ROOT>;
using vec = fpx::poly::vec<M>;

namespace {

using clk = std::chrono::steady_clock;

struct Options {
  size_t min_n = 1024;
  size_t max_n = 1u << 20;
  size_t exact_n = 0;
  int reps = 0;  // 5 for NTT, 3 for polynomial operations, as before
  int warmup = 1;
  std::string_view op = "all";
  bool csv = false;
  bool cold_scratch = false;
  bool memory = false;
};

void usage() {
  std::printf(
      "Usage: fastpoly_bench [max-size] [options]\n"
      "  --size N       Benchmark exactly N (any positive polynomial size;\n"
      "                 NTT/all require a power of two, N >= 2)\n"
      "  --min-size N   First size in a powers-of-two sweep (default 1024)\n"
      "  --op OP        all, ntt, ntt-forward, ntt-inverse, conv, series,\n"
      "                 inv, log, exp, sqrt, pow (default all); opt-in:\n"
      "                 square, skinny (40 x N), exp-linear (exp(x)),\n"
      "                 inv-short (1/(1-x)), pow-short ((1+x)^64),\n"
      "                 pow-linear ((1+x)^1000003), div-small (degree-40 divisor), plan\n"
      "                 inv-series (1/i table), scale, inv-linear (1/(1-7*x))\n"
      "                 plan requires --warmup 0, measures one cold build\n"
      "  --reps R       Timed repetitions (default 5 for NTT, 3 otherwise)\n"
      "  --warmup W     Untimed repetitions (default 1)\n"
      "  --csv          Print machine-readable results, including checksums\n"
      "  --cold-scratch Release idle scratch before each repetition\n"
      "  --memory       Report scratch pool statistics on stderr\n"
      "  --help         Show this help\n");
}

size_t number(std::string_view s) {
  size_t n = 0;
  const auto r = std::from_chars(s.data(), s.data() + s.size(), n);
  if (r.ec != std::errc() || r.ptr != s.data() + s.size())
    throw std::invalid_argument("invalid nonnegative integer: " + std::string(s));
  return n;
}

bool power_of_two(size_t n) { return n >= 2 && (n & (n - 1)) == 0; }

bool selected(const Options& opt, std::string_view op) {
  const bool extra = op == "square" || op == "skinny" || op == "exp-linear" ||
                     op == "inv-short" || op == "pow-short" || op == "pow-linear" ||
                     op == "div-small" || op == "plan" || op == "inv-series" ||
                     op == "scale" || op == "inv-linear";
  return (opt.op == "all" && !extra) || opt.op == op ||
         (opt.op == "ntt" && op.starts_with("ntt-")) ||
         (opt.op == "series" && !extra && op != "conv" && !op.starts_with("ntt-"));
}

Options parse(int argc, char** argv) {
  Options opt;
  bool positional = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--csv") { opt.csv = true; continue; }
    if (arg == "--cold-scratch") { opt.cold_scratch = true; continue; }
    if (arg == "--memory") { opt.memory = true; continue; }
    if (arg.starts_with("--")) {
      if (i + 1 == argc) throw std::invalid_argument("missing value for " + std::string(arg));
      const std::string_view value = argv[++i];
      if (arg == "--op") opt.op = value;
      else if (arg == "--size") {
        opt.exact_n = number(value);
        if (opt.exact_n == 0) throw std::invalid_argument("size must be positive");
      } else if (arg == "--min-size") opt.min_n = number(value);
      else if (arg == "--reps" || arg == "--warmup") {
        const size_t n = number(value);
        if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            (arg == "--reps" && n == 0))
          throw std::invalid_argument("invalid repetition count");
        if (arg == "--reps") opt.reps = static_cast<int>(n);
        else opt.warmup = static_cast<int>(n);
      } else throw std::invalid_argument("unknown option: " + std::string(arg));
    } else {
      if (positional) throw std::invalid_argument("only one max-size argument is accepted");
      opt.max_n = number(arg);
      positional = true;
    }
  }
  const std::string_view ops[] = {"all", "ntt", "ntt-forward", "ntt-inverse", "conv",
                                  "series", "inv", "log", "exp", "sqrt", "pow",
                                  "square", "skinny", "exp-linear", "inv-short", "pow-short",
                                  "pow-linear", "div-small", "plan", "inv-series", "scale", "inv-linear"};
  if (std::find(std::begin(ops), std::end(ops), opt.op) == std::end(ops))
    throw std::invalid_argument("unknown operation: " + std::string(opt.op));
  if (!power_of_two(opt.min_n)) throw std::invalid_argument("min-size must be a power of two >= 2");
  if (opt.op == "plan" && (opt.warmup != 0 || opt.reps > 1))
    throw std::invalid_argument("cold plans require --warmup 0 and at most one repetition; use fresh processes to repeat");
  if (opt.exact_n != 0 && (opt.op == "all" || opt.op == "plan" || opt.op.starts_with("ntt")) &&
      !power_of_two(opt.exact_n))
    throw std::invalid_argument("NTT size must be a power of two >= 2");
  if (opt.exact_n != 0) opt.min_n = opt.max_n = opt.exact_n;
  if (opt.max_n < opt.min_n) throw std::invalid_argument("max-size is below min-size");
  const size_t limit = (opt.op == "plan" || opt.op.starts_with("ntt")) ? fpx::ntt_max_size<M>()
                      : opt.op == "skinny" ? fpx::ntt_max_size<M>() - 39
                      : opt.op == "inv-series" ? M::mod - 1
                      : opt.op == "exp-linear" ? M::mod
                      : fpx::ntt_max_size<M>() / 2;
  if (opt.max_n > limit) throw std::invalid_argument("size exceeds the selected operation's NTT limit");
  return opt;
}

uint32_t limb(uint32_t x) { return x; }
uint32_t limb(M x) { return x.raw_val(); }

// Every output coefficient is observed outside the timed region. The volatile
// sink keeps this meaningful even with link-time optimization enabled.
volatile uint64_t checksum_sink = 0;
template <class V>
uint64_t checksum(const V& a) {
  uint64_t h = 14695981039346656037ull;
  for (const auto& x : a) { h ^= limb(x); h *= 1099511628211ull; }
  h ^= a.size();
  checksum_sink = h;
  return h;
}

struct Result { double best_ms, median_ms; uint64_t hash; int reps; };

// Setup (input restoration), output hashing and output destruction are untimed.
// Each repetition executes the same operation on the same deterministic input.
template <class Setup, class Run, class Observe>
Result measure(const Options& opt, int default_reps, Setup&& setup, Run&& run, Observe&& observe) {
  const int reps = opt.reps == 0 ? default_reps : opt.reps;
  auto prepare = [&] {
    setup();
#ifdef FASTPOLY_BENCH_HAS_SCRATCH_POOL
    if (opt.cold_scratch) fpx::release_scratch_memory();
#endif
  };
  for (int i = 0; i < opt.warmup; ++i) { prepare(); run(); (void)observe(); }
#ifdef FASTPOLY_BENCH_HAS_SCRATCH_POOL
  const auto before = fpx::scratch_memory_stats();
#endif
  std::vector<double> times;
  times.reserve(static_cast<size_t>(reps));
  uint64_t hash = 0;
  for (int i = 0; i < reps; ++i) {
    prepare();
    const auto t0 = clk::now();
    run();
    const auto t1 = clk::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    const uint64_t got = observe();
    if (i != 0 && got != hash) throw std::runtime_error("output changed between repetitions");
    hash = got;
  }
  std::sort(times.begin(), times.end());
  const size_t mid = times.size() / 2;
  const double median = times.size() % 2 != 0 ? times[mid] : (times[mid - 1] + times[mid]) * 0.5;
#ifdef FASTPOLY_BENCH_HAS_SCRATCH_POOL
  if (opt.memory) {
    const auto after = fpx::scratch_memory_stats();
    std::fprintf(stderr, "scratch: timed_reps=%d system_allocations=%zu cache_hits=%zu retained_bytes=%zu retained_blocks=%zu\n",
                 reps, after.system_allocations - before.system_allocations,
                 after.cache_hits - before.cache_hits, after.cached_bytes, after.cached_blocks);
  }
#endif
  return {times.front(), median, hash, reps};
}

void report(const Options& opt, const char* op, size_t n, const Result& r) {
  const double per = n > 1 ? 1e6 * r.best_ms / (double(n) * std::log2(double(n))) : 0.0;
  if (opt.csv) {
    std::printf("%s,%s,%zu,%d,%.9f,%.9f,%.6f,%016llx\n", fpx::simd_backend(), op,
                n, r.reps, r.best_ms, r.median_ms, per, static_cast<unsigned long long>(r.hash));
  } else {
    std::printf("  %-11s n=%-9zu best %9.3f ms  median %9.3f ms  %6.2f ns/(elem*log n)  checksum %016llx\n",
                op, n, r.best_ms, r.median_ms, per, static_cast<unsigned long long>(r.hash));
  }
}

vec rnd(size_t n, std::mt19937_64& rng) {
  vec a(n);
  for (auto& x : a) x = M::from_int(rng() % M::mod);
  return a;
}

void bench_ntt(const Options& opt, size_t n) {
  auto plan = fpx::NttPlan<M>::get(static_cast<uint32_t>(n));
  std::mt19937_64 rng(1 + n);
  const vec input = rnd(n, rng);
  std::vector<uint32_t> original(n), a(n);
  for (size_t i = 0; i < n; ++i) original[i] = input[i].raw_val();
  if (selected(opt, "ntt-forward"))
    report(opt, "ntt-forward", n, measure(opt, 5, [&] { a = original; },
        [&] { plan->forward(a.data()); }, [&] { return checksum(a); }));
  if (selected(opt, "ntt-inverse")) {
    plan->forward(original.data());
    report(opt, "ntt-inverse", n, measure(opt, 5, [&] { a = original; },
        [&] { plan->inverse(a.data()); }, [&] { return checksum(a); }));
  }
}

template <class F>
void bench_poly(const Options& opt, const char* op, size_t n, F&& f) {
  vec result;
  report(opt, op, n, measure(opt, 3, [&] { vec().swap(result); },
      [&] { result = f(); }, [&] { return checksum(result); }));
}

void bench_size(const Options& opt, size_t n) {
  if (opt.op == "inv-series") {
    bench_poly(opt, "inv-series", n, [&] { return fpx::poly::inv_series<M>(n); });
    return;
  }
  if (opt.op == "scale") {
    std::mt19937_64 rng(5 + n);
    const vec a = rnd(n, rng);
    bench_poly(opt, "scale", n, [&] { return fpx::poly::mul_scalar(a, M::from_int(7)); });
    return;
  }
  if (opt.op == "inv-linear") {
    const vec a{M::from_int(1), -M::from_int(7)};
    bench_poly(opt, "inv-linear", n, [&] { return fpx::poly::inv(a, n); });
    return;
  }
  if (opt.op == "inv-short") {
    const vec a{M::from_int(1), -M::from_int(1)};
    bench_poly(opt, "inv-short", n, [&] { return fpx::poly::inv(a, n); });
    return;
  }
  if (opt.op == "pow-linear") {
    const vec a{1, 1};
    bench_poly(opt, "pow-linear", n, [&] { return fpx::poly::pow(a, 1000003, n); });
    return;
  }
  if (opt.op == "pow-short") {
    const vec a{1, 1};
    bench_poly(opt, "pow-short", n, [&] { return fpx::poly::pow(a, 64, n); });
    return;
  }
  if (opt.op == "div-small") {
    std::mt19937_64 rng(4 + n);
    const vec a = rnd(n, rng), b = rnd(41, rng);
    vec q, r;
    report(opt, "div-small", n, measure(opt, 3, [&] { vec().swap(q); vec().swap(r); },
        [&] { auto result = fpx::poly::divmod(a, b); q = std::move(result.first); r = std::move(result.second); },
        [&] { return checksum(q) ^ std::rotl(checksum(r), 17); }));
    return;
  }
  if (opt.op == "plan") {
    std::mt19937_64 rng(1 + n);
    const vec input = rnd(n, rng);
    std::vector<uint32_t> a(n);
    for (size_t i = 0; i < n; ++i) a[i] = input[i].raw_val();
    const auto t0 = clk::now();
    auto plan = fpx::NttPlan<M>::get(static_cast<uint32_t>(n));
    const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    // Reading the generated tables is observable, but outside the build timer.
    plan->forward(a.data());
    report(opt, "plan", n, {ms, ms, checksum(a), 1});
    return;
  }
  if (selected(opt, "square")) {
    std::mt19937_64 rng(2 + n);
    const vec a = rnd(n, rng);
    bench_poly(opt, "square", n, [&] { return fpx::poly::conv(a, a); });
    return;
  }
  if (selected(opt, "skinny")) {
    std::mt19937_64 rng(4 + n);
    const vec a = rnd(40, rng), b = rnd(n, rng);
    bench_poly(opt, "skinny", n, [&] { return fpx::poly::conv(a, b); });
    return;
  }
  if (selected(opt, "exp-linear")) {
    const vec a{M(), M::from_int(1)};
    bench_poly(opt, "exp-linear", n, [&] { return fpx::poly::exp(a, n); });
    return;
  }
  if (selected(opt, "ntt-forward") || selected(opt, "ntt-inverse")) bench_ntt(opt, n);
  if (selected(opt, "conv")) {
    std::mt19937_64 rng(2 + n);
    const vec a = rnd(n, rng), b = rnd(n, rng);
    bench_poly(opt, "conv", n, [&] { return fpx::poly::conv(a, b); });
  }
  if (opt.op == "ntt" || opt.op.starts_with("ntt-") || opt.op == "conv") return;
  std::mt19937_64 rng(3 + n);
  vec a = rnd(n, rng);
  a[0] = M::from_int(1);
  if (selected(opt, "inv")) bench_poly(opt, "inv", n, [&] { return fpx::poly::inv(a, n); });
  if (selected(opt, "log")) bench_poly(opt, "log", n, [&] { return fpx::poly::log(a, n); });
  if (selected(opt, "exp")) {
    vec b = a;
    b[0] = M();
    bench_poly(opt, "exp", n, [&] { return fpx::poly::exp(b, n); });
  }
  if (selected(opt, "sqrt")) {
    const vec sq = fpx::poly::conv(a, a, n);
    bench_poly(opt, "sqrt", n, [&] { return fpx::poly::sqrt(sq, n); });
  }
  if (selected(opt, "pow"))
    bench_poly(opt, "pow", n, [&] { return fpx::poly::pow(a, 1000003, n); });
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--help") { usage(); return 0; }
  }
  try {
    const Options opt = parse(argc, argv);
    if (opt.csv) std::printf("backend,operation,size,reps,best_ms,median_ms,ns_per_elem_log2,checksum\n");
    else std::printf("fastpoly benchmark | simd backend: %s (%d x uint32 lanes) | mod = %u | warmup = %d\n",
                     fpx::simd_backend(), fpx::simd::lane, M::mod, opt.warmup);
    for (size_t n = opt.min_n; n <= opt.max_n; n <<= 1) bench_size(opt, n);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fastpoly_bench: %s\n", e.what());
    return 1;
  }
}
