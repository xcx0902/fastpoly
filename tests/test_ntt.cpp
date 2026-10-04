// fastpoly - tests for the NTT plan (correctness across sizes, moduli, arches).
#include <algorithm>
#include <array>
#include <barrier>
#include <cstddef>
#include <limits>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "fastpoly/modint.hpp"
#include "fastpoly/ntt.hpp"
#include "fp_test.hpp"

using namespace fpx;

namespace {

std::mt19937_64 rng(0xabcdef);

template <class M>
std::vector<uint32_t> naive_conv(const std::vector<uint32_t>& a,
                                 const std::vector<uint32_t>& b) {
  std::vector<uint32_t> r(a.size() + b.size() - 1, 0);
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < b.size(); ++j)
      r[i + j] = detail::s_add(r[i + j], M::reduce(uint64_t(a[i]) * b[j]), M::mod);
  return r;
}

/// forward followed by inverse must be the identity, for every supported size
/// (odd log2(n) exercises the trailing radix-2 stage).
template <class M>
void roundtrip(int max_log) {
  for (int k = 1; k <= max_log; ++k) {
    const uint32_t n = 1u << k;
    auto plan = NttPlan<M>::get(n);
    std::vector<uint32_t> a(n), b(n);
    for (uint32_t i = 0; i < n; ++i) a[i] = static_cast<uint32_t>(rng() % M::mod);
    b = a;
    plan->forward(b.data());
    plan->inverse(b.data());
    for (uint32_t i = 0; i < n; ++i) CHECK_MSG(a[i] == b[i], "roundtrip n=%u i=%u", n, i);
  }
}

/// A monomial x^j must transform to a permutation of {w^(j*k)}. This pins down
/// the transform as the true DFT, not merely a self-consistent pair.
template <class M>
void monomial_is_dft(int max_log) {
  for (int k = 1; k <= max_log; ++k) {
    const uint32_t n = 1u << k;
    auto plan = NttPlan<M>::get(n);
    const M w = M::from_int(M::primitive_root).pow((M::mod - 1) / n);
    for (uint32_t j = 0; j < n; j += std::max<uint32_t>(1, n / 4)) {
      std::vector<uint32_t> a(n, 0);
      a[j] = M::from_int(1).raw_val();
      plan->forward(a.data());
      std::multiset<uint32_t> got(a.begin(), a.end()), want;
      const M wj = w.pow(j);
      M cur = M::from_int(1);
      for (uint32_t t = 0; t < n; ++t) {
        want.insert(cur.raw_val());
        cur = cur * wj;
      }
      CHECK_MSG(got == want, "monomial DFT n=%u j=%u", n, j);
    }
  }
}

template <class M>
void convolution_vs_naive(int trials, size_t max_len) {
  for (int t = 0; t < trials; ++t) {
    const size_t la = 2 + rng() % max_len, lb = 2 + rng() % max_len;
    std::vector<uint32_t> a(la), b(lb);
    for (auto& x : a) x = static_cast<uint32_t>(rng() % M::mod);
    for (auto& x : b) x = static_cast<uint32_t>(rng() % M::mod);
    if (t % 9 == 0) { a[0] = M::mod - 1; b[0] = 0; }
    const uint32_t N = next_pow2(la + lb - 1);
    auto plan = NttPlan<M>::get(N);
    std::vector<uint32_t> fa(N, 0), fb(N, 0);
    std::copy(a.begin(), a.end(), fa.begin());
    std::copy(b.begin(), b.end(), fb.begin());
    plan->forward(fa.data());
    plan->forward(fb.data());
    for (uint32_t i = 0; i < N; ++i) fa[i] = M::reduce(uint64_t(fa[i]) * fb[i]);
    plan->inverse(fa.data());
    const auto want = naive_conv<M>(a, b);
    CHECK_EQ(want.size(), la + lb - 1);
    for (size_t i = 0; i < want.size(); ++i)
      CHECK_MSG(fa[i] == want[i], "conv t=%d i=%zu (la=%zu lb=%zu N=%u)", t, i, la, lb, N);
  }
}

template <class M>
void check_modulus(const char* name, int max_log = 12) {
  roundtrip<M>(max_log);
  monomial_is_dft<M>(std::min(max_log, 10));
  convolution_vs_naive<M>(60, 60);
  CHECK_EQ(ntt_max_size<M>(), 1u << ntt_max_log<M>());
  std::printf("  modulus %s ok\n", name);
}

}  // namespace

FP_TEST(ntt_mod998244353) { check_modulus<mod998244353>("998244353", 13); }
FP_TEST(ntt_mod1004535809) { check_modulus<mod1004535809>("1004535809"); }
FP_TEST(ntt_mod469762049) { check_modulus<mod469762049>("469762049"); }
FP_TEST(ntt_mod167772161) { check_modulus<mod167772161>("167772161"); }
FP_TEST(ntt_mod754974721) { check_modulus<mod754974721>("754974721"); }
FP_TEST(ntt_mod1224736769) { check_modulus<mod1224736769>("1224736769"); }

FP_TEST(ntt_small_sizes) {
  // smallest sizes, where every butterfly falls back to the scalar tail
  using M = mod998244353;
  for (uint32_t n : {2u, 4u, 8u, 16u, 32u}) {
    auto plan = NttPlan<M>::get(n);
    std::vector<uint32_t> a(n), b(n);
    for (uint32_t i = 0; i < n; ++i) a[i] = M::from_int(i + 1).raw_val();
    b = a;
    plan->forward(b.data());
    plan->inverse(b.data());
    for (uint32_t i = 0; i < n; ++i) CHECK_EQ(a[i], b[i]);
  }
}

/// The chunked small-m kernels only take over once `4*m <= n` and `m < lane`;
/// at sizes well past that boundary the transform must still be a true DFT (the
/// monomial check is O(n log n), so it scales where the naive one cannot) and
/// forward/inverse must still round-trip bit-exactly.  Both parities of
/// log2(n) are covered, which selects different `m` sequences (…4,1 vs …2).
FP_TEST(ntt_large_small_m_kernels) {
  using M = mod998244353;
  for (int k : {14, 15, 16, 17}) {
    const uint32_t n = 1u << k;
    auto plan = NttPlan<M>::get(n);
    const M w = M::from_int(M::primitive_root).pow((M::mod - 1) / n);
    for (uint32_t j = 0; j < n; j += std::max<uint32_t>(1, n / 8)) {
      std::vector<uint32_t> a(n, 0);
      a[j] = M::from_int(1).raw_val();
      plan->forward(a.data());
      std::multiset<uint32_t> got(a.begin(), a.end()), want;
      const M wj = w.pow(j);
      M cur = M::from_int(1);
      for (uint32_t t = 0; t < n; ++t) {
        want.insert(cur.raw_val());
        cur = cur * wj;
      }
      CHECK_MSG(got == want, "large monomial DFT n=%u j=%u", n, j);
    }
    std::vector<uint32_t> b(n), c(n);
    for (uint32_t i = 0; i < n; ++i) b[i] = M::from_int(rng() % M::mod).raw_val();
    c = b;
    plan->forward(c.data());
    plan->inverse(c.data());
    for (uint32_t i = 0; i < n; ++i) CHECK_MSG(b[i] == c[i], "large roundtrip n=%u i=%u", n, i);
  }
}

FP_TEST(ntt_plan_rejects_bad_sizes) {
  using M = mod998244353;
  bool threw = false;
  try {
    NttPlan<M>::get(3);
  } catch (const ntt_size_error&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    NttPlan<M>::get(1);
  } catch (const ntt_size_error&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {  // 998244353 - 1 is divisible by 2^23 only
    NttPlan<M>::get(1u << 24);
  } catch (const ntt_size_error&) {
    threw = true;
  }
  CHECK(threw);
  for (uint32_t n : {0u, 1u, 3u, 1u << 24}) {
    threw = false;
    try { (void)NttPlan<M>::get_ref(n); }
    catch (const ntt_size_error&) { threw = true; }
    CHECK(threw);
  }
  CHECK_EQ(ntt_max_log<mod998244353>(), 23);
  CHECK_EQ(next_pow2(1), 1u);
  CHECK_EQ(next_pow2(1025), 2048u);
}

FP_TEST(ntt_plan_is_cached) {
  using M = mod998244353;
  const auto owning = NttPlan<M>::get(1024);
  CHECK(owning == NttPlan<M>::get(1024));
  CHECK(&NttPlan<M>::get_ref(1024) == owning.get());
  CHECK_EQ(NttPlan<M>::get_ref(1024).size(), 1024u);
}

FP_TEST(ntt_plan_value_semantics) {
  using M = mod998244353;
  for (uint32_t n : {2u, 64u, 128u}) {
    const auto& cached = NttPlan<M>::get_ref(n);
    NttPlan<M> copy(cached);
    NttPlan<M> assigned(NttPlan<M>::get_ref(8));
    assigned = copy;
    std::vector<uint32_t> input(n), actual(n), expected(n);
    for (auto& x : input) x = M::from_int(rng() % M::mod).raw_val();
    actual = expected = input;
    copy.forward(actual.data());
    cached.forward(expected.data());
    CHECK(actual == expected);
    assigned.inverse(actual.data());
    CHECK(actual == input);
  }
}

FP_TEST(ntt_constant_and_impulse) {
  using M = mod998244353;
  const uint32_t n = 64;
  auto plan = NttPlan<M>::get(n);
  // constant polynomial -> only the k = 0 slot set (up to the output order)
  std::vector<uint32_t> a(n, M::from_int(5).raw_val());
  plan->forward(a.data());
  int nonzero = 0;
  for (uint32_t i = 0; i < n; ++i) nonzero += (a[i] != 0);
  CHECK_EQ(nonzero, 1);
  // the k = 0 slot is always output index 0 and equals n * a(0)
  CHECK_EQ(a[0], M::from_int(5ull * n).raw_val());
}

namespace {

template <class Field>
void lazy_forward_contract() {
  std::mt19937_64 gen(0x785a);
  constexpr uint32_t sentinel = 0xdecafbad;
  for (int log_n = 1; log_n <= 17; ++log_n) {
    const uint32_t n = 1u << log_n;
    const auto plan = NttPlan<Field>::get(n);
    std::vector<uint32_t> input(n + 2, sentinel);
    for (uint32_t i = 1; i <= n; ++i)
      input[i] = i % 7 == 0 ? Field::mod - 1 : i % 7 == 1 ? 0 : static_cast<uint32_t>(gen() % Field::mod);
    auto canonical = input, lazy = input;
    // Offset one limb also checks the public unaligned-buffer contract.
    plan->forward(canonical.data() + 1);
    plan->forward_lazy(lazy.data() + 1);
    CHECK_EQ(canonical.front(), sentinel);
    CHECK_EQ(canonical.back(), sentinel);
    CHECK_EQ(lazy.front(), sentinel);
    CHECK_EQ(lazy.back(), sentinel);
    for (uint32_t i = 1; i <= n; ++i) {
      CHECK_MSG(canonical[i] < Field::mod, "canonical bounds mod=%u n=%u i=%u", Field::mod, n, i);
      CHECK_MSG(lazy[i] < simd::rmod<Field::mod>, "lazy bounds mod=%u n=%u i=%u", Field::mod, n, i);
      CHECK_MSG(lazy[i] % Field::mod == canonical[i], "lazy residue mod=%u n=%u i=%u", Field::mod, n, i);
    }
    plan->inverse(lazy.data() + 1);
    CHECK_MSG(lazy == input, "lazy roundtrip mod=%u n=%u", Field::mod, n);
    if constexpr (simd::lazy_ok<Field::mod>) {
      // Exercise the entire allowed inverse input interval, independently of
      // which representatives the forward implementation happened to choose.
      for (uint32_t i = 1; i <= n; ++i) canonical[i] += Field::mod;
      plan->inverse(canonical.data() + 1);
      CHECK_MSG(canonical == input, "lifted inverse mod=%u n=%u", Field::mod, n);
    }
  }
}

}  // namespace

FP_TEST(ntt_lazy_contract_all_moduli) {
  lazy_forward_contract<mod998244353>();
  lazy_forward_contract<mod1004535809>();
  lazy_forward_contract<mod469762049>();
  lazy_forward_contract<mod167772161>();
  lazy_forward_contract<mod754974721>();
  lazy_forward_contract<mod1224736769>();
}

FP_TEST(ntt_concurrent_cold_plans) {
  // 3^5 is another primitive root because gcd(5, 998244352) == 1. This
  // distinct template instantiation gives the test a genuinely cold cache.
  using Field = Mont<998244353, 243>;
  using Plan = NttPlan<Field>;
  constexpr size_t threads = 8;
  constexpr std::array<uint32_t, threads> sizes{256, 512, 2048, 256, 512, 2048, 256, 512};
  std::array<std::shared_ptr<const Plan>, threads> plans;
  std::array<bool, threads> correct{};
  std::barrier start(static_cast<std::ptrdiff_t>(threads));
  std::array<std::thread, threads> workers;
  for (size_t t = 0; t < threads; ++t) {
    workers[t] = std::thread([&, t] {
      start.arrive_and_wait();
      try {
        const uint32_t n = sizes[t];
        if (t & 1) plans[t] = Plan::get(n);
        const auto& borrowed = Plan::get_ref(n);
        if (!(t & 1)) plans[t] = Plan::get(n);
        std::vector<uint32_t> a(n);
        for (uint32_t i = 0; i < n; ++i)
          a[i] = Field::from_int(uint64_t(i + 1) * (t + 1)).raw_val();
        const auto original = a;
        borrowed.forward_lazy(a.data());
        borrowed.inverse(a.data());
        correct[t] = a == original && plans[t] == Plan::get(n) &&
                     &borrowed == plans[t].get();
      } catch (...) {
        correct[t] = false;
      }
    });
  }
  for (auto& worker : workers) worker.join();
  for (size_t t = 0; t < threads; ++t) {
    CHECK_MSG(correct[t], "concurrent transform thread=%zu n=%u", t, sizes[t]);
    CHECK(plans[t] == Plan::get(sizes[t]));
    CHECK(plans[t].get() == &Plan::get_ref(sizes[t]));
    for (size_t u = 0; u < t; ++u)
      CHECK((plans[t] == plans[u]) == (sizes[t] == sizes[u]));
  }
}

FP_TEST(ntt_next_pow2_boundaries) {
  CHECK_EQ(next_pow2(0), 1u);
  for (uint32_t log_n = 1; log_n <= 31; ++log_n) {
    const uint32_t n = 1u << log_n;
    CHECK_EQ(next_pow2(n), n);
    if (n > 2) CHECK_EQ(next_pow2(n - 1), n);
    CHECK_EQ(next_pow2(uint64_t(n / 2) + 1), n);
  }
  for (const uint64_t invalid : {(uint64_t(1) << 31) + 1, uint64_t(1) << 32,
                                 std::numeric_limits<uint64_t>::max()}) {
    bool threw = false;
    try { (void)next_pow2(invalid); }
    catch (const ntt_size_error&) { threw = true; }
    CHECK_MSG(threw, "next_pow2 must reject %llu", static_cast<unsigned long long>(invalid));
  }
}

FP_TEST_MAIN()
