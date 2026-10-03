// fastpoly - tests for polynomial arithmetic mod x^n.
#include <random>
#include <vector>

#include "fastpoly/poly.hpp"
#include "fp_test.hpp"

using namespace fpx;
using M = mod998244353;
using vec = std::vector<M>;

namespace {

std::mt19937_64 rng(0x13572468);

vec rnd(size_t n) {
  vec a(n);
  for (auto& x : a) x = M::from_int(rng() % M::mod);
  return a;
}
vec rnd_nonzero0(size_t n) {
  vec a = rnd(n);
  if (a[0].is_zero()) a[0] = M::from_int(1);
  return a;
}
vec rnd_unit0(size_t n) {  // constant term 1
  vec a = rnd(n);
  a[0] = M::from_int(1);
  return a;
}
vec rnd_zero0(size_t n) {  // zero constant term
  vec a = rnd(n);
  a[0] = M();
  return a;
}

vec naive_conv(const vec& a, const vec& b, size_t n) {
  vec r(std::min(a.size() + b.size() - 1, n), M());
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < b.size() && i + j < r.size(); ++j) r[i + j] += a[i] * b[j];
  r.resize(n, M());
  return r;
}

template <class V>
bool same(const V& a, const V& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return false;
  return true;
}

template <class V>
V padded(const V& a, size_t n) {
  V r(n, typename V::value_type());
  for (size_t i = 0; i < std::min(a.size(), n); ++i) r[i] = a[i];
  return r;
}

/// The algebraic identities must hold for every NTT prime, not just the
/// default one: v2(mod-1), primitive root and mod % 8 all differ between them.
template <class M>
void series_identities(const char* name) {
  std::mt19937_64 r(0xfeed);
  for (int t = 0; t < 10; ++t) {
    const size_t n = 1 + r() % 150;
    std::vector<M> a(n), b(n);
    for (auto& x : a) x = M::from_int(r() % M::mod);
    for (auto& x : b) x = M::from_int(r() % M::mod);
    a[0] = M::from_int(1);
    b[0] = M();

    // a * inv(a) == 1
    std::vector<M> prod = poly::conv(a, poly::inv(a, n), n);
    prod.resize(n, M());
    for (size_t i = 0; i < n; ++i) CHECK_EQ(prod[i], i == 0 ? M::from_int(1) : M());
    // log/exp are inverse to each other
    CHECK(same(padded(poly::exp(poly::log(a, n), n), n), a));
    CHECK(same(padded(poly::log(poly::exp(b, n), n), n), b));
    // sqrt(a^2) == +-a : check by squaring
    std::vector<M> sq = poly::conv(a, a, n);
    sq.resize(n, M());
    try {
      std::vector<M> s = poly::sqrt(sq, n);
      CHECK(same(padded(poly::conv(s, s, n), n), sq));
    } catch (const poly::domain_error&) {
      CHECK(false);  // a perfect square must always have a root
    }
    // pow
    std::vector<M> p3 = poly::pow(a, 7, n);
    std::vector<M> acc(n, M());
    acc[0] = M::from_int(1);
    for (int i = 0; i < 7; ++i) {
      acc = poly::conv(acc, a, n);
      acc.resize(n, M());
    }
    CHECK(same(p3, acc));
    // big exponent through the exp/log route
    std::vector<M> big = poly::pow(a, 1000000007ull, n);
    std::vector<M> alt = poly::exp(poly::mul_scalar(poly::log(a, n), M::from_int(1000000007ull)), n);
    CHECK(same(big, alt));
  }
  std::printf("  series identities over %s ok\n", name);
}

}  // namespace

FP_TEST(poly_identities_mod469762049) { series_identities<mod469762049>("469762049"); }
FP_TEST(poly_identities_mod167772161) { series_identities<mod167772161>("167772161"); }
FP_TEST(poly_identities_mod754974721) { series_identities<mod754974721>("754974721"); }
FP_TEST(poly_identities_mod1224736769) { series_identities<mod1224736769>("1224736769"); }
FP_TEST(poly_identities_mod1004535809) { series_identities<mod1004535809>("1004535809"); }

FP_TEST(poly_basic_ops) {
  const vec a{1, 2, 3}, b{4, 5};
  CHECK_EQ(poly::add(a, b).size(), 3u);
  CHECK_EQ(poly::add(a, b)[0].val(), 5u);
  CHECK_EQ(poly::sub(a, b)[2].val(), 3u);
  CHECK_EQ(poly::neg(a)[0].val(), M::mod - 1);
  CHECK_EQ(poly::mul_scalar(a, M::from_int(2))[1].val(), 4u);
  CHECK_EQ(poly::shift(a, 2).size(), 5u);
  CHECK_EQ(poly::shift(a, 2)[3].val(), 2u);
  CHECK_EQ(poly::shift(a, 2, 3).size(), 3u);
  CHECK_EQ(poly::eval(a, M::from_int(2)).val(), 1u + 2u * 2u + 3u * 4u);
  // trim
  vec t{1, 0, 0};
  poly::trim(t);
  CHECK_EQ(t.size(), 1u);
  // derivative / integral are inverse up to the constant term
  for (int i = 0; i < 50; ++i) {
    vec p = rnd(1 + rng() % 60);
    vec q = poly::integral(poly::derivative(p));
    q.resize(p.size(), M());
    q[0] = p[0];
    CHECK(same(q, padded(p, q.size())));
  }
}

FP_TEST(poly_convolution) {
  for (int t = 0; t < 120; ++t) {
    const size_t la = 1 + rng() % 200, lb = 1 + rng() % 200;
    const size_t n = 1 + rng() % 300;
    vec a = rnd(la), b = rnd(lb);
    auto got = poly::conv(a, b, n);
    CHECK_EQ(got.size(), std::min(la + lb - 1, n));
    CHECK(same(padded(got, n), naive_conv(a, b, n)));
    if (la * lb <= 40000) CHECK(same(padded(poly::conv(a, b), n), naive_conv(a, b, n)));
  }
  // empty operand
  CHECK(poly::conv(vec{}, vec{1, 2}).empty());
  CHECK(poly::conv(vec{1, 2}, vec{1, 2}, 0).empty());
  // the naive/non-naive cross-over must be seamless
  for (size_t len : {size_t(39), size_t(40), size_t(41), size_t(80), size_t(81)}) {
    vec a = rnd(len), b = rnd(len);
    CHECK(same(padded(poly::conv(a, b), 2 * len), naive_conv(a, b, 2 * len)));
  }
}

FP_TEST(poly_inverse) {
  for (int t = 0; t < 80; ++t) {
    const size_t na = 1 + rng() % 150, n = 1 + rng() % 250;
    vec a = rnd_nonzero0(na);
    vec b = poly::inv(a, n);
    CHECK_EQ(b.size(), n);
    vec prod = naive_conv(a, b, n);
    for (size_t i = 0; i < n; ++i)
      CHECK_MSG(prod[i] == (i == 0 ? M::from_int(1) : M()), "inv t=%d n=%zu i=%zu", t, n, i);
  }
  // 1/(1-x) = 1 + x + x^2 + ...
  { vec a{M::from_int(1), -M::from_int(1)};
    vec b = poly::inv(a, 20);
    for (size_t i = 0; i < 20; ++i) CHECK_EQ(b[i].val(), 1u); }
  // constant polynomial
  { vec a{M::from_int(7)};
    vec b = poly::inv(a, 5);
    CHECK_EQ(b[0].val(), M::from_int(7).inv().val());
    for (size_t i = 1; i < 5; ++i) CHECK(b[i].is_zero()); }
  // a(0) == 0 must be rejected
  bool threw = false;
  try { poly::inv(vec{M(), M::from_int(1)}, 4); } catch (const poly::domain_error&) { threw = true; }
  CHECK(threw);
}

FP_TEST(poly_log_exp) {
  for (int t = 0; t < 60; ++t) {
    const size_t n = 1 + rng() % 400;
    vec b = padded(rnd_zero0(1 + rng() % n), n);
    vec a = padded(rnd_unit0(1 + rng() % n), n);
    vec e = poly::exp(b, n), l = poly::log(a, n);
    CHECK(same(poly::log(e, n), b));   // log(exp(b)) == b
    CHECK(same(poly::exp(l, n), a));   // exp(log(a)) == a
  }
  // exp of 0 is 1
  CHECK_EQ(poly::exp(vec{M()}, 4)[0].val(), 1u);
  CHECK(poly::exp(vec{M()}, 4)[3].is_zero());
  // log(1) == 0 and log(1+x) = x - x^2/2 + x^3/3
  CHECK(poly::log(vec{M::from_int(1)}, 5).empty() || poly::log(vec{M::from_int(1)}, 5)[4].val() == 0u);
  { vec l = poly::log(vec{M::from_int(1), M::from_int(1)}, 6);
    CHECK_EQ(l[1].val(), 1u);
    CHECK_EQ(l[2].val(), (M::mod - M::inv2) % M::mod);  // -1/2
    CHECK_EQ((l[3] * M::from_int(3)).val(), 1u);        // 1/3
  }
  // exp(x) = sum x^i / i!
  { vec e = poly::exp(vec{M(), M::from_int(1)}, 8), f{M::from_int(1)};
    for (size_t i = 0; i < 8; ++i) {
      CHECK_EQ(e[i], f[i]);
      f.push_back(f.back() * M::from_int(i + 1).inv());
    }
    CHECK_EQ(e[7].val(), f[7].val());
  }
  bool threw = false;
  try { poly::exp(vec{M::from_int(3)}, 4); } catch (const poly::domain_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { poly::log(vec{M::from_int(3)}, 4); } catch (const poly::domain_error&) { threw = true; }
  CHECK(threw);
}

FP_TEST(poly_exp_derivative_identity) {
  // exp(a)' == exp(a) * a'  -- independent characterisation of exp
  for (int t = 0; t < 40; ++t) {
    const size_t n = 2 + rng() % 300;
    vec a = padded(rnd_zero0(1 + rng() % n), n);
    vec e = poly::exp(a, n);
    vec lhs = poly::derivative(e);
    vec rhs = naive_conv(e, poly::derivative(a), n - 1);
    CHECK(same(lhs, rhs));
  }
}

FP_TEST(poly_sqrt) {
  for (int t = 0; t < 40; ++t) {
    const size_t n = 1 + rng() % 300;
    vec a = rnd_nonzero0(1 + rng() % n);
    vec s;
    bool ok = true;
    try { s = poly::sqrt(a, n); } catch (const poly::domain_error&) { ok = false; }
    if (!ok) continue;
    CHECK(same(padded(naive_conv(s, s, n), n), padded(a, n)));
  }
  // sqrt of a perfect square always exists
  for (int t = 0; t < 20; ++t) {
    const size_t n = 1 + rng() % 200;
    vec r = rnd_nonzero0(1 + rng() % 60);
    vec a = padded(poly::conv(r, r), n);
    CHECK(same(padded(naive_conv(poly::sqrt(a, n), poly::sqrt(a, n), n), n), padded(a, n)));
  }
  // x^2 * u  ->  x * sqrt(u)
  { vec u = rnd_nonzero0(1 + rng() % 20);
    vec a{M(), M()};
    for (auto x : u) a.push_back(x);
    vec s = poly::sqrt(a, 10);
    CHECK(s[0].is_zero());
    CHECK(same(padded(naive_conv(s, s, 10), 10), padded(a, 10))); }
  // odd valuation -> no square root
  bool threw = false;
  try { poly::sqrt(vec{M(), M::from_int(4)}, 5); } catch (const poly::domain_error&) { threw = true; }
  CHECK(threw);
  // zero polynomial
  CHECK(poly::sqrt(vec{M(), M(), M()}, 4)[3].is_zero());
}

FP_TEST(poly_pow) {
  for (int t = 0; t < 100; ++t) {
    const size_t n = 1 + rng() % 200;
    vec a = rnd(1 + rng() % 50);
    if (t % 4 == 0) a[0] = M();                    // positive valuation
    const uint64_t k = (t % 3 == 0) ? (rng() % 6)
                       : (t % 3 == 1) ? (rng() % 65)
                                      : (rng() % 1000000 + 200);
    vec got;
    try { got = poly::pow(a, k, n); } catch (const poly::domain_error&) { continue; }
    CHECK_EQ(got.size(), n);
    // brute force reference for small k
    if (k <= 40) {
      vec ref = poly::pow(a, k, n);  // (also checks determinism)
      vec acc(n, M());
      acc[0] = M::from_int(1);
      for (uint64_t i = 0; i < k; ++i) acc = naive_conv(acc, a, n);
      CHECK(same(got, acc));
      CHECK(same(ref, acc));
    }
    // a^k * a == a^(k+1)
    if (k > 0) {
      vec nxt = poly::pow(a, k + 1, n);
      CHECK(same(naive_conv(got, a, n), nxt));
    }
    // pow with the two internal routes (binary exponentiation vs exp/log) must
    // agree; the exp/log route needs a(0) == 1 so that no leading scalar is lost
    if (k > 64 && a[0] == M::from_int(1)) {
      vec alt = poly::exp(poly::mul_scalar(poly::log(a, n), M::from_int(k)), n);
      CHECK(same(alt, got));
    }
  }
  // (1+x)^k truncated: binomial coefficients
  { vec a{M::from_int(1), M::from_int(1)};
    vec g = poly::pow(a, 100, 6);
    // C(100,0..5) mod p
    uint64_t c = 1;
    for (uint32_t i = 0; i < 6; ++i) {
      CHECK_EQ(g[i].val(), c % M::mod);
      c = c * (100 - i) / (i + 1);
    } }
  // a^0 == 1, a^1 == a
  { vec a = rnd(5);
    CHECK_EQ(poly::pow(a, 0, 4)[0].val(), 1u);
    CHECK_EQ(poly::pow(a, 0, 4)[3].val(), 0u);
    CHECK(same(padded(poly::pow(a, 1, 5), 5), padded(a, 5))); }
  // x^k with k*v >= n collapses to zero
  { vec a{M(), M(), M::from_int(1)};
    vec g = poly::pow(a, 10, 5);
    for (size_t i = 0; i < 5; ++i) CHECK(g[i].is_zero()); }
}

FP_TEST(poly_divmod) {
  for (int t = 0; t < 200; ++t) {
    const size_t db = 1 + rng() % 50, dq = 1 + rng() % 50;
    vec b = rnd(db + 1);
    b[db] = M::from_int(1 + rng() % (M::mod - 1));  // make deg(b) exact
    if (t % 7 == 0) b[0] = M();
    vec q = rnd(dq + 1);
    vec a = naive_conv(b, q, db + dq + 2);
    if (rng() % 2) a.resize(a.size() - 1);
    a.resize(db + dq + 2, M());
    auto pr = poly::divmod(a, b);
    CHECK(pr.second.size() <= db);
    vec back = naive_conv(b, pr.first, a.size());
    for (size_t i = 0; i < pr.second.size(); ++i) back[i] += pr.second[i];
    CHECK(same(padded(back, a.size()), padded(a, a.size())));
    // div/mod agree with divmod
    CHECK(same(poly::div(a, b), pr.first));
    CHECK(same(poly::mod(a, b), pr.second));
  }
  // deg(a) < deg(b) -> q = 0, r = a
  { vec a{1, 2}, b{1, 1, 1};
    auto pr = poly::divmod(a, b);
    CHECK(pr.first.empty());
    CHECK_EQ(pr.second.size(), 2u); }
  // division by the zero polynomial must throw
  bool threw = false;
  try { poly::divmod(vec{1, 2}, vec{M(), M()}); } catch (const poly::domain_error&) { threw = true; }
  CHECK(threw);
}

FP_TEST(poly_class_api) {
  Poly<M> a{1, 2, 3}, b{4, 5};
  CHECK_EQ((a + b)[2].val(), 3u);
  CHECK_EQ((a - b)[0].val(), (M::mod - 3) % M::mod);
  CHECK_EQ((a * b).size(), 4u);
  CHECK_EQ((-a)[0].val(), M::mod - 1);
  CHECK_EQ((a << 1).size(), 4u);
  CHECK_EQ(a.degree(), 2u);
  CHECK(!a.is_zero());
  Poly<M> zero_poly{M(), M()};
  CHECK(zero_poly.is_zero());
  CHECK_EQ(a.eval(M::from_int(1)).val(), 6u);
  CHECK_EQ(a.mul(b, 3).size(), 3u);
  CHECK_EQ(a.derivative().size(), 2u);
  CHECK_EQ(a.integral().size(), 4u);
  Poly<M> inv = a.inv(6);
  CHECK_EQ(inv.size(), 6u);
  CHECK_EQ(a.to_ints()[0], 1u);
  CHECK_EQ(Poly<M>::from_ints({1, 2, 3})[2].val(), 3u);
  Poly<M> e = Poly<M>{M()}.exp(5);  // exp(0) = 1
  CHECK_EQ(e[0].val(), 1u);
  CHECK_EQ(a.pow(2, 4).size(), 4u);
  auto pr = a.divmod(b);
  CHECK_EQ(pr.first.size() + pr.second.size() > 0, true);
}

FP_TEST(poly_large_sizes) {
  // sizes that cross several NTT doublings plus a non power of two boundary
  for (size_t n : {size_t(1000), size_t(1024), size_t(1025), size_t(4096), size_t(10000)}) {
    vec a = rnd_nonzero0(1 + rng() % n);
    vec b = poly::inv(a, n);
    vec prod = naive_conv(a, b, n);
    for (size_t i = 0; i < n; ++i)
      CHECK_MSG(prod[i] == (i == 0 ? M::from_int(1) : M()), "large inv n=%zu i=%zu", n, i);
  }
}

FP_TEST_MAIN()
