// fastpoly - tests for polynomial arithmetic mod x^n.
#include <cstddef>
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

/// Cross-validation at production sizes: the series operations are checked
/// through `conv`, which is itself validated against the O(n^2) definition.
FP_TEST(poly_large_series_crosscheck) {
  for (const size_t n : {size_t(65536), size_t(100000)}) {
    const vec a = rnd_unit0(n);  // a(0) = 1, n coefficients
    // a * inv(a) == 1 mod x^n
    vec prod = poly::conv(a, poly::inv(a, n), n);
    prod.resize(n, M());
    for (size_t i = 0; i < n; i += 9973)
      CHECK_MSG(prod[i] == (i == 0 ? M::from_int(1) : M()), "large inv n=%zu i=%zu", n, i);
    // exp(log(a)) == a
    CHECK(same(poly::exp(poly::log(a, n), n), a));
    // exp(b)' == exp(b) * b'
    vec b = rnd_zero0(n);
    vec eb = poly::exp(b, n);
    CHECK(same(padded(poly::conv(eb, poly::derivative(b), n - 1), n - 1),
               padded(poly::derivative(eb), n - 1)));
    // sqrt(a^2) squared back
    vec sq = poly::conv(a, a, n);
    sq.resize(n, M());
    vec s = poly::sqrt(sq, n);
    vec back = poly::conv(s, s, n);
    back.resize(n, M());
    CHECK(same(back, sq));
    // pow (exp/log route) is consistent with one extra multiplication
    const uint64_t k = 1000003;
    vec pk = poly::pow(a, k, n), pk1 = poly::pow(a, k - 1, n);
    vec chk = poly::conv(pk1, a, n);
    chk.resize(n, M());
    CHECK(same(chk, pk));
  }
  std::printf("  large series cross-check ok\n");
}

namespace {

template <class Field>
std::vector<Field> reference_product(const std::vector<Field>& a,
                                     const std::vector<Field>& b, size_t n) {
  if (a.empty() || b.empty() || n == 0) return {};
  std::vector<Field> out(std::min(n, a.size() + b.size() - 1));
  for (size_t i = 0; i < std::min(a.size(), out.size()); ++i)
    for (size_t j = 0; j < b.size() && i + j < out.size(); ++j)
      out[i + j] += a[i] * b[j];
  return out;
}

// Coefficient recurrences provide independent references for the Newton paths.
template <class Field>
std::vector<Field> reference_inverse(const std::vector<Field>& a, size_t n) {
  std::vector<Field> out(n);
  out[0] = a[0].inv();
  for (size_t i = 1; i < n; ++i) {
    Field sum;
    for (size_t j = 1; j <= i && j < a.size(); ++j) sum += a[j] * out[i - j];
    out[i] = -sum * out[0];
  }
  return out;
}

template <class Field>
std::vector<Field> reference_exp(const std::vector<Field>& a, size_t n) {
  std::vector<Field> out(n);
  out[0] = Field::from_int(1);
  for (size_t i = 1; i < n; ++i) {
    Field sum;
    for (size_t j = 1; j <= i && j < a.size(); ++j)
      sum += Field::from_int(j) * a[j] * out[i - j];
    out[i] = sum / Field::from_int(i);
  }
  return out;
}

template <class Field>
void optimized_poly_boundaries() {
  std::mt19937_64 gen(0x8246);
  auto random = [&](size_t n) {
    std::vector<Field> a(n);
    for (size_t i = 0; i < n; ++i) {
      // Exercise canonical extrema in every vector, as well as random limbs.
      a[i] = Field::from_int(i % 7 == 0 ? Field::mod - 1 : gen() % Field::mod);
    }
    return a;
  };

  // The streamed direct-convolution path must handle either argument order,
  // output clipping, unaligned stores and vector tails at the 40 crossover.
  const auto long_operand = random(4097);
  for (const size_t short_n : {size_t(1), size_t(3), size_t(7), size_t(17),
                               size_t(31), size_t(39), size_t(40), size_t(41)}) {
    const auto short_operand = random(short_n);
    for (const size_t n : {size_t(0), size_t(1), size_t(32), size_t(65), size_t(257),
                           size_t(4095), size_t(4096), size_t(4097), size_t(4137)}) {
      const auto want = reference_product(long_operand, short_operand, n);
      CHECK_MSG(poly::conv(long_operand, short_operand, n) == want,
                "skinny conv mod=%u short=%zu n=%zu", Field::mod, short_n, n);
      CHECK_MSG(poly::conv(short_operand, long_operand, n) == want,
                "reversed skinny conv mod=%u short=%zu n=%zu", Field::mod, short_n, n);
    }
  }

  // Passing the same vector takes the one-transform square path; a distinct
  // equal copy takes the ordinary product path. Check both against the sum.
  for (const size_t len : {size_t(39), size_t(40), size_t(41), size_t(63), size_t(64),
                           size_t(65), size_t(127), size_t(128), size_t(129), size_t(257)}) {
    const auto a = random(len), copy = a;
    for (const size_t n : {size_t(1), len / 2, len, 2 * len - 1}) {
      const auto want = reference_product(a, a, n);
      CHECK_MSG(poly::conv(a, a, n) == want, "square mod=%u len=%zu n=%zu", Field::mod, len, n);
      CHECK_MSG(poly::conv(a, copy, n) == want, "equal-copy product mod=%u len=%zu n=%zu", Field::mod, len, n);
    }
  }

  // Seed-32 boundary and final partial Newton steps, with short, full and
  // longer-than-output inputs. No NTT-dependent identity is used as reference.
  for (const size_t n : {size_t(15), size_t(16), size_t(17), size_t(31), size_t(32),
                         size_t(33), size_t(63), size_t(64), size_t(65), size_t(127),
                         size_t(129), size_t(257), size_t(513)}) {
    for (const size_t len : {size_t(2), size_t(17), n + 5}) {
      auto a = random(len);
      a[0] = Field::from_int(Field::mod - 1);
      CHECK_MSG(poly::inv(a, n) == reference_inverse(a, n),
                "inverse boundary mod=%u len=%zu n=%zu", Field::mod, len, n);
    }
    auto a = random(n + 5);
    a[0] = Field();
    CHECK_MSG(poly::exp(a, n) == reference_exp(a, n), "exp boundary mod=%u n=%zu", Field::mod, n);
  }

  // Valued roots must retain exactly n-shift coefficients after removing
  // x^(2*shift), including seed boundaries and partial correction lengths.
  for (const size_t inner_n : {size_t(31), size_t(32), size_t(33), size_t(63), size_t(64),
                               size_t(65), size_t(127), size_t(128), size_t(129)}) {
    for (const size_t shift : {size_t(1), size_t(16), size_t(33)}) {
      const size_t n = inner_n + shift;
      auto base = random(inner_n);
      base[0] = Field::from_int(Field::mod - 1);
      std::vector<Field> root(n);
      std::copy(base.begin(), base.end(), root.begin() + static_cast<std::ptrdiff_t>(shift));
      const auto square = reference_product(root, root, 2 * n);
      const auto got = poly::sqrt(square, n);
      CHECK_EQ(got.size(), n);
      CHECK_MSG(reference_product(got, got, n + shift) == reference_product(root, root, n + shift),
                "valued sqrt mod=%u shift=%zu inner=%zu", Field::mod, shift, inner_n);
      for (size_t i = 0; i < shift; ++i) CHECK_EQ(got[i], Field());
      const Field sign = got[shift] / base[0];
      CHECK(sign == Field::from_int(1) || sign == -Field::from_int(1));
      for (size_t i = 0; i < inner_n; ++i) CHECK_EQ(got[shift + i], base[i] * sign);
    }
  }

  // SIMD differentiation/integration cover every possible remainder for the
  // widest backend, using scalar coefficient definitions as the reference.
  for (size_t len = 0; len <= 65; ++len) {
    const auto a = random(len);
    const auto d = poly::derivative(a), integral = poly::integral(a);
    CHECK_EQ(d.size(), len <= 1 ? size_t(0) : len - 1);
    CHECK_EQ(integral.size(), len == 0 ? size_t(0) : len + 1);
    for (size_t i = 1; i < len; ++i) CHECK_EQ(d[i - 1], a[i] * Field::from_int(i));
    if (!integral.empty()) CHECK_EQ(integral[0], Field());
    for (size_t i = 0; i < len; ++i) CHECK_EQ(integral[i + 1], a[i] / Field::from_int(i + 1));
  }
}

}  // namespace

FP_TEST(poly_optimized_boundaries_all_moduli) {
  optimized_poly_boundaries<mod998244353>();
  optimized_poly_boundaries<mod1004535809>();
  optimized_poly_boundaries<mod469762049>();
  optimized_poly_boundaries<mod167772161>();
  optimized_poly_boundaries<mod754974721>();
  optimized_poly_boundaries<mod1224736769>();
}

namespace {

template <class Field>
void inverse_table_and_linear_series() {
  for (const size_t n : {size_t(65535), size_t(65536), size_t(65537), size_t(262147)}) {
    const auto table = poly::inv_series<Field>(n);
    CHECK_EQ(table.size(), n + 1);
    CHECK(table[0].is_zero());
    for (size_t i = 1; i <= n; ++i) {
      CHECK(table[i].raw_val() < Field::mod);
      CHECK_EQ(uint64_t(table[i].raw_val()) * i % Field::mod, Field::one);
    }
  }
  for (const uint32_t c : {1u, 7u, Field::mod - 1}) {
    for (const size_t n : {size_t(255), size_t(256), size_t(257), size_t(271),
                           size_t(272), size_t(273), size_t(4095), size_t(4096),
                           size_t(4097), size_t(65539)}) {
      const std::vector<Field> a{Field(), Field::from_int(c), Field(), Field()};
      const auto e = poly::exp(a, n);
      CHECK_EQ(e.size(), n);
      CHECK_EQ(e[0].raw_val(), Field::one);
      // This coefficient definition uniquely pins exp(c*x), independently of
      // factorial tables, SIMD scans and the library's inverse implementation.
      for (size_t i = 1; i < n; ++i) {
        CHECK(e[i].raw_val() < Field::mod);
        CHECK_EQ(uint64_t(e[i].raw_val()) * i % Field::mod,
                 uint64_t(e[i-1].raw_val()) * c % Field::mod);
      }
    }
  }
  const uint32_t fourth_root = Field::from_int(Field::primitive_root).pow((Field::mod - 1) / 4).val();
  for (const uint32_t ratio : {1u, Field::mod - 1, 7u, fourth_root}) {
    const Field constant = Field::from_int(3);
    const std::vector<Field> a{constant, -constant * Field::from_int(ratio)};
    for (const size_t n : {size_t(1), size_t(15), size_t(16), size_t(17), size_t(31),
                           size_t(32), size_t(33), size_t(4097)}) {
      const auto b = poly::inv(a, n);
      CHECK_EQ(b.size(), n);
      uint32_t expected = constant.inv().raw_val();
      for (size_t i = 0; i < n; ++i) {
        CHECK_EQ(b[i].raw_val(), expected);
        expected = static_cast<uint32_t>(uint64_t(expected) * ratio % Field::mod);
      }
    }
  }
  std::vector<Field> a(4097);
  for (size_t i = 0; i < a.size(); ++i) a[i] = Field::from_int(i * 131 + 17);
  for (const uint32_t c : {0u, 1u, 7u, Field::mod - 1}) {
    const auto b = poly::mul_scalar(a, Field::from_int(c));
    CHECK_EQ(b.size(), a.size());
    for (size_t i = 0; i < a.size(); ++i)
      CHECK_EQ(b[i].raw_val(), uint64_t(a[i].raw_val()) * c % Field::mod);
  }
}

}  // namespace

FP_TEST(poly_inverse_tables_and_linear_series) {
  inverse_table_and_linear_series<mod998244353>();
  inverse_table_and_linear_series<mod1004535809>();
  inverse_table_and_linear_series<mod469762049>();
  inverse_table_and_linear_series<mod167772161>();
  inverse_table_and_linear_series<mod754974721>();
  inverse_table_and_linear_series<mod1224736769>();
  inverse_table_and_linear_series<Mont<1073479681u, 11>>();
  inverse_table_and_linear_series<Mont<2013265921u, 31>>();
}

FP_TEST_MAIN()
