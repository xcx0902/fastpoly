// fastpoly - polynomial arithmetic modulo x^n over GF(Mod), Mod a 32-bit NTT
// prime.  All "infinite" results (inverse, ln, exp, sqrt, pow, division) take
// an explicit truncation length `n` and are computed modulo x^n.
#ifndef FASTPOLY_POLY_HPP
#define FASTPOLY_POLY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fastpoly/modint.hpp"
#include "fastpoly/ntt.hpp"
#include "fastpoly/simd.hpp"

namespace fpx {

template <class M>
class Poly;

namespace poly {

/// Thrown for algebraic preconditions (a(0) == 0 in an inverse, a(0) != 1 in
/// a logarithm, a non-residue constant term in a square root, ...).
class domain_error : public std::runtime_error {
 public:
  explicit domain_error(const std::string& s) : std::runtime_error(s) {}
};

template <class M>
using vec = std::vector<M>;

/// Convolutions with fewer than this many operations are done directly.
inline constexpr size_t naive_threshold = 40;

/// dst[0, cnt) = src[0, cnt)
template <class M>
void copy_prefix(const vec<M>& src, size_t cnt, vec<M>& dst) {
  for (size_t i = 0; i < cnt; ++i) dst[i] = src[i];
}
/// dst[0, cnt) = src[from, from + cnt)
template <class M>
void copy_range(const vec<M>& src, size_t from, size_t cnt, vec<M>& dst) {
  for (size_t i = 0; i < cnt; ++i) dst[i] = src[from + i];
}

/// Pointwise Montgomery product of two transform-domain buffers:
/// `dst[i] *= b[i]`.  This is a full pass over the spectrum, so it runs on the
/// SIMD lane width; for Mod < 2^30 it uses the lazy kernel (the inverse
/// transform canonicalises afterwards).
template <class M>
void pointwise_mul(uint32_t* dst, const uint32_t* b, size_t n) {
  constexpr uint32_t MOD = M::mod;
  constexpr bool LAZY = simd::lazy_ok<MOD>;
  size_t i = 0;
  for (; i + simd::lane <= n; i += simd::lane) {
    simd::store(dst + i, simd::butterfly_mul<MOD, M::ninv, LAZY>(
                             simd::load(dst + i), simd::load(b + i)));
  }
  for (; i < n; ++i) dst[i] = M::reduce(uint64_t(dst[i]) * b[i]);
}

/// ---------------------------------------------------------------------------
/// basic operations
/// ---------------------------------------------------------------------------

/// Drop trailing zeros so that degree() is meaningful.
template <class M>
void trim(vec<M>& a) {
  while (!a.empty() && a.back().is_zero()) a.pop_back();
}

template <class M>
vec<M> add(const vec<M>& a, const vec<M>& b) {
  vec<M> r(std::max(a.size(), b.size()));
  for (size_t i = 0; i < r.size(); ++i) {
    M x = i < a.size() ? a[i] : M();
    M y = i < b.size() ? b[i] : M();
    r[i] = x + y;
  }
  return r;
}

template <class M>
vec<M> sub(const vec<M>& a, const vec<M>& b) {
  vec<M> r(std::max(a.size(), b.size()));
  for (size_t i = 0; i < r.size(); ++i) {
    M x = i < a.size() ? a[i] : M();
    M y = i < b.size() ? b[i] : M();
    r[i] = x - y;
  }
  return r;
}

template <class M>
vec<M> neg(vec<M> a) {
  for (auto& x : a) x = -x;
  return a;
}

template <class M>
vec<M> mul_scalar(const vec<M>& a, M c) {
  vec<M> r(a.size());
  constexpr uint32_t MOD = M::mod;
  const simd::native_t c1 = simd::set1(c.raw_val());
  size_t i = 0;
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  const auto* src = reinterpret_cast<const uint32_t*>(a.data());
  for (; i + simd::lane <= a.size(); i += simd::lane) {
    simd::store(dst + i, simd::mulmod<MOD, M::ninv>(simd::load(src + i), c1));
  }
  for (; i < a.size(); ++i) r[i] = a[i] * c;
  return r;
}

/// a << k  (multiply by x^k)
template <class M>
vec<M> shift(const vec<M>& a, size_t k, size_t n = SIZE_MAX) {
  vec<M> r(std::min(a.size() + k, n), M());
  for (size_t i = 0; i < a.size() && i + k < r.size(); ++i) r[i + k] = a[i];
  return r;
}

/// Horner evaluation at a point.
template <class M>
M eval(const vec<M>& a, M x) {
  M r = M();
  for (size_t i = a.size(); i-- > 0;) r = r * x + a[i];
  return r;
}

/// ---------------------------------------------------------------------------
/// convolution
/// ---------------------------------------------------------------------------

/// Internal: truncated product of two Montgomery-limb buffers. Both inputs are
/// already reduced to at most `lim` coefficients by the caller.
template <class M>
vec<M> conv_limbs(const vec<M>& a, const vec<M>& b, size_t lim) {
  if (a.empty() || b.empty() || lim == 0) return {};
  const size_t la = std::min(a.size(), lim), lb = std::min(b.size(), lim);
  const size_t full = la + lb - 1;
  const size_t out = std::min(full, lim);
  static_assert(sizeof(M) == sizeof(uint32_t), "Mont must be a single 32-bit limb");

  if (std::min(la, lb) <= naive_threshold) {
    vec<M> r(out, M());
    for (size_t i = 0; i < la; ++i)
      for (size_t j = 0; j < lb && i + j < out; ++j) r[i + j] += a[i] * b[j];
    return r;
  }

  const uint32_t N = next_pow2(full);
  auto plan = NttPlan<M>::get(N);
  vec<M> fa(N, M()), fb(N, M());
  copy_prefix(a, la, fa);
  copy_prefix(b, lb, fb);
  uint32_t* pa = reinterpret_cast<uint32_t*>(fa.data());
  uint32_t* pb = reinterpret_cast<uint32_t*>(fb.data());
  plan->forward(pa);
  plan->forward(pb);
  pointwise_mul<M>(pa, pb, N);
  plan->inverse(pa);
  fa.resize(out);
  return fa;
}

/// Product of a and b, truncated to n coefficients (n = SIZE_MAX -> full).
template <class M>
vec<M> conv(const vec<M>& a, const vec<M>& b, size_t n = SIZE_MAX) {
  return conv_limbs(a, b, n);
}

/// ---------------------------------------------------------------------------
/// derivative / integral
/// ---------------------------------------------------------------------------

template <class M>
vec<M> derivative(const vec<M>& a) {
  if (a.size() <= 1) return {};
  vec<M> r(a.size() - 1);
  for (size_t i = 1; i < a.size(); ++i) r[i - 1] = a[i] * M::from_int(i);
  return r;
}

/// Montgomery forms of 1/1, 1/2, ..., 1/n (linear recurrence, no divisions in
/// the field beyond one inverse). Requires n < mod.
///
/// Measured against a prefix-product "batch inversion" (one inverse + 3n
/// multiplications) this is ~5x faster: `mod` is a compile-time constant, so
/// `mod / i`, `mod % i` and the `from_int` reductions all strength-reduce to
/// multiply/shift, leaving ~6 cheap ALU ops per element against three full
/// 64-bit Montgomery products.
template <class M>
vec<M> inv_series(size_t n) {
  if (n >= M::mod) throw domain_error("poly::inv_series: length must be < mod");
  vec<M> r(n + 1, M());
  if (n == 0) return r;
  r[1] = M::from_int(1);
  for (size_t i = 2; i <= n; ++i) {
    // 1/i = -(mod/i) * (1/(mod%i)) mod mod
    uint64_t q = M::mod / i, rem = M::mod % i;
    M inner = r[static_cast<size_t>(rem)];
    r[i] = -(M::from_int(q) * inner);
  }
  return r;
}

template <class M>
vec<M> integral(const vec<M>& a) {
  if (a.empty()) return {};
  vec<M> inv = inv_series<M>(a.size());
  vec<M> r(a.size() + 1);
  for (size_t i = 0; i < a.size(); ++i) r[i + 1] = a[i] * inv[i + 1];
  return r;
}

/// ---------------------------------------------------------------------------
/// inverse   (requires a(0) != 0)
/// ---------------------------------------------------------------------------

/// b = a^-1 mod x^n.
///
/// Newton iteration with the *middle product* optimisation: at step m -> m2
/// (m2 <= 2m) we only need the high half of a*b, and that half is free of
/// cyclic aliasing at transform size 2m, so the transform stays at 2m instead
/// of 4m -- a 2x reduction in transform length per iteration.
template <class M>
vec<M> inv(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (a.empty() || a[0].is_zero())
    throw domain_error("poly::inv: constant term is zero");
  if (a.size() == 1) {  // constant: 1/a0 + 0*x + 0*x^2 + ...
    vec<M> r(n, M());
    r[0] = a[0].inv();
    return r;
  }
  static_assert(sizeof(M) == sizeof(uint32_t), "Mont must be a single 32-bit limb");

  vec<M> b{a[0].inv()};  // correct mod x^1
  for (size_t m = 1; m < n; m <<= 1) {
    const size_t m2 = std::min(2 * m, n);
    const uint32_t N = next_pow2(2 * m);
    auto plan = NttPlan<M>::get(N);

    vec<M> t(N, M());
    const size_t acnt = std::min(a.size(), 2 * m);
    copy_prefix(a, acnt, t);
    uint32_t* pt = reinterpret_cast<uint32_t*>(t.data());
    // u = b padded to N
    vec<M> u(N, M());
    std::copy(b.begin(), b.end(), u.begin());
    uint32_t* pu = reinterpret_cast<uint32_t*>(u.data());
    plan->forward(pu);
    plan->forward(pt);
    pointwise_mul<M>(pt, pu, N);
    plan->inverse(pt);
    // h = high half of a*b, i.e. coefficients [m, m2)
    const size_t hlen = m2 - m;
    vec<M> h(hlen, M());
    for (size_t i = 0; i < hlen; ++i) h[i] = -t[m + i];
    // c = b * h mod x^hlen  (transform of b is still in pu / recompute at N)
    vec<M> hv(N, M());
    std::copy(h.begin(), h.end(), hv.begin());
    uint32_t* ph = reinterpret_cast<uint32_t*>(hv.data());
    plan->forward(ph);
    pointwise_mul<M>(ph, pu, N);
    plan->inverse(ph);
    b.resize(m2);
    for (size_t i = 0; i < hlen; ++i) b[m + i] = hv[i];
  }
  b.resize(n);
  return b;
}

/// log(a) = integral(a' / a) mod x^n   (requires a(0) == 1)
template <class M>
vec<M> log(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (a.empty() || a[0] != M::from_int(1))
    throw domain_error("poly::log: constant term must be 1");
  vec<M> d = derivative(a);
  vec<M> ia = inv(a, n);
  vec<M> p = conv_limbs(d, ia, n - 1);  // (a'/a) mod x^(n-1)
  vec<M> r = integral(p);
  r.resize(n, M());
  return r;
}

/// exp(a) mod x^n   (requires a(0) == 0)
///
/// b_{2m} = b_m * (1 + a - log b_m) mod x^{2m}: each step costs one truncated
/// log plus two convolutions, all of size O(m), giving O(n log n) overall.
template <class M>
vec<M> exp(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (!a.empty() && !a[0].is_zero())
    throw domain_error("poly::exp: constant term must be 0");
  vec<M> b{M::from_int(1)};
  for (size_t m = 1; m < n; m <<= 1) {
    const size_t m2 = std::min(2 * m, n);
    vec<M> lb = log(b, m2);
    vec<M> t(m2, M());
    t[0] = M::from_int(1);
    for (size_t i = 0; i < m2; ++i) t[i] -= lb[i];
    for (size_t i = 0; i < std::min(a.size(), m2); ++i) t[i] += a[i];
    b = conv_limbs(b, t, m2);
    b.resize(m2);
  }
  return b;
}

/// Square root mod x^n.  Throws `domain_error` if a(0) is a non-residue or the
/// x-adic valuation of a is odd (then no square root exists in GF(Mod)[[x]]).
template <class M>
vec<M> sqrt(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  size_t v = 0;
  while (v < a.size() && a[v].is_zero()) ++v;
  if (v == a.size()) return vec<M>(n, M());  // zero polynomial
  if (v & 1) throw domain_error("poly::sqrt: odd valuation, no square root");
  M root0;
  if (!a[v].sqrt(root0)) throw domain_error("poly::sqrt: non-residue constant term");
  const size_t half = v / 2;
  if (half >= n) return vec<M>(n, M());
  // normalize: a = x^v * u, u(0) != 0; work on u up to n - half terms
  const size_t inner_n = n - half;
  vec<M> u(std::min(a.size() - v, inner_n));
  copy_range(a, v, u.size(), u);
  vec<M> b{root0};  // b = sqrt(u) mod x^1
  for (size_t m = 1; m < inner_n; m <<= 1) {
    const size_t m2 = std::min(2 * m, inner_n);
    vec<M> ib = inv(b, m2);
    vec<M> t = conv_limbs(u, ib, m2);
    t.resize(m2, M());
    for (size_t i = 0; i < m2; ++i) t[i] += (i < b.size() ? b[i] : M());
    M half_inv = M::from_int(M::inv2);
    b = mul_scalar(t, half_inv);
    b.resize(m2);
  }
  // result = x^(v/2) * b
  vec<M> r(n, M());
  for (size_t i = 0; i < b.size() && i + half < n; ++i) r[i + half] = b[i];
  return r;
}

/// a^k mod x^n for an arbitrary 64-bit exponent.
///
/// Uses direct binary exponentiation for small exponents and the
/// exp(k*log(a)) route otherwise.  The logarithm is legitimate for any integer
/// k: on the normalized series (1 + O(x)) the j-th coefficient of (1+u)^k is
/// binom(k, j) * ..., which depends on k only through k mod Mod, and j < n <= p.
template <class M>
vec<M> pow(const vec<M>& a, uint64_t k, size_t n) {
  if (n == 0) return {};
  if (k == 0) {
    vec<M> r(n, M());
    r[0] = M::from_int(1);
    return r;
  }
  size_t v = 0;
  while (v < a.size() && a[v].is_zero()) ++v;
  if (v == a.size()) return vec<M>(n, M());
  // x^(v*k) shift: bail out if the result is already >= x^n
  if (v != 0 && k > (n - 1) / v) return vec<M>(n, M());
  const size_t sh = static_cast<size_t>(v * k);
  const size_t inner_n = n - sh;
  vec<M> u(std::min(a.size() - v, inner_n));
  copy_range(a, v, u.size(), u);

  if (k <= 64) {  // binary exponentiation
    vec<M> r{M::from_int(1)}, base = u;
    for (uint64_t e = k; e; e >>= 1) {
      if (e & 1) { r = conv_limbs(r, base, inner_n); r.resize(inner_n); }
      if (e > 1) { base = conv_limbs(base, base, inner_n); base.resize(inner_n); }
    }
    vec<M> out(n, M());
    for (size_t i = 0; i < r.size() && i + sh < n; ++i) out[i + sh] = r[i];
    return out;
  }

  const M c = u[0];
  const vec<M> nrm = mul_scalar(u, c.inv());  // nrm(0) == 1
  vec<M> l = log(nrm, inner_n);
  for (auto& x : l) x *= M::from_int(k);      // k mod Mod is the correct exponent
  vec<M> r = exp(l, inner_n);
  r = mul_scalar(r, c.pow(k));
  vec<M> out(n, M());
  for (size_t i = 0; i < r.size() && i + sh < n; ++i) out[i + sh] = r[i];
  return out;
}

/// Euclidean division: a = b*q + r with deg(r) < deg(b). Both results trimmed.
template <class M>
std::pair<vec<M>, vec<M>> divmod(const vec<M>& a_in, const vec<M>& b_in) {
  vec<M> a = a_in, b = b_in;
  trim(a);
  trim(b);
  if (b.empty()) throw domain_error("poly::divmod: division by zero polynomial");
  if (a.size() < b.size()) return {vec<M>{}, a};
  const size_t dn = a.size() - b.size() + 1;
  // q = rev( rev(a) * inv(rev(b)) ) mod x^dn
  vec<M> ra(a.rbegin(), a.rend());
  vec<M> rb(b.rbegin(), b.rend());
  vec<M> ib = inv(rb, dn);
  vec<M> rq = conv_limbs(ra, ib, dn);
  rq.resize(dn, M());
  vec<M> q(rq.rbegin(), rq.rend());
  vec<M> r = sub(a, conv_limbs(b, q, a.size()));
  r.resize(a.size(), M());
  trim(r);
  return {q, r};
}

template <class M>
vec<M> div(const vec<M>& a, const vec<M>& b) { return divmod(a, b).first; }

template <class M>
vec<M> mod(const vec<M>& a, const vec<M>& b) { return divmod(a, b).second; }

}  // namespace poly

/// ---------------------------------------------------------------------------
/// public polynomial type: std::vector<M> plus method syntax
/// ---------------------------------------------------------------------------

template <class M>
class Poly : public std::vector<M> {
 public:
  using base = std::vector<M>;
  using value_type = M;

  Poly() = default;
  Poly(const base& v) : base(v) {}
  Poly(base&& v) : base(std::move(v)) {}
  Poly(std::initializer_list<M> il) : base(il) {}
  explicit Poly(size_t n, M v = M()) : base(n, v) {}

  using base::size;
  using base::operator[];

  size_t degree() const { return this->empty() ? 0 : this->size() - 1; }
  bool is_zero() const {
    for (const auto& x : *this)
      if (!x.is_zero()) return false;
    return true;
  }

  Poly& trim() { poly::trim(*this); return *this; }
  Poly& resize_(size_t n) { this->resize(n); return *this; }

  Poly operator+(const base& b) const { return poly::add(*this, b); }
  Poly operator-(const base& b) const { return poly::sub(*this, b); }
  Poly operator-() const { return poly::neg(*this); }
  Poly operator*(const base& b) const { return poly::conv(*this, b); }
  Poly operator*(M c) const { return poly::mul_scalar(*this, c); }
  Poly operator<<(size_t k) const { return poly::shift(*this, k); }
  Poly& operator+=(const base& b) { return *this = poly::add(*this, b); }
  Poly& operator-=(const base& b) { return *this = poly::sub(*this, b); }
  Poly& operator*=(const base& b) { return *this = poly::conv(*this, b); }

  /// Truncated product (mod x^n).
  Poly mul(const base& b, size_t n) const { return poly::conv(*this, b, n); }
  Poly derivative() const { return poly::derivative(*this); }
  Poly integral() const { return poly::integral(*this); }
  /// Series inverse mod x^n (a(0) must be invertible).
  Poly inv(size_t n) const { return poly::inv(*this, n); }
  /// Series logarithm mod x^n (a(0) must be 1).
  Poly log(size_t n) const { return poly::log(*this, n); }
  /// Series exponential mod x^n (a(0) must be 0).
  Poly exp(size_t n) const { return poly::exp(*this, n); }
  /// Series square root mod x^n.
  Poly sqrt(size_t n) const { return poly::sqrt(*this, n); }
  /// Series power mod x^n.
  Poly pow(uint64_t k, size_t n) const { return poly::pow(*this, k, n); }
  std::pair<Poly, Poly> divmod(const base& b) const {
    auto r = poly::divmod(*this, b);
    return {Poly(std::move(r.first)), Poly(std::move(r.second))};
  }
  Poly div(const base& b) const { return poly::div(*this, b); }
  Poly mod(const base& b) const { return poly::mod(*this, b); }
  M eval(M x) const { return poly::eval(*this, x); }

  /// Convenience: coefficients of this polynomial as a flat vector of ints.
  std::vector<uint32_t> to_ints() const {
    std::vector<uint32_t> r(this->size());
    for (size_t i = 0; i < this->size(); ++i) r[i] = (*this)[i].val();
    return r;
  }
  /// Build a polynomial from plain integers.
  static Poly from_ints(const std::vector<uint32_t>& v) {
    Poly r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = M::from_int(v[i]);
    return r;
  }
};

}  // namespace fpx

#endif  // FASTPOLY_POLY_HPP
