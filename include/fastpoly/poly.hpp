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

/// Convolutions with a short operand are done directly.
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
    // Broadcast the short operand and stream through the long operand/output.
    // This also makes a scalar-times-polynomial product one contiguous pass.
    const vec<M>& small = la <= lb ? a : b;
    const vec<M>& large = la <= lb ? b : a;
    const size_t short_n = std::min(la, lb), long_n = std::max(la, lb);
    auto* dst = reinterpret_cast<uint32_t*>(r.data());
    const auto* src = reinterpret_cast<const uint32_t*>(large.data());
    for (size_t i = 0; i < short_n && i < out; ++i) {
      const size_t cnt = std::min(long_n, out - i);
      const auto c = simd::set1(small[i].raw_val());
      size_t j = 0;
      for (; j + simd::lane <= cnt; j += simd::lane) {
        const auto v = simd::mulmod<M::mod, M::ninv>(simd::load(src + j), c);
        simd::store(dst + i + j, simd::add(simd::load(dst + i + j), v, M::mod));
      }
      for (; j < cnt; ++j) r[i + j] += small[i] * large[j];
    }
    return r;
  }

  const uint32_t N = next_pow2(full);
  auto plan = NttPlan<M>::get(N);
  vec<M> fa(N, M());
  copy_prefix(a, la, fa);
  uint32_t* pa = reinterpret_cast<uint32_t*>(fa.data());
  plan->forward_lazy(pa);
  if (a.data() == b.data() && la == lb) {
    pointwise_mul<M>(pa, pa, N);
  } else {
    vec<M> fb(N, M());
    copy_prefix(b, lb, fb);
    auto* pb = reinterpret_cast<uint32_t*>(fb.data());
    plan->forward_lazy(pb);
    pointwise_mul<M>(pa, pb, N);
  }
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

namespace detail {

/// Derivative of only the first cnt input coefficients.
template <class M>
vec<M> derivative_prefix(const vec<M>& a, size_t cnt) {
  cnt = std::min(a.size(), cnt);
  if (cnt <= 1) return {};
  vec<M> r(cnt - 1);
  alignas(64) uint32_t factors[simd::lane];
  for (int k = 0; k < simd::lane; ++k) factors[k] = M::from_int(uint32_t(k + 1)).raw_val();
  auto factor = simd::load(factors);
  const auto stride = simd::set1(M::from_int(simd::lane).raw_val());
  const auto* src = reinterpret_cast<const uint32_t*>(a.data());
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  size_t i = 1;
  for (; i + simd::lane <= cnt; i += simd::lane) {
    simd::store(dst + i - 1, simd::mulmod<M::mod, M::ninv>(simd::load(src + i), factor));
    factor = simd::add(factor, stride, M::mod);
  }
  for (; i < cnt; ++i) r[i - 1] = a[i] * M::from_int(i);
  return r;
}

}  // namespace detail

template <class M>
vec<M> derivative(const vec<M>& a) {
  return detail::derivative_prefix(a, a.size());
}

/// Montgomery forms of 1/1, 1/2, ..., 1/n using a linear recurrence.
/// Requires n < mod. Quotient and remainder share a 32-bit integer division.
/// Multiplication by the ordinary quotient preserves the raw inverse's
/// Montgomery factor, needing just one constant-modulus reduction per element.
template <class M>
vec<M> inv_series(size_t n) {
  if (n >= M::mod) throw domain_error("poly::inv_series: length must be < mod");
  vec<M> r(n + 1, M());
  if (n == 0) return r;
  r[1] = M::from_int(1);
  const uint32_t count = static_cast<uint32_t>(n);
  for (uint32_t i = 2; i <= count; ++i) {
    // 1/i = -(mod/i) * (1/(mod%i)) mod mod
    const uint32_t q = M::mod / i, rem = M::mod % i;
    const uint32_t product = static_cast<uint32_t>(
        (uint64_t(q) * r[rem].raw_val()) % M::mod);
    // With prime mod and 1 < i < mod, q and the inverse are both nonzero.
    r[i] = M::raw(M::mod - product);
  }
  return r;
}

template <class M>
vec<M> integral(const vec<M>& a) {
  if (a.empty()) return {};
  vec<M> inv = inv_series<M>(a.size());
  vec<M> r(a.size() + 1);
  const auto* src = reinterpret_cast<const uint32_t*>(a.data());
  const auto* divisors = reinterpret_cast<const uint32_t*>(inv.data());
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  size_t i = 0;
  for (; i + simd::lane <= a.size(); i += simd::lane) {
    simd::store(dst + i + 1, simd::mulmod<M::mod, M::ninv>(
                               simd::load(src + i), simd::load(divisors + i + 1)));
  }
  for (; i < a.size(); ++i) r[i + 1] = a[i] * inv[i + 1];
  return r;
}

/// ---------------------------------------------------------------------------
/// inverse   (requires a(0) != 0)
/// ---------------------------------------------------------------------------

namespace detail {

// Extend an inverse prefix c to wanted <= 2*c.size(), using only two
// scratch buffers and preserving the c spectrum through the correction.
template <class M>
void extend_inverse(const vec<M>& a, vec<M>& c, size_t wanted,
                  vec<M>& scratch, vec<M>& spectrum) {
  const size_t k = c.size(), hlen = wanted - k;
  // A handful of coefficients after a power-of-two boundary costs less than
  // another full transform. The new coefficients are filled in order.
  if (hlen <= 8) {
    const M inv0 = c[0];
    c.resize(wanted);
    for (size_t i = k; i < wanted; ++i) {
      M sum;
      for (size_t j = 1; j <= i && j < a.size(); ++j) sum += a[j]*c[i-j];
      c[i] = -sum*inv0;
    }
    return;
  }
  const uint32_t N = next_pow2(2*k);
  auto plan = NttPlan<M>::get(N);
  scratch.assign(N, M());
  spectrum.assign(N, M());
  std::copy_n(a.begin(), std::min(a.size(), wanted), scratch.begin());
  std::copy(c.begin(), c.end(), spectrum.begin());
  auto* pt = reinterpret_cast<uint32_t*>(scratch.data());
  auto* pc = reinterpret_cast<uint32_t*>(spectrum.data());
  plan->forward_lazy(pt);
  plan->forward_lazy(pc);
  pointwise_mul<M>(pt, pc, N);
  plan->inverse(pt);
  for (size_t i = 0; i < hlen; ++i) scratch[i] = -scratch[k+i];
  std::fill(scratch.data() + hlen, scratch.data() + scratch.size(), M());
  plan->forward_lazy(pt);
  pointwise_mul<M>(pt, pc, N);
  plan->inverse(pt);
  c.resize(wanted);
  std::copy_n(scratch.data(), hlen, c.data() + k);
}

template <class M>
vec<M> inverse_seed(const vec<M>& a, size_t n) {
  vec<M> c(n, M());
  c[0] = a[0].inv();
  for (size_t i = 1; i < n; ++i) {
    M sum;
    for (size_t j = 1; j <= i && j < a.size(); ++j) sum += a[j]*c[i-j];
    c[i] = -sum*c[0];
  }
  return c;
}

}  // namespace detail

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

  const size_t seed = std::min(n, size_t(32));
  vec<M> b = detail::inverse_seed(a, seed);
  vec<M> work, spectrum;
  for (size_t m = seed; m < n; m <<= 1)
    detail::extend_inverse(a, b, std::min(2 * m, n), work, spectrum);
  b.resize(n);
  return b;
}

/// log(a) = integral(a' / a) mod x^n   (requires a(0) == 1)
template <class M>
vec<M> log(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (a.empty() || a[0] != M::from_int(1))
    throw domain_error("poly::log: constant term must be 1");
  if (n == 1) return vec<M>(1, M());
  vec<M> d = detail::derivative_prefix(a, n);
  vec<M> ia = inv(a, n - 1);
  vec<M> p = conv_limbs(d, ia, n - 1);  // (a'/a) mod x^(n-1)
  vec<M> r = integral(p);
  r.resize(n, M());
  return r;
}

/// exp(a) mod x^n   (requires a(0) == 0)
///
/// Solve b' = a' b using the high differential residual at each doubling.
/// Dividing it by b and integrating gives (a-log b)[m,m2); multiplying only
/// this new half by b updates the result at transform size 2m. Keep the inverse
/// prefix between steps instead of restarting an inverse/logarithm each time.
template <class M>
vec<M> exp(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (!a.empty() && !a[0].is_zero())
    throw domain_error("poly::exp: constant term must be 0");
  if (n-1 >= M::mod)
    throw domain_error("poly::inv_series: length must be < mod");
  vec<M> aprime = detail::derivative_prefix(a, n);
  trim(aprime);
  if (aprime.empty()) {
    vec<M> b(n, M());
    b[0] = M::from_int(1);
    return b;
  }
  // Short input series admit an O(n*degree(a)) recurrence; in particular exp
  // of a linear polynomial is a single contiguous pass.
  const size_t seed = aprime.size() <= naive_threshold ? n : std::min<size_t>(n, 32);
  vec<M> denominators = inv_series<M>(n-1);
  vec<M> b(seed, M());
  b[0] = M::from_int(1);
  for (size_t i = 1; i < seed; ++i) {
    M sum;
    for (size_t j = 1; j <= i && j <= aprime.size(); ++j)
      sum += aprime[j-1]*b[i-j];
    b[i] = sum*denominators[i];
  }
  if (n == seed) return b;
  vec<M> c = detail::inverse_seed(b, seed/2);
  vec<M> spectrum, inverse_spectrum, work;
  for (size_t m = seed; m < n; m *= 2) {
    const size_t m2 = std::min(2*m, n), hlen = m2-m;
    if (hlen <= 8) {
      b.resize(m2);
      for (size_t i = m; i < m2; ++i) {
        M sum;
        for (size_t j = 1; j <= i && j <= aprime.size(); ++j)
          sum += aprime[j-1]*b[i-j];
        b[i] = sum*denominators[i];
      }
      continue;
    }
    if (c.size() < hlen)
      detail::extend_inverse(b, c, hlen, work, inverse_spectrum);
    const uint32_t N = next_pow2(2*m);
    auto plan = NttPlan<M>::get(N);
    spectrum.assign(N, M());
    inverse_spectrum.assign(N, M());
    work.assign(N, M());
    std::copy(b.begin(), b.end(), spectrum.begin());
    std::copy(c.begin(), c.end(), inverse_spectrum.begin());
    std::copy_n(aprime.begin(), std::min(aprime.size(), m2-1), work.begin());
    auto* pb = reinterpret_cast<uint32_t*>(spectrum.data());
    auto* pc = reinterpret_cast<uint32_t*>(inverse_spectrum.data());
    auto* pt = reinterpret_cast<uint32_t*>(work.data());
    plan->forward_lazy(pb);
    plan->forward_lazy(pt);
    pointwise_mul<M>(pt, pb, N);
    plan->inverse(pt);
    // a' b - b' is zero below degree m-1.  The wanted high range is
    // untouched by cyclic wraparound: deg(a'b) <= 3m-3.
    for (size_t i = 0; i < hlen; ++i) work[i] = work[m-1+i];
    std::fill(work.data() + hlen, work.data() + work.size(), M());
    plan->forward_lazy(pt);
    plan->forward_lazy(pc);
    pointwise_mul<M>(pt, pc, N);
    plan->inverse(pt);
    // Integrate the shifted high differential residual.
    pointwise_mul<M>(pt, reinterpret_cast<const uint32_t*>(denominators.data()) + m, hlen);
    std::fill(work.data() + hlen, work.data() + work.size(), M());
    plan->forward_lazy(pt);
    pointwise_mul<M>(pt, pb, N);
    plan->inverse(pt);
    b.resize(m2);
    std::copy_n(work.data(), hlen, b.data() + m);
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
  if (v == a.size()) return vec<M>(n, M());
  if (v & 1) throw domain_error("poly::sqrt: odd valuation, no square root");
  M root0;
  if (!a[v].sqrt(root0))
    throw domain_error("poly::sqrt: non-residue constant term");
  const size_t half = v/2;
  if (half >= n) return vec<M>(n, M());
  const size_t inner_n = n-half;
  const size_t ucnt = std::min(a.size()-v, inner_n);
  if (std::all_of(a.data()+v+1, a.data()+v+ucnt,
                  [](const M& x) { return x.is_zero(); })) {
    vec<M> r(n, M());
    r[half] = root0;
    return r;
  }
  const size_t seed = std::min<size_t>(inner_n, 32);
  const M half_inv = M::from_int(M::inv2);
  const M twice_root_inv = root0.inv()*half_inv;
  vec<M> b(seed, M());
  b[0] = root0;
  for (size_t i = 1; i < seed; ++i) {
    M residual = i < a.size()-v ? a[v+i] : M();
    for (size_t j = 1; j < i; ++j) residual -= b[j]*b[i-j];
    b[i] = residual*twice_root_inv;
  }
  if (inner_n > seed) {
    vec<M> c = detail::inverse_seed(b, seed/2);
    vec<M> work, inverse_spectrum;
    for (size_t m = seed; m < inner_n; m *= 2) {
      const size_t m2 = std::min(2*m, inner_n), hlen = m2-m;
      if (hlen <= 8) {
        b.resize(m2);
        for (size_t i = m; i < m2; ++i) {
          M residual = i < a.size()-v ? a[v+i] : M();
          for (size_t j = 1; j < i; ++j) residual -= b[j]*b[i-j];
          b[i] = residual*twice_root_inv;
        }
        continue;
      }
      if (c.size() < hlen)
        detail::extend_inverse(b, c, hlen, work, inverse_spectrum);
      const uint32_t N = next_pow2(2*m);
      auto plan = NttPlan<M>::get(N);
      work.assign(N, M());
      inverse_spectrum.assign(N, M());
      std::copy(b.begin(), b.end(), work.begin());
      std::copy(c.begin(), c.end(), inverse_spectrum.begin());
      auto* pt = reinterpret_cast<uint32_t*>(work.data());
      auto* pc = reinterpret_cast<uint32_t*>(inverse_spectrum.data());
      plan->forward_lazy(pt);
      pointwise_mul<M>(pt, pt, N);
      plan->inverse(pt);
      for (size_t i = 0; i < hlen; ++i)
        work[i] = ((m+i < a.size()-v) ? a[v+m+i] : M()) - work[m+i];
      std::fill(work.data() + hlen, work.data() + work.size(), M());
      plan->forward_lazy(pt);
      plan->forward_lazy(pc);
      pointwise_mul<M>(pt, pc, N);
      plan->inverse(pt);
      b.resize(m2);
      for (size_t i = 0; i < hlen; ++i) b[m+i] = work[i]*half_inv;
    }
  }
  if (half == 0) return b;
  vec<M> r(n, M());
  std::copy(b.begin(), b.end(), r.data() + half);
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
  if (k == 1) {
    vec<M> r(n, M());
    std::copy_n(a.begin(), std::min(a.size(), n), r.begin());
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
    if (sh == 0) return r;
    vec<M> out(n, M());
    for (size_t i = 0; i < r.size() && i + sh < n; ++i) out[i + sh] = r[i];
    return out;
  }

  const M c = u[0];
  const vec<M> nrm = mul_scalar(u, c.inv());  // nrm(0) == 1
  vec<M> l = log(nrm, inner_n);
  l = mul_scalar(l, M::from_int(k));         // k mod Mod is the correct exponent
  vec<M> r = exp(l, inner_n);
  r = mul_scalar(r, c.pow(k));
  if (sh == 0) return r;
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
