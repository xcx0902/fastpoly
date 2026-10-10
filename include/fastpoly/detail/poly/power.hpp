// fastpoly - truncated integer powers and binomial shortcuts.
#ifndef FASTPOLY_DETAIL_POLY_POWER_HPP
#define FASTPOLY_DETAIL_POLY_POWER_HPP

#include "fastpoly/detail/poly/exponential.hpp"

namespace fpx::poly {

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
  trim(u);
  if (u.size() == 1) {
    vec<M> r(n, M());
    r[sh] = u[0].pow(k);
    return r;
  }
  if (k > 64 && u.size() == 2 && inner_n <= M::mod) {
    // For j < p, binomial(k,j) depends only on k mod p. Coefficients
    // beyond that residue vanish, and all needed factorials are invertible.
    const uint32_t exponent = static_cast<uint32_t>(k % M::mod);
    const size_t cnt = std::min(inner_n, size_t(exponent) + 1);
    vec<M> divisors = inv_series<M>(cnt - 1);
    vec<M> r(n, M());
    r[sh] = u[0].pow(k);
    const M ratio = u[1]*u[0].inv();
    M factor = ratio*M::from_int(exponent);
    for (size_t j = 1; j < cnt; ++j) {
      r[sh+j] = r[sh+j-1]*factor*divisors[j];
      factor -= ratio;
    }
    return r;
  }

  if (k <= 64) {  // binary exponentiation
    vec<M> r{M::from_int(1)}, base = std::move(u);
    for (uint64_t e = k; e; e >>= 1) {
      // Keep the actual support until the final result: early powers of a
      // short polynomial need neither padded operands nor large transforms.
      if (e & 1) r = conv_limbs(r, base, inner_n);
      if (e > 1) base = conv_limbs(base, base, inner_n);
    }
    r.resize(inner_n);
    if (sh == 0) return r;
    vec<M> out(n, M());
    for (size_t i = 0; i < r.size() && i + sh < n; ++i) out[i + sh] = r[i];
    return out;
  }

  const M c = u[0];
  vec<M> nrm = std::move(u);
  detail::scale_inplace(nrm, c.inv());  // nrm(0) == 1
  vec<M> l = log(nrm, inner_n);
  detail::scale_inplace(l, M::from_int(k));  // k mod Mod is the correct exponent
  vec<M> r = exp(l, inner_n);
  detail::scale_inplace(r, c.pow(k));
  if (sh == 0) return r;
  vec<M> out(n, M());
  for (size_t i = 0; i < r.size() && i + sh < n; ++i) out[i + sh] = r[i];
  return out;
}

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_POWER_HPP
