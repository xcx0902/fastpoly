// fastpoly - derivatives, inverse denominators and integration.
#ifndef FASTPOLY_DETAIL_POLY_CALCULUS_HPP
#define FASTPOLY_DETAIL_POLY_CALCULUS_HPP

#include "fastpoly/detail/poly/base.hpp"

namespace fpx::poly {

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
template <class M, class Backend = simd::backend>
vec<M> inv_series(size_t n) {
  if (n >= M::mod) throw domain_error("poly::inv_series: length must be < mod");
  vec<M> r(n + 1, M());
  if (n == 0) return r;
  r[1] = M::from_int(1);
  const uint32_t count = static_cast<uint32_t>(n);
  uint32_t i = 2;
  for (; i <= std::min(count, uint32_t(65536)); ++i) {
    // 1/i = -(mod/i) * (1/(mod%i)) mod mod
    const uint32_t q = M::mod / i, rem = M::mod % i;
    const uint32_t product = static_cast<uint32_t>(
        (uint64_t(q) * r[rem].raw_val()) % M::mod);
    // With prime mod and 1 < i < mod, q and the inverse are both nonzero.
    r[i] = M::raw(M::mod - product);
  }
  // Past this point mod/i stays constant over several consecutive indices.
  // All remainders decrease within an interval and precede its first index,
  // so these products are independent and need no per-element division.
  while (i <= count) {
    const uint32_t q = M::mod / i;
    uint32_t rem = M::mod % i;
    const uint32_t end = std::min(count, M::mod / q);
    if constexpr (simd::compact_twiddle<M::mod>) {
      if (i + 3 <= end) {
        const auto factor = simd::fixed_twiddle<M::mod, M::ninv>(M::from_int(q).raw_val());
        auto* dst = reinterpret_cast<uint32_t*>(r.data());
        for (; i + 3 <= end; i += 4, rem -= 4 * q) {
          const typename Backend::native_type v{r[rem].raw_val(), r[rem-q].raw_val(),
                                               r[rem-2*q].raw_val(), r[rem-3*q].raw_val()};
          const auto product = simd::mul_twiddle_full<M::mod, M::ninv>(v, factor);
          // Prime mod, 1 < i < mod: q and every inverse are nonzero.
          simd::store(dst + i, simd::sub_wide(simd::set1(M::mod), product));
        }
      }
    }
    for (; i <= end; ++i, rem -= q) {
      const uint32_t product = static_cast<uint32_t>(uint64_t(q) * r[rem].raw_val() % M::mod);
      r[i] = M::raw(M::mod - product);
    }
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

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_CALCULUS_HPP
