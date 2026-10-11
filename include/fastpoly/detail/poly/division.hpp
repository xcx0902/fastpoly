// fastpoly - Euclidean division and remainders.
#ifndef FASTPOLY_DETAIL_POLY_DIVISION_HPP
#define FASTPOLY_DETAIL_POLY_DIVISION_HPP

#include "fastpoly/detail/poly/inverse.hpp"

namespace fpx::poly {

namespace detail {

template <bool Remainder, class M>
std::pair<vec<M>, vec<M>> divide(const vec<M>& a_in, const vec<M>& b_in) {
  size_t an = a_in.size(), bn = b_in.size();
  while (an != 0 && a_in[an-1].is_zero()) --an;
  while (bn != 0 && b_in[bn-1].is_zero()) --bn;
  if (bn == 0) throw domain_error("poly::divmod: division by zero polynomial");
  if (an < bn) {
    if constexpr (Remainder)
      return {vec<M>{}, vec<M>(a_in.begin(), a_in.begin() + static_cast<std::ptrdiff_t>(an))};
    else return {};
  }
  if (bn == 1) {
    vec<M> q(an);
    detail::scale_limbs(reinterpret_cast<uint32_t*>(q.data()),
                        reinterpret_cast<const uint32_t*>(a_in.data()), an, b_in[0].inv());
    return {std::move(q), vec<M>{}};
  }
  const size_t dn = an - bn + 1;
  if (bn <= 16 || dn <= 32) {
    // Long division avoids constructing a dense reciprocal for short divisors
    // or quotients. Quotient-only division never touches the low remainder.
    vec<M> work(a_in.begin(), a_in.begin()+static_cast<std::ptrdiff_t>(an)), q(dn);
    const M leading_inverse = b_in[bn-1].inv();
    auto* dst = reinterpret_cast<uint32_t*>(work.data());
    const auto* src = reinterpret_cast<const uint32_t*>(b_in.data());
    for (size_t i = dn; i-- > 0;) {
      const M factor = work[i+bn-1]*leading_inverse;
      q[i] = factor;
      if (factor.is_zero()) continue;
      const auto c = simd::fixed_twiddle<M::mod, M::ninv>(factor.raw_val());
      size_t j = Remainder || i >= bn-1 ? 0 : bn-1-i;
      if constexpr (M::mod != 2) {
        for (; j + simd::lane <= bn-1; j += simd::lane) {
          const auto v = simd::mul_twiddle_full<M::mod, M::ninv>(simd::load(src+j), c);
          simd::store(dst+i+j, simd::sub(simd::load(dst+i+j), v, M::mod));
        }
      }
      for (; j < bn-1; ++j) work[i+j] -= factor*b_in[j];
    }
    if constexpr (Remainder) {
      work.resize(bn-1);
      trim(work);
      return {std::move(q), std::move(work)};
    } else return {std::move(q), {}};
  }
  // q = rev( rev(a) * inv(rev(b)) ) mod x^dn
  vec<M> ra(std::make_reverse_iterator(a_in.data() + an),
            std::make_reverse_iterator(a_in.data() + an - dn));
  vec<M> rb(std::make_reverse_iterator(b_in.data() + bn),
            std::make_reverse_iterator(b_in.data() + bn - std::min(bn, dn)));
  vec<M> ib = inv(rb, dn);
  vec<M> rq = conv_limbs(ra, ib, dn);
  rq.resize(dn, M());
  std::reverse(rq.begin(), rq.end());
  vec<M> q = std::move(rq);
  if constexpr (!Remainder) return {std::move(q), {}};
  // Once q is known, the high coefficients of a-b*q cancel exactly. Compute
  // only the low remainder instead of transforming those cancelling terms.
  const size_t rn = bn - 1;
  vec<M> r(a_in.data(), a_in.data() + rn);
  vec<M> product = conv_limbs(b_in, q, rn);
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  const auto* src = reinterpret_cast<const uint32_t*>(product.data());
  size_t i = 0;
  for (; i + simd::lane <= product.size(); i += simd::lane)
    simd::store(dst + i, simd::sub(simd::load(dst + i), simd::load(src + i), M::mod));
  for (; i < product.size(); ++i) r[i] -= product[i];
  trim(r);
  return {std::move(q), std::move(r)};
}

}  // namespace detail

/// Euclidean division: a = b*q + r with deg(r) < deg(b). Both results trimmed.
template <class M>
std::pair<vec<M>, vec<M>> divmod(const vec<M>& a, const vec<M>& b) {
  return detail::divide<true>(a, b);
}

template <class M>
vec<M> div(const vec<M>& a, const vec<M>& b) { return detail::divide<false>(a, b).first; }

template <class M>
vec<M> mod(const vec<M>& a, const vec<M>& b) {
  size_t bn = b.size();
  while (bn != 0 && b[bn-1].is_zero()) --bn;
  if (bn == 0) throw domain_error("poly::mod: division by zero polynomial");
  if (bn == 1) return {};
  if (bn == 2) {
    const M remainder = eval(a, -b[0]*b[1].inv());
    return remainder.is_zero() ? vec<M>{} : vec<M>{remainder};
  }
  return divmod(a, b).second;
}

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_DIVISION_HPP
