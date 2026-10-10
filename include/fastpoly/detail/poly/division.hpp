// fastpoly - Euclidean division and remainders.
#ifndef FASTPOLY_DETAIL_POLY_DIVISION_HPP
#define FASTPOLY_DETAIL_POLY_DIVISION_HPP

#include "fastpoly/detail/poly/inverse.hpp"

namespace fpx::poly {

/// Euclidean division: a = b*q + r with deg(r) < deg(b). Both results trimmed.
template <class M>
std::pair<vec<M>, vec<M>> divmod(const vec<M>& a_in, const vec<M>& b_in) {
  size_t an = a_in.size(), bn = b_in.size();
  while (an != 0 && a_in[an-1].is_zero()) --an;
  while (bn != 0 && b_in[bn-1].is_zero()) --bn;
  if (bn == 0) throw domain_error("poly::divmod: division by zero polynomial");
  if (an < bn)
    return {vec<M>{}, vec<M>(a_in.begin(), a_in.begin() + static_cast<std::ptrdiff_t>(an))};
  if (bn == 1) {
    vec<M> q(an);
    detail::scale_limbs(reinterpret_cast<uint32_t*>(q.data()),
                        reinterpret_cast<const uint32_t*>(a_in.data()), an, b_in[0].inv());
    return {std::move(q), vec<M>{}};
  }
  const size_t dn = an - bn + 1;
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

template <class M>
vec<M> div(const vec<M>& a, const vec<M>& b) { return divmod(a, b).first; }

template <class M>
vec<M> mod(const vec<M>& a, const vec<M>& b) { return divmod(a, b).second; }

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_DIVISION_HPP
