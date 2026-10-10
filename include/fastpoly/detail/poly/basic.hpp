// fastpoly - coefficient arithmetic, shifts and evaluation.
#ifndef FASTPOLY_DETAIL_POLY_BASIC_HPP
#define FASTPOLY_DETAIL_POLY_BASIC_HPP

#include "fastpoly/detail/poly/base.hpp"

namespace fpx::poly {

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
  if (c.raw_val() == M::one) return a;
  vec<M> r(a.size());
  if (c.is_zero()) return r;
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  const auto* src = reinterpret_cast<const uint32_t*>(a.data());
  detail::scale_limbs(dst, src, a.size(), c);
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

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_BASIC_HPP
