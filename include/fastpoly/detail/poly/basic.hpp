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

namespace detail {

template <bool Subtract, class M>
void add_assign(vec<M>& a, const vec<M>& b) {
  a.resize(std::max(a.size(), b.size()));
  auto* dst = reinterpret_cast<uint32_t*>(a.data());
  const auto* src = reinterpret_cast<const uint32_t*>(b.data());
  size_t i = 0;
  for (; i + simd::lane <= b.size(); i += simd::lane) {
    const auto x = simd::load(dst+i), y = simd::load(src+i);
    if constexpr (Subtract) simd::store(dst+i, simd::sub(x, y, M::mod));
    else simd::store(dst+i, simd::add(x, y, M::mod));
  }
  for (; i < b.size(); ++i) {
    if constexpr (Subtract) a[i] -= b[i];
    else a[i] += b[i];
  }
}

}  // namespace detail

template <class M>
vec<M> add(vec<M>&& a, const vec<M>& b) {
  detail::add_assign<false>(a, b);
  return std::move(a);
}

template <class M>
vec<M> sub(vec<M>&& a, const vec<M>& b) {
  detail::add_assign<true>(a, b);
  return std::move(a);
}

template <class M>
vec<M> add(const vec<M>& a, const vec<M>& b) {
  // Copy the longer operand once; growing a copy of the shorter operand would
  // allocate twice and copy its coefficients again.
  vec<M> r = a.size() >= b.size() ? a : b;
  detail::add_assign<false>(r, a.size() >= b.size() ? b : a);
  return r;
}

template <class M>
vec<M> sub(const vec<M>& a, const vec<M>& b) {
  vec<M> r;
  r.reserve(std::max(a.size(), b.size()));
  r.insert(r.end(), a.begin(), a.end());
  detail::add_assign<true>(r, b);
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

/// Blocked Horner: independent SIMD chains evaluate residue classes modulo
/// 4*lane, then a short scalar Horner pass joins their values.
template <class M>
M eval(const vec<M>& a, M x) {
  if (a.empty()) return M();
  if (x.is_zero()) return a[0];
  constexpr size_t W = simd::lane, block = 4*W;
  if constexpr (M::mod != 2) {
    if (a.size() >= 4*block) {
      alignas(64) uint32_t seed[block]{};
      size_t i = a.size()/block*block;
      std::memcpy(seed, a.data()+i, (a.size()-i)*sizeof(uint32_t));
      auto v0 = simd::load(seed), v1 = simd::load(seed+W);
      auto v2 = simd::load(seed+2*W), v3 = simd::load(seed+3*W);
      const auto step = simd::fixed_twiddle<M::mod, M::ninv>(x.pow(block).raw_val());
      const auto* src = reinterpret_cast<const uint32_t*>(a.data());
      for (; i != 0;) {
        i -= block;
        v0 = simd::add(simd::mul_twiddle_full<M::mod, M::ninv>(v0, step), simd::load(src+i), M::mod);
        v1 = simd::add(simd::mul_twiddle_full<M::mod, M::ninv>(v1, step), simd::load(src+i+W), M::mod);
        v2 = simd::add(simd::mul_twiddle_full<M::mod, M::ninv>(v2, step), simd::load(src+i+2*W), M::mod);
        v3 = simd::add(simd::mul_twiddle_full<M::mod, M::ninv>(v3, step), simd::load(src+i+3*W), M::mod);
      }
      simd::store(seed, v0); simd::store(seed+W, v1);
      simd::store(seed+2*W, v2); simd::store(seed+3*W, v3);
      M r;
      for (size_t j = block; j-- > 0;) r = r*x+M::raw(seed[j]);
      return r;
    }
  }
  M r = M();
  for (size_t i = a.size(); i-- > 0;) r = r * x + a[i];
  return r;
}

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_BASIC_HPP
