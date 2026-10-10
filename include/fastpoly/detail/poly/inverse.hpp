// fastpoly - reciprocal seeds, short inputs and Newton extension.
#ifndef FASTPOLY_DETAIL_POLY_INVERSE_HPP
#define FASTPOLY_DETAIL_POLY_INVERSE_HPP

#include "fastpoly/detail/poly/convolution.hpp"

namespace fpx::poly {

/// ---------------------------------------------------------------------------
/// inverse   (requires a(0) != 0)
/// ---------------------------------------------------------------------------

namespace detail {

// Extend an inverse prefix c to wanted <= 2*c.size(), using only two
// scratch buffers and preserving the c spectrum through the correction.
template <class M>
void extend_inverse(const vec<M>& a, vec<M>& c, size_t wanted,
                  fpx::detail::ScratchBuffer& scratch,
                  fpx::detail::ScratchBuffer& spectrum) {
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
  const auto& plan = NttPlan<M>::get_ref(N);
  prepare_scratch(a, std::min(a.size(), wanted), scratch, N);
  prepare_scratch(c, c.size(), spectrum, N);
  auto* pt = scratch.data();
  auto* pc = spectrum.data();
  plan.forward_lazy(pt);
  plan.forward_lazy(pc);
  pointwise_mul<M>(pt, pc, N);
  plan.inverse(pt);
  size_t i = 0;
  const auto zero = simd::set1(0);
  for (; i + simd::lane <= hlen; i += simd::lane)
    simd::store(pt + i, simd::sub(zero, simd::load(pt + k + i), M::mod));
  for (; i < hlen; ++i) pt[i] = pt[k+i] == 0 ? 0 : M::mod - pt[k+i];
  std::memset(pt + hlen, 0, (N - hlen) * sizeof(uint32_t));
  plan.forward_lazy(pt);
  pointwise_mul<M>(pt, pc, N);
  plan.inverse(pt);
  c.resize(wanted);
  std::memcpy(c.data() + k, pt, hlen * sizeof(uint32_t));
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

// A linear reciprocal is a geometric progression. Four independent SIMD
// chains advance by ratio^(4*lane), avoiding a dependency between neighbours.
template <class M>
vec<M> inverse_linear(const vec<M>& a, size_t n) {
  const M first = a[0].inv(), ratio = -a[1]*first;
  if (ratio.raw_val() == M::one) return vec<M>(n, first);
  vec<M> b(n);
  constexpr size_t W = simd::lane, block = 4*W;
  if (n < block) {
    M x = first;
    for (size_t i = 0; i < n; ++i) { b[i] = x; x *= ratio; }
    return b;
  }
  alignas(64) uint32_t seed[block];
  M x = first;
  for (size_t i = 0; i < block; ++i) { seed[i] = x.raw_val(); x *= ratio; }
  auto x0 = simd::load(seed), x1 = simd::load(seed + W);
  auto x2 = simd::load(seed + 2*W), x3 = simd::load(seed + 3*W);
  const M step_value = ratio.pow(block);
  const auto step = simd::fixed_twiddle<M::mod, M::ninv>(step_value.raw_val());
  auto* dst = reinterpret_cast<uint32_t*>(b.data());
  size_t i = 0;
  if (step_value.raw_val() == M::one) {
    // A short-period geometric progression repeats the already canonical seed.
    for (; i + block <= n; i += block) std::memcpy(dst + i, seed, sizeof(seed));
    if (i != n) std::memcpy(dst + i, seed, (n - i) * sizeof(uint32_t));
    return b;
  }
  auto store_value = [](uint32_t* p, simd::native_t v) {
    if constexpr (simd::fixed_lazy<M::mod>) v = simd::reduce_full(v, M::mod);
    simd::store(p, v);
  };
  auto advance = [&](simd::native_t v) {
    if constexpr (simd::fixed_lazy<M::mod>) return simd::mul_twiddle<M::mod, M::ninv>(v, step);
    else return simd::mul_twiddle_full<M::mod, M::ninv>(v, step);
  };
  for (; i + block <= n; i += block) {
    store_value(dst + i, x0); store_value(dst + i + W, x1);
    store_value(dst + i + 2*W, x2); store_value(dst + i + 3*W, x3);
    x0 = advance(x0); x1 = advance(x1);
    x2 = advance(x2); x3 = advance(x3);
  }
  if (i != n) {
    store_value(seed, x0); store_value(seed + W, x1);
    store_value(seed + 2*W, x2); store_value(seed + 3*W, x3);
    std::memcpy(dst + i, seed, (n - i) * sizeof(uint32_t));
  }
  return b;
}

// Normalize the fixed short input once, then each new coefficient is a dot
// product of at most eight neighbours; no transform or scratch is required.
template <class M>
vec<M> inverse_short(const vec<M>& a, size_t cnt, size_t n) {
  vec<M> b(n);
  b[0] = a[0].inv();
  std::array<M, 8> coefficients;
  const size_t degree = cnt - 1;
  for (size_t j = 0; j < degree; ++j) coefficients[j] = -a[j+1]*b[0];
  for (size_t i = 1; i < n; ++i) {
    M sum;
    for (size_t j = 0; j < std::min(i, degree); ++j)
      sum += coefficients[j]*b[i-j-1];
    b[i] = sum;
  }
  return b;
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
  size_t cnt = std::min(a.size(), n);
  while (cnt > 1 && a[cnt-1].is_zero()) --cnt;
  if (cnt == 1) {  // Includes padded constants and a constant prefix mod x^n.
    vec<M> r(n, M());
    r[0] = a[0].inv();
    return r;
  }
  static_assert(sizeof(M) == sizeof(uint32_t), "Mont must be a single 32-bit limb");
  if (cnt == 2) return detail::inverse_linear(a, n);
  if (cnt <= 5 || (cnt <= 9 && n >= 4096))
    return detail::inverse_short(a, cnt, n);

  const size_t seed = std::min(n, size_t(32));
  const uint32_t scratch_n = detail::series_scratch_size<M>(n, seed);
  vec<M> b = detail::inverse_seed(a, seed);
  b.reserve(n);
  fpx::detail::ScratchBuffer work(scratch_n), spectrum(scratch_n);
  for (size_t m = seed; m < n; m <<= 1)
    detail::extend_inverse(a, b, std::min(2 * m, n), work, spectrum);
  b.resize(n);
  return b;
}

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_INVERSE_HPP
