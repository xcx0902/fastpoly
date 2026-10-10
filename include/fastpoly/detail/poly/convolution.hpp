// fastpoly - tiled direct products and NTT convolution.
#ifndef FASTPOLY_DETAIL_POLY_CONVOLUTION_HPP
#define FASTPOLY_DETAIL_POLY_CONVOLUTION_HPP

#include "fastpoly/detail/poly/basic.hpp"

namespace fpx::poly {

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
    // Finish each consecutive output tile while it stays in L1. For tiles
    // after the first, every shifted accumulation starts at the same output
    // address, avoiding repeatedly misaligned stores and full-size passes.
    const vec<M>& small = la <= lb ? a : b;
    const vec<M>& large = la <= lb ? b : a;
    const size_t short_n = std::min(la, lb), long_n = std::max(la, lb);
    auto* dst = reinterpret_cast<uint32_t*>(r.data());
    const auto* src = reinterpret_cast<const uint32_t*>(large.data());
    if (short_n == 1) {
      detail::scale_limbs(dst, src, out, small[0]);
      return r;
    }
    constexpr size_t tile = 4096;
    for (size_t base = 0; base < out; base += tile) {
      const size_t end = std::min(out, base + tile);
      auto accumulate = [&](size_t i) {
        if (small[i].is_zero()) return;
        const size_t begin = std::max(base, i);
        const size_t limit = std::min(end, i + long_n);
        const auto c = simd::fixed_twiddle<M::mod, M::ninv>(small[i].raw_val());
        size_t j = begin - i, cnt = limit - i;
        for (; j + simd::lane <= cnt; j += simd::lane) {
          const auto v = simd::mul_twiddle_full<M::mod, M::ninv>(simd::load(src + j), c);
          simd::store(dst + i + j, simd::add(simd::load(dst + i + j), v, M::mod));
        }
        for (; j < cnt; ++j) r[i + j] += small[i] * large[j];
      };
      size_t i = 0;
      if constexpr (simd::backend::q32 && simd::lazy_ok<M::mod>) {
        // With q = floor(w*2^32/MOD) and canonical a < MOD, the lazy
        // remainder a*w-floor(a*q/2^32)*MOD is < MOD + MOD^2/2^32.
        // Each product is thus < 5*MOD/4. Two
        // products plus a canonical accumulator are < 7*MOD/2 < 2^32.
        // Share the accumulator load/store and reduce only their sum.
        for (; i + 1 < short_n && i < end; i += 2) {
          if (small[i].is_zero() || small[i + 1].is_zero()) {
            accumulate(i);
            if (i + 1 < end) accumulate(i + 1);
            continue;
          }
          const auto c0 = simd::fixed_twiddle<M::mod, M::ninv>(small[i].raw_val());
          const auto c1 = simd::fixed_twiddle<M::mod, M::ninv>(small[i + 1].raw_val());
          if (i >= base) r[i] += small[i] * large[0];
          const size_t last = i + long_n;
          if (last >= base && last < end) r[last] += small[i + 1] * large[long_n - 1];
          size_t j = std::max(base, i + 1);
          const size_t limit = std::min(end, last);
          for (; j + simd::lane <= limit; j += simd::lane) {
            const auto v0 = simd::mul_twiddle<M::mod, M::ninv>(simd::load(src + j - i), c0);
            const auto v1 = simd::mul_twiddle<M::mod, M::ninv>(simd::load(src + j - i - 1), c1);
            const auto sum = simd::add_wide(simd::load(dst + j), simd::add_wide(v0, v1));
            simd::store(dst + j, simd::reduce_full(simd::reduce_full(sum, 2*M::mod), M::mod));
          }
          for (; j < limit; ++j)
            r[j] += small[i]*large[j - i] + small[i + 1]*large[j - i - 1];
        }
      }
      for (; i < short_n && i < end; ++i) accumulate(i);
    }
    return r;
  }

  const uint32_t N = next_pow2(full);
  const auto& plan = NttPlan<M>::get_ref(N);
  vec<M> fa;
  fa.reserve(N);
  fa.insert(fa.end(), a.data(), a.data() + la);
  fa.resize(N);  // Only the padding is initialized, after copying the prefix.
  uint32_t* pa = reinterpret_cast<uint32_t*>(fa.data());
  plan.forward_lazy(pa);
  if (a.data() == b.data() && la == lb) {
    pointwise_mul<M>(pa, pa, N);
  } else {
    fpx::detail::ScratchBuffer fb;
    detail::prepare_scratch(b, lb, fb, N);
    auto* pb = fb.data();
    plan.forward_lazy(pb);
    pointwise_mul<M>(pa, pb, N);
  }
  plan.inverse(pa);
  fa.resize(out);
  return fa;
}

/// Product of a and b, truncated to n coefficients (n = SIZE_MAX -> full).
template <class M>
vec<M> conv(const vec<M>& a, const vec<M>& b, size_t n = SIZE_MAX) {
  return conv_limbs(a, b, n);
}

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_CONVOLUTION_HPP
