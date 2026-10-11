// fastpoly - exact convolution independent of the target prime's roots.
#ifndef FASTPOLY_DETAIL_POLY_CRT_CONVOLUTION_HPP
#define FASTPOLY_DETAIL_POLY_CRT_CONVOLUTION_HPP

#include <span>
#include "fastpoly/detail/poly/base.hpp"

namespace fpx::poly::detail {

// All three auxiliary fields support 2^24 points. With at most 2^23
// terms per coefficient and target residues < 2^31, an integer coefficient is
// < 2^85. The product of the primes is > 2^85, so CRT reconstructs the
// integer coefficient uniquely, with no floating point.
using crt_prime1 = mod167772161;
using crt_prime2 = mod469762049;
// Keep every auxiliary modulus below 2^30 so all three transforms can use
// lazy butterflies. Their product still exceeds the worst-case 2^85 bound.
using crt_prime3 = mod754974721;
inline constexpr size_t crt_block_size = size_t(1) << 23;

// Lift Montgomery limbs as ordinary integers. The reconstructed integer
// product has two target Montgomery factors; one target REDC below removes
// the extra factor. This avoids decoding every input in every auxiliary field.
template <class P, class M>
void lift_limbs(uint32_t* dst, std::span<const M> src) {
  const auto factor = simd::set1(P::r2);
  const auto* input = reinterpret_cast<const uint32_t*>(src.data());
  size_t i = 0;
  for (; i + simd::lane <= src.size(); i += simd::lane)
    simd::store(dst + i, simd::mulmod<P::mod, P::ninv>(simd::load(input + i), factor));
  for (; i < src.size(); ++i) dst[i] = P::reduce(uint64_t(input[i])*P::r2);
}

template <class P, class M>
fpx::detail::ScratchBuffer auxiliary_product(std::span<const M> a, std::span<const M> b,
                                             size_t out, size_t offset = 0,
                                             uint32_t cyclic_size = 0) {
  if (std::min(a.size(), b.size()) <= naive_threshold) {
    fpx::detail::ScratchBuffer r(out), bb(b.size());
    if (out != 0) std::memset(r.data(), 0, out*sizeof(uint32_t));
    lift_limbs<P>(bb.data(), b);
    for (size_t i = 0; i < a.size() && i < offset+out; ++i) {
      const P x = P::raw(P::reduce(uint64_t(a[i].raw_val())*P::r2));
      for (size_t j = offset > i ? offset-i : 0;
           j < b.size() && j < offset+out-i; ++j)
        r[i+j-offset] = (P::raw(r[i+j-offset])+x*P::raw(bb[j])).raw_val();
    }
    return r;
  }
  const uint32_t N = cyclic_size ? cyclic_size : next_pow2(a.size()+b.size()-1);
  const auto& plan = NttPlan<P>::get_ref(N);
  fpx::detail::ScratchBuffer fa(N);
  auto* pa = fa.data();
  lift_limbs<P>(pa, a);
  plan.forward_prefix(pa, a.size());
  if (a.data() == b.data() && a.size() == b.size()) {
    pointwise_mul<P>(pa, pa, N);
  } else {
    fpx::detail::ScratchBuffer fb(N);
    auto* pb = fb.data();
    lift_limbs<P>(pb, b);
    plan.forward_prefix(pb, b.size());
    pointwise_mul<P>(pa, pb, N);
  }
  if (offset == N/2) plan.inverse_highhalf(pa);
  else plan.inverse_prefix(pa, offset+out);
  if (offset != 0) std::memmove(pa, pa+offset, out*sizeof(uint32_t));
  fa.resize_uninitialized(out);
  return fa;
}

// Strict integer bounds: equality with the CRT modulus would alias to zero.
// Use division before multiplication so this also works without uint128_t.
template <class M>
size_t crt_prime_count(size_t terms) {
  constexpr uint64_t square = uint64_t(M::mod-1)*(M::mod-1);
  if (square <= (crt_prime1::mod-1)/terms) return 1;
  if (square <= (uint64_t(crt_prime1::mod)*crt_prime2::mod-1)/terms) return 2;
  return 3;
}

// Garner digits and the target reconstruction entirely in SIMD lanes. Ordinary
// digits times a target constant followed by REDC directly produce the desired
// target Montgomery limb (the lifted input product contains two R factors).
template <unsigned Primes, class M>
size_t restore_crt(M* dst, size_t count, const fpx::detail::ScratchBuffer& c1,
                   const fpx::detail::ScratchBuffer& c2, const fpx::detail::ScratchBuffer& c3) {
  if constexpr (M::mod == 2 || simd::lane == 1) return 0;
  constexpr uint64_t p1 = crt_prime1::mod, p2 = crt_prime2::mod;
  constexpr auto inverse12 = crt_prime2::from_int(p1).inv();
  constexpr auto inverse123 = crt_prime3::from_int(p1*p2).inv();
  const auto inv12 = simd::fixed_twiddle<crt_prime2::mod, crt_prime2::ninv>(inverse12.raw_val());
  const auto inv123 = simd::fixed_twiddle<crt_prime3::mod, crt_prime3::ninv>(inverse123.raw_val());
  const auto p1_in3 = simd::fixed_twiddle<crt_prime3::mod, crt_prime3::ninv>(crt_prime3::from_int(p1).raw_val());
  const auto one = simd::set1(1), p1_target = simd::set1(p1%M::mod);
  const auto p12_target = simd::set1(p1*p2%M::mod);
  auto* output = reinterpret_cast<uint32_t*>(dst);
  const auto* a = c1.data();
  const auto* b = c2.data();
  const auto* c = c3.data();
  size_t i = 0;
  for (; i + simd::lane <= count; i += simd::lane) {
    const auto x1 = simd::mulmod<crt_prime1::mod, crt_prime1::ninv>(simd::load(a+i), one);
    auto value = simd::mulmod<M::mod, M::ninv>(x1, one);
    if constexpr (Primes >= 2) {
      const auto r2 = simd::mulmod<crt_prime2::mod, crt_prime2::ninv>(simd::load(b+i), one);
      const auto x2 = simd::mul_twiddle_full<crt_prime2::mod, crt_prime2::ninv>(
          simd::sub(r2, x1, crt_prime2::mod), inv12);
      value = simd::add(value, simd::mulmod<M::mod, M::ninv>(x2, p1_target), M::mod);
      if constexpr (Primes == 3) {
        const auto r3 = simd::mulmod<crt_prime3::mod, crt_prime3::ninv>(simd::load(c+i), one);
        const auto x12 = simd::add(x1,
            simd::mul_twiddle_full<crt_prime3::mod, crt_prime3::ninv>(x2, p1_in3), crt_prime3::mod);
        const auto x3 = simd::mul_twiddle_full<crt_prime3::mod, crt_prime3::ninv>(
            simd::sub(r3, x12, crt_prime3::mod), inv123);
        value = simd::add(value, simd::mulmod<M::mod, M::ninv>(x3, p12_target), M::mod);
      }
    }
    simd::store(output+i, simd::add(simd::load(output+i), value, M::mod));
  }
  return i;
}

// Accumulate a window of an exact integer product. A supplied cyclic size is
// allowed only when all wrapped coefficients lie strictly below offset.
template <class M>
void crt_product_window(std::span<const M> a, std::span<const M> b,
                        M* dst, size_t count, size_t offset = 0, uint32_t cyclic_size = 0) {
  constexpr uint64_t p1 = crt_prime1::mod, p2 = crt_prime2::mod, p3 = crt_prime3::mod;
  constexpr auto inv12 = crt_prime2::from_int(p1).inv();
  constexpr auto inv123 = crt_prime3::from_int(p1*p2).inv();
  constexpr uint64_t p12_target = p1*p2 % M::mod;
  const size_t primes = crt_prime_count<M>(std::min(a.size(), b.size()));
  const auto c1 = auxiliary_product<crt_prime1>(a, b, count, offset, cyclic_size);
  if (primes == 1) {
    size_t i = restore_crt<1>(dst, count, c1, {}, {});
    for (; i < count; ++i) dst[i] += M::raw(M::reduce(crt_prime1::raw(c1[i]).val()));
    return;
  }
  const auto c2 = auxiliary_product<crt_prime2>(a, b, count, offset, cyclic_size);
  auto digit2 = [&](size_t i, uint32_t x1) {
    // REDC with a Montgomery constant multiplies an ordinary integer and
    // returns an ordinary integer, saving a division-based conversion.
    return crt_prime2::reduce((crt_prime2::raw(c2[i]).val()+p2-x1)*inv12.raw_val());
  };
  if (primes == 2) {
    size_t i = restore_crt<2>(dst, count, c1, c2, {});
    for (; i < count; ++i) {
      const uint32_t x1 = crt_prime1::raw(c1[i]).val(), x2 = digit2(i, x1);
      dst[i] += M::raw(M::reduce((x1+p1*x2) % M::mod));
    }
    return;
  }
  const auto c3 = auxiliary_product<crt_prime3>(a, b, count, offset, cyclic_size);
  size_t i = restore_crt<3>(dst, count, c1, c2, c3);
  for (; i < count; ++i) {
    const uint32_t x1 = crt_prime1::raw(c1[i]).val(), x2 = digit2(i, x1);
    const uint64_t x12 = x1+p1*x2;
    const uint32_t x3 = crt_prime3::reduce((crt_prime3::raw(c3[i]).val()+p3-x12%p3)*inv123.raw_val());
    dst[i] += M::raw(M::reduce((x12 % M::mod + p12_target*x3) % M::mod));
  }
}

// BlockSize is a template parameter so the actual overlap/add implementation
// can be exercised with small allocations in the tests. The production limit
// enforces both the NTT root bound and the integer reconstruction bound.
template <class M, size_t BlockSize = crt_block_size>
vec<M> crt_convolution(const vec<M>& a, const vec<M>& b, size_t out) {
  static_assert(BlockSize > 0 && BlockSize <= crt_block_size);
  vec<M> r(out);
  const size_t la = std::min(a.size(), out), lb = std::min(b.size(), out);
  for (size_t ai = 0; ai < la;) {
    const size_t na = std::min(BlockSize, la-ai);
    const std::span<const M> aa(a.data()+ai, na);
    for (size_t bi = 0; bi < lb && bi < out-ai;) {
      const size_t nb = std::min(BlockSize, lb-bi);
      const std::span<const M> bb(b.data()+bi, nb);
      const size_t count = std::min(na+nb-1, out-ai-bi);
      crt_product_window(aa, bb, r.data()+ai+bi, count);
      bi += nb;
    }
    ai += na;
  }
  return r;
}

// At Newton precision m, deg(a) < 2m and deg(b) < m. Modulo x^(2m)-1,
// wraparound can only affect [0,m-1), so the high half is exact even over
// the integers lifted into the auxiliary fields.
template <class M>
vec<M> crt_middle_product(const vec<M>& a, const vec<M>& b, size_t m, size_t count,
                          size_t offset = SIZE_MAX) {
  if (offset == SIZE_MAX) offset = m;
  if (m > crt_block_size) {
    auto product = crt_convolution(a, b, offset+count);
    vec<M> r(count);
    if (product.size() > offset)
      std::copy(product.begin()+static_cast<std::ptrdiff_t>(offset), product.end(), r.begin());
    return r;
  }
  vec<M> r(count);
  const std::span<const M> aa(a.data(), std::min(a.size(), offset+count));
  const std::span<const M> bb(b.data(), std::min(b.size(), m));
  if (!aa.empty() && !bb.empty())
    crt_product_window(aa, bb, r.data(), count, offset, next_pow2(2*m));
  return r;
}

}  // namespace fpx::poly::detail

#endif  // FASTPOLY_DETAIL_POLY_CRT_CONVOLUTION_HPP
