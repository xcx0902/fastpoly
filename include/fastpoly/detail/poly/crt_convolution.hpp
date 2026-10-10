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
using crt_prime3 = mod1224736769;
inline constexpr size_t crt_block_size = size_t(1) << 23;

template <class P, class M>
vec<P> auxiliary_product(std::span<const M> a, std::span<const M> b, size_t out) {
  const size_t full = a.size() + b.size() - 1;
  if (std::min(a.size(), b.size()) <= naive_threshold) {
    vec<P> r(out);
    for (size_t i = 0; i < a.size() && i < out; ++i) {
      const P x = P::from_int(a[i].val());
      for (size_t j = 0; j < b.size() && j < out-i; ++j)
        r[i+j] += x*P::from_int(b[j].val());
    }
    return r;
  }
  const uint32_t N = next_pow2(full);
  const auto& plan = NttPlan<P>::get_ref(N);
  vec<P> fa(N);
  for (size_t i = 0; i < a.size(); ++i) fa[i] = P::from_int(a[i].val());
  auto* pa = reinterpret_cast<uint32_t*>(fa.data());
  plan.forward_lazy(pa);
  if (a.data() == b.data() && a.size() == b.size()) {
    pointwise_mul<P>(pa, pa, N);
  } else {
    fpx::detail::ScratchBuffer fb(N);
    auto* pb = fb.data();
    for (size_t i = 0; i < b.size(); ++i) pb[i] = P::from_int(b[i].val()).raw_val();
    std::memset(pb + b.size(), 0, (N-b.size())*sizeof(uint32_t));
    plan.forward_lazy(pb);
    pointwise_mul<P>(pa, pb, N);
  }
  plan.inverse(pa);
  fa.resize(out);
  return fa;
}

// BlockSize is a template parameter so the actual overlap/add implementation
// can be exercised with small allocations in the tests. The production limit
// enforces both the NTT root bound and the integer reconstruction bound.
template <class M, size_t BlockSize = crt_block_size>
vec<M> crt_convolution(const vec<M>& a, const vec<M>& b, size_t out) {
  static_assert(BlockSize > 0 && BlockSize <= crt_block_size);
  constexpr uint64_t p1 = crt_prime1::mod, p2 = crt_prime2::mod, p3 = crt_prime3::mod;
  constexpr uint64_t inv12 = crt_prime2::from_int(p1).inv().val();
  constexpr uint64_t inv123 = crt_prime3::from_int(p1*p2).inv().val();
  constexpr uint64_t p12_target = p1*p2 % M::mod;
  vec<M> r(out);
  const size_t la = std::min(a.size(), out), lb = std::min(b.size(), out);
  for (size_t ai = 0; ai < la;) {
    const size_t na = std::min(BlockSize, la-ai);
    const std::span<const M> aa(a.data()+ai, na);
    for (size_t bi = 0; bi < lb && bi < out-ai;) {
      const size_t nb = std::min(BlockSize, lb-bi);
      const std::span<const M> bb(b.data()+bi, nb);
      const size_t count = std::min(na+nb-1, out-ai-bi);
      const auto c1 = auxiliary_product<crt_prime1>(aa, bb, count);
      const auto c2 = auxiliary_product<crt_prime2>(aa, bb, count);
      const auto c3 = auxiliary_product<crt_prime3>(aa, bb, count);
      for (size_t i = 0; i < count; ++i) {
        const uint64_t x1 = c1[i].val();
        const uint64_t x2 = (c2[i].val()+p2-x1)*inv12 % p2;
        const uint64_t x12 = x1+p1*x2;
        const uint64_t x3 = (c3[i].val()+p3-x12%p3)*inv123 % p3;
        // Garner digits fit uint64_t; never materialize the 87-bit integer.
        r[ai+bi+i] += M::from_int(x12 % M::mod + p12_target*x3);
      }
      bi += nb;
    }
    ai += na;
  }
  return r;
}

}  // namespace fpx::poly::detail

#endif  // FASTPOLY_DETAIL_POLY_CRT_CONVOLUTION_HPP
