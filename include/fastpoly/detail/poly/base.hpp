// fastpoly - coefficient types, errors and Montgomery-limb helpers.
#ifndef FASTPOLY_DETAIL_POLY_BASE_HPP
#define FASTPOLY_DETAIL_POLY_BASE_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fastpoly/modint.hpp"
#include "fastpoly/memory.hpp"
#include "fastpoly/ntt.hpp"
#include "fastpoly/simd.hpp"

namespace fpx::poly {

/// Thrown for algebraic preconditions (a(0) == 0 in an inverse, a(0) != 1 in
/// a logarithm, a non-residue constant term in a square root, ...).
class domain_error : public std::runtime_error {
 public:
  explicit domain_error(const std::string& s) : std::runtime_error(s) {}
};

template <class M>
using vec = std::vector<M>;

namespace detail {

// Newton uses powers of two, except that a final <=8-coefficient tail is
// filled directly. Reserve the largest transform once, without allocating a
// needless doubled buffer for that small tail.
template <class M>
uint32_t series_scratch_size(size_t n, size_t seed) {
  if (n <= seed) return 0;
  const size_t floor = std::bit_floor(n);
  const uint32_t N = next_pow2(n - floor <= 8 ? floor : n);
  if (N <= seed) return 0;
  if (N > ntt_max_size<M>())
    throw ntt_size_error("NTT size exceeds 2^v2(mod-1) for this modulus");
  return N;
}

// A transform overwrites every limb. Copy the live coefficients first and
// initialize only its zero padding; recycled scratch needs no value objects.
template <class M>
void prepare_scratch(const vec<M>& src, size_t cnt,
                     fpx::detail::ScratchBuffer& dst, size_t n) {
  static_assert(sizeof(M) == sizeof(uint32_t), "Mont must be a single 32-bit limb");
  dst.resize_uninitialized(n);
  if (cnt != 0) std::memcpy(dst.data(), src.data(), cnt * sizeof(uint32_t));
  std::memset(dst.data() + cnt, 0, (n - cnt) * sizeof(uint32_t));
}

template <class M>
void scale_limbs(uint32_t* dst, const uint32_t* src, size_t n, M c) {
  if (c.raw_val() == M::one) {
    if (dst != src && n != 0) std::memcpy(dst, src, n * sizeof(uint32_t));
    return;
  }
  if (c.is_zero()) {
    if (n != 0) std::memset(dst, 0, n * sizeof(uint32_t));
    return;
  }
  const auto factor = simd::fixed_twiddle<M::mod, M::ninv>(c.raw_val());
  size_t i = 0;
  for (; i + simd::lane <= n; i += simd::lane)
    simd::store(dst + i, simd::mul_twiddle_full<M::mod, M::ninv>(simd::load(src + i), factor));
  for (; i < n; ++i) dst[i] = M::reduce(uint64_t(src[i]) * c.raw_val());
}

template <class M>
void scale_inplace(vec<M>& a, M c) {
  auto* p = reinterpret_cast<uint32_t*>(a.data());
  scale_limbs(p, p, a.size(), c);
}

}  // namespace detail

/// Convolutions with a short operand are done directly.
inline constexpr size_t naive_threshold = 40;

/// dst[0, cnt) = src[0, cnt)
template <class M>
void copy_prefix(const vec<M>& src, size_t cnt, vec<M>& dst) {
  for (size_t i = 0; i < cnt; ++i) dst[i] = src[i];
}
/// dst[0, cnt) = src[from, from + cnt)
template <class M>
void copy_range(const vec<M>& src, size_t from, size_t cnt, vec<M>& dst) {
  for (size_t i = 0; i < cnt; ++i) dst[i] = src[from + i];
}

/// Pointwise Montgomery product of two transform-domain buffers:
/// `dst[i] *= b[i]`.  This is a full pass over the spectrum, so it runs on the
/// SIMD lane width; for Mod < 2^30 it uses the lazy kernel (the inverse
/// transform canonicalises afterwards).
template <class M>
void pointwise_mul(uint32_t* dst, const uint32_t* b, size_t n) {
  constexpr uint32_t MOD = M::mod;
  constexpr bool LAZY = simd::lazy_ok<MOD>;
  size_t i = 0;
  constexpr size_t W = simd::lane;
  // Four independent products overlap the Montgomery multiply chains while
  // retaining consecutive loads/stores for every SIMD backend.
  for (; i + 4*W <= n; i += 4*W) {
    const auto x0 = simd::load(dst + i), y0 = simd::load(b + i);
    const auto x1 = simd::load(dst + i + W), y1 = simd::load(b + i + W);
    const auto x2 = simd::load(dst + i + 2*W), y2 = simd::load(b + i + 2*W);
    const auto x3 = simd::load(dst + i + 3*W), y3 = simd::load(b + i + 3*W);
    const auto v0 = simd::butterfly_mul<MOD, M::ninv, LAZY>(x0, y0);
    const auto v1 = simd::butterfly_mul<MOD, M::ninv, LAZY>(x1, y1);
    const auto v2 = simd::butterfly_mul<MOD, M::ninv, LAZY>(x2, y2);
    const auto v3 = simd::butterfly_mul<MOD, M::ninv, LAZY>(x3, y3);
    simd::store(dst + i, v0);
    simd::store(dst + i + W, v1);
    simd::store(dst + i + 2*W, v2);
    simd::store(dst + i + 3*W, v3);
  }
  for (; i + simd::lane <= n; i += simd::lane) {
    simd::store(dst + i, simd::butterfly_mul<MOD, M::ninv, LAZY>(
                             simd::load(dst + i), simd::load(b + i)));
  }
  for (; i < n; ++i) dst[i] = M::reduce(uint64_t(dst[i]) * b[i]);
}

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_BASE_HPP
