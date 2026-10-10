// fastpoly - avx2 native primitives; included by simd.hpp.
#ifndef FASTPOLY_DETAIL_SIMD_AVX2_HPP
#define FASTPOLY_DETAIL_SIMD_AVX2_HPP

namespace fpx::simd {

namespace x86 {
/// 4-way de-interleave of 32 consecutive uint32 into four 8-lane vectors:
/// The row order inside a column is (0,2,4,6,1,3,5,7); store4x8 undoes it.
/// AVX2 network; the AVX-512 backend has its own full-width network.
inline void load4x8(const uint32_t* p, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3) {
  const __m256i a = _mm256_loadu_si256((const __m256i*)(p + 0));
  const __m256i b = _mm256_loadu_si256((const __m256i*)(p + 8));
  const __m256i c = _mm256_loadu_si256((const __m256i*)(p + 16));
  const __m256i d = _mm256_loadu_si256((const __m256i*)(p + 24));
  const __m256i t0 = _mm256_unpacklo_epi32(a, b), t1 = _mm256_unpackhi_epi32(a, b);
  const __m256i t2 = _mm256_unpacklo_epi32(c, d), t3 = _mm256_unpackhi_epi32(c, d);
  x0 = _mm256_unpacklo_epi64(t0, t2); x1 = _mm256_unpackhi_epi64(t0, t2);
  x2 = _mm256_unpacklo_epi64(t1, t3); x3 = _mm256_unpackhi_epi64(t1, t3);
}
/// Exact inverse of load4x8.
inline void store4x8(uint32_t* p, __m256i y0, __m256i y1, __m256i y2, __m256i y3) {
  const __m256i t0 = _mm256_unpacklo_epi32(y0, y1), t1 = _mm256_unpackhi_epi32(y0, y1);
  const __m256i t2 = _mm256_unpacklo_epi32(y2, y3), t3 = _mm256_unpackhi_epi32(y2, y3);
  _mm256_storeu_si256((__m256i*)(p + 0), _mm256_unpacklo_epi64(t0, t2));
  _mm256_storeu_si256((__m256i*)(p + 8), _mm256_unpackhi_epi64(t0, t2));
  _mm256_storeu_si256((__m256i*)(p + 16), _mm256_unpacklo_epi64(t1, t3));
  _mm256_storeu_si256((__m256i*)(p + 24), _mm256_unpackhi_epi64(t1, t3));
}
/// Chunk width 2 (four 8-element blocks): xj lane i gets p[(i/2)*8 + 2j + i%2].
inline void load4m2x8(const uint32_t* p, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3) {
  const __m256i w0 = _mm256_loadu_si256((const __m256i*)(p + 0));
  const __m256i w1 = _mm256_loadu_si256((const __m256i*)(p + 8));
  const __m256i w2 = _mm256_loadu_si256((const __m256i*)(p + 16));
  const __m256i w3 = _mm256_loadu_si256((const __m256i*)(p + 24));
  const __m256i a = _mm256_permute2x128_si256(w0, w1, 0x20);
  const __m256i b = _mm256_permute2x128_si256(w0, w1, 0x31);
  const __m256i c = _mm256_permute2x128_si256(w2, w3, 0x20);
  const __m256i d = _mm256_permute2x128_si256(w2, w3, 0x31);
  x0 = _mm256_unpacklo_epi64(a, c);
  x1 = _mm256_unpackhi_epi64(a, c);
  x2 = _mm256_unpacklo_epi64(b, d);
  x3 = _mm256_unpackhi_epi64(b, d);
}
inline void store4m2x8(uint32_t* p, __m256i y0, __m256i y1, __m256i y2, __m256i y3) {
  const __m256i a = _mm256_unpacklo_epi64(y0, y1);
  const __m256i c = _mm256_unpackhi_epi64(y0, y1);
  const __m256i b = _mm256_unpacklo_epi64(y2, y3);
  const __m256i d = _mm256_unpackhi_epi64(y2, y3);
  _mm256_storeu_si256((__m256i*)(p + 0), _mm256_permute2x128_si256(a, b, 0x20));
  _mm256_storeu_si256((__m256i*)(p + 8), _mm256_permute2x128_si256(a, b, 0x31));
  _mm256_storeu_si256((__m256i*)(p + 16), _mm256_permute2x128_si256(c, d, 0x20));
  _mm256_storeu_si256((__m256i*)(p + 24), _mm256_permute2x128_si256(c, d, 0x31));
}
/// Chunk width 4 (two 16-element blocks): xj lane i gets p[(i/4)*16 + 4j + i%4].
inline void load4m4x8(const uint32_t* p, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3) {
  const __m256i v0 = _mm256_loadu_si256((const __m256i*)(p + 0));
  const __m256i v1 = _mm256_loadu_si256((const __m256i*)(p + 8));
  const __m256i v2 = _mm256_loadu_si256((const __m256i*)(p + 16));
  const __m256i v3 = _mm256_loadu_si256((const __m256i*)(p + 24));
  x0 = _mm256_permute2x128_si256(v0, v2, 0x20);
  x1 = _mm256_permute2x128_si256(v0, v2, 0x31);
  x2 = _mm256_permute2x128_si256(v1, v3, 0x20);
  x3 = _mm256_permute2x128_si256(v1, v3, 0x31);
}
inline void store4m4x8(uint32_t* p, __m256i y0, __m256i y1, __m256i y2, __m256i y3) {
  _mm256_storeu_si256((__m256i*)(p + 0), _mm256_set_m128i(_mm256_castsi256_si128(y1), _mm256_castsi256_si128(y0)));
  _mm256_storeu_si256((__m256i*)(p + 8), _mm256_set_m128i(_mm256_castsi256_si128(y3), _mm256_castsi256_si128(y2)));
  _mm256_storeu_si256((__m256i*)(p + 16), _mm256_set_m128i(_mm256_extracti128_si256(y1, 1), _mm256_extracti128_si256(y0, 1)));
  _mm256_storeu_si256((__m256i*)(p + 24), _mm256_set_m128i(_mm256_extracti128_si256(y3, 1), _mm256_extracti128_si256(y2, 1)));
}
}  // namespace x86

using native_t = __m256i;
inline constexpr int lane = 8;
inline constexpr const char* name = "avx2";

inline native_t load(const uint32_t* p) { return _mm256_loadu_si256((const __m256i*)p); }
inline void store(uint32_t* p, native_t x) { _mm256_storeu_si256((__m256i*)p, x); }
inline native_t set1(uint32_t x) { return _mm256_set1_epi32((int)x); }
inline native_t zero() { return _mm256_setzero_si256(); }
inline native_t add(native_t a, native_t b, uint32_t R) {
  native_t r = _mm256_add_epi32(a, b);
  return _mm256_min_epu32(r, _mm256_sub_epi32(r, _mm256_set1_epi32((int)R)));
}
inline native_t sub(native_t a, native_t b, uint32_t R) {
  // a-b mod R in 3 ops: min(a-b, a-b+R) picks the wrapped branch when a < b
  // because a-b is a large unsigned value when it wraps.
  const __m256i vr = _mm256_set1_epi32((int)R);
  native_t y = _mm256_sub_epi32(a, b);
  return _mm256_min_epu32(y, _mm256_add_epi32(y, vr));
}
/// Montgomery multiplication of 8 lanes (even/odd halves via mul_epu32),
/// full reduction: result in [0, MOD).
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod(native_t a, native_t b) {
  const native_t vn = _mm256_set1_epi32((int)NINV);
  const native_t vm = _mm256_set1_epi32((int)MOD);
  __m256i t0 = _mm256_mul_epu32(a, b);  // lanes 0,2,4,6 -> 64-bit products
  __m256i t1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
  __m256i m0 = _mm256_mul_epu32(t0, vn);  // low 32 bits are the Montgomery digit
  __m256i m1 = _mm256_mul_epu32(t1, vn);
  __m256i u0 = _mm256_srli_epi64(_mm256_add_epi64(t0, _mm256_mul_epu32(m0, vm)), 32);
  __m256i u1 = _mm256_srli_epi64(_mm256_add_epi64(t1, _mm256_mul_epu32(m1, vm)), 32);
  __m256i r = _mm256_or_si256(u0, _mm256_slli_epi64(u1, 32));
  // r < 2*MOD, so an unsigned min against r-MOD is the conditional subtract.
  return _mm256_min_epu32(r, _mm256_sub_epi32(r, vm));
}
/// Lazy Montgomery multiplication (MOD < 2^30, inputs < 2*MOD):
/// result in [0, 2*MOD), computed without the final conditional subtraction.
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod_lazy(native_t a, native_t b) {
  const native_t vn = _mm256_set1_epi32((int)NINV);
  const native_t vm = _mm256_set1_epi32((int)MOD);
  __m256i t0 = _mm256_mul_epu32(a, b);
  __m256i t1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
  __m256i m0 = _mm256_mul_epu32(t0, vn);
  __m256i m1 = _mm256_mul_epu32(t1, vn);
  __m256i u0 = _mm256_srli_epi64(_mm256_add_epi64(t0, _mm256_mul_epu32(m0, vm)), 32);
  __m256i u1 = _mm256_srli_epi64(_mm256_add_epi64(t1, _mm256_mul_epu32(m1, vm)), 32);
  return _mm256_or_si256(u0, _mm256_slli_epi64(u1, 32));
}
/// 4-way de-interleave of 4*lane = 32 consecutive values.
inline void load4(const uint32_t* p, native_t& x0, native_t& x1, native_t& x2, native_t& x3) {
  x86::load4x8(p, x0, x1, x2, x3);
}
inline void store4(uint32_t* p, native_t y0, native_t y1, native_t y2, native_t y3) {
  x86::store4x8(p, y0, y1, y2, y3);
}
template <int M>
inline void load4m(const uint32_t* p, native_t& x0, native_t& x1, native_t& x2, native_t& x3) {
  if constexpr (M == static_cast<int>(lane)) {
    x0 = load(p);
    x1 = load(p + M);
    x2 = load(p + 2 * M);
    x3 = load(p + 3 * M);
  } else if constexpr (M == 1) {
    x86::load4x8(p, x0, x1, x2, x3);
  } else if constexpr (M == 2) {
    x86::load4m2x8(p, x0, x1, x2, x3);
  } else {
    static_assert(M == 4, "unsupported small chunk width");
    x86::load4m4x8(p, x0, x1, x2, x3);
  }
}
template <int M>
inline void store4m(uint32_t* p, native_t y0, native_t y1, native_t y2, native_t y3) {
  if constexpr (M == static_cast<int>(lane)) {
    store(p, y0);
    store(p + M, y1);
    store(p + 2 * M, y2);
    store(p + 3 * M, y3);
  } else if constexpr (M == 1) {
    x86::store4x8(p, y0, y1, y2, y3);
  } else if constexpr (M == 2) {
    x86::store4m2x8(p, y0, y1, y2, y3);
  } else {
    x86::store4m4x8(p, y0, y1, y2, y3);
  }
}
/// Widest chunk width with a hand-written kernel (must be < lane).
inline constexpr int small_m_max = 4;

/// Swap the two 32-bit halves of every 64-bit group (pairs (2i, 2i+1)).
inline native_t swap_pairs(native_t v) { return _mm256_shuffle_epi32(v, 0xB1); }
/// Even lanes from `a`, odd lanes from `b`.
inline native_t pick_odd(native_t a, native_t b) { return _mm256_blend_epi32(a, b, 0xAA); }
/// Canonicalise a lazy value (< 2*MOD) into [0, MOD).
inline native_t reduce_full(native_t v, uint32_t MOD) {
  return _mm256_min_epu32(v, _mm256_sub_epi32(v, _mm256_set1_epi32((int)MOD)));
}
/// Unreduced sum of two lazy values (< 2*MOD each), for fixed multiplication.
inline native_t add_wide(native_t a, native_t b) { return _mm256_add_epi32(a, b); }

inline native_t sub_wide(native_t a, native_t b) { return _mm256_sub_epi32(a, b); }

// Capabilities and the instructions specific to Q32 fixed multiplication.
struct backend {
  static constexpr bool q31 = false;
  static constexpr bool q32 = true;
  static constexpr bool split_radix8 = false;
  static constexpr bool parallel_linear_exp = false;

  // A width-2 stage has unity twiddles in even lanes. Only odd lanes multiply.
  template <uint32_t MOD, uint32_t NINV>
  static native_t mul_unity_even(native_t a, native_t twiddle) {
    const native_t odd = _mm256_srli_epi64(a, 32), w = _mm256_srli_epi64(twiddle, 32);
    const native_t t = _mm256_mul_epu32(odd, w);
    const native_t m = _mm256_mul_epu32(odd, _mm256_mullo_epi32(w, set1(NINV)));
    const native_t u = _mm256_add_epi64(t, _mm256_mul_epu32(m, set1(MOD)));
    const native_t r = _mm256_blend_epi32(a, u, 0xAA);
    if constexpr (lazy_ok<MOD>) return r;
    else return reduce_full(r, MOD);
  }

  // Convert canonical Montgomery limbs into ordinary values and Q32 quotients.
  template <uint32_t MOD, uint32_t NINV>
  static void fixed_vector(native_t mont, native_t& value, native_t& quotient) {
    const native_t q = _mm256_mullo_epi32(mont, set1(NINV));
    const native_t lo = _mm256_and_si256(mont, _mm256_set1_epi64x(0xffffffffu));
    const native_t hi = _mm256_srli_epi64(mont, 32);
    const native_t v0 = _mm256_add_epi64(_mm256_mul_epu32(q, set1(MOD)), lo);
    const native_t v1 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(q, 32), set1(MOD)), hi);
    value = _mm256_blend_epi32(_mm256_srli_epi64(v0, 32), v1, 0xAA);
    quotient = q;
  }

  // The high products estimate the quotient in parallel with the low product.
  template <uint32_t MOD>
  static native_t mul_q32(native_t a, native_t value, native_t quotient) {
    const native_t q0 = _mm256_mul_epu32(a, quotient);
    const native_t q1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(quotient, 32));
    const native_t q = _mm256_blend_epi32(_mm256_srli_epi64(q0, 32), q1, 0xAA);
    const native_t r = _mm256_sub_epi32(_mm256_mullo_epi32(a, value), _mm256_mullo_epi32(q, set1(MOD)));
    if constexpr (lazy_ok<MOD>) return r;
    else return reduce_full(r, MOD);
  }
};

}  // namespace fpx::simd

#endif  // FASTPOLY_DETAIL_SIMD_AVX2_HPP
