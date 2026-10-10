// fastpoly - avx512 native primitives; included by simd.hpp.
#ifndef FASTPOLY_DETAIL_SIMD_AVX512_HPP
#define FASTPOLY_DETAIL_SIMD_AVX512_HPP

namespace fpx::simd {

using native_t = __m512i;
inline constexpr int lane = 16;
inline constexpr const char* name = "avx512";

inline native_t load(const uint32_t* p) { return _mm512_loadu_si512((const void*)p); }
inline void store(uint32_t* p, native_t x) { _mm512_storeu_si512((void*)p, x); }
inline native_t set1(uint32_t x) { return _mm512_set1_epi32((int)x); }
inline native_t zero() { return _mm512_setzero_si512(); }
inline native_t add(native_t a, native_t b, uint32_t R) {
  native_t r = _mm512_add_epi32(a, b);
  // r < 2*R < 2^32; min(r, r-R) folds it back (unsigned).
  return _mm512_min_epu32(r, _mm512_sub_epi32(r, _mm512_set1_epi32((int)R)));
}
inline native_t sub(native_t a, native_t b, uint32_t R) {
  // a-b mod R, in 3 ops: take y = a-b and z = y+R; the unsigned min of the two
  // is a-b when a >= b and y+R (the wrapped value) when a < b.  No compare and
  // no borrow mask are needed.
  const __m512i vr = _mm512_set1_epi32((int)R);
  native_t y = _mm512_sub_epi32(a, b);
  return _mm512_min_epu32(y, _mm512_add_epi32(y, vr));
}
/// Full Montgomery reduction: result in [0, MOD).
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod(native_t a, native_t b) {
  const native_t vn = _mm512_set1_epi32((int)NINV);
  const native_t vm = _mm512_set1_epi32((int)MOD);
  __m512i ah = _mm512_srli_epi64(a, 32), bh = _mm512_srli_epi64(b, 32);
  __m512i t0 = _mm512_mul_epu32(a, b);
  __m512i t1 = _mm512_mul_epu32(ah, bh);
  __m512i m0 = _mm512_mul_epu32(t0, vn);
  __m512i m1 = _mm512_mul_epu32(t1, vn);
  __m512i u0 = _mm512_srli_epi64(_mm512_add_epi64(t0, _mm512_mul_epu32(m0, vm)), 32);
  __m512i u1 = _mm512_srli_epi64(_mm512_add_epi64(t1, _mm512_mul_epu32(m1, vm)), 32);
  __m512i r = _mm512_or_si512(u0, _mm512_slli_epi64(u1, 32));
  return _mm512_min_epu32(r, _mm512_sub_epi32(r, vm));
}
/// Lazy Montgomery reduction (MOD < 2^30, inputs < 2*MOD): result in [0, 2*MOD).
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod_lazy(native_t a, native_t b) {
  const native_t vn = _mm512_set1_epi32((int)NINV);
  const native_t vm = _mm512_set1_epi32((int)MOD);
  __m512i ah = _mm512_srli_epi64(a, 32), bh = _mm512_srli_epi64(b, 32);
  __m512i t0 = _mm512_mul_epu32(a, b);
  __m512i t1 = _mm512_mul_epu32(ah, bh);
  __m512i m0 = _mm512_mul_epu32(t0, vn);
  __m512i m1 = _mm512_mul_epu32(t1, vn);
  __m512i u0 = _mm512_srli_epi64(_mm512_add_epi64(t0, _mm512_mul_epu32(m0, vm)), 32);
  __m512i u1 = _mm512_srli_epi64(_mm512_add_epi64(t1, _mm512_mul_epu32(m1, vm)), 32);
  return _mm512_or_si512(u0, _mm512_slli_epi64(u1, 32));
}
/// Four full-width loads and eight lane-local unpacks. The row permutation
/// (0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15) is undone by store4.
inline void load4(const uint32_t* p, native_t& x0, native_t& x1, native_t& x2, native_t& x3) {
  const native_t a = load(p), b = load(p + 16), c = load(p + 32), d = load(p + 48);
  const native_t t0 = _mm512_unpacklo_epi32(a, b), t1 = _mm512_unpackhi_epi32(a, b);
  const native_t t2 = _mm512_unpacklo_epi32(c, d), t3 = _mm512_unpackhi_epi32(c, d);
  x0 = _mm512_unpacklo_epi64(t0, t2); x1 = _mm512_unpackhi_epi64(t0, t2);
  x2 = _mm512_unpacklo_epi64(t1, t3); x3 = _mm512_unpackhi_epi64(t1, t3);
}
inline void store4(uint32_t* p, native_t y0, native_t y1, native_t y2, native_t y3) {
  const native_t t0 = _mm512_unpacklo_epi32(y0, y1), t1 = _mm512_unpackhi_epi32(y0, y1);
  const native_t t2 = _mm512_unpacklo_epi32(y2, y3), t3 = _mm512_unpackhi_epi32(y2, y3);
  store(p, _mm512_unpacklo_epi64(t0, t2)); store(p + 16, _mm512_unpackhi_epi64(t0, t2));
  store(p + 32, _mm512_unpacklo_epi64(t1, t3)); store(p + 48, _mm512_unpackhi_epi64(t1, t3));
}
/// Small stages keep all sixteen lanes in native-width shuffle networks.
template <int M>
inline void load4m(const uint32_t* p, native_t& x0, native_t& x1, native_t& x2, native_t& x3) {
  if constexpr (M == static_cast<int>(lane)) {
    x0 = load(p);
    x1 = load(p + M);
    x2 = load(p + 2 * M);
    x3 = load(p + 3 * M);
  } else if constexpr (M == 1) {
    load4(p, x0, x1, x2, x3);
  } else if constexpr (M == 8) {
    const native_t a = load(p), b = load(p + 16);
    const native_t c = load(p + 32), d = load(p + 48);
    x0 = _mm512_shuffle_i64x2(a, c, 0x44);
    x1 = _mm512_shuffle_i64x2(a, c, 0xEE);
    x2 = _mm512_shuffle_i64x2(b, d, 0x44);
    x3 = _mm512_shuffle_i64x2(b, d, 0xEE);
  } else {
    static_assert(M == 2 || M == 4, "unsupported small chunk width");
    const native_t a = load(p), b = load(p + 16), c = load(p + 32), d = load(p + 48);
    if constexpr (M == 2) {
      const native_t u = _mm512_shuffle_i64x2(a, b, 0x88);
      const native_t v = _mm512_shuffle_i64x2(a, b, 0xDD);
      const native_t w = _mm512_shuffle_i64x2(c, d, 0x88);
      const native_t z = _mm512_shuffle_i64x2(c, d, 0xDD);
      x0 = _mm512_unpacklo_epi64(u, w); x1 = _mm512_unpackhi_epi64(u, w);
      x2 = _mm512_unpacklo_epi64(v, z); x3 = _mm512_unpackhi_epi64(v, z);
    } else {
      const native_t u = _mm512_shuffle_i64x2(a, b, 0x44);
      const native_t v = _mm512_shuffle_i64x2(a, b, 0xEE);
      const native_t w = _mm512_shuffle_i64x2(c, d, 0x44);
      const native_t z = _mm512_shuffle_i64x2(c, d, 0xEE);
      x0 = _mm512_shuffle_i64x2(u, w, 0x88); x1 = _mm512_shuffle_i64x2(u, w, 0xDD);
      x2 = _mm512_shuffle_i64x2(v, z, 0x88); x3 = _mm512_shuffle_i64x2(v, z, 0xDD);
    }
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
    store4(p, y0, y1, y2, y3);
  } else if constexpr (M == 8) {
    store(p, _mm512_shuffle_i64x2(y0, y1, 0x44));
    store(p + 16, _mm512_shuffle_i64x2(y2, y3, 0x44));
    store(p + 32, _mm512_shuffle_i64x2(y0, y1, 0xEE));
    store(p + 48, _mm512_shuffle_i64x2(y2, y3, 0xEE));
  } else {
    static_assert(M == 2 || M == 4, "unsupported small chunk width");
    if constexpr (M == 2) {
      const native_t u = _mm512_unpacklo_epi64(y0, y1), w = _mm512_unpackhi_epi64(y0, y1);
      const native_t v = _mm512_unpacklo_epi64(y2, y3), z = _mm512_unpackhi_epi64(y2, y3);
      const native_t lo = _mm512_set_epi64(11,10,3,2,9,8,1,0);
      const native_t hi = _mm512_set_epi64(15,14,7,6,13,12,5,4);
      store(p, _mm512_permutex2var_epi64(u, lo, v));
      store(p + 16, _mm512_permutex2var_epi64(u, hi, v));
      store(p + 32, _mm512_permutex2var_epi64(w, lo, z));
      store(p + 48, _mm512_permutex2var_epi64(w, hi, z));
    } else {
      const native_t u = _mm512_shuffle_i64x2(y0, y1, 0x44);
      const native_t v = _mm512_shuffle_i64x2(y0, y1, 0xEE);
      const native_t w = _mm512_shuffle_i64x2(y2, y3, 0x44);
      const native_t z = _mm512_shuffle_i64x2(y2, y3, 0xEE);
      store(p, _mm512_shuffle_i64x2(u, w, 0x88)); store(p + 16, _mm512_shuffle_i64x2(u, w, 0xDD));
      store(p + 32, _mm512_shuffle_i64x2(v, z, 0x88)); store(p + 48, _mm512_shuffle_i64x2(v, z, 0xDD));
    }
  }
}
/// Widest chunk width with a hand-written kernel (must be < lane).
inline constexpr int small_m_max = 8;

/// Swap the two 32-bit halves of every 64-bit group (pairs (2i, 2i+1)).
inline native_t swap_pairs(native_t v) { return _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0xB1); }
/// Even lanes from `a`, odd lanes from `b`.
inline native_t pick_odd(native_t a, native_t b) { return _mm512_mask_blend_epi32(0xAAAAu, a, b); }
/// Canonicalise a lazy value (< 2*MOD) into [0, MOD).
inline native_t reduce_full(native_t v, uint32_t MOD) {
  return _mm512_min_epu32(v, _mm512_sub_epi32(v, _mm512_set1_epi32((int)MOD)));
}
/// Unreduced sum of two lazy values (< 2*MOD each). Only valid as the operand
/// of a Montgomery multiply against a twiddle (< MOD), and only for MOD < 2^30:
/// then the sum < 4*MOD and the product stays below MOD*2^32.
inline native_t add_wide(native_t a, native_t b) { return _mm512_add_epi32(a, b); }

inline native_t sub_wide(native_t a, native_t b) { return _mm512_sub_epi32(a, b); }

// Capabilities and the instructions specific to Q32 fixed multiplication.
struct backend {
  static constexpr bool q31 = false;
  static constexpr bool q32 = true;
  static constexpr bool split_radix8 = true;
  static constexpr bool parallel_linear_exp = false;

  // Keep the k=0/1 split lane-local; no cross-ZMM permutation is needed.
  static native_t unzip_even(native_t a, native_t b) {
    a = _mm512_shuffle_epi32(a, (_MM_PERM_ENUM)0xD8);
    b = _mm512_shuffle_epi32(b, (_MM_PERM_ENUM)0xD8);
    return _mm512_unpacklo_epi64(a, b);
  }
  static native_t unzip_odd(native_t a, native_t b) {
    a = _mm512_shuffle_epi32(a, (_MM_PERM_ENUM)0xD8);
    b = _mm512_shuffle_epi32(b, (_MM_PERM_ENUM)0xD8);
    return _mm512_unpackhi_epi64(a, b);
  }
  static native_t zip_low(native_t a, native_t b) { return _mm512_unpacklo_epi32(a, b); }
  static native_t zip_high(native_t a, native_t b) { return _mm512_unpackhi_epi32(a, b); }

  // A width-2 stage has unity twiddles in even lanes. Only odd lanes multiply.
  template <uint32_t MOD, uint32_t NINV>
  static native_t mul_unity_even(native_t a, native_t twiddle) {
    const native_t odd = _mm512_srli_epi64(a, 32), w = _mm512_srli_epi64(twiddle, 32);
    const native_t t = _mm512_mul_epu32(odd, w);
    const native_t m = _mm512_mul_epu32(odd, _mm512_mullo_epi32(w, set1(NINV)));
    const native_t u = _mm512_add_epi64(t, _mm512_mul_epu32(m, set1(MOD)));
    const native_t r = _mm512_mask_blend_epi32(0xAAAAu, a, u);
    if constexpr (lazy_ok<MOD>) return r;
    else return reduce_full(r, MOD);
  }

  // Convert canonical Montgomery limbs into ordinary values and Q32 quotients.
  template <uint32_t MOD, uint32_t NINV>
  static void fixed_vector(native_t mont, native_t& value, native_t& quotient) {
    const native_t q = _mm512_mullo_epi32(mont, set1(NINV));
    const native_t lo = _mm512_and_si512(mont, _mm512_set1_epi64(0xffffffffu));
    const native_t hi = _mm512_srli_epi64(mont, 32);
    const native_t v0 = _mm512_add_epi64(_mm512_mul_epu32(q, set1(MOD)), lo);
    const native_t v1 = _mm512_add_epi64(_mm512_mul_epu32(_mm512_srli_epi64(q, 32), set1(MOD)), hi);
    value = _mm512_mask_blend_epi32(0xAAAAu, _mm512_srli_epi64(v0, 32), v1);
    quotient = q;
  }

  // The high products estimate the quotient in parallel with the low product.
  template <uint32_t MOD>
  static native_t mul_q32(native_t a, native_t value, native_t quotient) {
    const native_t q0 = _mm512_mul_epu32(a, quotient);
    const native_t q1 = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(quotient, 32));
    const native_t q = _mm512_mask_blend_epi32(0xAAAAu, _mm512_srli_epi64(q0, 32), q1);
    const native_t r = _mm512_sub_epi32(_mm512_mullo_epi32(a, value), _mm512_mullo_epi32(q, set1(MOD)));
    if constexpr (lazy_ok<MOD>) return r;
    else return reduce_full(r, MOD);
  }
};

}  // namespace fpx::simd

#endif  // FASTPOLY_DETAIL_SIMD_AVX512_HPP
