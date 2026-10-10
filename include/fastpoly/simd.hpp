// fastpoly - portable SIMD layer for Montgomery butterflies.
//
// Backends:
//   AVX2   (x86-64, 8 x uint32 lanes)  -- requires -mavx2
//   AVX-512F/VL (x86-64, 16 lanes)     -- requires -mavx512f -mavx512vl (optional)
//   NEON   (arm64,  4 x uint32 lanes)  -- baseline on AArch64
//   Scalar (fallback)
//
// All operations are on Montgomery residues, so `mulmod` is a Montgomery
// multiplication (mul-lo/mul-hi + shift), never a division.
//
// Two reduction conventions are provided:
//
//   * `mulmod` / `add` / `sub` with R = MOD keep every value in [0, MOD).
//     These have the exact scalar semantics and are what the unit tests pin.
//   * `mulmod_lazy` plus `add` / `sub` with R = rmod<MOD> keep every value in
//     [0, R) with R = 2*MOD.  For MOD < 2^30 the bound 4*MOD^2 < MOD*2^32 means
//     a Montgomery product of two values < R lands back in [0, R) *without* the
//     final conditional subtraction, so the butterfly drops that step on every
//     one of its four multiplications.  R is still a multiple of MOD, so the
//     results represent the same residues mod MOD.
#ifndef FASTPOLY_SIMD_HPP
#define FASTPOLY_SIMD_HPP

#include <cstdint>
#include <cstddef>

#if defined(__AVX512F__) && defined(__AVX512VL__)
#define FPX_SIMD_AVX512 1
#define FPX_HAVE_AVX2_INTRIN 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define FPX_SIMD_AVX2 1
#define FPX_HAVE_AVX2_INTRIN 1
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#define FPX_SIMD_NEON 1
#include <arm_neon.h>
#endif

#if defined(FPX_HAVE_AVX2_INTRIN)
namespace fpx {
namespace simd {
namespace x86 {
/// 4-way de-interleave of 32 consecutive uint32 into four 8-lane vectors:
/// The row order inside a column is (0,2,4,6,1,3,5,7); store4x8 undoes it.
/// AVX2 network; the AVX-512 backend uses its own full-width network below.
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
}  // namespace simd
}  // namespace fpx
#endif

namespace fpx {
namespace simd {

/// True when the modulus is small enough to use the lazy [0, 2*MOD) convention.
/// The exact condition is 4*MOD^2 < MOD*2^32, i.e. MOD < 2^30.
template <uint32_t MOD>
inline constexpr bool lazy_ok = MOD < (1u << 30);

/// Reduction modulus of the lazy convention: the ring the residues live in.
/// `R` is a multiple of MOD, so reducing mod R still produces a valid residue.
template <uint32_t MOD>
inline constexpr uint32_t rmod = lazy_ok<MOD> ? (2u * MOD) : MOD;

#if defined(FPX_SIMD_AVX512)

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

#elif defined(FPX_SIMD_AVX2)

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
  // (see the AVX-512 comment above).
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
/// Unreduced sum of two lazy values (< 2*MOD each); see the AVX-512 note above.
inline native_t add_wide(native_t a, native_t b) { return _mm256_add_epi32(a, b); }

#elif defined(FPX_SIMD_NEON)

using native_t = uint32x4_t;
inline constexpr int lane = 4;
inline constexpr const char* name = "neon";

inline native_t load(const uint32_t* p) { return vld1q_u32(p); }
inline void store(uint32_t* p, native_t x) { vst1q_u32(p, x); }
inline native_t set1(uint32_t x) { return vdupq_n_u32(x); }
inline native_t zero() { return vdupq_n_u32(0); }
inline native_t add(native_t a, native_t b, uint32_t R) {
  native_t r = vaddq_u32(a, b);
  return vminq_u32(r, vsubq_u32(r, vdupq_n_u32(R)));
}
inline native_t sub(native_t a, native_t b, uint32_t R) {
  // a-b mod R in 3 ops: min(a-b, a-b+R) picks the wrapped branch when a < b.
  native_t y = vsubq_u32(a, b);
  return vminq_u32(y, vaddq_u32(y, vdupq_n_u32(R)));
}
/// Montgomery multiplication of 4 lanes (half-width widening multiplies),
/// full reduction: result in [0, MOD).
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod(native_t a, native_t b) {
  const uint32x4_t vn = vdupq_n_u32(NINV);
  const uint32x4_t vm = vdupq_n_u32(MOD);
  const uint64x2_t t0 = vmull_u32(vget_low_u32(a), vget_low_u32(b));
  const uint64x2_t t1 = vmull_u32(vget_high_u32(a), vget_high_u32(b));
  // Compute all four Montgomery digits independently of the widening
  // products. This removes narrowing shuffles and exposes the two multiply
  // chains to the instruction scheduler. Widening multiply-add folds t+m*Mod
  // into one instruction; unzip selects the high halves in their lane order.
  const native_t m = vmulq_u32(vmulq_u32(a, b), vn);
  const uint64x2_t u0 = vmlal_u32(t0, vget_low_u32(m), vget_low_u32(vm));
  const uint64x2_t u1 = vmlal_u32(t1, vget_high_u32(m), vget_high_u32(vm));
  native_t r = vuzpq_u32(vreinterpretq_u32_u64(u0), vreinterpretq_u32_u64(u1)).val[1];
  return vminq_u32(r, vsubq_u32(r, vdupq_n_u32(MOD)));
}
/// Lazy Montgomery multiplication (MOD < 2^30, inputs < 2*MOD):
/// result in [0, 2*MOD), computed without the final conditional subtraction.
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod_lazy(native_t a, native_t b) {
  const uint32x4_t vn = vdupq_n_u32(NINV);
  const uint32x4_t vm = vdupq_n_u32(MOD);
  const uint64x2_t t0 = vmull_u32(vget_low_u32(a), vget_low_u32(b));
  const uint64x2_t t1 = vmull_u32(vget_high_u32(a), vget_high_u32(b));
  const native_t m = vmulq_u32(vmulq_u32(a, b), vn);
  const uint64x2_t u0 = vmlal_u32(t0, vget_low_u32(m), vget_low_u32(vm));
  const uint64x2_t u1 = vmlal_u32(t1, vget_high_u32(m), vget_high_u32(vm));
  return vuzpq_u32(vreinterpretq_u32_u64(u0), vreinterpretq_u32_u64(u1)).val[1];
}
/// 4-way de-interleave of 4*lane = 16 consecutive values (`ld4` one-shot).
inline void load4(const uint32_t* p, native_t& x0, native_t& x1, native_t& x2, native_t& x3) {
  uint32x4x4_t v = vld4q_u32(p);
  x0 = v.val[0];
  x1 = v.val[1];
  x2 = v.val[2];
  x3 = v.val[3];
}
inline void store4(uint32_t* p, native_t y0, native_t y1, native_t y2, native_t y3) {
  uint32x4x4_t v;
  v.val[0] = y0;
  v.val[1] = y1;
  v.val[2] = y2;
  v.val[3] = y3;
  vst4q_u32(p, v);
}
/// Chunk width 2 (two 8-element blocks) via 64-bit half recombination.
template <int M>
inline void load4m(const uint32_t* p, native_t& x0, native_t& x1, native_t& x2, native_t& x3) {
  if constexpr (M == static_cast<int>(lane)) {
    x0 = load(p);
    x1 = load(p + M);
    x2 = load(p + 2 * M);
    x3 = load(p + 3 * M);
  } else if constexpr (M == 1) {
    load4(p, x0, x1, x2, x3);
  } else {
    static_assert(M == 2, "unsupported small chunk width");
    const uint32x4_t u = load(p + 0), v = load(p + 4), w = load(p + 8), z = load(p + 12);
    x0 = vcombine_u32(vget_low_u32(u), vget_low_u32(w));
    x1 = vcombine_u32(vget_high_u32(u), vget_high_u32(w));
    x2 = vcombine_u32(vget_low_u32(v), vget_low_u32(z));
    x3 = vcombine_u32(vget_high_u32(v), vget_high_u32(z));
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
  } else {
    store(p + 0, vcombine_u32(vget_low_u32(y0), vget_low_u32(y1)));
    store(p + 4, vcombine_u32(vget_low_u32(y2), vget_low_u32(y3)));
    store(p + 8, vcombine_u32(vget_high_u32(y0), vget_high_u32(y1)));
    store(p + 12, vcombine_u32(vget_high_u32(y2), vget_high_u32(y3)));
  }
}
/// Widest chunk width with a hand-written kernel (must be < lane).
inline constexpr int small_m_max = 2;

/// Swap the two 32-bit halves of every 64-bit group (pairs (2i, 2i+1)).
inline native_t swap_pairs(native_t v) { return vrev64q_u32(v); }
/// Even lanes from `a`, odd lanes from `b`.
inline native_t pick_odd(native_t a, native_t b) {
  const uint32x4_t m = vreinterpretq_u32_u64(vdupq_n_u64(0xFFFFFFFF00000000ull));
  return vbslq_u32(m, b, a);
}
/// Canonicalise a lazy value (< 2*MOD) into [0, MOD).
inline native_t reduce_full(native_t v, uint32_t MOD) {
  return vminq_u32(v, vsubq_u32(v, vdupq_n_u32(MOD)));
}
/// Unreduced sum of two lazy values (< 2*MOD each); see the AVX-512 note above.
inline native_t add_wide(native_t a, native_t b) { return vaddq_u32(a, b); }

#else  // ---------------------------------------------------------------- scalar

using native_t = uint32_t;
inline constexpr int lane = 1;
inline constexpr const char* name = "scalar";

inline native_t load(const uint32_t* p) { return *p; }
inline void store(uint32_t* p, native_t x) { *p = x; }
inline native_t set1(uint32_t x) { return x; }
inline native_t zero() { return 0; }
inline native_t add(native_t a, native_t b, uint32_t R) {
  uint32_t s = a + b;
  return s >= R ? s - R : s;
}
inline native_t sub(native_t a, native_t b, uint32_t R) {
  return a >= b ? a - b : a - b + R;
}
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod(native_t a, native_t b) {
  uint64_t t = (uint64_t)a * b;
  uint32_t m = (uint32_t)t * NINV;
  uint32_t r = (uint32_t)((t + (uint64_t)m * MOD) >> 32);
  return r >= MOD ? r - MOD : r;
}
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod_lazy(native_t a, native_t b) {
  uint64_t t = (uint64_t)a * b;
  uint32_t m = (uint32_t)t * NINV;
  return (uint32_t)((t + (uint64_t)m * MOD) >> 32);
}
/// Scalar variants also serve the terminal radix-4 unity-twiddle kernel.
inline void load4(const uint32_t* p, native_t& x0, native_t& x1, native_t& x2, native_t& x3) {
  x0 = p[0];
  x1 = p[1];
  x2 = p[2];
  x3 = p[3];
}
inline void store4(uint32_t* p, native_t y0, native_t y1, native_t y2, native_t y3) {
  p[0] = y0;
  p[1] = y1;
  p[2] = y2;
  p[3] = y3;
}
inline native_t swap_pairs(native_t v) { return v; }
template <int M>
inline void load4m(const uint32_t* p, native_t& x0, native_t& x1, native_t& x2, native_t& x3) {
  x0 = p[0];
  x1 = p[M];
  x2 = p[2 * M];
  x3 = p[3 * M];
}
template <int M>
inline void store4m(uint32_t* p, native_t y0, native_t y1, native_t y2, native_t y3) {
  p[0] = y0;
  p[M] = y1;
  p[2 * M] = y2;
  p[3 * M] = y3;
}
/// Widest chunk width with a hand-written kernel (must be < lane); none.
inline constexpr int small_m_max = 0;
inline native_t pick_odd(native_t a, native_t) { return a; }
inline native_t reduce_full(native_t v, uint32_t MOD) { return v >= MOD ? v - MOD : v; }
inline native_t add_wide(native_t a, native_t b) { return a + b; }

#endif

/// Montgomery multiply for the NTT butterflies: picks the lazy kernel whenever
/// the modulus allows it, otherwise falls back to the fully reduced one.
template <uint32_t MOD, uint32_t NINV, bool LAZY>
inline native_t butterfly_mul(native_t a, native_t b) {
  if (LAZY) {
    return mulmod_lazy<MOD, NINV>(a, b);
  } else {
    return mulmod<MOD, NINV>(a, b);
  }
}

#if defined(FPX_HAVE_AVX2_INTRIN)
/// A width-2 NTT stage has unity twiddles in every even lane. Preserve those
/// lanes and run one widening multiply chain for the odd lanes only.
template <uint32_t MOD, uint32_t NINV>
inline native_t mul_unity_even(native_t a, native_t twiddle) {
#if defined(FPX_SIMD_AVX512)
  const native_t odd = _mm512_srli_epi64(a, 32), w = _mm512_srli_epi64(twiddle, 32);
  const native_t t = _mm512_mul_epu32(odd, w);
  const native_t m = _mm512_mul_epu32(odd, _mm512_mullo_epi32(w, set1(NINV)));
  const native_t u = _mm512_add_epi64(t, _mm512_mul_epu32(m, set1(MOD)));
  const native_t r = _mm512_mask_blend_epi32(0xAAAAu, a, u);
#else
  const native_t odd = _mm256_srli_epi64(a, 32), w = _mm256_srli_epi64(twiddle, 32);
  const native_t t = _mm256_mul_epu32(odd, w);
  const native_t m = _mm256_mul_epu32(odd, _mm256_mullo_epi32(w, set1(NINV)));
  const native_t u = _mm256_add_epi64(t, _mm256_mul_epu32(m, set1(MOD)));
  const native_t r = _mm256_blend_epi32(a, u, 0xAA);
#endif
  if constexpr (lazy_ok<MOD>) return r;
  else return reduce_full(r, MOD);
}
#endif

// NEON tables store compact Q31 quotients. Fixed x86 multipliers use Q32;
// large x86 stages keep Montgomery tables to limit twiddle traffic.
template <uint32_t MOD>
inline constexpr bool compact_twiddle =
#if defined(FPX_SIMD_NEON)
    lazy_ok<MOD>;
#else
    false;
#endif

template <uint32_t MOD>
inline constexpr bool fixed_lazy =
#if defined(FPX_HAVE_AVX2_INTRIN)
    lazy_ok<MOD>;
#else
    compact_twiddle<MOD>;
#endif

struct Twiddle { native_t value, quotient; };
struct FixedTwiddle : Twiddle {};

/// Encode an already canonical Montgomery limb as a Q31 quotient. ROUND
/// selects nearest (signed butterfly differences) or floor (positive inputs).
/// For t < MOD, m = t*NINV mod 2^32 and w = (t+m*MOD)/2^32 < MOD:
///   w*2^31/MOD = m/2 + t/(2*MOD).
/// Thus floor is m>>1 and nearest is ceil(m/2), without a division or REDC.
template <uint32_t MOD, uint32_t NINV, bool ROUND>
constexpr uint32_t encode_twiddle_scalar(uint32_t t) {
  if constexpr (compact_twiddle<MOD>) {
    const uint32_t m = t * NINV;
    return ROUND ? m - (m >> 1) : m >> 1;
  } else {
    return t;
  }
}

template <uint32_t MOD, uint32_t NINV, bool ROUND>
inline native_t encode_twiddle(native_t t) {
#if defined(FPX_SIMD_NEON)
  if constexpr (compact_twiddle<MOD>) {
    const native_t m = vmulq_u32(t, set1(NINV));
    if constexpr (ROUND) return vhaddq_u32(m, set1(1));
    else return vshrq_n_u32(m, 1);
  } else
#endif
  return t;
}

/// Reconstruct w exactly from either Q31 encoding. Its error after scaling
/// back is < MOD/2^31 < 1/2, so one rounded high multiply recovers w.
template <uint32_t MOD>
inline Twiddle decode_twiddle(native_t q) {
#if defined(FPX_SIMD_NEON)
  if constexpr (compact_twiddle<MOD>) {
    return {vreinterpretq_u32_s32(vqrdmulhq_s32(
                vreinterpretq_s32_u32(q), vdupq_n_s32(static_cast<int32_t>(MOD)))), q};
  } else
#endif
  return {q, zero()};
}

template <uint32_t MOD, uint32_t NINV, bool ROUND = false>
inline FixedTwiddle fixed_twiddle(uint32_t mont) {
  const uint32_t q = encode_twiddle_scalar<MOD, NINV, ROUND>(mont);
  if constexpr (compact_twiddle<MOD>) {
    const uint32_t w = static_cast<uint32_t>(
        (uint64_t(q) * MOD + (uint64_t(1) << 30)) >> 31);
    return {{set1(w), set1(q)}};
  } else {
#if defined(FPX_HAVE_AVX2_INTRIN)
    const uint32_t digit = mont*NINV;
    const uint32_t value = static_cast<uint32_t>((uint64_t(digit)*MOD + mont) >> 32);
    // value*2^32/MOD = digit + mont/MOD, and mont is canonical.
    const uint32_t quotient = digit;
    return {{set1(value), set1(quotient)}};
#else
    return {{set1(q), zero()}};
#endif
  }
}

#if defined(FPX_HAVE_AVX2_INTRIN)
template <uint32_t MOD, uint32_t NINV>
inline FixedTwiddle fixed_twiddle_vector(native_t mont) {
#if defined(FPX_SIMD_AVX512)
  const native_t q = _mm512_mullo_epi32(mont, set1(NINV));
  const native_t lo = _mm512_and_si512(mont, _mm512_set1_epi64(0xffffffffu));
  const native_t hi = _mm512_srli_epi64(mont, 32);
  const native_t v0 = _mm512_add_epi64(_mm512_mul_epu32(q, set1(MOD)), lo);
  const native_t v1 = _mm512_add_epi64(_mm512_mul_epu32(_mm512_srli_epi64(q, 32), set1(MOD)), hi);
  return {{_mm512_mask_blend_epi32(0xAAAAu, _mm512_srli_epi64(v0, 32), v1), q}};
#else
  const native_t q = _mm256_mullo_epi32(mont, set1(NINV));
  const native_t lo = _mm256_and_si256(mont, _mm256_set1_epi64x(0xffffffffu));
  const native_t hi = _mm256_srli_epi64(mont, 32);
  const native_t v0 = _mm256_add_epi64(_mm256_mul_epu32(q, set1(MOD)), lo);
  const native_t v1 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(q, 32), set1(MOD)), hi);
  return {{_mm256_blend_epi32(_mm256_srli_epi64(v0, 32), v1, 0xAA), q}};
#endif
}
#endif

/// Positive fixed multiplication: a < 2*MOD and a floor Q31 quotient.
/// floor(a*q/2^31) underestimates a*w/MOD by < 2. The difference is in
/// [0, 2*MOD); SQDMULH + MUL + MLS handles four lanes in three instructions.
template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle(native_t a, Twiddle w) {
#if defined(FPX_SIMD_NEON)
  if constexpr (compact_twiddle<MOD>) {
    const native_t q = vreinterpretq_u32_s32(vqdmulhq_s32(
        vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(w.quotient)));
    return vmlsq_u32(vmulq_u32(a, w.value), q, set1(MOD));
  } else
#endif
  return butterfly_mul<MOD, NINV, lazy_ok<MOD>>(a, w.value);
}

template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_full(native_t a, Twiddle w) {
  if constexpr (compact_twiddle<MOD>)
    return reduce_full(mul_twiddle<MOD, NINV>(a, w), MOD);
  else
    return mulmod<MOD, NINV>(a, w.value);
}

#if defined(FPX_SIMD_NEON)
/// Signed x in [-2*MOD, 2*MOD) and a nearest Q31 quotient. The reciprocal
/// error is < |x|/2^32 < 1/2; rounding adds at most 1/2. Hence x*w-Q*MOD
/// lies in (-MOD, MOD). Adding MOD restores the lazy positive interval.
template <uint32_t MOD>
inline native_t mul_twiddle_centered(native_t x, Twiddle w) {
  const native_t q = vreinterpretq_u32_s32(vqrdmulhq_s32(
      vreinterpretq_s32_u32(x), vreinterpretq_s32_u32(w.quotient)));
  const native_t r = vmlsq_u32(vmulq_u32(x, w.value), q, set1(MOD));
  return vaddq_u32(r, set1(MOD));
}
#endif

template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_diff(native_t a, native_t b, Twiddle w) {
#if defined(FPX_SIMD_NEON)
  if constexpr (compact_twiddle<MOD>)
    return mul_twiddle_centered<MOD>(vsubq_u32(a, b), w);
  else
#endif
    return butterfly_mul<MOD, NINV, lazy_ok<MOD>>(sub(a, b, rmod<MOD>), w.value);
}

template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_sum(native_t a, native_t b, Twiddle w) {
#if defined(FPX_SIMD_NEON)
  if constexpr (compact_twiddle<MOD>)
    return mul_twiddle_centered<MOD>(vsubq_u32(vaddq_u32(a, b), set1(2 * MOD)), w);
  else
#endif
  if constexpr (lazy_ok<MOD>)
    return mulmod_lazy<MOD, NINV>(add_wide(a, b), w.value);
  else
    return mulmod<MOD, NINV>(add(a, b, MOD), w.value);
}

// A fixed ordinary multiplier uses a floor Q32 quotient. For a < 2^32 the
// estimated quotient undershoots by less than two, so the remainder is < 2*MOD.
// Its two wide high products run in parallel with the all-lane low product.
template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle(native_t a, FixedTwiddle w) {
#if defined(FPX_SIMD_AVX512)
  const native_t q0 = _mm512_mul_epu32(a, w.quotient);
  const native_t q1 = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(w.quotient, 32));
  const native_t q = _mm512_mask_blend_epi32(0xAAAAu, _mm512_srli_epi64(q0, 32), q1);
  const native_t r = _mm512_sub_epi32(_mm512_mullo_epi32(a, w.value), _mm512_mullo_epi32(q, set1(MOD)));
#elif defined(FPX_SIMD_AVX2)
  const native_t q0 = _mm256_mul_epu32(a, w.quotient);
  const native_t q1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(w.quotient, 32));
  const native_t q = _mm256_blend_epi32(_mm256_srli_epi64(q0, 32), q1, 0xAA);
  const native_t r = _mm256_sub_epi32(_mm256_mullo_epi32(a, w.value), _mm256_mullo_epi32(q, set1(MOD)));
#else
  return mul_twiddle<MOD, NINV>(a, static_cast<Twiddle>(w));
#endif
#if defined(FPX_HAVE_AVX2_INTRIN)
  if constexpr (lazy_ok<MOD>) return r;
  else return reduce_full(r, MOD);
#endif
}
template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_full(native_t a, FixedTwiddle w) {
  if constexpr (lazy_ok<MOD>) return reduce_full(mul_twiddle<MOD, NINV>(a, w), MOD);
  else return mul_twiddle<MOD, NINV>(a, w);
}
template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_diff(native_t a, native_t b, FixedTwiddle w) {
#if defined(FPX_HAVE_AVX2_INTRIN)
  return mul_twiddle<MOD, NINV>(sub(a, b, rmod<MOD>), w);
#else
  return mul_twiddle_diff<MOD, NINV>(a, b, static_cast<Twiddle>(w));
#endif
}
template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_sum(native_t a, native_t b, FixedTwiddle w) {
#if defined(FPX_HAVE_AVX2_INTRIN)
  if constexpr (lazy_ok<MOD>) return mul_twiddle<MOD, NINV>(add_wide(a, b), w);
  else return mul_twiddle<MOD, NINV>(add(a, b, MOD), w);
#else
  return mul_twiddle_sum<MOD, NINV>(a, b, static_cast<Twiddle>(w));
#endif
}

}  // namespace simd
}  // namespace fpx

#endif  // FASTPOLY_SIMD_HPP
