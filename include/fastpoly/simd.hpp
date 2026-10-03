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
/// lane i of `xj` receives p[4*i + j] for i = 0..7.  The row order inside a
/// column is a fixed permutation (0,2,1,3,4,6,5,7); store4x8 undoes it exactly.
/// Shared by the AVX2 and AVX-512 backends, the latter composing two halves.
inline void load4x8(const uint32_t* p, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3) {
  const __m128i a0 = _mm_loadu_si128((const __m128i*)(p + 0));
  const __m128i a1 = _mm_loadu_si128((const __m128i*)(p + 4));
  const __m128i a2 = _mm_loadu_si128((const __m128i*)(p + 8));
  const __m128i a3 = _mm_loadu_si128((const __m128i*)(p + 12));
  const __m128i b0 = _mm_loadu_si128((const __m128i*)(p + 16));
  const __m128i b1 = _mm_loadu_si128((const __m128i*)(p + 20));
  const __m128i b2 = _mm_loadu_si128((const __m128i*)(p + 24));
  const __m128i b3 = _mm_loadu_si128((const __m128i*)(p + 28));
  const __m128i t0 = _mm_unpacklo_epi32(a0, a1), t1 = _mm_unpackhi_epi32(a0, a1);
  const __m128i t2 = _mm_unpacklo_epi32(a2, a3), t3 = _mm_unpackhi_epi32(a2, a3);
  const __m128i u0 = _mm_unpacklo_epi32(b0, b1), u1 = _mm_unpackhi_epi32(b0, b1);
  const __m128i u2 = _mm_unpacklo_epi32(b2, b3), u3 = _mm_unpackhi_epi32(b2, b3);
  x0 = _mm256_set_m128i(_mm_unpacklo_epi32(u0, u2), _mm_unpacklo_epi32(t0, t2));
  x1 = _mm256_set_m128i(_mm_unpackhi_epi32(u0, u2), _mm_unpackhi_epi32(t0, t2));
  x2 = _mm256_set_m128i(_mm_unpacklo_epi32(u1, u3), _mm_unpacklo_epi32(t1, t3));
  x3 = _mm256_set_m128i(_mm_unpackhi_epi32(u1, u3), _mm_unpackhi_epi32(t1, t3));
}
/// Exact inverse of load4x8.
inline void store4x8(uint32_t* p, __m256i y0, __m256i y1, __m256i y2, __m256i y3) {
  __m128i c0 = _mm256_castsi256_si128(y0), d0 = _mm256_extracti128_si256(y0, 1);
  __m128i c1 = _mm256_castsi256_si128(y1), d1 = _mm256_extracti128_si256(y1, 1);
  __m128i c2 = _mm256_castsi256_si128(y2), d2 = _mm256_extracti128_si256(y2, 1);
  __m128i c3 = _mm256_castsi256_si128(y3), d3 = _mm256_extracti128_si256(y3, 1);
  const __m128i w0 = _mm_unpacklo_epi32(c0, c1), w1 = _mm_unpackhi_epi32(c0, c1);
  const __m128i w2 = _mm_unpacklo_epi32(c2, c3), w3 = _mm_unpackhi_epi32(c2, c3);
  _mm_storeu_si128((__m128i*)(p + 0), _mm_unpacklo_epi64(w0, w2));
  _mm_storeu_si128((__m128i*)(p + 4), _mm_unpacklo_epi64(w1, w3));
  _mm_storeu_si128((__m128i*)(p + 8), _mm_unpackhi_epi64(w0, w2));
  _mm_storeu_si128((__m128i*)(p + 12), _mm_unpackhi_epi64(w1, w3));
  const __m128i z0 = _mm_unpacklo_epi32(d0, d1), z1 = _mm_unpackhi_epi32(d0, d1);
  const __m128i z2 = _mm_unpacklo_epi32(d2, d3), z3 = _mm_unpackhi_epi32(d2, d3);
  _mm_storeu_si128((__m128i*)(p + 16), _mm_unpacklo_epi64(z0, z2));
  _mm_storeu_si128((__m128i*)(p + 20), _mm_unpacklo_epi64(z1, z3));
  _mm_storeu_si128((__m128i*)(p + 24), _mm_unpackhi_epi64(z0, z2));
  _mm_storeu_si128((__m128i*)(p + 28), _mm_unpackhi_epi64(z1, z3));
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
/// 4-way de-interleave of 4*lane consecutive values (lane == 16 here); built
/// from two AVX2-width transposes so it shares the tested 256-bit network.
inline void load4(const uint32_t* p, native_t& x0, native_t& x1, native_t& x2, native_t& x3) {
  __m256i l0, l1, l2, l3, h0, h1, h2, h3;
  x86::load4x8(p, l0, l1, l2, l3);
  x86::load4x8(p + 32, h0, h1, h2, h3);
  x0 = _mm512_inserti64x4(_mm512_castsi256_si512(l0), h0, 1);
  x1 = _mm512_inserti64x4(_mm512_castsi256_si512(l1), h1, 1);
  x2 = _mm512_inserti64x4(_mm512_castsi256_si512(l2), h2, 1);
  x3 = _mm512_inserti64x4(_mm512_castsi256_si512(l3), h3, 1);
}
inline void store4(uint32_t* p, native_t y0, native_t y1, native_t y2, native_t y3) {
  x86::store4x8(p, _mm512_castsi512_si256(y0), _mm512_castsi512_si256(y1),
                _mm512_castsi512_si256(y2), _mm512_castsi512_si256(y3));
  x86::store4x8(p + 32, _mm512_extracti64x4_epi64(y0, 1), _mm512_extracti64x4_epi64(y1, 1),
                _mm512_extracti64x4_epi64(y2, 1), _mm512_extracti64x4_epi64(y3, 1));
}
/// Chunk width for the small-m stages: M < lane, M | lane.  Composed from two
/// AVX2-width transposes, mirroring load4/store4.
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
    static_assert(M == 2 || M == 4, "unsupported small chunk width");
    __m256i l0, l1, l2, l3, h0, h1, h2, h3;
    if constexpr (M == 2) {
      x86::load4m2x8(p, l0, l1, l2, l3);
      x86::load4m2x8(p + 32, h0, h1, h2, h3);
    } else {
      x86::load4m4x8(p, l0, l1, l2, l3);
      x86::load4m4x8(p + 32, h0, h1, h2, h3);
    }
    x0 = _mm512_inserti64x4(_mm512_castsi256_si512(l0), h0, 1);
    x1 = _mm512_inserti64x4(_mm512_castsi256_si512(l1), h1, 1);
    x2 = _mm512_inserti64x4(_mm512_castsi256_si512(l2), h2, 1);
    x3 = _mm512_inserti64x4(_mm512_castsi256_si512(l3), h3, 1);
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
    static_assert(M == 2 || M == 4, "unsupported small chunk width");
    if constexpr (M == 2) {
      x86::store4m2x8(p, _mm512_castsi512_si256(y0), _mm512_castsi512_si256(y1),
                      _mm512_castsi512_si256(y2), _mm512_castsi512_si256(y3));
      x86::store4m2x8(p + 32, _mm512_extracti64x4_epi64(y0, 1), _mm512_extracti64x4_epi64(y1, 1),
                      _mm512_extracti64x4_epi64(y2, 1), _mm512_extracti64x4_epi64(y3, 1));
    } else {
      x86::store4m4x8(p, _mm512_castsi512_si256(y0), _mm512_castsi512_si256(y1),
                      _mm512_castsi512_si256(y2), _mm512_castsi512_si256(y3));
      x86::store4m4x8(p + 32, _mm512_extracti64x4_epi64(y0, 1), _mm512_extracti64x4_epi64(y1, 1),
                      _mm512_extracti64x4_epi64(y2, 1), _mm512_extracti64x4_epi64(y3, 1));
    }
  }
}
/// Widest chunk width with a hand-written kernel (must be < lane).
inline constexpr int small_m_max = 4;

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
  const uint32x2_t vn = vdup_n_u32(NINV);
  const uint32x2_t vm = vdup_n_u32(MOD);
  uint64x2_t t0 = vmull_u32(vget_low_u32(a), vget_low_u32(b));
  uint64x2_t t1 = vmull_u32(vget_high_u32(a), vget_high_u32(b));
  // Only the low 32 bits of the product feed the Montgomery digit, so a plain
  // 32-bit multiply is enough (one op less than widen + truncate).
  uint32x2_t m0 = vmul_u32(vmovn_u64(t0), vn);
  uint32x2_t m1 = vmul_u32(vmovn_u64(t1), vn);
  uint32x2_t u0 = vshrn_n_u64(vaddq_u64(t0, vmull_u32(m0, vm)), 32);
  uint32x2_t u1 = vshrn_n_u64(vaddq_u64(t1, vmull_u32(m1, vm)), 32);
  native_t r = vcombine_u32(u0, u1);
  return vminq_u32(r, vsubq_u32(r, vdupq_n_u32(MOD)));
}
/// Lazy Montgomery multiplication (MOD < 2^30, inputs < 2*MOD):
/// result in [0, 2*MOD), computed without the final conditional subtraction.
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod_lazy(native_t a, native_t b) {
  const uint32x2_t vn = vdup_n_u32(NINV);
  const uint32x2_t vm = vdup_n_u32(MOD);
  uint64x2_t t0 = vmull_u32(vget_low_u32(a), vget_low_u32(b));
  uint64x2_t t1 = vmull_u32(vget_high_u32(a), vget_high_u32(b));
  uint32x2_t m0 = vmul_u32(vmovn_u64(t0), vn);
  uint32x2_t m1 = vmul_u32(vmovn_u64(t1), vn);
  uint32x2_t u0 = vshrn_n_u64(vaddq_u64(t0, vmull_u32(m0, vm)), 32);
  uint32x2_t u1 = vshrn_n_u64(vaddq_u64(t1, vmull_u32(m1, vm)), 32);
  return vcombine_u32(u0, u1);
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
/// Trivial scalar variants; the len==4 kernel is only used for lane > 1.
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

}  // namespace simd
}  // namespace fpx

#endif  // FASTPOLY_SIMD_HPP
