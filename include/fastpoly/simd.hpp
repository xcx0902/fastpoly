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
#ifndef FASTPOLY_SIMD_HPP
#define FASTPOLY_SIMD_HPP

#include <cstdint>
#include <cstddef>

#if defined(__AVX512F__) && defined(__AVX512VL__)
#define FPX_SIMD_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define FPX_SIMD_AVX2 1
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#define FPX_SIMD_NEON 1
#include <arm_neon.h>
#endif

namespace fpx {
namespace simd {

#if defined(FPX_SIMD_AVX512)

using native_t = __m512i;
inline constexpr int lane = 16;
inline constexpr const char* name = "avx512";

inline native_t load(const uint32_t* p) { return _mm512_loadu_si512((const void*)p); }
inline void store(uint32_t* p, native_t x) { _mm512_storeu_si512((void*)p, x); }
inline native_t set1(uint32_t x) { return _mm512_set1_epi32((int)x); }
inline native_t zero() { return _mm512_setzero_si512(); }
inline native_t add(native_t a, native_t b, uint32_t MOD) {
  native_t r = _mm512_add_epi32(a, b);
  // r < 2*MOD < 2^32; min(r, r-MOD) folds it back (unsigned).
  return _mm512_min_epu32(r, _mm512_sub_epi32(r, _mm512_set1_epi32((int)MOD)));
}
inline native_t sub(native_t a, native_t b, uint32_t MOD) {
  native_t r = _mm512_sub_epi32(a, b);
  // borrow mask from the sign bit of the (wrapped) difference
  native_t m = _mm512_srai_epi32(r, 31);
  return _mm512_add_epi32(r, _mm512_and_si512(m, _mm512_set1_epi32((int)MOD)));
}
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
  __mmask16 ge = _mm512_cmpgt_epi32_mask(r, _mm512_set1_epi32((int)MOD - 1));
  return _mm512_mask_sub_epi32(r, ge, r, vm);
}

#elif defined(FPX_SIMD_AVX2)

using native_t = __m256i;
inline constexpr int lane = 8;
inline constexpr const char* name = "avx2";

inline native_t load(const uint32_t* p) { return _mm256_loadu_si256((const __m256i*)p); }
inline void store(uint32_t* p, native_t x) { _mm256_storeu_si256((__m256i*)p, x); }
inline native_t set1(uint32_t x) { return _mm256_set1_epi32((int)x); }
inline native_t zero() { return _mm256_setzero_si256(); }
inline native_t add(native_t a, native_t b, uint32_t MOD) {
  native_t r = _mm256_add_epi32(a, b);
  return _mm256_min_epu32(r, _mm256_sub_epi32(r, _mm256_set1_epi32((int)MOD)));
}
inline native_t sub(native_t a, native_t b, uint32_t MOD) {
  native_t r = _mm256_sub_epi32(a, b);
  // arithmetic shift of the sign bit -> 0xffffffff iff a < b
  native_t m = _mm256_srai_epi32(r, 31);
  return _mm256_add_epi32(r, _mm256_and_si256(m, _mm256_set1_epi32((int)MOD)));
}
/// Montgomery multiplication of 8 lanes (even/odd halves via mul_epu32).
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
  __m256i ge = _mm256_cmpgt_epi32(r, _mm256_set1_epi32((int)MOD - 1));
  return _mm256_sub_epi32(r, _mm256_and_si256(ge, vm));
}

#elif defined(FPX_SIMD_NEON)

using native_t = uint32x4_t;
inline constexpr int lane = 4;
inline constexpr const char* name = "neon";

inline native_t load(const uint32_t* p) { return vld1q_u32(p); }
inline void store(uint32_t* p, native_t x) { vst1q_u32(p, x); }
inline native_t set1(uint32_t x) { return vdupq_n_u32(x); }
inline native_t zero() { return vdupq_n_u32(0); }
inline native_t add(native_t a, native_t b, uint32_t MOD) {
  native_t r = vaddq_u32(a, b);
  return vminq_u32(r, vsubq_u32(r, vdupq_n_u32(MOD)));
}
inline native_t sub(native_t a, native_t b, uint32_t MOD) {
  native_t r = vsubq_u32(a, b);
  return vaddq_u32(r, vandq_u32(vcltq_u32(a, b), vdupq_n_u32(MOD)));
}
/// Montgomery multiplication of 4 lanes (half-width widening multiplies).
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod(native_t a, native_t b) {
  const uint32x2_t vn = vdup_n_u32(NINV);
  const uint32x2_t vm = vdup_n_u32(MOD);
  uint64x2_t t0 = vmull_u32(vget_low_u32(a), vget_low_u32(b));
  uint64x2_t t1 = vmull_u32(vget_high_u32(a), vget_high_u32(b));
  uint32x2_t m0 = vmovn_u64(vmull_u32(vmovn_u64(t0), vn));
  uint32x2_t m1 = vmovn_u64(vmull_u32(vmovn_u64(t1), vn));
  uint32x2_t u0 = vshrn_n_u64(vaddq_u64(t0, vmull_u32(m0, vm)), 32);
  uint32x2_t u1 = vshrn_n_u64(vaddq_u64(t1, vmull_u32(m1, vm)), 32);
  native_t r = vcombine_u32(u0, u1);
  return vsubq_u32(r, vandq_u32(vcgeq_u32(r, vdupq_n_u32(MOD)), vdupq_n_u32(MOD)));
}

#else  // ---------------------------------------------------------------- scalar

using native_t = uint32_t;
inline constexpr int lane = 1;
inline constexpr const char* name = "scalar";

inline native_t load(const uint32_t* p) { return *p; }
inline void store(uint32_t* p, native_t x) { *p = x; }
inline native_t set1(uint32_t x) { return x; }
inline native_t zero() { return 0; }
inline native_t add(native_t a, native_t b, uint32_t MOD) {
  uint32_t s = a + b;
  return s >= MOD ? s - MOD : s;
}
inline native_t sub(native_t a, native_t b, uint32_t MOD) {
  return a >= b ? a - b : a - b + MOD;
}
template <uint32_t MOD, uint32_t NINV>
inline native_t mulmod(native_t a, native_t b) {
  uint64_t t = (uint64_t)a * b;
  uint32_t m = (uint32_t)t * NINV;
  uint32_t r = (uint32_t)((t + (uint64_t)m * MOD) >> 32);
  return r >= MOD ? r - MOD : r;
}

#endif

}  // namespace simd
}  // namespace fpx

#endif  // FASTPOLY_SIMD_HPP
