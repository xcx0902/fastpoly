// fastpoly - single-header, include-and-use (C++20, header only).
//
//   fpx::Mont<Mod,Root>       Montgomery field element over a 32-bit NTT prime
//   fpx::mod998244353, ...    ready-made prime types (-1 divides a large 2^k)
//   fpx::NttPlan<M>::get(n)   cached, permutation-free, SIMD radix-4 NTT
//   fpx::Poly<M>              polynomial over GF(Mod), truncated mod x^n
//   fpx::poly::*              free-function API: conv/inv/log/exp/sqrt/pow/...
//   fpx::simd_backend()       "avx512" | "avx2" | "neon" | "scalar"
//
// x86-64: compile with -mavx2 (-march=native turns on AVX-512 when present).
// AArch64/NEON is baseline and needs no flags.  No external dependencies.
//
// This is a minimised distribution build: every name reachable from user code
// is unchanged, only comments, whitespace and implementation-private names
// (class-private members, locals, parameters) were shortened.
#ifndef FASTPOLY_SINGLE_HPP
#define FASTPOLY_SINGLE_HPP
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
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
namespace fpx {
constexpr uint32_t inv_mod_2_32(uint32_t a) {
  uint32_t x = 1;
  for (int i = 0; i < 5; ++i) x *= 2u - a * x;
  return x;
}
// Montgomery form (a * 2^32 mod p): multiplication is mul + shift + condition,
// never a 64-bit division -- the shape the SIMD butterflies need.
template <uint32_t Mod, uint32_t Root>
class Mont {
  static_assert(Mod < (1u << 31), "modulus must fit a signed 32-bit lane");
  static_assert(Mod % 2 == 1, "modulus must be odd");
 public:
  static constexpr uint32_t mod = Mod;
  static constexpr uint32_t primitive_root = Root;
  static constexpr uint32_t ninv = 0u - inv_mod_2_32(Mod);
  static constexpr uint32_t one = uint32_t((uint64_t(1) << 32) % Mod);
  static constexpr uint32_t r2 = uint32_t(uint64_t(one) * one % Mod);
  static constexpr uint32_t inv2 = (Mod + 1) / 2;
  constexpr Mont() : v(0) {}
  template <class I, class = std::enable_if_t<std::is_integral_v<I> && !std::is_same_v<I, bool>>>
  constexpr Mont(I x) : v(fi(uint64_t(x))) {}
  static constexpr Mont from_int(uint64_t x) { return w(fi(x)); }
  static constexpr Mont raw(uint32_t x) { return w(x); }
  static constexpr uint32_t reduce(uint64_t t) {
    uint32_t m = uint32_t(t) * ninv;
    uint64_t u = (t + uint64_t(m) * Mod) >> 32;
    return uint32_t(u >= Mod ? u - Mod : u);
  }
  constexpr uint32_t val() const { return reduce(v); }
  constexpr uint32_t raw_val() const { return v; }
  friend constexpr Mont operator+(Mont a, Mont b) {
    uint32_t s = a.v + b.v;
    return w(s >= Mod ? s - Mod : s);
  }
  friend constexpr Mont operator-(Mont a, Mont b) {
    return w(a.v >= b.v ? a.v - b.v : a.v - b.v + Mod);
  }
  friend constexpr Mont operator-(Mont a) { return w(a.v == 0 ? 0 : Mod - a.v); }
  friend constexpr Mont operator*(Mont a, Mont b) { return w(reduce(uint64_t(a.v) * b.v)); }
  friend constexpr Mont operator/(Mont a, Mont b) { return a * b.inv(); }
  constexpr Mont& operator+=(Mont b) { return *this = *this + b; }
  constexpr Mont& operator-=(Mont b) { return *this = *this - b; }
  constexpr Mont& operator*=(Mont b) { return *this = *this * b; }
  constexpr Mont& operator/=(Mont b) { return *this = *this / b; }
  friend constexpr bool operator==(Mont a, Mont b) { return a.v == b.v; }
  friend constexpr bool operator!=(Mont a, Mont b) { return a.v != b.v; }
  constexpr bool is_zero() const { return v == 0; }
  constexpr bool operator!() const { return v == 0; }
  constexpr explicit operator bool() const { return v != 0; }
  constexpr Mont pow(uint64_t e) const {
    Mont r = from_int(1), b = *this;
    for (; e; e >>= 1) {
      if (e & 1) r *= b;
      b *= b;
    }
    return r;
  }
  constexpr Mont inv() const { return pow(Mod - 2); }
  bool sqrt(Mont& o) const {
    const Mont u = from_int(1);
    if (v == 0) { o = *this; return true; }
    if (pow((Mod - 1) / 2) != u) return false;
    if (Mod % 4 == 3) {
      o = pow((uint64_t(Mod) + 1) / 4);
      return true;
    }
    uint64_t q = Mod - 1;
    int s = 0;
    while (q % 2 == 0) { q /= 2; ++s; }
    Mont z = from_int(2);
    while (z.pow((Mod - 1) / 2) != -u) z += u;
    Mont c = z.pow(q), x = pow((q + 1) / 2), t = pow(q);
    int m = s;
    while (t != u) {
      int i = 0;
      for (Mont t2 = t; t2 != u; t2 *= t2) ++i;
      Mont b = c.pow(uint64_t(1) << (m - i - 1));
      x *= b;
      c = b * b;
      t *= c;
      m = i;
    }
    o = x;
    return true;
  }
  std::string str() const { return std::to_string(val()); }
 private:
  static constexpr uint32_t fi(uint64_t x) { return uint32_t(x % Mod) * uint64_t(one) % Mod; }
  static constexpr Mont w(uint32_t x) {
    Mont r;
    r.v = x;
    return r;
  }
  uint32_t v;
};
template <uint32_t M, uint32_t R>
std::ostream& operator<<(std::ostream& os, const Mont<M, R>& a) { return os << a.val(); }
template <uint32_t M, uint32_t R>
std::istream& operator>>(std::istream& is, Mont<M, R>& a) {
  uint64_t x = 0;
  is >> x;
  a = Mont<M, R>::from_int(x);
  return is;
}
using mod998244353 = Mont<998244353u, 3>;
using mod1004535809 = Mont<1004535809u, 3>;
using mod469762049 = Mont<469762049u, 3>;
using mod167772161 = Mont<167772161u, 3>;
using mod754974721 = Mont<754974721u, 11>;
using mod1224736769 = Mont<1224736769u, 3>;
// -------------------------------------------------------------------- SIMD ---
// Primitives run on Montgomery residues; `mulmod*` is a Montgomery multiply.
// Two reduction conventions: `mulmod`/`add`/`sub` with R = MOD keep values in
// [0, MOD); `mulmod_lazy` plus `add`/`sub` with R = 2*MOD keep them in [0, 2*MOD)
// and are what the butterflies use.
namespace simd {
#if defined(FPX_HAVE_AVX2_INTRIN)
namespace x86 {
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
inline void store4x8(uint32_t* p, __m256i y0, __m256i y1, __m256i y2, __m256i y3) {
  const __m128i c0 = _mm256_castsi256_si128(y0), d0 = _mm256_extracti128_si256(y0, 1);
  const __m128i c1 = _mm256_castsi256_si128(y1), d1 = _mm256_extracti128_si256(y1, 1);
  const __m128i c2 = _mm256_castsi256_si128(y2), d2 = _mm256_extracti128_si256(y2, 1);
  const __m128i c3 = _mm256_castsi256_si128(y3), d3 = _mm256_extracti128_si256(y3, 1);
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
  _mm256_storeu_si256((__m256i*)(p + 0),
                      _mm256_set_m128i(_mm256_castsi256_si128(y1), _mm256_castsi256_si128(y0)));
  _mm256_storeu_si256((__m256i*)(p + 8),
                      _mm256_set_m128i(_mm256_castsi256_si128(y3), _mm256_castsi256_si128(y2)));
  _mm256_storeu_si256((__m256i*)(p + 16), _mm256_set_m128i(_mm256_extracti128_si256(y1, 1),
                                                           _mm256_extracti128_si256(y0, 1)));
  _mm256_storeu_si256((__m256i*)(p + 24), _mm256_set_m128i(_mm256_extracti128_si256(y3, 1),
                                                           _mm256_extracti128_si256(y2, 1)));
}
}
#endif
// Lazy convention: MOD < 2^30 makes 4*MOD^2 < MOD*2^32, so a Montgomery product
// of two values < 2*MOD lands back in [0, 2*MOD) with no final conditional
// subtraction.  R is a multiple of MOD, so the residues stay valid; both NTT
// entry points canonicalise, keeping the public contract ([0, MOD)) intact.
template <uint32_t MOD>
inline constexpr bool lazy_ok = MOD < (1u << 30);
template <uint32_t MOD>
inline constexpr uint32_t rmod = lazy_ok<MOD> ? (2u * MOD) : MOD;
#if defined(FPX_SIMD_AVX512)
using native_t = __m512i;
using V = native_t;
inline constexpr int lane = 16;
inline constexpr const char* name = "avx512";
inline V load(const uint32_t* p) { return _mm512_loadu_si512((const void*)p); }
inline void store(uint32_t* p, V x) { _mm512_storeu_si512((void*)p, x); }
inline V set1(uint32_t x) { return _mm512_set1_epi32((int)x); }
inline V zero() { return _mm512_setzero_si512(); }
inline V reduce_full(V v, uint32_t MOD) {
  return _mm512_min_epu32(v, _mm512_sub_epi32(v, _mm512_set1_epi32((int)MOD)));
}
// add/sub in 3 ops each, no compare mask and no arithmetic shift: r < 2*R, so
// the unsigned min(r, r-R) is the conditional subtract; for sub, a < b wraps
// a-b to a huge value and min picks the other branch, a-b+R.
inline V add(V a, V b, uint32_t R) {
  V r = _mm512_add_epi32(a, b);
  return _mm512_min_epu32(r, _mm512_sub_epi32(r, _mm512_set1_epi32((int)R)));
}
inline V sub(V a, V b, uint32_t R) {
  V y = _mm512_sub_epi32(a, b);
  return _mm512_min_epu32(y, _mm512_add_epi32(y, _mm512_set1_epi32((int)R)));
}
// Lazy Montgomery multiply: inputs < 2*MOD -> result in [0, 2*MOD).
template <uint32_t MOD, uint32_t NINV>
inline V mulmod_lazy(V a, V b) {
  const V vn = _mm512_set1_epi32((int)NINV);
  const V vm = _mm512_set1_epi32((int)MOD);
  __m512i t0 = _mm512_mul_epu32(a, b);
  __m512i t1 = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(b, 32));
  __m512i m0 = _mm512_mul_epu32(t0, vn);
  __m512i m1 = _mm512_mul_epu32(t1, vn);
  __m512i u0 = _mm512_srli_epi64(_mm512_add_epi64(t0, _mm512_mul_epu32(m0, vm)), 32);
  __m512i u1 = _mm512_srli_epi64(_mm512_add_epi64(t1, _mm512_mul_epu32(m1, vm)), 32);
  return _mm512_or_si512(u0, _mm512_slli_epi64(u1, 32));
}
// Full reduction, valid for any input < 2^31 (REDC output is always < 2*MOD).
template <uint32_t MOD, uint32_t NINV>
inline V mulmod(V a, V b) {
  return reduce_full(mulmod_lazy<MOD, NINV>(a, b), MOD);
}
inline void load4(const uint32_t* p, V& x0, V& x1, V& x2, V& x3) {
  __m256i l0, l1, l2, l3, h0, h1, h2, h3;
  x86::load4x8(p, l0, l1, l2, l3);
  x86::load4x8(p + 32, h0, h1, h2, h3);
  x0 = _mm512_inserti64x4(_mm512_castsi256_si512(l0), h0, 1);
  x1 = _mm512_inserti64x4(_mm512_castsi256_si512(l1), h1, 1);
  x2 = _mm512_inserti64x4(_mm512_castsi256_si512(l2), h2, 1);
  x3 = _mm512_inserti64x4(_mm512_castsi256_si512(l3), h3, 1);
}
inline void store4(uint32_t* p, V y0, V y1, V y2, V y3) {
  x86::store4x8(p, _mm512_castsi512_si256(y0), _mm512_castsi512_si256(y1),
                _mm512_castsi512_si256(y2), _mm512_castsi512_si256(y3));
  x86::store4x8(p + 32, _mm512_extracti64x4_epi64(y0, 1), _mm512_extracti64x4_epi64(y1, 1),
                _mm512_extracti64x4_epi64(y2, 1), _mm512_extracti64x4_epi64(y3, 1));
}
template <int M>
inline void load4m(const uint32_t* p, V& x0, V& x1, V& x2, V& x3) {
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
inline void store4m(uint32_t* p, V y0, V y1, V y2, V y3) {
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
inline constexpr int small_m_max = 4;
inline V swap_pairs(V v) { return _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0xB1); }
inline V pick_odd(V a, V b) { return _mm512_mask_blend_epi32(0xAAAAu, a, b); }
inline V add_wide(V a, V b) { return _mm512_add_epi32(a, b); }
#elif defined(FPX_SIMD_AVX2)
using native_t = __m256i;
using V = native_t;
inline constexpr int lane = 8;
inline constexpr const char* name = "avx2";
inline V load(const uint32_t* p) { return _mm256_loadu_si256((const __m256i*)p); }
inline void store(uint32_t* p, V x) { _mm256_storeu_si256((__m256i*)p, x); }
inline V set1(uint32_t x) { return _mm256_set1_epi32((int)x); }
inline V zero() { return _mm256_setzero_si256(); }
inline V reduce_full(V v, uint32_t MOD) {
  return _mm256_min_epu32(v, _mm256_sub_epi32(v, _mm256_set1_epi32((int)MOD)));
}
inline V add(V a, V b, uint32_t R) {
  V r = _mm256_add_epi32(a, b);
  return _mm256_min_epu32(r, _mm256_sub_epi32(r, _mm256_set1_epi32((int)R)));
}
inline V sub(V a, V b, uint32_t R) {
  V y = _mm256_sub_epi32(a, b);
  return _mm256_min_epu32(y, _mm256_add_epi32(y, _mm256_set1_epi32((int)R)));
}
template <uint32_t MOD, uint32_t NINV>
inline V mulmod_lazy(V a, V b) {
  const V vn = _mm256_set1_epi32((int)NINV);
  const V vm = _mm256_set1_epi32((int)MOD);
  __m256i t0 = _mm256_mul_epu32(a, b);
  __m256i t1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
  __m256i m0 = _mm256_mul_epu32(t0, vn);
  __m256i m1 = _mm256_mul_epu32(t1, vn);
  __m256i u0 = _mm256_srli_epi64(_mm256_add_epi64(t0, _mm256_mul_epu32(m0, vm)), 32);
  __m256i u1 = _mm256_srli_epi64(_mm256_add_epi64(t1, _mm256_mul_epu32(m1, vm)), 32);
  return _mm256_or_si256(u0, _mm256_slli_epi64(u1, 32));
}
template <uint32_t MOD, uint32_t NINV>
inline V mulmod(V a, V b) {
  return reduce_full(mulmod_lazy<MOD, NINV>(a, b), MOD);
}
inline void load4(const uint32_t* p, V& x0, V& x1, V& x2, V& x3) {
  x86::load4x8(p, x0, x1, x2, x3);
}
inline void store4(uint32_t* p, V y0, V y1, V y2, V y3) { x86::store4x8(p, y0, y1, y2, y3); }
template <int M>
inline void load4m(const uint32_t* p, V& x0, V& x1, V& x2, V& x3) {
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
inline void store4m(uint32_t* p, V y0, V y1, V y2, V y3) {
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
inline constexpr int small_m_max = 4;
inline V swap_pairs(V v) { return _mm256_shuffle_epi32(v, 0xB1); }
inline V pick_odd(V a, V b) { return _mm256_blend_epi32(a, b, 0xAA); }
inline V add_wide(V a, V b) { return _mm256_add_epi32(a, b); }
#elif defined(FPX_SIMD_NEON)
using native_t = uint32x4_t;
using V = native_t;
inline constexpr int lane = 4;
inline constexpr const char* name = "neon";
inline V load(const uint32_t* p) { return vld1q_u32(p); }
inline void store(uint32_t* p, V x) { vst1q_u32(p, x); }
inline V set1(uint32_t x) { return vdupq_n_u32(x); }
inline V zero() { return vdupq_n_u32(0); }
inline V reduce_full(V v, uint32_t MOD) {
  return vminq_u32(v, vsubq_u32(v, vdupq_n_u32(MOD)));
}
inline V add(V a, V b, uint32_t R) {
  V r = vaddq_u32(a, b);
  return vminq_u32(r, vsubq_u32(r, vdupq_n_u32(R)));
}
inline V sub(V a, V b, uint32_t R) {
  V y = vsubq_u32(a, b);
  return vminq_u32(y, vaddq_u32(y, vdupq_n_u32(R)));
}
template <uint32_t MOD, uint32_t NINV>
inline V mulmod_lazy(V a, V b) {
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
template <uint32_t MOD, uint32_t NINV>
inline V mulmod(V a, V b) {
  return reduce_full(mulmod_lazy<MOD, NINV>(a, b), MOD);
}
inline void load4(const uint32_t* p, V& x0, V& x1, V& x2, V& x3) {
  uint32x4x4_t v = vld4q_u32(p);
  x0 = v.val[0];
  x1 = v.val[1];
  x2 = v.val[2];
  x3 = v.val[3];
}
inline void store4(uint32_t* p, V y0, V y1, V y2, V y3) {
  uint32x4x4_t v;
  v.val[0] = y0;
  v.val[1] = y1;
  v.val[2] = y2;
  v.val[3] = y3;
  vst4q_u32(p, v);
}
template <int M>
inline void load4m(const uint32_t* p, V& x0, V& x1, V& x2, V& x3) {
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
inline void store4m(uint32_t* p, V y0, V y1, V y2, V y3) {
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
inline constexpr int small_m_max = 2;
inline V swap_pairs(V v) { return vrev64q_u32(v); }
inline V pick_odd(V a, V b) {
  const uint32x4_t m = vreinterpretq_u32_u64(vdupq_n_u64(0xFFFFFFFF00000000ull));
  return vbslq_u32(m, b, a);
}
inline V add_wide(V a, V b) { return vaddq_u32(a, b); }
#else
using native_t = uint32_t;
using V = native_t;
inline constexpr int lane = 1;
inline constexpr const char* name = "scalar";
inline V load(const uint32_t* p) { return *p; }
inline void store(uint32_t* p, V x) { *p = x; }
inline V set1(uint32_t x) { return x; }
inline V zero() { return 0; }
inline V reduce_full(V v, uint32_t MOD) { return v >= MOD ? v - MOD : v; }
inline V add(V a, V b, uint32_t R) {
  uint32_t s = a + b;
  return s >= R ? s - R : s;
}
inline V sub(V a, V b, uint32_t R) { return a >= b ? a - b : a - b + R; }
template <uint32_t MOD, uint32_t NINV>
inline V mulmod_lazy(V a, V b) {
  uint64_t t = uint64_t(a) * b;
  uint32_t m = uint32_t(t) * NINV;
  return uint32_t((t + uint64_t(m) * MOD) >> 32);
}
template <uint32_t MOD, uint32_t NINV>
inline V mulmod(V a, V b) { return reduce_full(mulmod_lazy<MOD, NINV>(a, b), MOD); }
inline void load4(const uint32_t* p, V& x0, V& x1, V& x2, V& x3) {
  x0 = p[0];
  x1 = p[1];
  x2 = p[2];
  x3 = p[3];
}
inline void store4(uint32_t* p, V y0, V y1, V y2, V y3) {
  p[0] = y0;
  p[1] = y1;
  p[2] = y2;
  p[3] = y3;
}
inline V swap_pairs(V v) { return v; }
template <int M>
inline void load4m(const uint32_t* p, V& x0, V& x1, V& x2, V& x3) {
  x0 = p[0];
  x1 = p[M];
  x2 = p[2 * M];
  x3 = p[3 * M];
}
template <int M>
inline void store4m(uint32_t* p, V y0, V y1, V y2, V y3) {
  p[0] = y0;
  p[M] = y1;
  p[2 * M] = y2;
  p[3 * M] = y3;
}
inline constexpr int small_m_max = 0;
inline V pick_odd(V a, V) { return a; }
inline V add_wide(V a, V b) { return a + b; }
#endif
template <uint32_t MOD, uint32_t NINV, bool LAZY>
inline V butterfly_mul(V a, V b) {
  if (LAZY) {
    return mulmod_lazy<MOD, NINV>(a, b);
  } else {
    return mulmod<MOD, NINV>(a, b);
  }
}
}
// --------------------------------------------------------------------- NTT ---
// Forward = radix-4/radix-2 DIF pass sequence, inverse = the matching DIT
// sequence; they compose to n * identity, so no bit-reversal permutation is
// needed and the pointwise products live in the same base-4 digit-reversed
// order.  Radix-4 folds two levels into one butterfly: same multiplication
// count, half the memory passes and twiddle traffic.  Twiddles are
// precomputed per stage into contiguous arrays so the kernels load vectors.
template <class M>
constexpr int ntt_max_log() {
  uint64_t t = M::mod - 1;
  int k = 0;
  while (t % 2 == 0) {
    t /= 2;
    ++k;
  }
  return k;
}
template <class M>
constexpr uint32_t ntt_max_size() { return uint32_t(1) << ntt_max_log<M>(); }
inline uint32_t next_pow2(uint64_t need) {
  uint32_t n = 1;
  while (n < need) n <<= 1;
  return n;
}
class ntt_size_error : public std::runtime_error {
 public:
  explicit ntt_size_error(const std::string& s) : std::runtime_error(s) {}
};
namespace detail {
inline constexpr uint32_t s_add(uint32_t a, uint32_t b, uint32_t m) {
  uint32_t s = a + b;
  return s >= m ? s - m : s;
}
inline constexpr uint32_t s_sub(uint32_t a, uint32_t b, uint32_t m) {
  return a >= b ? a - b : a - b + m;
}
}
/// Precomputed, shared, immutable transform plan for one size.
template <class M>
class NttPlan {
 public:
  using simd_t = simd::native_t;
  static constexpr uint32_t MOD = M::mod;
  static constexpr uint32_t NINV = M::ninv;
  static constexpr uint32_t R = simd::rmod<MOD>;
  static constexpr bool LAZY = simd::lazy_ok<MOD>;
  static std::shared_ptr<const NttPlan> get(uint32_t n) {
    static std::mutex mu;
    static std::unordered_map<uint32_t, std::shared_ptr<const NttPlan>> c;
    std::lock_guard<std::mutex> lk(mu);
    auto it = c.find(n);
    if (it != c.end()) return it->second;
    auto p = std::shared_ptr<const NttPlan>(new NttPlan(n));
    c.emplace(n, p);
    return p;
  }
  uint32_t size() const { return n; }
  int log_size() const { return lg; }
  void forward(uint32_t* a) const {
    const bool r4 = (lg & 1) == 0;
    for (size_t s = 0; s < ls.size(); ++s) {
      if (r4 && s + 1 == ls.size()) {
        s4<true>(a, ls[s], s);
      } else {
        s4<false>(a, ls[s], s);
      }
    }
    if (lg & 1) f2<true>(a);
  }
  void inverse(uint32_t* a) const {
    if (lg & 1) f2<false>(a);
    for (size_t s = ls.size(); s-- > 0;) s4i(a, ls[s], s);
    const uint32_t in = inn;
    const simd_t vin = simd::set1(in);
    uint32_t k = 0;
    for (; k + simd::lane <= n; k += simd::lane) {
      simd_t v = simd::butterfly_mul<MOD, NINV, LAZY>(simd::load(a + k), vin);
      if constexpr (LAZY) v = simd::reduce_full(v, MOD);
      simd::store(a + k, v);
    }
    for (; k < n; ++k) a[k] = M::reduce(uint64_t(a[k]) * in);
  }
 private:
  explicit NttPlan(uint32_t sz) : n(sz) {
    if (sz < 2 || (sz & (sz - 1)) != 0)
      throw ntt_size_error("NTT size must be a power of two");
    if (sz > ntt_max_size<M>())
      throw ntt_size_error("NTT size exceeds 2^v2(mod-1) for this modulus");
    lg = 0;
    while ((uint32_t(1) << lg) < sz) ++lg;
    mk();
  }
  static simd_t bm(simd_t a, simd_t b) {
    return simd::butterfly_mul<MOD, NINV, LAZY>(a, b);
  }
  static simd_t fa(simd_t a, simd_t b) {
    if constexpr (LAZY) {
      return simd::add_wide(a, b);
    } else {
      return simd::add(a, b, R);
    }
  }
  void mk() {
    const M w = M::from_int(M::primitive_root).pow((MOD - 1) / n);
    q = w.pow(n / 4).raw_val();
    qi = w.pow(n / 4).inv().raw_val();
    for (uint32_t L = n; L >= 4; L >>= 2) ls.push_back(L);
    size_t tot = 0;
    for (uint32_t L : ls) tot += 3 * (L >> 2);
    fw.resize(tot);
    iv.resize(tot);
    of.resize(ls.size() + 1, 0);
    size_t pos = 0;
    for (size_t s = 0; s < ls.size(); ++s) {
      of[s] = pos;
      const uint32_t m = ls[s] >> 2;
      M wr = w;
      for (size_t t = 0; t < s; ++t) {
        wr = wr * wr;
        wr = wr * wr;
      }
      const M w2 = wr * wr, w3 = w2 * wr;
      const M wi = wr.inv(), wi2 = wi * wi, wi3 = wi2 * wi;
      const uint32_t a1 = wr.raw_val(), a2 = w2.raw_val(), a3 = w3.raw_val();
      const uint32_t b1 = wi.raw_val(), b2 = wi2.raw_val(), b3 = wi3.raw_val();
      uint32_t c1 = M::from_int(1).raw_val(), c2 = c1, c3 = c1;
      uint32_t d1 = c1, d2 = c1, d3 = c1;
      uint32_t* pa = fw.data() + pos;
      uint32_t* pb = pa + m;
      uint32_t* pc = pb + m;
      uint32_t* qa = iv.data() + pos;
      uint32_t* qb = qa + m;
      uint32_t* qc = qb + m;
      for (uint32_t j = 0; j < m; ++j) {
        pa[j] = c1;
        pb[j] = c2;
        pc[j] = c3;
        qa[j] = d1;
        qb[j] = d2;
        qc[j] = d3;
        c1 = M::reduce(uint64_t(c1) * a1);
        c2 = M::reduce(uint64_t(c2) * a2);
        c3 = M::reduce(uint64_t(c3) * a3);
        d1 = M::reduce(uint64_t(d1) * b1);
        d2 = M::reduce(uint64_t(d2) * b2);
        d3 = M::reduce(uint64_t(d3) * b3);
      }
      pos += 3 * m;
    }
    of[ls.size()] = pos;
    inn = M::from_int(n).inv().raw_val();
  }
  template <bool LAST>
  void s4(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t m = len >> 2;
    if constexpr (simd::small_m_max >= 1) {
      if (m == 1) {
        sm<1, LAST>(a, len, s);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 2) {
      if (m == 2) {
        sm<2, LAST>(a, len, s);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 4) {
      if (m == 4) {
        sm<4, LAST>(a, len, s);
        return;
      }
    }
    f4<LAST>(a, len, s);
  }
  void s4i(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t m = len >> 2;
    if constexpr (simd::small_m_max >= 1) {
      if (m == 1) {
        smi<1>(a, len, s);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 2) {
      if (m == 2) {
        smi<2>(a, len, s);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 4) {
      if (m == 4) {
        smi<4>(a, len, s);
        return;
      }
    }
    i4(a, len, s);
  }
  // Specialised stage for chunk width W == len/4 < lane.  With m below the lane
  // count the generic kernel would run the whole stage scalar; instead lane/W
  // consecutive blocks go through one vector (chunk de-interleave + matching
  // store), so only the lane *order* of the blocks differs along the way.
  template <int W, bool LAST>
  void sm(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t* A = fw.data() + of[s];
    const uint32_t* B = A + W;
    const uint32_t* C = B + W;
    alignas(64) uint32_t tw[3][64];
    for (int i = 0; i < simd::lane; ++i) {
      tw[0][i] = A[i % W];
      tw[1][i] = B[i % W];
      tw[2][i] = C[i % W];
    }
    const simd_t wa = simd::load(tw[0]), wb = simd::load(tw[1]), wc = simd::load(tw[2]);
    const simd_t vm = simd::set1(q);
    const uint32_t step = 4u * static_cast<uint32_t>(simd::lane);
    uint32_t base = 0;
    for (; base + step <= n; base += step) {
      uint32_t* p = a + base;
      simd_t x0, x1, x2, x3;
      simd::load4m<W>(p, x0, x1, x2, x3);
      simd_t t0 = simd::add(x0, x2, R);
      simd_t t1 = simd::sub(x0, x2, R);
      simd_t t2 = simd::add(x1, x3, R);
      simd_t t3 = bm(simd::sub(x1, x3, R), vm);
      simd_t y0 = simd::add(t0, t2, R);
      simd_t y1 = bm(simd::sub(t0, t2, R), wb);
      simd_t y2 = bm(fa(t1, t3), wa);
      simd_t y3 = bm(simd::sub(t1, t3, R), wc);
      if constexpr (LAST) {
        y0 = simd::reduce_full(y0, MOD);
        y1 = simd::reduce_full(y1, MOD);
        y2 = simd::reduce_full(y2, MOD);
        y3 = simd::reduce_full(y3, MOD);
      }
      simd::store4m<W>(p, y0, y1, y2, y3);
    }
    for (; base < n; base += len) {
      uint32_t* p = a + base;
      for (uint32_t k = 0; k < static_cast<uint32_t>(W); ++k) {
        uint32_t x0 = p[k], x1 = p[k + W], x2 = p[k + 2 * W], x3 = p[k + 3 * W];
        uint32_t t0 = detail::s_add(x0, x2, R), t1 = detail::s_sub(x0, x2, R);
        uint32_t t2 = detail::s_add(x1, x3, R);
        uint32_t t3 = M::reduce(uint64_t(detail::s_sub(x1, x3, R)) * q);
        uint32_t z0 = detail::s_add(t0, t2, R);
        if constexpr (LAST) z0 = z0 >= MOD ? z0 - MOD : z0;
        p[k] = z0;
        p[k + W] = M::reduce(uint64_t(detail::s_sub(t0, t2, R)) * B[k]);
        p[k + 2 * W] = M::reduce(uint64_t(detail::s_add(t1, t3, R)) * A[k]);
        p[k + 3 * W] = M::reduce(uint64_t(detail::s_sub(t1, t3, R)) * C[k]);
      }
    }
  }
  template <int W>
  void smi(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t* IA = iv.data() + of[s];
    const uint32_t* IB = IA + W;
    const uint32_t* IC = IB + W;
    alignas(64) uint32_t tw[3][64];
    for (int i = 0; i < simd::lane; ++i) {
      tw[0][i] = IA[i % W];
      tw[1][i] = IB[i % W];
      tw[2][i] = IC[i % W];
    }
    const simd_t iw1 = simd::load(tw[0]), iw2 = simd::load(tw[1]), iw3 = simd::load(tw[2]);
    const simd_t vm = simd::set1(qi);
    const uint32_t step = 4u * static_cast<uint32_t>(simd::lane);
    uint32_t base = 0;
    for (; base + step <= n; base += step) {
      uint32_t* p = a + base;
      simd_t b0, b1, b2, b3;
      simd::load4m<W>(p, b0, b1, b2, b3);
      simd_t c1 = bm(b1, iw2);
      simd_t c2 = bm(b2, iw1);
      simd_t c3 = bm(b3, iw3);
      simd_t d0 = simd::add(b0, c1, R), d1 = simd::sub(b0, c1, R);
      simd_t d2 = simd::add(c2, c3, R), d3 = simd::sub(c2, c3, R);
      simd_t e = bm(d3, vm);
      simd::store4m<W>(p, simd::add(d0, d2, R), simd::add(d1, e, R), simd::sub(d0, d2, R),
                       simd::sub(d1, e, R));
    }
    for (; base < n; base += len) {
      uint32_t* p = a + base;
      for (uint32_t k = 0; k < static_cast<uint32_t>(W); ++k) {
        uint32_t b0 = p[k], b1 = p[k + W], b2 = p[k + 2 * W], b3 = p[k + 3 * W];
        uint32_t c1 = M::reduce(uint64_t(b1) * IB[k]);
        uint32_t c2 = M::reduce(uint64_t(b2) * IA[k]);
        uint32_t c3 = M::reduce(uint64_t(b3) * IC[k]);
        uint32_t d0 = detail::s_add(b0, c1, R), d1 = detail::s_sub(b0, c1, R);
        uint32_t d2 = detail::s_add(c2, c3, R), d3 = detail::s_sub(c2, c3, R);
        uint32_t e = M::reduce(uint64_t(d3) * qi);
        p[k] = detail::s_add(d0, d2, R);
        p[k + W] = detail::s_add(d1, e, R);
        p[k + 2 * W] = detail::s_sub(d0, d2, R);
        p[k + 3 * W] = detail::s_sub(d1, e, R);
      }
    }
  }
  template <bool LAST>
  void f4(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t m = len >> 2;
    const uint32_t* A = fw.data() + of[s];
    const uint32_t* B = A + m;
    const uint32_t* C = B + m;
    const simd_t vm = simd::set1(q);
    for (uint32_t base = 0; base < n; base += len) {
      uint32_t* p = a + base;
      auto st = [&](uint32_t k) {
        simd_t x0 = simd::load(p + k), x1 = simd::load(p + k + m);
        simd_t x2 = simd::load(p + k + 2 * m), x3 = simd::load(p + k + 3 * m);
        simd_t wa = simd::load(A + k), wb = simd::load(B + k), wc = simd::load(C + k);
        simd_t t0 = simd::add(x0, x2, R);
        simd_t t1 = simd::sub(x0, x2, R);
        simd_t t2 = simd::add(x1, x3, R);
        simd_t t3 = bm(simd::sub(x1, x3, R), vm);
        simd_t y0 = simd::add(t0, t2, R);
        simd_t y1 = bm(simd::sub(t0, t2, R), wb);
        simd_t y2 = bm(fa(t1, t3), wa);
        simd_t y3 = bm(simd::sub(t1, t3, R), wc);
        if constexpr (LAST) {
          y0 = simd::reduce_full(y0, MOD);
          y1 = simd::reduce_full(y1, MOD);
          y2 = simd::reduce_full(y2, MOD);
          y3 = simd::reduce_full(y3, MOD);
        }
        simd::store(p + k, y0);
        simd::store(p + k + m, y1);
        simd::store(p + k + 2 * m, y2);
        simd::store(p + k + 3 * m, y3);
      };
      uint32_t k = 0;
      if constexpr (simd::lane > 1) {
        for (; k + 2 * simd::lane <= m; k += 2 * simd::lane) {
          st(k);
          st(k + simd::lane);
        }
      }
      for (; k + simd::lane <= m; k += simd::lane) st(k);
      for (; k < m; ++k) {
        uint32_t x0 = p[k], x1 = p[k + m], x2 = p[k + 2 * m], x3 = p[k + 3 * m];
        uint32_t t0 = detail::s_add(x0, x2, R);
        uint32_t t1 = detail::s_sub(x0, x2, R);
        uint32_t t2 = detail::s_add(x1, x3, R);
        uint32_t t3 = M::reduce(uint64_t(detail::s_sub(x1, x3, R)) * q);
        uint32_t y0 = detail::s_add(t0, t2, R);
        if constexpr (LAST) y0 = y0 >= MOD ? y0 - MOD : y0;
        p[k] = y0;
        p[k + m] = M::reduce(uint64_t(detail::s_sub(t0, t2, R)) * B[k]);
        p[k + 2 * m] = M::reduce(uint64_t(detail::s_add(t1, t3, R)) * A[k]);
        p[k + 3 * m] = M::reduce(uint64_t(detail::s_sub(t1, t3, R)) * C[k]);
      }
    }
  }
  void i4(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t m = len >> 2;
    const uint32_t* IA = iv.data() + of[s];
    const uint32_t* IB = IA + m;
    const uint32_t* IC = IB + m;
    const simd_t vm = simd::set1(qi);
    for (uint32_t base = 0; base < n; base += len) {
      uint32_t* p = a + base;
      auto st = [&](uint32_t k) {
        simd_t b0 = simd::load(p + k), b1 = simd::load(p + k + m);
        simd_t b2 = simd::load(p + k + 2 * m), b3 = simd::load(p + k + 3 * m);
        simd_t iw1 = simd::load(IA + k), iw2 = simd::load(IB + k), iw3 = simd::load(IC + k);
        simd_t c1 = bm(b1, iw2);
        simd_t c2 = bm(b2, iw1);
        simd_t c3 = bm(b3, iw3);
        simd_t d0 = simd::add(b0, c1, R), d1 = simd::sub(b0, c1, R);
        simd_t d2 = simd::add(c2, c3, R), d3 = simd::sub(c2, c3, R);
        simd_t e = bm(d3, vm);
        simd::store(p + k, simd::add(d0, d2, R));
        simd::store(p + k + m, simd::add(d1, e, R));
        simd::store(p + k + 2 * m, simd::sub(d0, d2, R));
        simd::store(p + k + 3 * m, simd::sub(d1, e, R));
      };
      uint32_t k = 0;
      if constexpr (simd::lane > 1) {
        for (; k + 2 * simd::lane <= m; k += 2 * simd::lane) {
          st(k);
          st(k + simd::lane);
        }
      }
      for (; k + simd::lane <= m; k += simd::lane) st(k);
      for (; k < m; ++k) {
        uint32_t b0 = p[k], b1 = p[k + m], b2 = p[k + 2 * m], b3 = p[k + 3 * m];
        uint32_t c1 = M::reduce(uint64_t(b1) * IB[k]);
        uint32_t c2 = M::reduce(uint64_t(b2) * IA[k]);
        uint32_t c3 = M::reduce(uint64_t(b3) * IC[k]);
        uint32_t d0 = detail::s_add(b0, c1, R), d1 = detail::s_sub(b0, c1, R);
        uint32_t d2 = detail::s_add(c2, c3, R), d3 = detail::s_sub(c2, c3, R);
        uint32_t e = M::reduce(uint64_t(d3) * qi);
        p[k] = detail::s_add(d0, d2, R);
        p[k + m] = detail::s_add(d1, e, R);
        p[k + 2 * m] = detail::s_sub(d0, d2, R);
        p[k + 3 * m] = detail::s_sub(d1, e, R);
      }
    }
  }
  // Trailing radix-2 stage for odd log2(n) (len == 2, twiddle == 1); the
  // butterfly is its own transpose, so forward and inverse share it.  lane
  // values == lane/2 contiguous pairs, so no gather is needed.
  template <bool LAST>
  void f2(uint32_t* a) const {
    uint32_t k = 0;
    if constexpr (simd::lane > 1) {
      for (; k + simd::lane <= n; k += simd::lane) {
        simd_t v = simd::load(a + k);
        simd_t w = simd::swap_pairs(v);
        simd_t s = simd::add(v, w, R);
        simd_t d = simd::sub(w, v, R);
        simd_t r = simd::pick_odd(s, d);
        if constexpr (LAST) r = simd::reduce_full(r, MOD);
        simd::store(a + k, r);
      }
    }
    for (; k + 1 < n; k += 2) {
      uint32_t x = a[k], y = a[k + 1];
      uint32_t s = detail::s_add(x, y, R);
      uint32_t t = detail::s_sub(x, y, R);
      if constexpr (LAST) {
        s = s >= MOD ? s - MOD : s;
        t = t >= MOD ? t - MOD : t;
      }
      a[k] = s;
      a[k + 1] = t;
    }
  }
  uint32_t n = 0;
  int lg = 0;
  uint32_t q = 0, qi = 0, inn = 0;
  std::vector<uint32_t> ls;
  std::vector<uint32_t> fw, iv;
  std::vector<size_t> of;
};
// -------------------------------------------------------------------- Poly ---
// Every "infinite" result (inverse, log, exp, sqrt, pow, division) takes an
// explicit truncation length n and is computed modulo x^n.
template <class M>
class Poly;
namespace poly {
class domain_error : public std::runtime_error {
 public:
  explicit domain_error(const std::string& s) : std::runtime_error(s) {}
};
template <class M>
using vec = std::vector<M>;
inline constexpr size_t naive_threshold = 40;
template <class M>
void copy_prefix(const vec<M>& src, size_t cnt, vec<M>& dst) {
  for (size_t i = 0; i < cnt; ++i) dst[i] = src[i];
}
template <class M>
void copy_range(const vec<M>& src, size_t from, size_t cnt, vec<M>& dst) {
  for (size_t i = 0; i < cnt; ++i) dst[i] = src[from + i];
}
// A full pass over the spectrum, so it runs on the SIMD lane width.
template <class M>
void pointwise_mul(uint32_t* dst, const uint32_t* b, size_t n) {
  constexpr uint32_t MOD = M::mod;
  constexpr bool LZ = simd::lazy_ok<MOD>;
  size_t i = 0;
  for (; i + simd::lane <= n; i += simd::lane) {
    simd::store(dst + i,
                simd::butterfly_mul<MOD, M::ninv, LZ>(simd::load(dst + i), simd::load(b + i)));
  }
  for (; i < n; ++i) dst[i] = M::reduce(uint64_t(dst[i]) * b[i]);
}
template <class M>
void trim(vec<M>& a) {
  while (!a.empty() && a.back().is_zero()) a.pop_back();
}
template <class M>
vec<M> add(const vec<M>& a, const vec<M>& b) {
  vec<M> r(std::max(a.size(), b.size()));
  for (size_t i = 0; i < r.size(); ++i) {
    M x = i < a.size() ? a[i] : M();
    M y = i < b.size() ? b[i] : M();
    r[i] = x + y;
  }
  return r;
}
template <class M>
vec<M> sub(const vec<M>& a, const vec<M>& b) {
  vec<M> r(std::max(a.size(), b.size()));
  for (size_t i = 0; i < r.size(); ++i) {
    M x = i < a.size() ? a[i] : M();
    M y = i < b.size() ? b[i] : M();
    r[i] = x - y;
  }
  return r;
}
template <class M>
vec<M> neg(vec<M> a) {
  for (auto& x : a) x = -x;
  return a;
}
template <class M>
vec<M> mul_scalar(const vec<M>& a, M c) {
  vec<M> r(a.size());
  constexpr uint32_t MOD = M::mod;
  const simd::native_t c1 = simd::set1(c.raw_val());
  size_t i = 0;
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  const auto* src = reinterpret_cast<const uint32_t*>(a.data());
  for (; i + simd::lane <= a.size(); i += simd::lane) {
    simd::store(dst + i, simd::mulmod<MOD, M::ninv>(simd::load(src + i), c1));
  }
  for (; i < a.size(); ++i) r[i] = a[i] * c;
  return r;
}
template <class M>
vec<M> shift(const vec<M>& a, size_t k, size_t n = SIZE_MAX) {
  vec<M> r(std::min(a.size() + k, n), M());
  for (size_t i = 0; i < a.size() && i + k < r.size(); ++i) r[i + k] = a[i];
  return r;
}
template <class M>
M eval(const vec<M>& a, M x) {
  M r = M();
  for (size_t i = a.size(); i-- > 0;) r = r * x + a[i];
  return r;
}
template <class M>
vec<M> conv_limbs(const vec<M>& a, const vec<M>& b, size_t lim) {
  if (a.empty() || b.empty() || lim == 0) return {};
  const size_t la = std::min(a.size(), lim), lb = std::min(b.size(), lim);
  const size_t full = la + lb - 1;
  const size_t out = std::min(full, lim);
  static_assert(sizeof(M) == sizeof(uint32_t), "Mont must be a single 32-bit limb");
  if (std::min(la, lb) <= naive_threshold) {
    vec<M> r(out, M());
    for (size_t i = 0; i < la; ++i)
      for (size_t j = 0; j < lb && i + j < out; ++j) r[i + j] += a[i] * b[j];
    return r;
  }
  const uint32_t N = next_pow2(full);
  auto plan = NttPlan<M>::get(N);
  vec<M> fa(N, M()), fb(N, M());
  copy_prefix(a, la, fa);
  copy_prefix(b, lb, fb);
  uint32_t* pa = reinterpret_cast<uint32_t*>(fa.data());
  uint32_t* pb = reinterpret_cast<uint32_t*>(fb.data());
  plan->forward(pa);
  plan->forward(pb);
  pointwise_mul<M>(pa, pb, N);
  plan->inverse(pa);
  fa.resize(out);
  return fa;
}
template <class M>
vec<M> conv(const vec<M>& a, const vec<M>& b, size_t n = SIZE_MAX) {
  return conv_limbs(a, b, n);
}
template <class M>
vec<M> derivative(const vec<M>& a) {
  if (a.size() <= 1) return {};
  vec<M> r(a.size() - 1);
  for (size_t i = 1; i < a.size(); ++i) r[i - 1] = a[i] * M::from_int(i);
  return r;
}
// Montgomery forms of 1/1 .. 1/n by 1/i = -(mod/i) /(mod%i); needs n < mod.
// ~5x faster than a batch inversion: mod is constexpr, so mod/i, mod%i and
// from_int strength-reduce to multiply/shift instead of 3 Montgomery products.
template <class M>
vec<M> inv_series(size_t n) {
  if (n >= M::mod) throw domain_error("poly::inv_series: length must be < mod");
  vec<M> r(n + 1, M());
  if (n == 0) return r;
  r[1] = M::from_int(1);
  for (size_t i = 2; i <= n; ++i) {
    r[i] = -(M::from_int(M::mod / i) * r[static_cast<size_t>(M::mod % i)]);
  }
  return r;
}
template <class M>
vec<M> integral(const vec<M>& a) {
  if (a.empty()) return {};
  vec<M> iv = inv_series<M>(a.size());
  vec<M> r(a.size() + 1);
  for (size_t i = 0; i < a.size(); ++i) r[i + 1] = a[i] * iv[i + 1];
  return r;
}
// Newton iteration with the *middle product* optimisation: step m -> m2 only
// needs the high half of a*b, and that half is free of cyclic aliasing at
// transform size 2m (aliasing only pollutes the low half), so the transform
// stays at 2m instead of 4m -- halving the transform length per iteration.
// log/exp/sqrt are all built on this inv.
template <class M>
vec<M> inv(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (a.empty() || a[0].is_zero()) throw domain_error("poly::inv: constant term is zero");
  if (a.size() == 1) {
    vec<M> r(n, M());
    r[0] = a[0].inv();
    return r;
  }
  static_assert(sizeof(M) == sizeof(uint32_t), "Mont must be a single 32-bit limb");
  vec<M> b{a[0].inv()};
  for (size_t m = 1; m < n; m <<= 1) {
    const size_t m2 = std::min(2 * m, n);
    const uint32_t N = next_pow2(2 * m);
    auto plan = NttPlan<M>::get(N);
    vec<M> t(N, M());
    const size_t acnt = std::min(a.size(), 2 * m);
    copy_prefix(a, acnt, t);
    uint32_t* pt = reinterpret_cast<uint32_t*>(t.data());
    vec<M> u(N, M());
    std::copy(b.begin(), b.end(), u.begin());
    uint32_t* pu = reinterpret_cast<uint32_t*>(u.data());
    plan->forward(pu);
    plan->forward(pt);
    pointwise_mul<M>(pt, pu, N);
    plan->inverse(pt);
    const size_t hlen = m2 - m;
    vec<M> h(hlen, M());
    for (size_t i = 0; i < hlen; ++i) h[i] = -t[m + i];
    vec<M> hv(N, M());
    std::copy(h.begin(), h.end(), hv.begin());
    uint32_t* ph = reinterpret_cast<uint32_t*>(hv.data());
    plan->forward(ph);
    pointwise_mul<M>(ph, pu, N);
    plan->inverse(ph);
    b.resize(m2);
    for (size_t i = 0; i < hlen; ++i) b[m + i] = hv[i];
  }
  b.resize(n);
  return b;
}
// log(a) = integral(a' / a) mod x^n; requires a(0) == 1.
template <class M>
vec<M> log(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (a.empty() || a[0] != M::from_int(1))
    throw domain_error("poly::log: constant term must be 1");
  vec<M> d = derivative(a);
  vec<M> ia = inv(a, n);
  vec<M> p = conv_limbs(d, ia, n - 1);
  vec<M> r = integral(p);
  r.resize(n, M());
  return r;
}
// exp(a) mod x^n; requires a(0) == 0.  b_{2m} = b_m * (1 + a - log b_m) mod x^{2m}.
template <class M>
vec<M> exp(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (!a.empty() && !a[0].is_zero()) throw domain_error("poly::exp: constant term must be 0");
  vec<M> b{M::from_int(1)};
  for (size_t m = 1; m < n; m <<= 1) {
    const size_t m2 = std::min(2 * m, n);
    vec<M> lb = log(b, m2);
    vec<M> t(m2, M());
    t[0] = M::from_int(1);
    for (size_t i = 0; i < m2; ++i) t[i] -= lb[i];
    for (size_t i = 0; i < std::min(a.size(), m2); ++i) t[i] += a[i];
    b = conv_limbs(b, t, m2);
    b.resize(m2);
  }
  return b;
}
// Square root mod x^n; throws domain_error when a(0) is a non-residue or the
// x-adic valuation of a is odd (then no root exists in GF(Mod)[[x]]).
template <class M>
vec<M> sqrt(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  size_t v = 0;
  while (v < a.size() && a[v].is_zero()) ++v;
  if (v == a.size()) return vec<M>(n, M());
  if (v & 1) throw domain_error("poly::sqrt: odd valuation, no square root");
  M root0;
  if (!a[v].sqrt(root0)) throw domain_error("poly::sqrt: non-residue constant term");
  const size_t half = v / 2;
  if (half >= n) return vec<M>(n, M());
  const size_t inner_n = n - half;
  vec<M> u(std::min(a.size() - v, inner_n));
  copy_range(a, v, u.size(), u);
  vec<M> b{root0};
  for (size_t m = 1; m < inner_n; m <<= 1) {
    const size_t m2 = std::min(2 * m, inner_n);
    vec<M> ib = inv(b, m2);
    vec<M> t = conv_limbs(u, ib, m2);
    t.resize(m2, M());
    for (size_t i = 0; i < m2; ++i) t[i] += (i < b.size() ? b[i] : M());
    b = mul_scalar(t, M::from_int(M::inv2));
    b.resize(m2);
  }
  vec<M> r(n, M());
  for (size_t i = 0; i < b.size() && i + half < n; ++i) r[i + half] = b[i];
  return r;
}
// a^k mod x^n, any 64-bit k: binary exponentiation when k is small, else
// exp(k*log a).  That log is legitimate for any integer k, since on the
// normalized series (1 + O(x)) the j-th coefficient of (1+u)^k sees k only
// through k mod Mod, and j < n <= Mod.
template <class M>
vec<M> pow(const vec<M>& a, uint64_t k, size_t n) {
  if (n == 0) return {};
  if (k == 0) {
    vec<M> r(n, M());
    r[0] = M::from_int(1);
    return r;
  }
  size_t v = 0;
  while (v < a.size() && a[v].is_zero()) ++v;
  if (v == a.size()) return vec<M>(n, M());
  if (v != 0 && k > (n - 1) / v) return vec<M>(n, M());
  const size_t sh = static_cast<size_t>(v * k);
  const size_t inner_n = n - sh;
  vec<M> u(std::min(a.size() - v, inner_n));
  copy_range(a, v, u.size(), u);
  if (k <= 64) {
    vec<M> r{M::from_int(1)}, base = u;
    for (uint64_t e = k; e; e >>= 1) {
      if (e & 1) {
        r = conv_limbs(r, base, inner_n);
        r.resize(inner_n);
      }
      if (e > 1) {
        base = conv_limbs(base, base, inner_n);
        base.resize(inner_n);
      }
    }
    vec<M> out(n, M());
    for (size_t i = 0; i < r.size() && i + sh < n; ++i) out[i + sh] = r[i];
    return out;
  }
  const M c = u[0];
  const vec<M> nrm = mul_scalar(u, c.inv());
  vec<M> l = log(nrm, inner_n);
  for (auto& x : l) x *= M::from_int(k);
  vec<M> r = exp(l, inner_n);
  r = mul_scalar(r, c.pow(k));
  vec<M> out(n, M());
  for (size_t i = 0; i < r.size() && i + sh < n; ++i) out[i + sh] = r[i];
  return out;
}
template <class M>
std::pair<vec<M>, vec<M>> divmod(const vec<M>& a_in, const vec<M>& b_in) {
  vec<M> a = a_in, b = b_in;
  trim(a);
  trim(b);
  if (b.empty()) throw domain_error("poly::divmod: division by zero polynomial");
  if (a.size() < b.size()) return {vec<M>{}, a};
  const size_t dn = a.size() - b.size() + 1;
  vec<M> ra(a.rbegin(), a.rend());
  vec<M> rb(b.rbegin(), b.rend());
  vec<M> ib = inv(rb, dn);
  vec<M> rq = conv_limbs(ra, ib, dn);
  rq.resize(dn, M());
  vec<M> q(rq.rbegin(), rq.rend());
  vec<M> r = sub(a, conv_limbs(b, q, a.size()));
  r.resize(a.size(), M());
  trim(r);
  return {q, r};
}
template <class M>
vec<M> div(const vec<M>& a, const vec<M>& b) {
  return divmod(a, b).first;
}
template <class M>
vec<M> mod(const vec<M>& a, const vec<M>& b) {
  return divmod(a, b).second;
}
}
template <class M>
class Poly : public std::vector<M> {
 public:
  using base = std::vector<M>;
  using value_type = M;
  Poly() = default;
  Poly(const base& v) : base(v) {}
  Poly(base&& v) : base(std::move(v)) {}
  Poly(std::initializer_list<M> il) : base(il) {}
  explicit Poly(size_t n, M v = M()) : base(n, v) {}
  using base::operator[];
  using base::size;
  size_t degree() const { return this->empty() ? 0 : this->size() - 1; }
  bool is_zero() const {
    for (const auto& x : *this)
      if (!x.is_zero()) return false;
    return true;
  }
  Poly& trim() {
    poly::trim(*this);
    return *this;
  }
  Poly& resize_(size_t n) {
    this->resize(n);
    return *this;
  }
  Poly operator+(const base& b) const { return poly::add(*this, b); }
  Poly operator-(const base& b) const { return poly::sub(*this, b); }
  Poly operator-() const { return poly::neg(*this); }
  Poly operator*(const base& b) const { return poly::conv(*this, b); }
  Poly operator*(M c) const { return poly::mul_scalar(*this, c); }
  Poly operator<<(size_t k) const { return poly::shift(*this, k); }
  Poly& operator+=(const base& b) { return *this = poly::add(*this, b); }
  Poly& operator-=(const base& b) { return *this = poly::sub(*this, b); }
  Poly& operator*=(const base& b) { return *this = poly::conv(*this, b); }
  Poly mul(const base& b, size_t n) const { return poly::conv(*this, b, n); }
  Poly derivative() const { return poly::derivative(*this); }
  Poly integral() const { return poly::integral(*this); }
  Poly inv(size_t n) const { return poly::inv(*this, n); }
  Poly log(size_t n) const { return poly::log(*this, n); }
  Poly exp(size_t n) const { return poly::exp(*this, n); }
  Poly sqrt(size_t n) const { return poly::sqrt(*this, n); }
  Poly pow(uint64_t k, size_t n) const { return poly::pow(*this, k, n); }
  std::pair<Poly, Poly> divmod(const base& b) const {
    auto r = poly::divmod(*this, b);
    return {Poly(std::move(r.first)), Poly(std::move(r.second))};
  }
  Poly div(const base& b) const { return poly::div(*this, b); }
  Poly mod(const base& b) const { return poly::mod(*this, b); }
  M eval(M x) const { return poly::eval(*this, x); }
  std::vector<uint32_t> to_ints() const {
    std::vector<uint32_t> r(this->size());
    for (size_t i = 0; i < this->size(); ++i) r[i] = (*this)[i].val();
    return r;
  }
  static Poly from_ints(const std::vector<uint32_t>& v) {
    Poly r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = M::from_int(v[i]);
    return r;
  }
};
inline const char* simd_backend() { return simd::name; }
}
#endif
