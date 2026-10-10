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

namespace fpx::simd {

/// True when the modulus is small enough to use the lazy [0, 2*MOD) convention.
/// The exact condition is 4*MOD^2 < MOD*2^32, i.e. MOD < 2^30.
template <uint32_t MOD>
inline constexpr bool lazy_ok = MOD < (1u << 30);

/// Reduction modulus of the lazy convention: the ring the residues live in.
/// `R` is a multiple of MOD, so reducing mod R still produces a valid residue.
template <uint32_t MOD>
inline constexpr uint32_t rmod = lazy_ok<MOD> ? (2u * MOD) : MOD;

}  // namespace fpx::simd

// This is the only architecture selection point. Algorithms use capabilities.
#if defined(__AVX512F__) && defined(__AVX512VL__)
#define FPX_SIMD_AVX512 1
#define FPX_HAVE_AVX2_INTRIN 1
#include <immintrin.h>
#include "fastpoly/detail/simd/avx512.hpp"
#elif defined(__AVX2__)
#define FPX_SIMD_AVX2 1
#define FPX_HAVE_AVX2_INTRIN 1
#include <immintrin.h>
#include "fastpoly/detail/simd/avx2.hpp"
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#define FPX_SIMD_NEON 1
#include <arm_neon.h>
#include "fastpoly/detail/simd/neon.hpp"
#else
#include "fastpoly/detail/simd/scalar.hpp"
#endif

namespace fpx::simd {

// Optional kernels have fixed lane layouts. Reject inconsistent new backends
// here, before a capability can route a transform to the wrong layout.
static_assert(!(backend::q31 && backend::q32));
static_assert(!backend::split_radix8 || lane == 4);
static_assert(!backend::parallel_linear_exp || lane == 4);

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

/// A width-2 stage's even lanes have unity twiddles.
template <uint32_t MOD, uint32_t NINV, class Backend = backend>
inline native_t mul_unity_even(native_t a, native_t twiddle) {
  return Backend::template mul_unity_even<MOD, NINV>(a, twiddle);
}

// NEON tables store compact Q31 quotients. Fixed x86 multipliers use Q32;
// large x86 stages keep Montgomery tables to limit twiddle traffic.
template <uint32_t MOD>
inline constexpr bool compact_twiddle = backend::q31 && lazy_ok<MOD>;

template <uint32_t MOD>
inline constexpr bool fixed_lazy = (backend::q32 || backend::q31) && lazy_ok<MOD>;

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

template <uint32_t MOD, uint32_t NINV, bool ROUND, class Backend = backend>
inline native_t encode_twiddle(native_t t) {
  if constexpr (compact_twiddle<MOD>)
    return Backend::template encode_q31<NINV, ROUND>(t);
  else
    return t;
}

/// Reconstruct w exactly from either Q31 encoding. Its error after scaling
/// back is < MOD/2^31 < 1/2, so one rounded high multiply recovers w.
template <uint32_t MOD, class Backend = backend>
inline Twiddle decode_twiddle(native_t q) {
  if constexpr (compact_twiddle<MOD>)
    return {Backend::template decode_q31<MOD>(q), q};
  else
    return {q, zero()};
}

template <uint32_t MOD, uint32_t NINV, bool ROUND = false>
inline FixedTwiddle fixed_twiddle(uint32_t mont) {
  const uint32_t q = encode_twiddle_scalar<MOD, NINV, ROUND>(mont);
  if constexpr (compact_twiddle<MOD>) {
    const uint32_t w = static_cast<uint32_t>(
        (uint64_t(q) * MOD + (uint64_t(1) << 30)) >> 31);
    return {{set1(w), set1(q)}};
  } else if constexpr (backend::q32) {
    const uint32_t digit = mont*NINV;
    const uint32_t value = static_cast<uint32_t>((uint64_t(digit)*MOD + mont) >> 32);
    // value*2^32/MOD = digit + mont/MOD, and mont is canonical.
    const uint32_t quotient = digit;
    return {{set1(value), set1(quotient)}};
  } else {
    return {{set1(q), zero()}};
  }
}

template <uint32_t MOD, uint32_t NINV, class Backend = backend>
inline FixedTwiddle fixed_twiddle_vector(native_t mont) {
  FixedTwiddle result;
  Backend::template fixed_vector<MOD, NINV>(mont, result.value, result.quotient);
  return result;
}

/// Positive fixed multiplication: a < 2*MOD and a floor Q31 quotient.
/// floor(a*q/2^31) underestimates a*w/MOD by < 2. The difference is in
/// [0, 2*MOD); SQDMULH + MUL + MLS handles four lanes in three instructions.
template <uint32_t MOD, uint32_t NINV, class Backend = backend>
inline native_t mul_twiddle(native_t a, Twiddle w) {
  if constexpr (compact_twiddle<MOD>)
    return Backend::template mul_q31<MOD>(a, w.value, w.quotient);
  else
    return butterfly_mul<MOD, NINV, lazy_ok<MOD>>(a, w.value);
}

template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_full(native_t a, Twiddle w) {
  if constexpr (compact_twiddle<MOD>)
    return reduce_full(mul_twiddle<MOD, NINV>(a, w), MOD);
  else
    return mulmod<MOD, NINV>(a, w.value);
}

/// Signed x in [-2*MOD, 2*MOD), with a nearest Q31 quotient.
template <uint32_t MOD, class Backend = backend>
inline native_t mul_twiddle_centered(native_t x, Twiddle w) {
  return Backend::template mul_q31_centered<MOD>(x, w.value, w.quotient);
}

template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_diff(native_t a, native_t b, Twiddle w) {
  if constexpr (compact_twiddle<MOD>)
    return mul_twiddle_centered<MOD>(sub_wide(a, b), w);
  else
    return butterfly_mul<MOD, NINV, lazy_ok<MOD>>(sub(a, b, rmod<MOD>), w.value);
}

template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_sum(native_t a, native_t b, Twiddle w) {
  if constexpr (compact_twiddle<MOD>)
    return mul_twiddle_centered<MOD>(sub_wide(add_wide(a, b), set1(2 * MOD)), w);
  else if constexpr (lazy_ok<MOD>)
    return mulmod_lazy<MOD, NINV>(add_wide(a, b), w.value);
  else
    return mulmod<MOD, NINV>(add(a, b, MOD), w.value);
}

// A fixed ordinary multiplier uses a floor Q32 quotient. For a < 2^32 the
// estimated quotient undershoots by less than two, so the remainder is < 2*MOD.
// Its two wide high products run in parallel with the all-lane low product.
template <uint32_t MOD, uint32_t NINV, class Backend = backend>
inline native_t mul_twiddle(native_t a, FixedTwiddle w) {
  if constexpr (Backend::q32)
    return Backend::template mul_q32<MOD>(a, w.value, w.quotient);
  else
    return mul_twiddle<MOD, NINV>(a, static_cast<Twiddle>(w));
}

template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_full(native_t a, FixedTwiddle w) {
  if constexpr (lazy_ok<MOD>) return reduce_full(mul_twiddle<MOD, NINV>(a, w), MOD);
  else return mul_twiddle<MOD, NINV>(a, w);
}
template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_diff(native_t a, native_t b, FixedTwiddle w) {
  if constexpr (backend::q32)
    return mul_twiddle<MOD, NINV>(sub(a, b, rmod<MOD>), w);
  else
    return mul_twiddle_diff<MOD, NINV>(a, b, static_cast<Twiddle>(w));
}
template <uint32_t MOD, uint32_t NINV>
inline native_t mul_twiddle_sum(native_t a, native_t b, FixedTwiddle w) {
  if constexpr (backend::q32) {
    if constexpr (lazy_ok<MOD>) return mul_twiddle<MOD, NINV>(add_wide(a, b), w);
    else return mul_twiddle<MOD, NINV>(add(a, b, MOD), w);
  } else {
    return mul_twiddle_sum<MOD, NINV>(a, b, static_cast<Twiddle>(w));
  }
}

}  // namespace fpx::simd

#endif  // FASTPOLY_SIMD_HPP
