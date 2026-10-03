// fastpoly - umbrella header.
//
//   fpx::Mont<Mod, Root>      Montgomery field element for an NTT prime
//   fpx::NttPlan<M>::get(n)   cached, SIMD-accelerated, permutation-free NTT
//   fpx::Poly<M>              polynomial over GF(Mod), truncated mod x^n
//
// Everything is header only; compile with -mavx2 on x86-64 (AVX-512 and NEON
// are picked up automatically when available).
#ifndef FASTPOLY_FASTPOLY_HPP
#define FASTPOLY_FASTPOLY_HPP

#include "fastpoly/modint.hpp"
#include "fastpoly/ntt.hpp"
#include "fastpoly/poly.hpp"
#include "fastpoly/simd.hpp"

namespace fpx {

/// Simd backend actually in use for this translation unit.
inline const char* simd_backend() { return simd::name; }

}  // namespace fpx

#endif  // FASTPOLY_FASTPOLY_HPP
