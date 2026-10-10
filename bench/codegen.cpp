// Compile with -S to inspect each backend's native shuffle/multiply kernels.
// Example: clang++ -std=c++20 -O2 -mavx512f -mavx512vl -Iinclude -S bench/codegen.cpp
#include "fastpoly/fastpoly.hpp"

namespace {
template <int W>
void columns(const uint32_t* src, uint32_t* dst) {
  fpx::simd::native_t x0, x1, x2, x3;
  fpx::simd::load4m<W>(src, x0, x1, x2, x3);
  fpx::simd::store(dst, x0); fpx::simd::store(dst + fpx::simd::lane, x1);
  fpx::simd::store(dst + 2*fpx::simd::lane, x2); fpx::simd::store(dst + 3*fpx::simd::lane, x3);
}
template <int W>
void rows(const uint32_t* src, uint32_t* dst) {
  fpx::simd::store4m<W>(dst, fpx::simd::load(src), fpx::simd::load(src + fpx::simd::lane),
                       fpx::simd::load(src + 2*fpx::simd::lane), fpx::simd::load(src + 3*fpx::simd::lane));
}
}

extern "C" {
void fastpoly_columns1(const uint32_t* src, uint32_t* dst) { columns<1>(src, dst); }
void fastpoly_rows1(const uint32_t* src, uint32_t* dst) { rows<1>(src, dst); }
#if defined(FPX_HAVE_AVX2_INTRIN)
void fastpoly_columns2(const uint32_t* src, uint32_t* dst) { columns<2>(src, dst); }
void fastpoly_rows2(const uint32_t* src, uint32_t* dst) { rows<2>(src, dst); }
void fastpoly_columns4(const uint32_t* src, uint32_t* dst) { columns<4>(src, dst); }
void fastpoly_rows4(const uint32_t* src, uint32_t* dst) { rows<4>(src, dst); }
#endif
#if defined(FPX_SIMD_AVX512)
void fastpoly_columns8(const uint32_t* src, uint32_t* dst) { columns<8>(src, dst); }
void fastpoly_rows8(const uint32_t* src, uint32_t* dst) { rows<8>(src, dst); }
#endif
void fastpoly_fixed7(const uint32_t* src, uint32_t* dst) {
  using M = fpx::mod998244353;
  const auto factor = fpx::simd::fixed_twiddle<M::mod, M::ninv>(M::from_int(7).raw_val());
  fpx::simd::store(dst, fpx::simd::mul_twiddle_full<M::mod, M::ninv>(fpx::simd::load(src), factor));
}
// Variable twiddles keep the multiplication visible instead of folding a
// small scalar coefficient into shifts. Both paths accept lazy input limbs.
void fastpoly_twiddle_diff(const uint32_t* a, const uint32_t* b,
                          const uint32_t* twiddle, uint32_t* dst) {
  using M = fpx::mod998244353;
  const auto factor = fpx::simd::decode_twiddle<M::mod>(fpx::simd::load(twiddle));
  fpx::simd::store(dst, fpx::simd::mul_twiddle_diff<M::mod, M::ninv>(
      fpx::simd::load(a), fpx::simd::load(b), factor));
}
#if defined(FPX_HAVE_AVX2_INTRIN)
void fastpoly_q32_diff(const uint32_t* a, const uint32_t* b,
                      const uint32_t* value, const uint32_t* quotient, uint32_t* dst) {
  using M = fpx::mod998244353;
  const fpx::simd::FixedTwiddle factor{{fpx::simd::load(value), fpx::simd::load(quotient)}};
  fpx::simd::store(dst, fpx::simd::mul_twiddle_diff<M::mod, M::ninv>(
      fpx::simd::load(a), fpx::simd::load(b), factor));
}
#endif
}
