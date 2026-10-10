// fastpoly - scalar native primitives; included by simd.hpp.
#ifndef FASTPOLY_DETAIL_SIMD_SCALAR_HPP
#define FASTPOLY_DETAIL_SIMD_SCALAR_HPP

namespace fpx::simd {

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

inline native_t sub_wide(native_t a, native_t b) { return a - b; }

struct backend {
  static constexpr bool q31 = false;
  static constexpr bool q32 = false;
  static constexpr bool split_radix8 = false;
  static constexpr bool parallel_linear_exp = false;
};

}  // namespace fpx::simd

#endif  // FASTPOLY_DETAIL_SIMD_SCALAR_HPP
