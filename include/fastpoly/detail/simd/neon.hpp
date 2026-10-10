// fastpoly - neon native primitives; included by simd.hpp.
#ifndef FASTPOLY_DETAIL_SIMD_NEON_HPP
#define FASTPOLY_DETAIL_SIMD_NEON_HPP

namespace fpx::simd {

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
/// Unreduced sum of two lazy values (< 2*MOD each), for fixed multiplication.
inline native_t add_wide(native_t a, native_t b) { return vaddq_u32(a, b); }

inline native_t sub_wide(native_t a, native_t b) { return vsubq_u32(a, b); }

// Compact Q31 tables, column splits and four-lane scans are NEON capabilities.
struct backend {
  using native_type = native_t;
  static constexpr bool q31 = true;
  static constexpr bool q32 = false;
  static constexpr bool split_radix8 = true;
  static constexpr bool parallel_linear_exp = true;

  template <uint32_t NINV, bool ROUND>
  static native_t encode_q31(native_t t) {
    const native_t m = vmulq_u32(t, set1(NINV));
    if constexpr (ROUND) return vhaddq_u32(m, set1(1));
    else return vshrq_n_u32(m, 1);
  }
  template <uint32_t MOD>
  static native_t decode_q31(native_t q) {
    return vreinterpretq_u32_s32(vqrdmulhq_s32(
        vreinterpretq_s32_u32(q), vdupq_n_s32(static_cast<int32_t>(MOD))));
  }
  template <uint32_t MOD>
  static native_t mul_q31(native_t a, native_t value, native_t quotient) {
    const native_t q = vreinterpretq_u32_s32(vqdmulhq_s32(
        vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(quotient)));
    return vmlsq_u32(vmulq_u32(a, value), q, set1(MOD));
  }
  // Signed x in [-2*MOD, 2*MOD), nearest Q31 quotient: remainder in (-MOD, MOD).
  template <uint32_t MOD>
  static native_t mul_q31_centered(native_t x, native_t value, native_t quotient) {
    const native_t q = vreinterpretq_u32_s32(vqrdmulhq_s32(
        vreinterpretq_s32_u32(x), vreinterpretq_s32_u32(quotient)));
    const native_t r = vmlsq_u32(vmulq_u32(x, value), q, set1(MOD));
    return vaddq_u32(r, set1(MOD));
  }

  static native_t unzip_even(native_t a, native_t b) { return vuzp1q_u32(a, b); }
  static native_t unzip_odd(native_t a, native_t b) { return vuzp2q_u32(a, b); }
  static native_t zip_low(native_t a, native_t b) { return vzip1q_u32(a, b); }
  static native_t zip_high(native_t a, native_t b) { return vzip2q_u32(a, b); }
  template <int I>
  static native_t broadcast(native_t x) { return vdupq_laneq_u32(x, I); }
  template <int N>
  static native_t shift_right_lanes(native_t fill, native_t x) {
    return vextq_u32(fill, x, lane - N);
  }
  // Extract endpoints as one vector primitive, preserving the independent
  // scan scheduling without exposing compiler vector indexing to algorithms.
  static native_t endpoints(native_t x0, native_t x1, native_t x2, native_t x3) {
    return native_t{vgetq_lane_u32(x0, 3), vgetq_lane_u32(x1, 3),
                    vgetq_lane_u32(x2, 3), vgetq_lane_u32(x3, 3)};
  }
  static uint32_t last_lane(native_t x) { return vgetq_lane_u32(x, 3); }
  static native_t reverse(native_t x) {
    x = vrev64q_u32(x);
    return vextq_u32(x, x, 2);
  }
};

}  // namespace fpx::simd

#endif  // FASTPOLY_DETAIL_SIMD_NEON_HPP
