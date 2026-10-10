// fastpoly - polynomial arithmetic modulo x^n over GF(Mod), Mod a 32-bit NTT
// prime.  All "infinite" results (inverse, ln, exp, sqrt, pow, division) take
// an explicit truncation length `n` and are computed modulo x^n.
#ifndef FASTPOLY_POLY_HPP
#define FASTPOLY_POLY_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fastpoly/modint.hpp"
#include "fastpoly/memory.hpp"
#include "fastpoly/ntt.hpp"
#include "fastpoly/simd.hpp"

namespace fpx {

template <class M>
class Poly;

namespace poly {

/// Thrown for algebraic preconditions (a(0) == 0 in an inverse, a(0) != 1 in
/// a logarithm, a non-residue constant term in a square root, ...).
class domain_error : public std::runtime_error {
 public:
  explicit domain_error(const std::string& s) : std::runtime_error(s) {}
};

template <class M>
using vec = std::vector<M>;

namespace detail {

// Newton uses powers of two, except that a final <=8-coefficient tail is
// filled directly. Reserve the largest transform once, without allocating a
// needless doubled buffer for that small tail.
template <class M>
uint32_t series_scratch_size(size_t n, size_t seed) {
  if (n <= seed) return 0;
  const size_t floor = std::bit_floor(n);
  const uint32_t N = next_pow2(n - floor <= 8 ? floor : n);
  if (N <= seed) return 0;
  if (N > ntt_max_size<M>())
    throw ntt_size_error("NTT size exceeds 2^v2(mod-1) for this modulus");
  return N;
}

// A transform overwrites every limb. Copy the live coefficients first and
// initialize only its zero padding; recycled scratch needs no value objects.
template <class M>
void prepare_scratch(const vec<M>& src, size_t cnt,
                     fpx::detail::ScratchBuffer& dst, size_t n) {
  static_assert(sizeof(M) == sizeof(uint32_t), "Mont must be a single 32-bit limb");
  dst.resize_uninitialized(n);
  if (cnt != 0) std::memcpy(dst.data(), src.data(), cnt * sizeof(uint32_t));
  std::memset(dst.data() + cnt, 0, (n - cnt) * sizeof(uint32_t));
}

template <class M>
void scale_limbs(uint32_t* dst, const uint32_t* src, size_t n, M c) {
  if (c.raw_val() == M::one) {
    if (dst != src && n != 0) std::memcpy(dst, src, n * sizeof(uint32_t));
    return;
  }
  if (c.is_zero()) {
    if (n != 0) std::memset(dst, 0, n * sizeof(uint32_t));
    return;
  }
  const auto factor = simd::fixed_twiddle<M::mod, M::ninv>(c.raw_val());
  size_t i = 0;
  for (; i + simd::lane <= n; i += simd::lane)
    simd::store(dst + i, simd::mul_twiddle_full<M::mod, M::ninv>(simd::load(src + i), factor));
  for (; i < n; ++i) dst[i] = M::reduce(uint64_t(src[i]) * c.raw_val());
}

template <class M>
void scale_inplace(vec<M>& a, M c) {
  auto* p = reinterpret_cast<uint32_t*>(a.data());
  scale_limbs(p, p, a.size(), c);
}

}  // namespace detail

/// Convolutions with a short operand are done directly.
inline constexpr size_t naive_threshold = 40;

/// dst[0, cnt) = src[0, cnt)
template <class M>
void copy_prefix(const vec<M>& src, size_t cnt, vec<M>& dst) {
  for (size_t i = 0; i < cnt; ++i) dst[i] = src[i];
}
/// dst[0, cnt) = src[from, from + cnt)
template <class M>
void copy_range(const vec<M>& src, size_t from, size_t cnt, vec<M>& dst) {
  for (size_t i = 0; i < cnt; ++i) dst[i] = src[from + i];
}

/// Pointwise Montgomery product of two transform-domain buffers:
/// `dst[i] *= b[i]`.  This is a full pass over the spectrum, so it runs on the
/// SIMD lane width; for Mod < 2^30 it uses the lazy kernel (the inverse
/// transform canonicalises afterwards).
template <class M>
void pointwise_mul(uint32_t* dst, const uint32_t* b, size_t n) {
  constexpr uint32_t MOD = M::mod;
  constexpr bool LAZY = simd::lazy_ok<MOD>;
  size_t i = 0;
  constexpr size_t W = simd::lane;
  // Four independent products overlap the Montgomery multiply chains while
  // retaining consecutive loads/stores for every SIMD backend.
  for (; i + 4*W <= n; i += 4*W) {
    const auto x0 = simd::load(dst + i), y0 = simd::load(b + i);
    const auto x1 = simd::load(dst + i + W), y1 = simd::load(b + i + W);
    const auto x2 = simd::load(dst + i + 2*W), y2 = simd::load(b + i + 2*W);
    const auto x3 = simd::load(dst + i + 3*W), y3 = simd::load(b + i + 3*W);
    const auto v0 = simd::butterfly_mul<MOD, M::ninv, LAZY>(x0, y0);
    const auto v1 = simd::butterfly_mul<MOD, M::ninv, LAZY>(x1, y1);
    const auto v2 = simd::butterfly_mul<MOD, M::ninv, LAZY>(x2, y2);
    const auto v3 = simd::butterfly_mul<MOD, M::ninv, LAZY>(x3, y3);
    simd::store(dst + i, v0);
    simd::store(dst + i + W, v1);
    simd::store(dst + i + 2*W, v2);
    simd::store(dst + i + 3*W, v3);
  }
  for (; i + simd::lane <= n; i += simd::lane) {
    simd::store(dst + i, simd::butterfly_mul<MOD, M::ninv, LAZY>(
                             simd::load(dst + i), simd::load(b + i)));
  }
  for (; i < n; ++i) dst[i] = M::reduce(uint64_t(dst[i]) * b[i]);
}

/// ---------------------------------------------------------------------------
/// basic operations
/// ---------------------------------------------------------------------------

/// Drop trailing zeros so that degree() is meaningful.
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
  if (c.raw_val() == M::one) return a;
  vec<M> r(a.size());
  if (c.is_zero()) return r;
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  const auto* src = reinterpret_cast<const uint32_t*>(a.data());
  detail::scale_limbs(dst, src, a.size(), c);
  return r;
}

/// a << k  (multiply by x^k)
template <class M>
vec<M> shift(const vec<M>& a, size_t k, size_t n = SIZE_MAX) {
  vec<M> r(std::min(a.size() + k, n), M());
  for (size_t i = 0; i < a.size() && i + k < r.size(); ++i) r[i + k] = a[i];
  return r;
}

/// Horner evaluation at a point.
template <class M>
M eval(const vec<M>& a, M x) {
  M r = M();
  for (size_t i = a.size(); i-- > 0;) r = r * x + a[i];
  return r;
}

/// ---------------------------------------------------------------------------
/// convolution
/// ---------------------------------------------------------------------------

/// Internal: truncated product of two Montgomery-limb buffers. Both inputs are
/// already reduced to at most `lim` coefficients by the caller.
template <class M>
vec<M> conv_limbs(const vec<M>& a, const vec<M>& b, size_t lim) {
  if (a.empty() || b.empty() || lim == 0) return {};
  const size_t la = std::min(a.size(), lim), lb = std::min(b.size(), lim);
  const size_t full = la + lb - 1;
  const size_t out = std::min(full, lim);
  static_assert(sizeof(M) == sizeof(uint32_t), "Mont must be a single 32-bit limb");

  if (std::min(la, lb) <= naive_threshold) {
    vec<M> r(out, M());
    // Finish each consecutive output tile while it stays in L1. For tiles
    // after the first, every shifted accumulation starts at the same output
    // address, avoiding repeatedly misaligned stores and full-size passes.
    const vec<M>& small = la <= lb ? a : b;
    const vec<M>& large = la <= lb ? b : a;
    const size_t short_n = std::min(la, lb), long_n = std::max(la, lb);
    auto* dst = reinterpret_cast<uint32_t*>(r.data());
    const auto* src = reinterpret_cast<const uint32_t*>(large.data());
    if (short_n == 1) {
      detail::scale_limbs(dst, src, out, small[0]);
      return r;
    }
    constexpr size_t tile = 4096;
    for (size_t base = 0; base < out; base += tile) {
      const size_t end = std::min(out, base + tile);
      for (size_t i = 0; i < short_n && i < end; ++i) {
        if (small[i].is_zero()) continue;
        const size_t begin = std::max(base, i);
        const size_t limit = std::min(end, i + long_n);
        const auto c = simd::fixed_twiddle<M::mod, M::ninv>(small[i].raw_val());
        size_t j = begin - i, cnt = limit - i;
        for (; j + simd::lane <= cnt; j += simd::lane) {
          const auto v = simd::mul_twiddle_full<M::mod, M::ninv>(simd::load(src + j), c);
          simd::store(dst + i + j, simd::add(simd::load(dst + i + j), v, M::mod));
        }
        for (; j < cnt; ++j) r[i + j] += small[i] * large[j];
      }
    }
    return r;
  }

  const uint32_t N = next_pow2(full);
  const auto& plan = NttPlan<M>::get_ref(N);
  vec<M> fa;
  fa.reserve(N);
  fa.insert(fa.end(), a.data(), a.data() + la);
  fa.resize(N);  // Only the padding is initialized, after copying the prefix.
  uint32_t* pa = reinterpret_cast<uint32_t*>(fa.data());
  plan.forward_lazy(pa);
  if (a.data() == b.data() && la == lb) {
    pointwise_mul<M>(pa, pa, N);
  } else {
    fpx::detail::ScratchBuffer fb;
    detail::prepare_scratch(b, lb, fb, N);
    auto* pb = fb.data();
    plan.forward_lazy(pb);
    pointwise_mul<M>(pa, pb, N);
  }
  plan.inverse(pa);
  fa.resize(out);
  return fa;
}

/// Product of a and b, truncated to n coefficients (n = SIZE_MAX -> full).
template <class M>
vec<M> conv(const vec<M>& a, const vec<M>& b, size_t n = SIZE_MAX) {
  return conv_limbs(a, b, n);
}

/// ---------------------------------------------------------------------------
/// derivative / integral
/// ---------------------------------------------------------------------------

namespace detail {

/// Derivative of only the first cnt input coefficients.
template <class M>
vec<M> derivative_prefix(const vec<M>& a, size_t cnt) {
  cnt = std::min(a.size(), cnt);
  if (cnt <= 1) return {};
  vec<M> r(cnt - 1);
  alignas(64) uint32_t factors[simd::lane];
  for (int k = 0; k < simd::lane; ++k) factors[k] = M::from_int(uint32_t(k + 1)).raw_val();
  auto factor = simd::load(factors);
  const auto stride = simd::set1(M::from_int(simd::lane).raw_val());
  const auto* src = reinterpret_cast<const uint32_t*>(a.data());
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  size_t i = 1;
  for (; i + simd::lane <= cnt; i += simd::lane) {
    simd::store(dst + i - 1, simd::mulmod<M::mod, M::ninv>(simd::load(src + i), factor));
    factor = simd::add(factor, stride, M::mod);
  }
  for (; i < cnt; ++i) r[i - 1] = a[i] * M::from_int(i);
  return r;
}

}  // namespace detail

template <class M>
vec<M> derivative(const vec<M>& a) {
  return detail::derivative_prefix(a, a.size());
}

/// Montgomery forms of 1/1, 1/2, ..., 1/n using a linear recurrence.
/// Requires n < mod. Quotient and remainder share a 32-bit integer division.
/// Multiplication by the ordinary quotient preserves the raw inverse's
/// Montgomery factor, needing just one constant-modulus reduction per element.
template <class M>
vec<M> inv_series(size_t n) {
  if (n >= M::mod) throw domain_error("poly::inv_series: length must be < mod");
  vec<M> r(n + 1, M());
  if (n == 0) return r;
  r[1] = M::from_int(1);
  const uint32_t count = static_cast<uint32_t>(n);
  uint32_t i = 2;
  for (; i <= std::min(count, uint32_t(65536)); ++i) {
    // 1/i = -(mod/i) * (1/(mod%i)) mod mod
    const uint32_t q = M::mod / i, rem = M::mod % i;
    const uint32_t product = static_cast<uint32_t>(
        (uint64_t(q) * r[rem].raw_val()) % M::mod);
    // With prime mod and 1 < i < mod, q and the inverse are both nonzero.
    r[i] = M::raw(M::mod - product);
  }
  // Past this point mod/i stays constant over several consecutive indices.
  // All remainders decrease within an interval and precede its first index,
  // so these products are independent and need no per-element division.
  while (i <= count) {
    const uint32_t q = M::mod / i;
    uint32_t rem = M::mod % i;
    const uint32_t end = std::min(count, M::mod / q);
#if defined(FPX_SIMD_NEON)
    if constexpr (simd::compact_twiddle<M::mod>) {
      if (i + 3 <= end) {
        const auto factor = simd::fixed_twiddle<M::mod, M::ninv>(M::from_int(q).raw_val());
        auto* dst = reinterpret_cast<uint32_t*>(r.data());
        for (; i + 3 <= end; i += 4, rem -= 4 * q) {
          const uint32x4_t v{r[rem].raw_val(), r[rem-q].raw_val(),
                            r[rem-2*q].raw_val(), r[rem-3*q].raw_val()};
          const auto product = simd::mul_twiddle_full<M::mod, M::ninv>(v, factor);
          // Prime mod, 1 < i < mod: q and every inverse are nonzero.
          simd::store(dst + i, vsubq_u32(simd::set1(M::mod), product));
        }
      }
    }
#endif
    for (; i <= end; ++i, rem -= q) {
      const uint32_t product = static_cast<uint32_t>(uint64_t(q) * r[rem].raw_val() % M::mod);
      r[i] = M::raw(M::mod - product);
    }
  }
  return r;
}

template <class M>
vec<M> integral(const vec<M>& a) {
  if (a.empty()) return {};
  vec<M> inv = inv_series<M>(a.size());
  vec<M> r(a.size() + 1);
  const auto* src = reinterpret_cast<const uint32_t*>(a.data());
  const auto* divisors = reinterpret_cast<const uint32_t*>(inv.data());
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  size_t i = 0;
  for (; i + simd::lane <= a.size(); i += simd::lane) {
    simd::store(dst + i + 1, simd::mulmod<M::mod, M::ninv>(
                               simd::load(src + i), simd::load(divisors + i + 1)));
  }
  for (; i < a.size(); ++i) r[i + 1] = a[i] * inv[i + 1];
  return r;
}

/// ---------------------------------------------------------------------------
/// inverse   (requires a(0) != 0)
/// ---------------------------------------------------------------------------

namespace detail {

// Extend an inverse prefix c to wanted <= 2*c.size(), using only two
// scratch buffers and preserving the c spectrum through the correction.
template <class M>
void extend_inverse(const vec<M>& a, vec<M>& c, size_t wanted,
                  fpx::detail::ScratchBuffer& scratch,
                  fpx::detail::ScratchBuffer& spectrum) {
  const size_t k = c.size(), hlen = wanted - k;
  // A handful of coefficients after a power-of-two boundary costs less than
  // another full transform. The new coefficients are filled in order.
  if (hlen <= 8) {
    const M inv0 = c[0];
    c.resize(wanted);
    for (size_t i = k; i < wanted; ++i) {
      M sum;
      for (size_t j = 1; j <= i && j < a.size(); ++j) sum += a[j]*c[i-j];
      c[i] = -sum*inv0;
    }
    return;
  }
  const uint32_t N = next_pow2(2*k);
  const auto& plan = NttPlan<M>::get_ref(N);
  prepare_scratch(a, std::min(a.size(), wanted), scratch, N);
  prepare_scratch(c, c.size(), spectrum, N);
  auto* pt = scratch.data();
  auto* pc = spectrum.data();
  plan.forward_lazy(pt);
  plan.forward_lazy(pc);
  pointwise_mul<M>(pt, pc, N);
  plan.inverse(pt);
  size_t i = 0;
  const auto zero = simd::set1(0);
  for (; i + simd::lane <= hlen; i += simd::lane)
    simd::store(pt + i, simd::sub(zero, simd::load(pt + k + i), M::mod));
  for (; i < hlen; ++i) pt[i] = pt[k+i] == 0 ? 0 : M::mod - pt[k+i];
  std::memset(pt + hlen, 0, (N - hlen) * sizeof(uint32_t));
  plan.forward_lazy(pt);
  pointwise_mul<M>(pt, pc, N);
  plan.inverse(pt);
  c.resize(wanted);
  std::memcpy(c.data() + k, pt, hlen * sizeof(uint32_t));
}

template <class M>
vec<M> inverse_seed(const vec<M>& a, size_t n) {
  vec<M> c(n, M());
  c[0] = a[0].inv();
  for (size_t i = 1; i < n; ++i) {
    M sum;
    for (size_t j = 1; j <= i && j < a.size(); ++j) sum += a[j]*c[i-j];
    c[i] = -sum*c[0];
  }
  return c;
}

// A linear reciprocal is a geometric progression. Four independent SIMD
// chains advance by ratio^(4*lane), avoiding a dependency between neighbours.
template <class M>
vec<M> inverse_linear(const vec<M>& a, size_t n) {
  const M first = a[0].inv(), ratio = -a[1]*first;
  if (ratio.raw_val() == M::one) return vec<M>(n, first);
  vec<M> b(n);
  constexpr size_t W = simd::lane, block = 4*W;
  if (n < block) {
    M x = first;
    for (size_t i = 0; i < n; ++i) { b[i] = x; x *= ratio; }
    return b;
  }
  alignas(64) uint32_t seed[block];
  M x = first;
  for (size_t i = 0; i < block; ++i) { seed[i] = x.raw_val(); x *= ratio; }
  auto x0 = simd::load(seed), x1 = simd::load(seed + W);
  auto x2 = simd::load(seed + 2*W), x3 = simd::load(seed + 3*W);
  const M step_value = ratio.pow(block);
  const auto step = simd::fixed_twiddle<M::mod, M::ninv>(step_value.raw_val());
  auto* dst = reinterpret_cast<uint32_t*>(b.data());
  size_t i = 0;
  if (step_value.raw_val() == M::one) {
    // A short-period geometric progression repeats the already canonical seed.
    for (; i + block <= n; i += block) std::memcpy(dst + i, seed, sizeof(seed));
    if (i != n) std::memcpy(dst + i, seed, (n - i) * sizeof(uint32_t));
    return b;
  }
  auto store_value = [](uint32_t* p, simd::native_t v) {
    if constexpr (simd::fixed_lazy<M::mod>) v = simd::reduce_full(v, M::mod);
    simd::store(p, v);
  };
  auto advance = [&](simd::native_t v) {
    if constexpr (simd::fixed_lazy<M::mod>) return simd::mul_twiddle<M::mod, M::ninv>(v, step);
    else return simd::mul_twiddle_full<M::mod, M::ninv>(v, step);
  };
  for (; i + block <= n; i += block) {
    store_value(dst + i, x0); store_value(dst + i + W, x1);
    store_value(dst + i + 2*W, x2); store_value(dst + i + 3*W, x3);
    x0 = advance(x0); x1 = advance(x1);
    x2 = advance(x2); x3 = advance(x3);
  }
  if (i != n) {
    store_value(seed, x0); store_value(seed + W, x1);
    store_value(seed + 2*W, x2); store_value(seed + 3*W, x3);
    std::memcpy(dst + i, seed, (n - i) * sizeof(uint32_t));
  }
  return b;
}

// Normalize the fixed short input once, then each new coefficient is a dot
// product of at most eight neighbours; no transform or scratch is required.
template <class M>
vec<M> inverse_short(const vec<M>& a, size_t cnt, size_t n) {
  vec<M> b(n);
  b[0] = a[0].inv();
  std::array<M, 8> coefficients;
  const size_t degree = cnt - 1;
  for (size_t j = 0; j < degree; ++j) coefficients[j] = -a[j+1]*b[0];
  for (size_t i = 1; i < n; ++i) {
    M sum;
    for (size_t j = 0; j < std::min(i, degree); ++j)
      sum += coefficients[j]*b[i-j-1];
    b[i] = sum;
  }
  return b;
}

#if defined(FPX_SIMD_NEON)
// Inclusive four-lane product scan, with Montgomery one in the shifted lanes.
template <class M>
simd::native_t prefix_product4(simd::native_t x) {
  const auto one = simd::set1(M::one);
  x = simd::butterfly_mul<M::mod, M::ninv, simd::lazy_ok<M::mod>>(
      x, vextq_u32(one, x, 3));
  return simd::butterfly_mul<M::mod, M::ninv, simd::lazy_ok<M::mod>>(
      x, vextq_u32(one, x, 2));
}

// exp(c*x)[i] = c^i/i!. Compute the last coefficient using one factorial
// inversion, then generate the output backwards. Four local scans and a scan
// of their endpoints expose sixteen independent coefficients per block; only
// the scalar endpoint carries a dependency between successive blocks.
template <class M>
vec<M> exp_linear(M c, size_t n) {
  constexpr size_t block = 16;
  const size_t degree = n - 1, full = degree / block * block;
  alignas(64) uint32_t seed[block];
  for (size_t k = 0; k < block; ++k) seed[k] = M::from_int(k + 1).raw_val();
  simd::native_t products[4], factors[4];
  const auto stride = simd::set1(M::from_int(block).raw_val());
  for (size_t j = 0; j < 4; ++j) {
    products[j] = simd::set1(M::one);
    factors[j] = simd::load(seed + 4*j);
  }
  // Sixteen independent progressions compute degree! without a prefix table.
  for (size_t i = 0; i < full; i += block) {
    for (size_t j = 0; j < 4; ++j) {
      products[j] = simd::butterfly_mul<M::mod, M::ninv, simd::lazy_ok<M::mod>>(products[j], factors[j]);
      factors[j] = simd::add(factors[j], stride, M::mod);
    }
  }
  for (size_t j = 0; j < 4; ++j) simd::store(seed + 4*j, products[j]);
  M factorial = M::from_int(1);
  for (const uint32_t x : seed) factorial *= M::raw(x);
  for (size_t i = full + 1; i <= degree; ++i) factorial *= M::from_int(i);
  M previous = c.pow(degree) * factorial.inv();
  const M inverse_c = c.inv();
  vec<M> b(n);
  b[degree] = previous;
  auto* dst = reinterpret_cast<uint32_t*>(b.data());
  for (size_t k = 0; k < block; ++k) seed[k] = (M::from_int(degree-k)*inverse_c).raw_val();
  for (size_t j = 0; j < 4; ++j) factors[j] = simd::load(seed + 4*j);
  const auto decrement = simd::set1((M::from_int(block)*inverse_c).raw_val());
  size_t pos = degree;
  for (; pos >= block; pos -= block) {
    auto x0 = prefix_product4<M>(factors[0]), x1 = prefix_product4<M>(factors[1]);
    auto x2 = prefix_product4<M>(factors[2]), x3 = prefix_product4<M>(factors[3]);
    const uint32x4_t ends{vgetq_lane_u32(x0, 3), vgetq_lane_u32(x1, 3),
                          vgetq_lane_u32(x2, 3), vgetq_lane_u32(x3, 3)};
    const auto joined = prefix_product4<M>(ends);
    const auto scales = simd::butterfly_mul<M::mod, M::ninv, simd::lazy_ok<M::mod>>(
        simd::set1(previous.raw_val()), vextq_u32(simd::set1(M::one), joined, 3));
    x0 = simd::mulmod<M::mod, M::ninv>(x0, vdupq_laneq_u32(scales, 0));
    x1 = simd::mulmod<M::mod, M::ninv>(x1, vdupq_laneq_u32(scales, 1));
    x2 = simd::mulmod<M::mod, M::ninv>(x2, vdupq_laneq_u32(scales, 2));
    x3 = simd::mulmod<M::mod, M::ninv>(x3, vdupq_laneq_u32(scales, 3));
    previous = M::raw(vgetq_lane_u32(x3, 3));
    auto reverse = [](simd::native_t x) { x = vrev64q_u32(x); return vextq_u32(x, x, 2); };
    simd::store(dst + pos - 4, reverse(x0)); simd::store(dst + pos - 8, reverse(x1));
    simd::store(dst + pos - 12, reverse(x2)); simd::store(dst + pos - 16, reverse(x3));
    for (auto& factor : factors) factor = simd::sub(factor, decrement, M::mod);
  }
  for (; pos != 0; --pos) {
    previous *= M::from_int(pos)*inverse_c;
    b[pos-1] = previous;
  }
  return b;
}
#endif

}  // namespace detail

/// b = a^-1 mod x^n.
///
/// Newton iteration with the *middle product* optimisation: at step m -> m2
/// (m2 <= 2m) we only need the high half of a*b, and that half is free of
/// cyclic aliasing at transform size 2m, so the transform stays at 2m instead
/// of 4m -- a 2x reduction in transform length per iteration.
template <class M>
vec<M> inv(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (a.empty() || a[0].is_zero())
    throw domain_error("poly::inv: constant term is zero");
  size_t cnt = std::min(a.size(), n);
  while (cnt > 1 && a[cnt-1].is_zero()) --cnt;
  if (cnt == 1) {  // Includes padded constants and a constant prefix mod x^n.
    vec<M> r(n, M());
    r[0] = a[0].inv();
    return r;
  }
  static_assert(sizeof(M) == sizeof(uint32_t), "Mont must be a single 32-bit limb");
  if (cnt == 2) return detail::inverse_linear(a, n);
  if (cnt <= 5 || (cnt <= 9 && n >= 4096))
    return detail::inverse_short(a, cnt, n);

  const size_t seed = std::min(n, size_t(32));
  const uint32_t scratch_n = detail::series_scratch_size<M>(n, seed);
  vec<M> b = detail::inverse_seed(a, seed);
  b.reserve(n);
  fpx::detail::ScratchBuffer work(scratch_n), spectrum(scratch_n);
  for (size_t m = seed; m < n; m <<= 1)
    detail::extend_inverse(a, b, std::min(2 * m, n), work, spectrum);
  b.resize(n);
  return b;
}

/// log(a) = integral(a' / a) mod x^n   (requires a(0) == 1)
template <class M>
vec<M> log(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (a.empty() || a[0] != M::from_int(1))
    throw domain_error("poly::log: constant term must be 1");
  if (n == 1) return vec<M>(1, M());
  vec<M> d = detail::derivative_prefix(a, n);
  vec<M> ia = inv(a, n - 1);
  vec<M> p = conv_limbs(d, ia, n - 1);  // (a'/a) mod x^(n-1)
  vec<M> r = integral(p);
  r.resize(n, M());
  return r;
}

/// exp(a) mod x^n   (requires a(0) == 0)
///
/// Solve b' = a' b using the high differential residual at each doubling.
/// Dividing it by b and integrating gives (a-log b)[m,m2); multiplying only
/// this new half by b updates the result at transform size 2m. Keep the inverse
/// prefix between steps instead of restarting an inverse/logarithm each time.
template <class M>
vec<M> exp(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (!a.empty() && !a[0].is_zero())
    throw domain_error("poly::exp: constant term must be 0");
  if (n-1 >= M::mod)
    throw domain_error("poly::inv_series: length must be < mod");
  vec<M> aprime = detail::derivative_prefix(a, n);
  trim(aprime);
  if (aprime.empty()) {
    vec<M> b(n, M());
    b[0] = M::from_int(1);
    return b;
  }
#if defined(FPX_SIMD_NEON)
  if (aprime.size() == 1 && n >= 256) return detail::exp_linear(aprime[0], n);
#endif
  // Short input series admit an O(n*degree(a)) recurrence; in particular exp
  // of a linear polynomial is a single contiguous pass.
  const size_t seed = aprime.size() <= naive_threshold ? n : std::min<size_t>(n, 32);
  const uint32_t scratch_n = detail::series_scratch_size<M>(n, seed);
  vec<M> denominators = inv_series<M>(n-1);
  vec<M> b(seed, M());
  b[0] = M::from_int(1);
  if (aprime.size() == 1) {
    // exp(c*x)[i] = exp(c*x)[i-1] * (c/i). Compute c/i independently
    // from the previous coefficient, shortening the dependent multiply chain.
    M previous = b[0];
    const M factor = aprime[0];
    for (size_t i = 1; i < n; ++i) {
      previous *= factor*denominators[i];
      b[i] = previous;
    }
    return b;
  }
  for (size_t i = 1; i < seed; ++i) {
    M sum;
    for (size_t j = 1; j <= i && j <= aprime.size(); ++j)
      sum += aprime[j-1]*b[i-j];
    b[i] = sum*denominators[i];
  }
  if (n == seed) return b;
  b.reserve(n);
  vec<M> c = detail::inverse_seed(b, seed/2);
  c.reserve(n/2);
  fpx::detail::ScratchBuffer spectrum(scratch_n), inverse_spectrum(scratch_n), work(scratch_n);
  for (size_t m = seed; m < n; m *= 2) {
    const size_t m2 = std::min(2*m, n), hlen = m2-m;
    if (hlen <= 8) {
      b.resize(m2);
      for (size_t i = m; i < m2; ++i) {
        M sum;
        for (size_t j = 1; j <= i && j <= aprime.size(); ++j)
          sum += aprime[j-1]*b[i-j];
        b[i] = sum*denominators[i];
      }
      continue;
    }
    if (c.size() < hlen)
      detail::extend_inverse(b, c, hlen, work, inverse_spectrum);
    const uint32_t N = next_pow2(2*m);
    const auto& plan = NttPlan<M>::get_ref(N);
    detail::prepare_scratch(b, b.size(), spectrum, N);
    detail::prepare_scratch(c, c.size(), inverse_spectrum, N);
    detail::prepare_scratch(aprime, std::min(aprime.size(), m2-1), work, N);
    auto* pb = spectrum.data();
    auto* pc = inverse_spectrum.data();
    auto* pt = work.data();
    plan.forward_lazy(pb);
    plan.forward_lazy(pt);
    pointwise_mul<M>(pt, pb, N);
    plan.inverse(pt);
    // a' b - b' is zero below degree m-1.  The wanted high range is
    // untouched by cyclic wraparound: deg(a'b) <= 3m-3.
    std::memmove(pt, pt + m - 1, hlen * sizeof(uint32_t));
    std::memset(pt + hlen, 0, (N - hlen) * sizeof(uint32_t));
    plan.forward_lazy(pt);
    plan.forward_lazy(pc);
    pointwise_mul<M>(pt, pc, N);
    plan.inverse(pt);
    // Integrate the shifted high differential residual.
    pointwise_mul<M>(pt, reinterpret_cast<const uint32_t*>(denominators.data()) + m, hlen);
    std::memset(pt + hlen, 0, (N - hlen) * sizeof(uint32_t));
    plan.forward_lazy(pt);
    pointwise_mul<M>(pt, pb, N);
    plan.inverse(pt);
    b.resize(m2);
    std::memcpy(b.data() + m, pt, hlen * sizeof(uint32_t));
  }
  return b;
}


/// Square root mod x^n.  Throws `domain_error` if a(0) is a non-residue or the
/// x-adic valuation of a is odd (then no square root exists in GF(Mod)[[x]]).
template <class M>
vec<M> sqrt(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  size_t v = 0;
  while (v < a.size() && a[v].is_zero()) ++v;
  if (v == a.size()) return vec<M>(n, M());
  if (v & 1) throw domain_error("poly::sqrt: odd valuation, no square root");
  M root0;
  if (!a[v].sqrt(root0))
    throw domain_error("poly::sqrt: non-residue constant term");
  const size_t half = v/2;
  if (half >= n) return vec<M>(n, M());
  const size_t inner_n = n-half;
  const size_t ucnt = std::min(a.size()-v, inner_n);
  if (std::all_of(a.data()+v+1, a.data()+v+ucnt,
                  [](const M& x) { return x.is_zero(); })) {
    vec<M> r(n, M());
    r[half] = root0;
    return r;
  }
  const size_t seed = std::min<size_t>(inner_n, 32);
  const uint32_t scratch_n = detail::series_scratch_size<M>(inner_n, seed);
  const M half_inv = M::from_int(M::inv2);
  const M twice_root_inv = root0.inv()*half_inv;
  vec<M> b(seed, M());
  b[0] = root0;
  for (size_t i = 1; i < seed; ++i) {
    M residual = i < a.size()-v ? a[v+i] : M();
    for (size_t j = 1; j < i; ++j) residual -= b[j]*b[i-j];
    b[i] = residual*twice_root_inv;
  }
  if (inner_n > seed) {
    b.reserve(inner_n);
    vec<M> c = detail::inverse_seed(b, seed/2);
    c.reserve(inner_n/2);
    fpx::detail::ScratchBuffer work(scratch_n), inverse_spectrum(scratch_n);
    for (size_t m = seed; m < inner_n; m *= 2) {
      const size_t m2 = std::min(2*m, inner_n), hlen = m2-m;
      if (hlen <= 8) {
        b.resize(m2);
        for (size_t i = m; i < m2; ++i) {
          M residual = i < a.size()-v ? a[v+i] : M();
          for (size_t j = 1; j < i; ++j) residual -= b[j]*b[i-j];
          b[i] = residual*twice_root_inv;
        }
        continue;
      }
      if (c.size() < hlen)
        detail::extend_inverse(b, c, hlen, work, inverse_spectrum);
      const uint32_t N = next_pow2(2*m);
      const auto& plan = NttPlan<M>::get_ref(N);
      detail::prepare_scratch(b, b.size(), work, N);
      detail::prepare_scratch(c, c.size(), inverse_spectrum, N);
      auto* pt = work.data();
      auto* pc = inverse_spectrum.data();
      plan.forward_lazy(pt);
      pointwise_mul<M>(pt, pt, N);
      plan.inverse(pt);
      for (size_t i = 0; i < hlen; ++i)
        pt[i] = (((m+i < a.size()-v) ? a[v+m+i] : M()) - M::raw(pt[m+i])).raw_val();
      std::memset(pt + hlen, 0, (N - hlen) * sizeof(uint32_t));
      plan.forward_lazy(pt);
      plan.forward_lazy(pc);
      pointwise_mul<M>(pt, pc, N);
      plan.inverse(pt);
      b.resize(m2);
      auto* dst = reinterpret_cast<uint32_t*>(b.data() + m);
      detail::scale_limbs(dst, pt, hlen, half_inv);
    }
  }
  if (half == 0) return b;
  vec<M> r(n, M());
  std::copy(b.begin(), b.end(), r.data() + half);
  return r;
}

/// a^k mod x^n for an arbitrary 64-bit exponent.
///
/// Uses direct binary exponentiation for small exponents and the
/// exp(k*log(a)) route otherwise.  The logarithm is legitimate for any integer
/// k: on the normalized series (1 + O(x)) the j-th coefficient of (1+u)^k is
/// binom(k, j) * ..., which depends on k only through k mod Mod, and j < n <= p.
template <class M>
vec<M> pow(const vec<M>& a, uint64_t k, size_t n) {
  if (n == 0) return {};
  if (k == 0) {
    vec<M> r(n, M());
    r[0] = M::from_int(1);
    return r;
  }
  if (k == 1) {
    vec<M> r(n, M());
    std::copy_n(a.begin(), std::min(a.size(), n), r.begin());
    return r;
  }
  size_t v = 0;
  while (v < a.size() && a[v].is_zero()) ++v;
  if (v == a.size()) return vec<M>(n, M());
  // x^(v*k) shift: bail out if the result is already >= x^n
  if (v != 0 && k > (n - 1) / v) return vec<M>(n, M());
  const size_t sh = static_cast<size_t>(v * k);
  const size_t inner_n = n - sh;
  vec<M> u(std::min(a.size() - v, inner_n));
  copy_range(a, v, u.size(), u);
  trim(u);
  if (u.size() == 1) {
    vec<M> r(n, M());
    r[sh] = u[0].pow(k);
    return r;
  }
  if (k > 64 && u.size() == 2 && inner_n <= M::mod) {
    // For j < p, binomial(k,j) depends only on k mod p. Coefficients
    // beyond that residue vanish, and all needed factorials are invertible.
    const uint32_t exponent = static_cast<uint32_t>(k % M::mod);
    const size_t cnt = std::min(inner_n, size_t(exponent) + 1);
    vec<M> divisors = inv_series<M>(cnt - 1);
    vec<M> r(n, M());
    r[sh] = u[0].pow(k);
    const M ratio = u[1]*u[0].inv();
    M factor = ratio*M::from_int(exponent);
    for (size_t j = 1; j < cnt; ++j) {
      r[sh+j] = r[sh+j-1]*factor*divisors[j];
      factor -= ratio;
    }
    return r;
  }

  if (k <= 64) {  // binary exponentiation
    vec<M> r{M::from_int(1)}, base = std::move(u);
    for (uint64_t e = k; e; e >>= 1) {
      // Keep the actual support until the final result: early powers of a
      // short polynomial need neither padded operands nor large transforms.
      if (e & 1) r = conv_limbs(r, base, inner_n);
      if (e > 1) base = conv_limbs(base, base, inner_n);
    }
    r.resize(inner_n);
    if (sh == 0) return r;
    vec<M> out(n, M());
    for (size_t i = 0; i < r.size() && i + sh < n; ++i) out[i + sh] = r[i];
    return out;
  }

  const M c = u[0];
  vec<M> nrm = std::move(u);
  detail::scale_inplace(nrm, c.inv());  // nrm(0) == 1
  vec<M> l = log(nrm, inner_n);
  detail::scale_inplace(l, M::from_int(k));  // k mod Mod is the correct exponent
  vec<M> r = exp(l, inner_n);
  detail::scale_inplace(r, c.pow(k));
  if (sh == 0) return r;
  vec<M> out(n, M());
  for (size_t i = 0; i < r.size() && i + sh < n; ++i) out[i + sh] = r[i];
  return out;
}

/// Euclidean division: a = b*q + r with deg(r) < deg(b). Both results trimmed.
template <class M>
std::pair<vec<M>, vec<M>> divmod(const vec<M>& a_in, const vec<M>& b_in) {
  size_t an = a_in.size(), bn = b_in.size();
  while (an != 0 && a_in[an-1].is_zero()) --an;
  while (bn != 0 && b_in[bn-1].is_zero()) --bn;
  if (bn == 0) throw domain_error("poly::divmod: division by zero polynomial");
  if (an < bn)
    return {vec<M>{}, vec<M>(a_in.begin(), a_in.begin() + static_cast<std::ptrdiff_t>(an))};
  if (bn == 1) {
    vec<M> q(an);
    detail::scale_limbs(reinterpret_cast<uint32_t*>(q.data()),
                        reinterpret_cast<const uint32_t*>(a_in.data()), an, b_in[0].inv());
    return {std::move(q), vec<M>{}};
  }
  const size_t dn = an - bn + 1;
  // q = rev( rev(a) * inv(rev(b)) ) mod x^dn
  vec<M> ra(std::make_reverse_iterator(a_in.data() + an),
            std::make_reverse_iterator(a_in.data() + an - dn));
  vec<M> rb(std::make_reverse_iterator(b_in.data() + bn),
            std::make_reverse_iterator(b_in.data() + bn - std::min(bn, dn)));
  vec<M> ib = inv(rb, dn);
  vec<M> rq = conv_limbs(ra, ib, dn);
  rq.resize(dn, M());
  std::reverse(rq.begin(), rq.end());
  vec<M> q = std::move(rq);
  // Once q is known, the high coefficients of a-b*q cancel exactly. Compute
  // only the low remainder instead of transforming those cancelling terms.
  const size_t rn = bn - 1;
  vec<M> r(a_in.data(), a_in.data() + rn);
  vec<M> product = conv_limbs(b_in, q, rn);
  auto* dst = reinterpret_cast<uint32_t*>(r.data());
  const auto* src = reinterpret_cast<const uint32_t*>(product.data());
  size_t i = 0;
  for (; i + simd::lane <= product.size(); i += simd::lane)
    simd::store(dst + i, simd::sub(simd::load(dst + i), simd::load(src + i), M::mod));
  for (; i < product.size(); ++i) r[i] -= product[i];
  trim(r);
  return {std::move(q), std::move(r)};
}

template <class M>
vec<M> div(const vec<M>& a, const vec<M>& b) { return divmod(a, b).first; }

template <class M>
vec<M> mod(const vec<M>& a, const vec<M>& b) { return divmod(a, b).second; }

}  // namespace poly

/// ---------------------------------------------------------------------------
/// public polynomial type: std::vector<M> plus method syntax
/// ---------------------------------------------------------------------------

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

  using base::size;
  using base::operator[];

  size_t degree() const { return this->empty() ? 0 : this->size() - 1; }
  bool is_zero() const {
    for (const auto& x : *this)
      if (!x.is_zero()) return false;
    return true;
  }

  Poly& trim() { poly::trim(*this); return *this; }
  Poly& resize_(size_t n) { this->resize(n); return *this; }

  Poly operator+(const base& b) const { return poly::add(*this, b); }
  Poly operator-(const base& b) const { return poly::sub(*this, b); }
  Poly operator-() const { return poly::neg(*this); }
  Poly operator*(const base& b) const { return poly::conv(*this, b); }
  Poly operator*(M c) const { return poly::mul_scalar(*this, c); }
  Poly operator<<(size_t k) const { return poly::shift(*this, k); }
  Poly& operator+=(const base& b) { return *this = poly::add(*this, b); }
  Poly& operator-=(const base& b) { return *this = poly::sub(*this, b); }
  Poly& operator*=(const base& b) { return *this = poly::conv(*this, b); }

  /// Truncated product (mod x^n).
  Poly mul(const base& b, size_t n) const { return poly::conv(*this, b, n); }
  Poly derivative() const { return poly::derivative(*this); }
  Poly integral() const { return poly::integral(*this); }
  /// Series inverse mod x^n (a(0) must be invertible).
  Poly inv(size_t n) const { return poly::inv(*this, n); }
  /// Series logarithm mod x^n (a(0) must be 1).
  Poly log(size_t n) const { return poly::log(*this, n); }
  /// Series exponential mod x^n (a(0) must be 0).
  Poly exp(size_t n) const { return poly::exp(*this, n); }
  /// Series square root mod x^n.
  Poly sqrt(size_t n) const { return poly::sqrt(*this, n); }
  /// Series power mod x^n.
  Poly pow(uint64_t k, size_t n) const { return poly::pow(*this, k, n); }
  std::pair<Poly, Poly> divmod(const base& b) const {
    auto r = poly::divmod(*this, b);
    return {Poly(std::move(r.first)), Poly(std::move(r.second))};
  }
  Poly div(const base& b) const { return poly::div(*this, b); }
  Poly mod(const base& b) const { return poly::mod(*this, b); }
  M eval(M x) const { return poly::eval(*this, x); }

  /// Convenience: coefficients of this polynomial as a flat vector of ints.
  std::vector<uint32_t> to_ints() const {
    std::vector<uint32_t> r(this->size());
    for (size_t i = 0; i < this->size(); ++i) r[i] = (*this)[i].val();
    return r;
  }
  /// Build a polynomial from plain integers.
  static Poly from_ints(const std::vector<uint32_t>& v) {
    Poly r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = M::from_int(v[i]);
    return r;
  }
};

}  // namespace fpx

#endif  // FASTPOLY_POLY_HPP
