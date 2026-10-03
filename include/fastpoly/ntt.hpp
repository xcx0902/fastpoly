// fastpoly - number theoretic transform over GF(Mod), Mod a 32-bit NTT prime.
//
// Design notes
// ------------
// * Forward transform is a **decimation-in-frequency (DIF)** radix-4 / radix-2
//   pass sequence; the inverse is the matching **decimation-in-time (DIT)**
//   sequence.  Composing them yields n * identity, so *no bit-reversal
//   permutation is ever needed* - the pointwise products are all evaluated in
//   the same permuted (base-4 digit reversed) order.
// * Radix-4 folds two radix-2 levels into one butterfly: same number of
//   modular multiplications, but half the passes over memory and half the
//   twiddle traffic.
// * Twiddles are precomputed per stage into *contiguous* arrays (w^k, w^2k,
//   w^3k) so the SIMD kernel can issue straight vector loads.
// * Butterflies are Montgomery multiplications (see simd.hpp), so the whole
//   transform is divide-free and vectorises to AVX2 / AVX-512 / NEON.
//
// Lazy reduction (Mod < 2^30)
// ---------------------------
// Every intermediate lives in [0, R) with R = 2*Mod instead of [0, Mod).  Because
// 4*Mod^2 < Mod*2^32 in that range, a Montgomery product of two values < R lands
// back in [0, R) *without* the final conditional subtraction, which the butterfly
// therefore drops on all four of its multiplications.  Additions/subtractions
// reduce modulo R (still a valid residue, since R is a multiple of Mod).
// Both entry points canonicalise their output, so the public contract is
// unchanged: `forward` and `inverse` return values in [0, Mod).
#ifndef FASTPOLY_NTT_HPP
#define FASTPOLY_NTT_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "fastpoly/modint.hpp"
#include "fastpoly/simd.hpp"

namespace fpx {

/// Largest power of two dividing Mod - 1: an NTT of size 2^k exists for k <= this.
template <class M>
constexpr int ntt_max_log() {
  uint64_t t = M::mod - 1;
  int k = 0;
  while (t % 2 == 0) { t /= 2; ++k; }
  return k;
}
template <class M>
constexpr uint32_t ntt_max_size() { return uint32_t(1) << ntt_max_log<M>(); }

/// Smallest power of two >= need (need >= 1).
inline uint32_t next_pow2(uint64_t need) {
  uint32_t n = 1;
  while (n < need) n <<= 1;
  return n;
}

/// Thrown when a transform size is not representable for the modulus.
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

}  // namespace detail

/// Precomputed transform plan for one size. Instances are shared and immutable.
template <class M>
class NttPlan {
 public:
  using simd_t = simd::native_t;

  static constexpr uint32_t MOD = M::mod;
  static constexpr uint32_t NINV = M::ninv;
  /// Reduction modulus of the lazy convention: R = 2*Mod when that is provably
  /// safe (Mod < 2^30), otherwise R = Mod (fully reduced everywhere).
  static constexpr uint32_t R = simd::rmod<MOD>;
  static constexpr bool LAZY = simd::lazy_ok<MOD>;

  /// Get (possibly cached) plan for size `n`, a power of two <= ntt_max_size<M>().
  static std::shared_ptr<const NttPlan> get(uint32_t n) {
    static std::mutex mu;
    static std::unordered_map<uint32_t, std::shared_ptr<const NttPlan>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(n);
    if (it != cache.end()) return it->second;
    auto p = std::shared_ptr<const NttPlan>(new NttPlan(n));
    cache.emplace(n, p);
    return p;
  }

  uint32_t size() const { return n_; }
  int log_size() const { return logn_; }

  /// Forward DIF transform, in place, on Montgomery limbs. Output in [0, Mod).
  void forward(uint32_t* a) const {
    const bool radix4_last = (logn_ & 1) == 0;
    for (size_t s = 0; s < lens_.size(); ++s) {
      const bool last = radix4_last && s + 1 == lens_.size();
      if (last) {
        radix4_stage<true>(a, lens_[s], s);
      } else {
        radix4_stage<false>(a, lens_[s], s);
      }
    }
    if (logn_ & 1) radix2_fwd<true>(a);
  }

  /// Inverse DIT transform (already scaled by 1/n), in place. Output in [0, Mod).
  void inverse(uint32_t* a) const {
    if (logn_ & 1) radix2_fwd<false>(a);  // len == 2 butterflies are self-transpose
    for (size_t s = lens_.size(); s-- > 0;) radix4_stage_inv(a, lens_[s], s);
    // Scale by 1/n. This is the last touch on the data, so it also restores the
    // canonical [0, Mod) representation that the lazy convention relaxed.
    const uint32_t in = inv_n_;
    const simd_t vin = simd::set1(in);
    uint32_t k = 0;
    for (; k + simd::lane <= n_; k += simd::lane) {
      simd_t v = simd::butterfly_mul<MOD, NINV, LAZY>(simd::load(a + k), vin);
      if constexpr (LAZY) v = simd::reduce_full(v, MOD);
      simd::store(a + k, v);
    }
    for (; k < n_; ++k) a[k] = M::reduce(uint64_t(a[k]) * in);
  }

 private:
  explicit NttPlan(uint32_t n) : n_(n) {
    if (n < 2 || (n & (n - 1)) != 0)
      throw ntt_size_error("NTT size must be a power of two");
    if (n > ntt_max_size<M>())
      throw ntt_size_error("NTT size exceeds 2^v2(mod-1) for this modulus");
    logn_ = 0;
    while ((uint32_t(1) << logn_) < n_) ++logn_;
    build();
  }

  /// One lazy Montgomery multiplication (the butterfly's four multiplies).
  static simd_t bmul(simd_t a, simd_t b) {
    return simd::butterfly_mul<MOD, NINV, LAZY>(a, b);
  }
  /// Sum whose only consumer is a multiply against a twiddle (< Mod). Under the
  /// lazy convention the bound 4*Mod^2 < Mod*2^32 lets it stay unreduced.
  static simd_t feed_add(simd_t a, simd_t b) {
    if constexpr (LAZY) {
      return simd::add_wide(a, b);
    } else {
      return simd::add(a, b, R);
    }
  }

  void build() {
    const M w_n_m = M::from_int(M::primitive_root).pow((MOD - 1) / n_);
    w4_ = w_n_m.pow(n_ / 4).raw_val();            // primitive 4th root of unity
    w4_inv_ = w_n_m.pow(n_ / 4).inv().raw_val();  // == -w4_

    // Radix-4 stages: len = n, n/4, n/16, ... (while len >= 4)
    for (uint32_t len = n_; len >= 4; len >>= 2) lens_.push_back(len);
    // Each stage stores w^k, w^2k, w^3k (and inverses) for k < len/4.
    size_t total = 0;
    for (uint32_t len : lens_) total += 3 * (len >> 2);
    fwd_.resize(total);
    inv_.resize(total);
    off_.resize(lens_.size() + 1, 0);
    size_t pos = 0;
    for (size_t s = 0; s < lens_.size(); ++s) {
      off_[s] = pos;
      const uint32_t m = lens_[s] >> 2;
      // w = root of order len = w_n^(n/len) = w_n^(4^s)
      M w = w_n_m;
      for (size_t t = 0; t < s; ++t) { w = w * w; w = w * w; }
      const M w2 = w * w, w3 = w2 * w;
      const M wi = w.inv(), wi2 = wi * wi, wi3 = wi2 * wi;
      uint32_t cur = M::from_int(1).raw_val();
      uint32_t cur2 = cur, cur3 = cur, icur = cur, icur2 = cur, icur3 = cur;
      const uint32_t wv = w.raw_val(), w2v = w2.raw_val(), w3v = w3.raw_val();
      const uint32_t wiv = wi.raw_val(), wi2v = wi2.raw_val(), wi3v = wi3.raw_val();
      uint32_t* A = fwd_.data() + pos;
      uint32_t* B = A + m;
      uint32_t* C = B + m;
      uint32_t* IA = inv_.data() + pos;
      uint32_t* IB = IA + m;
      uint32_t* IC = IB + m;
      for (uint32_t j = 0; j < m; ++j) {
        A[j] = cur; B[j] = cur2; C[j] = cur3;
        IA[j] = icur; IB[j] = icur2; IC[j] = icur3;
        cur = M::reduce(uint64_t(cur) * wv);
        cur2 = M::reduce(uint64_t(cur2) * w2v);
        cur3 = M::reduce(uint64_t(cur3) * w3v);
        icur = M::reduce(uint64_t(icur) * wiv);
        icur2 = M::reduce(uint64_t(icur2) * wi2v);
        icur3 = M::reduce(uint64_t(icur3) * wi3v);
      }
      pos += 3 * m;
    }
    off_[lens_.size()] = pos;
    inv_n_ = M::from_int(n_).inv().raw_val();
  }

  /// Route one forward stage: the last stages have chunk width m = len/4 below
  /// the SIMD lane count, where the generic kernel would run entirely on its
  /// scalar tail.  Those go to the specialised chunked kernels instead.
  template <bool LAST>
  void radix4_stage(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t m = len >> 2;
    if constexpr (simd::small_m_max >= 1) {
      if (m == 1) {
        radix4_small<1, LAST>(a, len, s);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 2) {
      if (m == 2) {
        radix4_small<2, LAST>(a, len, s);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 4) {
      if (m == 4) {
        radix4_small<4, LAST>(a, len, s);
        return;
      }
    }
    radix4_fwd<LAST>(a, len, s);
  }

  /// Same routing for the inverse direction.
  void radix4_stage_inv(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t m = len >> 2;
    if constexpr (simd::small_m_max >= 1) {
      if (m == 1) {
        radix4_small_inv<1>(a, len, s);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 2) {
      if (m == 2) {
        radix4_small_inv<2>(a, len, s);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 4) {
      if (m == 4) {
        radix4_small_inv<4>(a, len, s);
        return;
      }
    }
    radix4_inv(a, len, s);
  }

  /// Specialised stage for chunk width W == len/4 < lane.  The stage is a set of
  /// independent 4W-point transforms on *consecutive* blocks; with m < lane the
  /// generic kernel would fall back to scalar arithmetic for the whole stage, so
  /// instead lane/W blocks go through one vector: a chunk de-interleave gathers
  /// the four sub-blocks and the twiddle vectors are replicated to the lane
  /// pattern (each lane of a row shares the same k, so only the lane *order* of
  /// the blocks differs, which the matching store undoes exactly).
  template <int W, bool LAST>
  void radix4_small(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t* A = fwd_.data() + off_[s];
    const uint32_t* B = A + W;
    const uint32_t* C = B + W;
    alignas(64) uint32_t tw[3][64];
    for (int i = 0; i < simd::lane; ++i) {
      tw[0][i] = A[i % W];
      tw[1][i] = B[i % W];
      tw[2][i] = C[i % W];
    }
    const simd_t wa = simd::load(tw[0]), wb = simd::load(tw[1]), wc = simd::load(tw[2]);
    const simd_t vm = simd::set1(w4_);
    const uint32_t step = 4u * static_cast<uint32_t>(simd::lane);
    uint32_t base = 0;
    for (; base + step <= n_; base += step) {
      uint32_t* p = a + base;
      simd_t x0, x1, x2, x3;
      simd::load4m<W>(p, x0, x1, x2, x3);
      simd_t t0 = simd::add(x0, x2, R);
      simd_t t1 = simd::sub(x0, x2, R);
      simd_t t2 = simd::add(x1, x3, R);
      simd_t t3 = bmul(simd::sub(x1, x3, R), vm);
      simd_t y0 = simd::add(t0, t2, R);
      simd_t y1 = bmul(simd::sub(t0, t2, R), wb);
      simd_t y2 = bmul(feed_add(t1, t3), wa);
      simd_t y3 = bmul(simd::sub(t1, t3, R), wc);
      if constexpr (LAST) {
        y0 = simd::reduce_full(y0, MOD);
        y1 = simd::reduce_full(y1, MOD);
        y2 = simd::reduce_full(y2, MOD);
        y3 = simd::reduce_full(y3, MOD);
      }
      simd::store4m<W>(p, y0, y1, y2, y3);
    }
    for (; base < n_; base += len) {  // n < 4*lane: whole blocks, scalar
      uint32_t* p = a + base;
      for (uint32_t k = 0; k < static_cast<uint32_t>(W); ++k) {
        uint32_t x0 = p[k], x1 = p[k + W], x2 = p[k + 2 * W], x3 = p[k + 3 * W];
        uint32_t t0 = detail::s_add(x0, x2, R), t1 = detail::s_sub(x0, x2, R);
        uint32_t t2 = detail::s_add(x1, x3, R);
        uint32_t t3 = M::reduce(uint64_t(detail::s_sub(x1, x3, R)) * w4_);
        uint32_t z0 = detail::s_add(t0, t2, R);
        if constexpr (LAST) z0 = z0 >= MOD ? z0 - MOD : z0;
        p[k] = z0;
        p[k + W] = M::reduce(uint64_t(detail::s_sub(t0, t2, R)) * B[k]);
        p[k + 2 * W] = M::reduce(uint64_t(detail::s_add(t1, t3, R)) * A[k]);
        p[k + 3 * W] = M::reduce(uint64_t(detail::s_sub(t1, t3, R)) * C[k]);
      }
    }
  }

  /// Inverse counterpart of radix4_small.
  template <int W>
  void radix4_small_inv(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t* IA = inv_.data() + off_[s];
    const uint32_t* IB = IA + W;
    const uint32_t* IC = IB + W;
    alignas(64) uint32_t tw[3][64];
    for (int i = 0; i < simd::lane; ++i) {
      tw[0][i] = IA[i % W];
      tw[1][i] = IB[i % W];
      tw[2][i] = IC[i % W];
    }
    const simd_t iw1 = simd::load(tw[0]), iw2 = simd::load(tw[1]), iw3 = simd::load(tw[2]);
    const simd_t vm = simd::set1(w4_inv_);
    const uint32_t step = 4u * static_cast<uint32_t>(simd::lane);
    uint32_t base = 0;
    for (; base + step <= n_; base += step) {
      uint32_t* p = a + base;
      simd_t b0, b1, b2, b3;
      simd::load4m<W>(p, b0, b1, b2, b3);
      simd_t c1 = bmul(b1, iw2);
      simd_t c2 = bmul(b2, iw1);
      simd_t c3 = bmul(b3, iw3);
      simd_t d0 = simd::add(b0, c1, R), d1 = simd::sub(b0, c1, R);
      simd_t d2 = simd::add(c2, c3, R), d3 = simd::sub(c2, c3, R);
      simd_t e = bmul(d3, vm);
      simd::store4m<W>(p, simd::add(d0, d2, R), simd::add(d1, e, R),
                       simd::sub(d0, d2, R), simd::sub(d1, e, R));
    }
    for (; base < n_; base += len) {
      uint32_t* p = a + base;
      for (uint32_t k = 0; k < static_cast<uint32_t>(W); ++k) {
        uint32_t b0 = p[k], b1 = p[k + W], b2 = p[k + 2 * W], b3 = p[k + 3 * W];
        uint32_t c1 = M::reduce(uint64_t(b1) * IB[k]);
        uint32_t c2 = M::reduce(uint64_t(b2) * IA[k]);
        uint32_t c3 = M::reduce(uint64_t(b3) * IC[k]);
        uint32_t d0 = detail::s_add(b0, c1, R), d1 = detail::s_sub(b0, c1, R);
        uint32_t d2 = detail::s_add(c2, c3, R), d3 = detail::s_sub(c2, c3, R);
        uint32_t e = M::reduce(uint64_t(d3) * w4_inv_);
        p[k] = detail::s_add(d0, d2, R);
        p[k + W] = detail::s_add(d1, e, R);
        p[k + 2 * W] = detail::s_sub(d0, d2, R);
        p[k + 3 * W] = detail::s_sub(d1, e, R);
      }
    }
  }

  /// One radix-4 DIF butterfly stage over blocks of length `len`.
  /// `LAST` additionally canonicalises the output to [0, Mod) (only the final
  /// forward stage pays for it).
  template <bool LAST>
  void radix4_fwd(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t m = len >> 2;
    const uint32_t* A = fwd_.data() + off_[s];
    const uint32_t* B = A + m;
    const uint32_t* C = B + m;
    const simd_t vm = simd::set1(w4_);
    for (uint32_t base = 0; base < n_; base += len) {
      uint32_t* p = a + base;
      // One vector butterfly over lanes of k. Written as a lambda so the main
      // loop can be unrolled: the four multiplications have no mutual
      // dependencies, and two in flight hides their ~6-op latency chain.
      auto step = [&](uint32_t k) {
        simd_t x0 = simd::load(p + k), x1 = simd::load(p + k + m);
        simd_t x2 = simd::load(p + k + 2 * m), x3 = simd::load(p + k + 3 * m);
        simd_t wa = simd::load(A + k), wb = simd::load(B + k), wc = simd::load(C + k);
        simd_t t0 = simd::add(x0, x2, R);
        simd_t t1 = simd::sub(x0, x2, R);
        simd_t t2 = simd::add(x1, x3, R);
        simd_t t3 = bmul(simd::sub(x1, x3, R), vm);
        simd_t y0 = simd::add(t0, t2, R);
        simd_t y1 = bmul(simd::sub(t0, t2, R), wb);
        simd_t y2 = bmul(feed_add(t1, t3), wa);
        simd_t y3 = bmul(simd::sub(t1, t3, R), wc);
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
          step(k);
          step(k + simd::lane);
        }
      }
      for (; k + simd::lane <= m; k += simd::lane) step(k);
      for (; k < m; ++k) {
        uint32_t x0 = p[k], x1 = p[k + m], x2 = p[k + 2 * m], x3 = p[k + 3 * m];
        uint32_t t0 = detail::s_add(x0, x2, R);
        uint32_t t1 = detail::s_sub(x0, x2, R);
        uint32_t t2 = detail::s_add(x1, x3, R);
        uint32_t t3 = M::reduce(uint64_t(detail::s_sub(x1, x3, R)) * w4_);
        uint32_t y0 = detail::s_add(t0, t2, R);
        if constexpr (LAST) y0 = y0 >= MOD ? y0 - MOD : y0;
        p[k] = y0;
        p[k + m] = M::reduce(uint64_t(detail::s_sub(t0, t2, R)) * B[k]);
        p[k + 2 * m] = M::reduce(uint64_t(detail::s_add(t1, t3, R)) * A[k]);
        p[k + 3 * m] = M::reduce(uint64_t(detail::s_sub(t1, t3, R)) * C[k]);
      }
    }
  }

  /// One radix-4 DIT butterfly stage (inverse of radix4_fwd at the same `len`).
  void radix4_inv(uint32_t* a, uint32_t len, size_t s) const {
    const uint32_t m = len >> 2;
    const uint32_t* IA = inv_.data() + off_[s];
    const uint32_t* IB = IA + m;
    const uint32_t* IC = IB + m;
    const simd_t vm = simd::set1(w4_inv_);
    for (uint32_t base = 0; base < n_; base += len) {
      uint32_t* p = a + base;
      auto step = [&](uint32_t k) {
        simd_t b0 = simd::load(p + k), b1 = simd::load(p + k + m);
        simd_t b2 = simd::load(p + k + 2 * m), b3 = simd::load(p + k + 3 * m);
        simd_t iw1 = simd::load(IA + k), iw2 = simd::load(IB + k), iw3 = simd::load(IC + k);
        simd_t c1 = bmul(b1, iw2);
        simd_t c2 = bmul(b2, iw1);
        simd_t c3 = bmul(b3, iw3);
        simd_t d0 = simd::add(b0, c1, R), d1 = simd::sub(b0, c1, R);
        simd_t d2 = simd::add(c2, c3, R), d3 = simd::sub(c2, c3, R);
        simd_t e = bmul(d3, vm);
        simd::store(p + k, simd::add(d0, d2, R));
        simd::store(p + k + m, simd::add(d1, e, R));
        simd::store(p + k + 2 * m, simd::sub(d0, d2, R));
        simd::store(p + k + 3 * m, simd::sub(d1, e, R));
      };
      uint32_t k = 0;
      if constexpr (simd::lane > 1) {
        for (; k + 2 * simd::lane <= m; k += 2 * simd::lane) {
          step(k);
          step(k + simd::lane);
        }
      }
      for (; k + simd::lane <= m; k += simd::lane) step(k);
      for (; k < m; ++k) {
        uint32_t b0 = p[k], b1 = p[k + m], b2 = p[k + 2 * m], b3 = p[k + 3 * m];
        uint32_t c1 = M::reduce(uint64_t(b1) * IB[k]);
        uint32_t c2 = M::reduce(uint64_t(b2) * IA[k]);
        uint32_t c3 = M::reduce(uint64_t(b3) * IC[k]);
        uint32_t d0 = detail::s_add(b0, c1, R), d1 = detail::s_sub(b0, c1, R);
        uint32_t d2 = detail::s_add(c2, c3, R), d3 = detail::s_sub(c2, c3, R);
        uint32_t e = M::reduce(uint64_t(d3) * w4_inv_);
        p[k] = detail::s_add(d0, d2, R);
        p[k + m] = detail::s_add(d1, e, R);
        p[k + 2 * m] = detail::s_sub(d0, d2, R);
        p[k + 3 * m] = detail::s_sub(d1, e, R);
      }
    }
  }

  /// Trailing radix-2 stage used when log2(n) is odd (len == 2, twiddle == 1).
  /// The butterfly is its own transpose, so this serves both directions.
  template <bool LAST>
  void radix2_fwd(uint32_t* a) const {
    uint32_t k = 0;
    if constexpr (simd::lane > 1) {
      // simd::lane values == simd::lane/2 contiguous pairs, so one vector load
      // per iteration and no gather is needed.
      for (; k + simd::lane <= n_; k += simd::lane) {
        // swap the two halves of every pair, then recombine: even lanes carry
        // x+y, odd lanes carry x-y (both reduced mod R).
        simd_t v = simd::load(a + k);
        simd_t w = simd::swap_pairs(v);
        simd_t s = simd::add(v, w, R);
        simd_t d = simd::sub(w, v, R);
        simd_t r = simd::pick_odd(s, d);
        if constexpr (LAST) r = simd::reduce_full(r, MOD);
        simd::store(a + k, r);
      }
    }
    for (; k + 1 < n_; k += 2) {
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

  uint32_t n_ = 0;
  int logn_ = 0;
  uint32_t w4_ = 0, w4_inv_ = 0, inv_n_ = 0;
  std::vector<uint32_t> lens_;         // forward radix-4 block lengths
  std::vector<uint32_t> fwd_, inv_;    // per-stage twiddle tables
  std::vector<size_t> off_;            // stage -> offset into fwd_/inv_
};

}  // namespace fpx

#endif  // FASTPOLY_NTT_HPP
