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
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
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

/// Thrown when a transform size is not representable for the modulus.
class ntt_size_error : public std::runtime_error {
 public:
  explicit ntt_size_error(const std::string& s) : std::runtime_error(s) {}
};

/// Smallest representable power of two >= need (zero maps to one).
inline uint32_t next_pow2(uint64_t need) {
  if (need > (uint64_t(1) << 31))
    throw ntt_size_error("NTT size is not representable in a 32-bit limb");
  return std::bit_ceil(static_cast<uint32_t>(need));
}

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
    if (n < 2 || (n & (n - 1)) != 0)
      throw ntt_size_error("NTT size must be a power of two");
    if (n > ntt_max_size<M>())
      throw ntt_size_error("NTT size exceeds 2^v2(mod-1) for this modulus");
    constexpr size_t count = size_t(ntt_max_log<M>()) + 1;
    struct Entry {
      std::once_flag ready;
      std::shared_ptr<const NttPlan> plan;
    };
    static std::array<Entry, count> cache;
    // Warm lookups touch only this thread's slots; different cold sizes build
    // independently, and call_once publishes exactly one plan for each size.
    thread_local std::array<std::shared_ptr<const NttPlan>, count> local;
    const size_t slot = static_cast<size_t>(std::countr_zero(n));
    if (!local[slot]) {
      Entry& entry = cache[slot];
      std::call_once(entry.ready, [&entry, n] {
        entry.plan = std::shared_ptr<const NttPlan>(new NttPlan(n));
      });
      local[slot] = entry.plan;
    }
    return local[slot];
  }

  uint32_t size() const { return n_; }
  int log_size() const { return logn_; }

  /// Forward DIF transform, in place, on Montgomery limbs. Output in [0, Mod).
  void forward(uint32_t* a) const { forward_impl<true>(a); }

  /// Internal convolution entry point: retain lazy [0, 2*Mod) residues when
  /// supported. Matching inverse accepts these directly; the permutation and
  /// residues are identical to forward(), without its terminal reduction.
  void forward_lazy(uint32_t* a) const { forward_impl<false>(a); }

  /// Inverse DIT transform, normalized by 1/n, in place. Output in [0, Mod).
  /// Normalization is fused into the final butterfly and its pre-scaled twiddles.
  void inverse(uint32_t* a) const {
    if (n_ == 2) {
      const uint32_t x = a[0], y = a[1];
      a[0] = M::reduce(uint64_t(detail::s_add(x, y, R)) * inv_n_);
      a[1] = M::reduce(uint64_t(detail::s_sub(x, y, R)) * inv_n_);
      return;
    }
    size_t split = 0;
    while (split < lens_.size() && lens_[split] > CACHE_BLOCK) ++split;
    const uint32_t block = split < lens_.size() ? lens_[split] : n_;
    for (uint32_t base = 0; base < n_; base += block) {
      uint32_t* p = a + base;
      if (logn_ & 1) radix2_fwd<false>(p, block);
      for (size_t s = lens_.size(); s-- > split;) {
        if (s == 0) radix4_stage_inv<true>(p, lens_[s], s, block);
        else radix4_stage_inv<false>(p, lens_[s], s, block);
      }
    }
    for (size_t s = split; s-- > 0;) {
      if (s == 0) radix4_stage_inv<true>(a, lens_[s], s, n_);
      else radix4_stage_inv<false>(a, lens_[s], s, n_);
    }
  }

 private:
  // Data plus the active direction's twiddles consume about 2*block words.
  static constexpr uint32_t CACHE_BLOCK = 4096;

  template <bool CANON>
  void forward_impl(uint32_t* a) const {
    // Upper stages span the whole input. Once subtransforms fit in L1, finish
    // every remaining stage while each consecutive block is still resident.
    size_t split = 0;
    while (split < lens_.size() && lens_[split] > CACHE_BLOCK) {
      radix4_stage<false>(a, lens_[split], split, n_);
      ++split;
    }
    const uint32_t block = split < lens_.size() ? lens_[split] : n_;
    const bool radix4_last = (logn_ & 1) == 0;
    for (uint32_t base = 0; base < n_; base += block) {
      uint32_t* p = a + base;
      for (size_t s = split; s < lens_.size(); ++s) {
        if (CANON && radix4_last && s + 1 == lens_.size())
          radix4_stage<true>(p, lens_[s], s, block);
        else
          radix4_stage<false>(p, lens_[s], s, block);
      }
      if (logn_ & 1) radix2_fwd<CANON>(p, block);
    }
  }

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
    inv_n_ = M::from_int(n_).inv().raw_val();
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
      uint32_t cur2 = cur, cur3 = cur;
      // The final DIT stage absorbs normalization into its three twiddles.
      uint32_t icur = s == 0 ? inv_n_ : cur, icur2 = icur, icur3 = icur;
      const uint32_t wv = w.raw_val(), w2v = w2.raw_val(), w3v = w3.raw_val();
      const uint32_t wiv = wi.raw_val(), wi2v = wi2.raw_val(), wi3v = wi3.raw_val();
      uint32_t* A = fwd_.data() + pos;
      uint32_t* B = A + m;
      uint32_t* C = B + m;
      uint32_t* IA = inv_.data() + pos;
      uint32_t* IB = IA + m;
      uint32_t* IC = IB + m;
      uint32_t j = 0;
      if constexpr (simd::lane > 1) {
        // Amortize vector initialization over at least four full blocks.
        if (m >= 4u * simd::lane) {
          alignas(64) uint32_t seed[6][simd::lane];
          for (int k = 0; k < simd::lane; ++k) {
            seed[0][k] = cur; seed[1][k] = cur2; seed[2][k] = cur3;
            seed[3][k] = icur; seed[4][k] = icur2; seed[5][k] = icur3;
            cur = M::reduce(uint64_t(cur) * wv);
            cur2 = M::reduce(uint64_t(cur2) * w2v);
            cur3 = M::reduce(uint64_t(cur3) * w3v);
            icur = M::reduce(uint64_t(icur) * wiv);
            icur2 = M::reduce(uint64_t(icur2) * wi2v);
            icur3 = M::reduce(uint64_t(icur3) * wi3v);
          }
          simd_t va = simd::load(seed[0]), vb = simd::load(seed[1]), vc = simd::load(seed[2]);
          simd_t via = simd::load(seed[3]), vib = simd::load(seed[4]), vic = simd::load(seed[5]);
          // Forward cursors already advanced from Montgomery one to w^lane.
          const simd_t da = simd::set1(cur), db = simd::set1(cur2), dc = simd::set1(cur3);
          const M istep = wi.pow(simd::lane), istep2 = istep * istep, istep3 = istep2 * istep;
          const simd_t dia = simd::set1(istep.raw_val());
          const simd_t dib = simd::set1(istep2.raw_val());
          const simd_t dic = simd::set1(istep3.raw_val());
          for (; j + simd::lane <= m; j += simd::lane) {
            simd::store(A + j, va); simd::store(B + j, vb); simd::store(C + j, vc);
            simd::store(IA + j, via); simd::store(IB + j, vib); simd::store(IC + j, vic);
            va = simd::mulmod<MOD, NINV>(va, da);
            vb = simd::mulmod<MOD, NINV>(vb, db);
            vc = simd::mulmod<MOD, NINV>(vc, dc);
            via = simd::mulmod<MOD, NINV>(via, dia);
            vib = simd::mulmod<MOD, NINV>(vib, dib);
            vic = simd::mulmod<MOD, NINV>(vic, dic);
          }
          if (j < m) {
            simd::store(seed[0], va); simd::store(seed[1], vb); simd::store(seed[2], vc);
            simd::store(seed[3], via); simd::store(seed[4], vib); simd::store(seed[5], vic);
            cur = seed[0][0]; cur2 = seed[1][0]; cur3 = seed[2][0];
            icur = seed[3][0]; icur2 = seed[4][0]; icur3 = seed[5][0];
          }
        }
      }
      for (; j < m; ++j) {
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
  }

  /// Route one forward stage: the last stages have chunk width m = len/4 below
  /// the SIMD lane count, where the generic kernel would run entirely on its
  /// scalar tail.  Those go to the specialised chunked kernels instead.
  template <bool LAST>
  void radix4_stage(uint32_t* a, uint32_t len, size_t s, uint32_t count) const {
    const uint32_t m = len >> 2;
    if (m == 1) {
      radix4_small<1, LAST>(a, len, s, count);
      return;
    }
    if constexpr (simd::small_m_max >= 2) {
      if (m == 2) {
        radix4_small<2, LAST>(a, len, s, count);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 4) {
      if (m == 4) {
        radix4_small<4, LAST>(a, len, s, count);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 8) {
      if (m == 8) {
        radix4_small<8, LAST>(a, len, s, count);
        return;
      }
    }
    radix4_fwd<LAST>(a, len, s, count);
  }

  /// Same routing for the inverse direction.
  template <bool LAST>
  void radix4_stage_inv(uint32_t* a, uint32_t len, size_t s, uint32_t count) const {
    const uint32_t m = len >> 2;
    if (m == 1) {
      radix4_small_inv<1, LAST>(a, len, s, count);
      return;
    }
    if constexpr (simd::small_m_max >= 2) {
      if (m == 2) {
        radix4_small_inv<2, LAST>(a, len, s, count);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 4) {
      if (m == 4) {
        radix4_small_inv<4, LAST>(a, len, s, count);
        return;
      }
    }
    if constexpr (simd::small_m_max >= 8) {
      if (m == 8) {
        radix4_small_inv<8, LAST>(a, len, s, count);
        return;
      }
    }
    radix4_inv<LAST>(a, len, s, count);
  }

  /// Specialised stage for chunk width W == len/4 < lane.  The stage is a set of
  /// independent 4W-point transforms on *consecutive* blocks; with m < lane the
  /// generic kernel would fall back to scalar arithmetic for the whole stage, so
  /// instead lane/W blocks go through one vector: a chunk de-interleave gathers
  /// the four sub-blocks and the twiddle vectors are replicated to the lane
  /// pattern (each lane of a row shares the same k, so only the lane *order* of
  /// the blocks differs, which the matching store undoes exactly).
  template <int W, bool LAST>
  void radix4_small(uint32_t* a, uint32_t len, size_t s, uint32_t count) const {
    const uint32_t* A = fwd_.data() + off_[s];
    const uint32_t* B = A + W;
    const uint32_t* C = B + W;
    alignas(64) uint32_t tw[3][simd::lane];
    for (int i = 0; i < simd::lane; ++i) {
      tw[0][i] = A[i % W];
      tw[1][i] = B[i % W];
      tw[2][i] = C[i % W];
    }
    const simd_t wa = simd::load(tw[0]), wb = simd::load(tw[1]), wc = simd::load(tw[2]);
    const simd_t vm = simd::set1(w4_);
    const uint32_t step = 4u * static_cast<uint32_t>(simd::lane);
    uint32_t base = 0;
    for (; base + step <= count; base += step) {
      uint32_t* p = a + base;
      simd_t x0, x1, x2, x3;
      simd::load4m<W>(p, x0, x1, x2, x3);
      simd_t t0 = simd::add(x0, x2, R);
      simd_t t1 = simd::sub(x0, x2, R);
      simd_t t2 = simd::add(x1, x3, R);
      simd_t t3 = bmul(simd::sub(x1, x3, R), vm);
      simd_t y0 = simd::add(t0, t2, R);
      simd_t y1, y2, y3;
      if constexpr (W == 1) {
        // All three twiddles are Montgomery one for this terminal stage.
        y1 = simd::sub(t0, t2, R);
        y2 = simd::add(t1, t3, R);
        y3 = simd::sub(t1, t3, R);
      } else {
        y1 = bmul(simd::sub(t0, t2, R), wb);
        y2 = bmul(feed_add(t1, t3), wa);
        y3 = bmul(simd::sub(t1, t3, R), wc);
      }
      if constexpr (LAST && LAZY) {
        y0 = simd::reduce_full(y0, MOD);
        y1 = simd::reduce_full(y1, MOD);
        y2 = simd::reduce_full(y2, MOD);
        y3 = simd::reduce_full(y3, MOD);
      }
      simd::store4m<W>(p, y0, y1, y2, y3);
    }
    for (; base < count; base += len) {  // n < 4*lane: whole blocks, scalar
      uint32_t* p = a + base;
      for (uint32_t k = 0; k < static_cast<uint32_t>(W); ++k) {
        uint32_t x0 = p[k], x1 = p[k + W], x2 = p[k + 2 * W], x3 = p[k + 3 * W];
        uint32_t t0 = detail::s_add(x0, x2, R), t1 = detail::s_sub(x0, x2, R);
        uint32_t t2 = detail::s_add(x1, x3, R);
        uint32_t t3 = M::reduce(uint64_t(detail::s_sub(x1, x3, R)) * w4_);
        uint32_t z0 = detail::s_add(t0, t2, R);
        uint32_t z1 = detail::s_sub(t0, t2, R);
        uint32_t z2 = detail::s_add(t1, t3, R), z3 = detail::s_sub(t1, t3, R);
        if constexpr (W != 1) {
          z1 = M::reduce(uint64_t(z1) * B[k]);
          z2 = M::reduce(uint64_t(z2) * A[k]);
          z3 = M::reduce(uint64_t(z3) * C[k]);
        }
        if constexpr (LAST && LAZY) {
          z0 = z0 >= MOD ? z0 - MOD : z0; z1 = z1 >= MOD ? z1 - MOD : z1;
          z2 = z2 >= MOD ? z2 - MOD : z2; z3 = z3 >= MOD ? z3 - MOD : z3;
        }
        p[k] = z0; p[k + W] = z1;
        p[k + 2 * W] = z2; p[k + 3 * W] = z3;
      }
    }
  }

  /// Inverse counterpart of radix4_small.
  template <int W, bool LAST>
  void radix4_small_inv(uint32_t* a, uint32_t len, size_t s, uint32_t count) const {
    const uint32_t* IA = inv_.data() + off_[s];
    const uint32_t* IB = IA + W;
    const uint32_t* IC = IB + W;
    alignas(64) uint32_t tw[3][simd::lane];
    for (int i = 0; i < simd::lane; ++i) {
      tw[0][i] = IA[i % W];
      tw[1][i] = IB[i % W];
      tw[2][i] = IC[i % W];
    }
    const simd_t iw1 = simd::load(tw[0]), iw2 = simd::load(tw[1]), iw3 = simd::load(tw[2]);
    const simd_t vm = simd::set1(w4_inv_);
    const uint32_t step = 4u * static_cast<uint32_t>(simd::lane);
    uint32_t base = 0;
    for (; base + step <= count; base += step) {
      uint32_t* p = a + base;
      simd_t b0, b1, b2, b3;
      simd::load4m<W>(p, b0, b1, b2, b3);
      if constexpr (LAST) b0 = bmul(b0, simd::set1(inv_n_));
      simd_t c1, c2, c3;
      if constexpr (W == 1 && !LAST) {
        c1 = b1; c2 = b2; c3 = b3;
      } else {
        c1 = bmul(b1, iw2); c2 = bmul(b2, iw1); c3 = bmul(b3, iw3);
      }
      simd_t d0 = simd::add(b0, c1, R), d1 = simd::sub(b0, c1, R);
      simd_t d2 = simd::add(c2, c3, R), d3 = simd::sub(c2, c3, R);
      simd_t e = bmul(d3, vm);
      simd_t y0 = simd::add(d0, d2, R), y1 = simd::add(d1, e, R);
      simd_t y2 = simd::sub(d0, d2, R), y3 = simd::sub(d1, e, R);
      if constexpr (LAST && LAZY) {
        y0 = simd::reduce_full(y0, MOD); y1 = simd::reduce_full(y1, MOD);
        y2 = simd::reduce_full(y2, MOD); y3 = simd::reduce_full(y3, MOD);
      }
      simd::store4m<W>(p, y0, y1, y2, y3);
    }
    for (; base < count; base += len) {
      uint32_t* p = a + base;
      for (uint32_t k = 0; k < static_cast<uint32_t>(W); ++k) {
        uint32_t b0 = p[k], b1 = p[k + W], b2 = p[k + 2 * W], b3 = p[k + 3 * W];
        if constexpr (LAST) b0 = M::reduce(uint64_t(b0) * inv_n_);
        uint32_t c1 = b1, c2 = b2, c3 = b3;
        if constexpr (W != 1 || LAST) {
          c1 = M::reduce(uint64_t(b1) * IB[k]);
          c2 = M::reduce(uint64_t(b2) * IA[k]);
          c3 = M::reduce(uint64_t(b3) * IC[k]);
        }
        uint32_t d0 = detail::s_add(b0, c1, R), d1 = detail::s_sub(b0, c1, R);
        uint32_t d2 = detail::s_add(c2, c3, R), d3 = detail::s_sub(c2, c3, R);
        uint32_t e = M::reduce(uint64_t(d3) * w4_inv_);
        uint32_t y0 = detail::s_add(d0, d2, R), y1 = detail::s_add(d1, e, R);
        uint32_t y2 = detail::s_sub(d0, d2, R), y3 = detail::s_sub(d1, e, R);
        if constexpr (LAST && LAZY) {
          y0 = y0 >= MOD ? y0 - MOD : y0; y1 = y1 >= MOD ? y1 - MOD : y1;
          y2 = y2 >= MOD ? y2 - MOD : y2; y3 = y3 >= MOD ? y3 - MOD : y3;
        }
        p[k] = y0; p[k + W] = y1;
        p[k + 2 * W] = y2; p[k + 3 * W] = y3;
      }
    }
  }

  /// One radix-4 DIF butterfly stage over blocks of length `len`.
  /// `LAST` additionally canonicalises the output to [0, Mod) (only the final
  /// forward stage pays for it).
  template <bool LAST>
  void radix4_fwd(uint32_t* a, uint32_t len, size_t s, uint32_t count) const {
    const uint32_t m = len >> 2;
    const uint32_t* A = fwd_.data() + off_[s];
    const uint32_t* B = A + m;
    const uint32_t* C = B + m;
    const simd_t vm = simd::set1(w4_);
    for (uint32_t base = 0; base < count; base += len) {
      uint32_t* p = a + base;
      // One vector butterfly over lanes of k. Written as a lambda so the main
      // loop can be unrolled: the t3 multiply feeds y2/y3, while y1 is
      // independent. Two butterflies in flight expose further parallelism.
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
        if constexpr (LAST && LAZY) {
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
  template <bool LAST>
  void radix4_inv(uint32_t* a, uint32_t len, size_t s, uint32_t count) const {
    const uint32_t m = len >> 2;
    const uint32_t* IA = inv_.data() + off_[s];
    const uint32_t* IB = IA + m;
    const uint32_t* IC = IB + m;
    const simd_t vm = simd::set1(w4_inv_);
    for (uint32_t base = 0; base < count; base += len) {
      uint32_t* p = a + base;
      auto step = [&](uint32_t k) {
        simd_t b0 = simd::load(p + k), b1 = simd::load(p + k + m);
        simd_t b2 = simd::load(p + k + 2 * m), b3 = simd::load(p + k + 3 * m);
        simd_t iw1 = simd::load(IA + k), iw2 = simd::load(IB + k), iw3 = simd::load(IC + k);
        if constexpr (LAST) b0 = bmul(b0, simd::set1(inv_n_));
        simd_t c1 = bmul(b1, iw2);
        simd_t c2 = bmul(b2, iw1);
        simd_t c3 = bmul(b3, iw3);
        simd_t d0 = simd::add(b0, c1, R), d1 = simd::sub(b0, c1, R);
        simd_t d2 = simd::add(c2, c3, R), d3 = simd::sub(c2, c3, R);
        simd_t e = bmul(d3, vm);
        simd_t y0 = simd::add(d0, d2, R), y1 = simd::add(d1, e, R);
        simd_t y2 = simd::sub(d0, d2, R), y3 = simd::sub(d1, e, R);
        if constexpr (LAST && LAZY) {
          y0 = simd::reduce_full(y0, MOD); y1 = simd::reduce_full(y1, MOD);
          y2 = simd::reduce_full(y2, MOD); y3 = simd::reduce_full(y3, MOD);
        }
        simd::store(p + k, y0); simd::store(p + k + m, y1);
        simd::store(p + k + 2 * m, y2); simd::store(p + k + 3 * m, y3);
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
        if constexpr (LAST) b0 = M::reduce(uint64_t(b0) * inv_n_);
        uint32_t c1 = M::reduce(uint64_t(b1) * IB[k]);
        uint32_t c2 = M::reduce(uint64_t(b2) * IA[k]);
        uint32_t c3 = M::reduce(uint64_t(b3) * IC[k]);
        uint32_t d0 = detail::s_add(b0, c1, R), d1 = detail::s_sub(b0, c1, R);
        uint32_t d2 = detail::s_add(c2, c3, R), d3 = detail::s_sub(c2, c3, R);
        uint32_t e = M::reduce(uint64_t(d3) * w4_inv_);
        uint32_t y0 = detail::s_add(d0, d2, R), y1 = detail::s_add(d1, e, R);
        uint32_t y2 = detail::s_sub(d0, d2, R), y3 = detail::s_sub(d1, e, R);
        if constexpr (LAST && LAZY) {
          y0 = y0 >= MOD ? y0 - MOD : y0; y1 = y1 >= MOD ? y1 - MOD : y1;
          y2 = y2 >= MOD ? y2 - MOD : y2; y3 = y3 >= MOD ? y3 - MOD : y3;
        }
        p[k] = y0; p[k + m] = y1;
        p[k + 2 * m] = y2; p[k + 3 * m] = y3;
      }
    }
  }

  /// Trailing radix-2 stage used when log2(n) is odd (len == 2, twiddle == 1).
  /// The butterfly is its own transpose, so this serves both directions.
  template <bool LAST>
  void radix2_fwd(uint32_t* a, uint32_t count) const {
    uint32_t k = 0;
    if constexpr (simd::lane > 1) {
      // simd::lane values == simd::lane/2 contiguous pairs, so one vector load
      // per iteration and no gather is needed.
      for (; k + simd::lane <= count; k += simd::lane) {
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
    for (; k + 1 < count; k += 2) {
      uint32_t x = a[k], y = a[k + 1];
      uint32_t s = detail::s_add(x, y, R);
      uint32_t t = detail::s_sub(x, y, R);
      if constexpr (LAST && LAZY) {
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
