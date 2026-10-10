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
// * Butterflies use Montgomery multiplication, compact Q31 on small-modulus
//   NEON, or Q32 fixed multiplication in cached x86 stages (see simd.hpp).
//   No transform performs division.
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
#include <utility>

#include "fastpoly/memory.hpp"
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
  static constexpr bool COMPACT = simd::compact_twiddle<MOD>;
  static constexpr bool X86_SHOUP = simd::backend::q32;
  static constexpr uint32_t SHOUP_BLOCK = 2048;
  // Keep the terminal radix-2 in the width-2 kernel, avoiding another pass.
  static constexpr bool FUSE_RADIX2 = COMPACT || X86_SHOUP;
  using Twiddle = simd::Twiddle;

  // Preserve the plan's value semantics while keeping its large tables in
  // aligned storage. Cached plans are const and never take this copy path.
  NttPlan(const NttPlan& other)
      : n_(other.n_), logn_(other.logn_), w4_(other.w4_),
        w4_inv_(other.w4_inv_), inv_n_(other.inv_n_),
        stage_count_(other.stage_count_), lens_(other.lens_),
        fwd_(other.fwd_.size()), inv_(other.inv_.size()), off_(other.off_) {
    if (fwd_.size() != 0) {
      std::memcpy(fwd_.data(), other.fwd_.data(), fwd_.size() * sizeof(uint32_t));
      std::memcpy(inv_.data(), other.inv_.data(), inv_.size() * sizeof(uint32_t));
    }
  }
  NttPlan& operator=(const NttPlan& other) {
    if (this != &other) {
      NttPlan copy(other);
      *this = std::move(copy);
    }
    return *this;
  }
  NttPlan(NttPlan&&) noexcept = default;
  NttPlan& operator=(NttPlan&&) noexcept = default;

  /// Shared ownership is retained for compatibility with existing callers.
  static std::shared_ptr<const NttPlan> get(uint32_t n) { return cached_plan(n); }

  /// Borrow the immutable cached plan without touching its reference count.
  /// The cache owns the plan for its entire lifetime; transform hot paths should
  /// use this accessor when they do not need a separate owning handle.
  static const NttPlan& get_ref(uint32_t n) { return *cached_plan(n); }

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
    while (split < stage_count_ && lens_[split] > CACHE_BLOCK) ++split;
    const uint32_t block = split < stage_count_ ? lens_[split] : n_;
    for (uint32_t base = 0; base < n_; base += block) {
      uint32_t* p = a + base;
      if ((logn_ & 1) && !FUSE_RADIX2) radix2_fwd<false>(p, block);
      for (size_t s = stage_count_; s-- > split;) {
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
  static const std::shared_ptr<const NttPlan>& cached_plan(uint32_t n) {
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
    // Thread-local pointers remove ownership bookkeeping and TLS destructors.
    // call_once still publishes exactly one immutable plan per size.
    thread_local std::array<const Entry*, count> local{};
    const size_t slot = static_cast<size_t>(std::countr_zero(n));
    if (!local[slot]) {
      Entry& entry = cache[slot];
      std::call_once(entry.ready, [&entry, n] {
        struct Builder final : NttPlan {
          explicit Builder(uint32_t size) : NttPlan(size) {}
        };
        entry.plan = std::make_shared<Builder>(n);
      });
      local[slot] = &entry;
    }
    return local[slot]->plan;
  }

  // Cached x86 stages have value and quotient planes: about 3*block words.
  // At most 24 KiB leaves room in a 32 KiB L1 cache. Larger stages retain
  // single-word Montgomery tables; quotient storage stays below 16 KiB total.
  static constexpr uint32_t CACHE_BLOCK = X86_SHOUP ? SHOUP_BLOCK : 4096;

  template <bool CANON>
  void forward_impl(uint32_t* a) const {
    // Upper stages span the whole input. Once subtransforms fit in L1, finish
    // every remaining stage while each consecutive block is still resident.
    size_t split = 0;
    while (split < stage_count_ && lens_[split] > CACHE_BLOCK) {
      radix4_stage<false>(a, lens_[split], split, n_);
      ++split;
    }
    const uint32_t block = split < stage_count_ ? lens_[split] : n_;
    const bool radix4_last = (logn_ & 1) == 0 || FUSE_RADIX2;
    for (uint32_t base = 0; base < n_; base += block) {
      uint32_t* p = a + base;
      for (size_t s = split; s < stage_count_; ++s) {
        if (CANON && radix4_last && s + 1 == stage_count_)
          radix4_stage<true>(p, lens_[s], s, block);
        else
          radix4_stage<false>(p, lens_[s], s, block);
      }
      if ((logn_ & 1) && (!FUSE_RADIX2 || n_ == 2)) radix2_fwd<CANON>(p, block);
    }
  }

  explicit NttPlan(uint32_t n) : n_(n) {
    if (n < 2 || (n & (n - 1)) != 0)
      throw ntt_size_error("NTT size must be a power of two");
    if (n > ntt_max_size<M>())
      throw ntt_size_error("NTT size exceeds 2^v2(mod-1) for this modulus");
    logn_ = static_cast<int>(std::countr_zero(n_));
    build();
  }

  template <bool ROUND = true>
  static simd::FixedTwiddle tw_const(uint32_t mont) {
    return simd::fixed_twiddle<MOD, NINV, ROUND>(mont);
  }
  static auto cursor_step(uint32_t mont) {
    if constexpr (X86_SHOUP) {
      // Six independent cursors already hide the Montgomery dependency. A
      // single-word factor avoids spilling twelve Shoup constants on AVX2.
      return Twiddle{simd::set1(mont), simd::zero()};
    } else {
      return simd::fixed_twiddle<MOD, NINV>(mont);
    }
  }
  template <bool INVERSE>
  auto tw_fourth() const {
    if constexpr (X86_SHOUP) {
      constexpr uint32_t root = M::from_int(M::primitive_root).pow((MOD - 1)/4).raw_val();
      return tw_const(INVERSE ? MOD - root : root);
    } else {
      return tw_const(INVERSE ? w4_inv_ : w4_);
    }
  }
  template <int POWER, bool INVERSE = false>
  static auto tw_eighth_mont() {
    constexpr M root = M::from_int(M::primitive_root).pow((MOD - 1)/8);
    constexpr uint32_t mont = (INVERSE ? root.inv() : root).pow(POWER).raw_val();
    return simd::set1(mont);
  }
  static Twiddle tw_decode(simd_t q) { return simd::decode_twiddle<MOD>(q); }
  template <bool SHOUP = X86_SHOUP>
  static auto tw_load(const uint32_t* p, uint32_t qoff) {
    if constexpr (SHOUP) return simd::FixedTwiddle{{simd::load(p), simd::load(p + qoff)}};
    else return tw_decode(simd::load(p));
  }
  template <class T>
  static simd_t bmul(simd_t a, T w) { return simd::mul_twiddle<MOD, NINV>(a, w); }
  template <class T>
  static simd_t submul(simd_t a, simd_t b, T w) {
    return simd::mul_twiddle_diff<MOD, NINV>(a, b, w);
  }
  template <class T>
  static simd_t addmul(simd_t a, simd_t b, T w) {
    return simd::mul_twiddle_sum<MOD, NINV>(a, b, w);
  }

  // Scalar tails use the same compact tables. Ordinary Montgomery constants
  // (w4 and inv_n) continue to use M::reduce in those small tails.
  template <bool INVERSE = false, bool SHOUP = X86_SHOUP>
  static uint32_t smul(uint32_t a, uint32_t q) {
    if constexpr (SHOUP) {
      return static_cast<uint32_t>(uint64_t(a)*q % MOD);
    } else if constexpr (COMPACT) {
      const uint32_t w = static_cast<uint32_t>(
          (uint64_t(q) * MOD + (uint64_t(1) << 30)) >> 31);
      if constexpr (INVERSE)
        return static_cast<uint32_t>(uint64_t(a) * w - ((uint64_t(a) * q) >> 31) * MOD);
      else {
        const int64_t x = int64_t(a) - MOD;
        const int64_t quotient = (x * q + (int64_t(1) << 30)) >> 31;
        return static_cast<uint32_t>(x * w - quotient * MOD + MOD);
      }
    } else {
      return M::reduce(uint64_t(a) * q);
    }
  }

  void build() {
    inv_n_ = M::from_int(M::inv2).pow(static_cast<uint32_t>(logn_)).raw_val();
    const M w_n_m = M::from_int(M::primitive_root).pow((MOD - 1) / n_);
    w4_ = w_n_m.pow(n_ / 4).raw_val();            // primitive 4th root of unity
    w4_inv_ = n_ >= 4 ? MOD - w4_ : w4_;        // == -w4_ for n >= 4

    // Radix-4 stages: len = n, n/4, n/16, ... (while len >= 4)
    for (uint32_t len = n_; len >= 4; len >>= 2) lens_[stage_count_++] = len;
    // Each stage stores w^k, w^2k, w^3k (and inverses) for k < len/4.
    // NEON encodes Q31 quotients in one word. Cached x86 stages store separate
    // ordinary multiplier and Q32 planes; larger stages keep one Montgomery
    // word. Generate canonical cursors first, then encode only the bounded
    // cached planes. Separating conversion keeps the six-chain loop compact.
    size_t total = 0;
    for (size_t s = 0; s < stage_count_; ++s)
      total += 3*(lens_[s] >> 2)*(X86_SHOUP && lens_[s] <= SHOUP_BLOCK ? 2 : 1);
    fwd_.resize_uninitialized(total);
    inv_.resize_uninitialized(total);
    size_t pos = 0;
    M w = w_n_m, wi = w_n_m.inv();
    for (size_t s = 0; s < stage_count_; ++s) {
      off_[s] = pos;
      const uint32_t m = lens_[s] >> 2;
      const uint32_t qoff = X86_SHOUP && lens_[s] <= SHOUP_BLOCK ? 3*m : 0;
      // w = root of order len = w_n^(n/len) = w_n^(4^s)
      const M w2 = w * w, w3 = w2 * w;
      const M wi2 = wi * wi, wi3 = wi2 * wi;
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
          const auto da = cursor_step(cur);
          const auto db = cursor_step(cur2);
          const auto dc = cursor_step(cur3);
          const M istep = wi.pow(simd::lane), istep2 = istep * istep, istep3 = istep2 * istep;
          const auto dia = cursor_step(istep.raw_val());
          const auto dib = cursor_step(istep2.raw_val());
          const auto dic = cursor_step(istep3.raw_val());
          for (; j + simd::lane <= m; j += simd::lane) {
            simd::store(A + j, simd::encode_twiddle<MOD, NINV, true>(va));
            simd::store(B + j, simd::encode_twiddle<MOD, NINV, true>(vb));
            simd::store(C + j, simd::encode_twiddle<MOD, NINV, true>(vc));
            simd::store(IA + j, simd::encode_twiddle<MOD, NINV, false>(via));
            simd::store(IB + j, simd::encode_twiddle<MOD, NINV, false>(vib));
            simd::store(IC + j, simd::encode_twiddle<MOD, NINV, false>(vic));
            va = simd::mul_twiddle_full<MOD, NINV>(va, da);
            vb = simd::mul_twiddle_full<MOD, NINV>(vb, db);
            vc = simd::mul_twiddle_full<MOD, NINV>(vc, dc);
            via = simd::mul_twiddle_full<MOD, NINV>(via, dia);
            vib = simd::mul_twiddle_full<MOD, NINV>(vib, dib);
            vic = simd::mul_twiddle_full<MOD, NINV>(vic, dic);
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
        A[j] = simd::encode_twiddle_scalar<MOD, NINV, true>(cur);
        B[j] = simd::encode_twiddle_scalar<MOD, NINV, true>(cur2);
        C[j] = simd::encode_twiddle_scalar<MOD, NINV, true>(cur3);
        IA[j] = simd::encode_twiddle_scalar<MOD, NINV, false>(icur);
        IB[j] = simd::encode_twiddle_scalar<MOD, NINV, false>(icur2);
        IC[j] = simd::encode_twiddle_scalar<MOD, NINV, false>(icur3);
        cur = M::reduce(uint64_t(cur) * wv);
        cur2 = M::reduce(uint64_t(cur2) * w2v);
        cur3 = M::reduce(uint64_t(cur3) * w3v);
        icur = M::reduce(uint64_t(icur) * wiv);
        icur2 = M::reduce(uint64_t(icur2) * wi2v);
        icur3 = M::reduce(uint64_t(icur3) * wi3v);
      }
      if constexpr (X86_SHOUP) {
        if (qoff != 0) {
          for (uint32_t* p : {A, IA}) {
            uint32_t k = 0;
            for (; k + simd::lane <= qoff; k += simd::lane) {
              const auto t = simd::fixed_twiddle_vector<MOD, NINV>(simd::load(p + k));
              simd::store(p + k, t.value);
              simd::store(p + k + qoff, t.quotient);
            }
            for (; k < qoff; ++k) {
              const uint32_t mont = p[k], q = mont*NINV;
              p[k] = static_cast<uint32_t>((uint64_t(q)*MOD + mont) >> 32);
              p[k + qoff] = q;
            }
          }
        }
      }
      pos += 3*m + qoff;
      w = w2 * w2;
      wi = wi2 * wi2;
    }
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
    if constexpr (X86_SHOUP) {
      if (len <= SHOUP_BLOCK) { radix4_fwd<LAST, true>(a, len, s, count); return; }
    }
    radix4_fwd<LAST, false>(a, len, s, count);
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
    if constexpr (X86_SHOUP) {
      if (len <= SHOUP_BLOCK) { radix4_inv<LAST, true>(a, len, s, count); return; }
    }
    radix4_inv<LAST, false>(a, len, s, count);
  }

  // Four eight-point blocks occupy 32 consecutive limbs. Separate their k=0
  // and k=1 columns: k=0 has only unity twiddles, while the paired radix-2
  // now computes each sum/difference once, using every lane. The inverse's
  // normalized n = 8 stage stays in the ordinary small kernel below.
  template <bool INVERSE, bool UNITY>
  static void split4_core(simd_t* x, Twiddle wa, Twiddle wb, Twiddle wc, Twiddle vm) {
    if constexpr (INVERSE) {
      simd_t c1 = x[1], c2 = x[2], c3 = x[3];
      if constexpr (!UNITY) { c1 = bmul(c1, wb); c2 = bmul(c2, wa); c3 = bmul(c3, wc); }
      const auto d0 = simd::add(x[0], c1, R), d1 = simd::sub(x[0], c1, R);
      const auto d2 = simd::add(c2, c3, R), e = submul(c2, c3, vm);
      x[0] = simd::add(d0, d2, R); x[1] = simd::add(d1, e, R);
      x[2] = simd::sub(d0, d2, R); x[3] = simd::sub(d1, e, R);
    } else {
      const auto t0 = simd::add(x[0], x[2], R), t1 = simd::sub(x[0], x[2], R);
      const auto t2 = simd::add(x[1], x[3], R), t3 = submul(x[1], x[3], vm);
      x[0] = simd::add(t0, t2, R);
      if constexpr (UNITY) {
        x[1] = simd::sub(t0, t2, R); x[2] = simd::add(t1, t3, R); x[3] = simd::sub(t1, t3, R);
      } else {
        x[1] = submul(t0, t2, wb); x[2] = addmul(t1, t3, wa); x[3] = submul(t1, t3, wc);
      }
    }
  }
  template <bool INVERSE, bool CANON, class Backend = simd::backend>
  void radix8_split(uint32_t* a, size_t s, uint32_t count) const {
    const auto* table = (INVERSE ? inv_.data() : fwd_.data()) + off_[s];
    const Twiddle wa = tw_decode(simd::set1(table[1])), wb = tw_decode(simd::set1(table[3]));
    const Twiddle wc = tw_decode(simd::set1(table[5])), vm = tw_const(INVERSE ? w4_inv_ : w4_);
    for (uint32_t base = 0; base < count; base += 32) {
      simd_t lo[4], hi[4], x[4], y[4];
      simd::load4m<2>(a + base, lo[0], lo[1], lo[2], lo[3]);
      simd::load4m<2>(a + base + 16, hi[0], hi[1], hi[2], hi[3]);
      for (int j = 0; j < 4; ++j) {
        x[j] = Backend::unzip_even(lo[j], hi[j]);
        y[j] = Backend::unzip_odd(lo[j], hi[j]);
      }
      if constexpr (INVERSE) {
        for (int j = 0; j < 4; ++j) {
          const auto u = x[j];
          x[j] = simd::add(u, y[j], R); y[j] = simd::sub(u, y[j], R);
        }
      }
      split4_core<INVERSE, true>(x, wa, wb, wc, vm);
      split4_core<INVERSE, false>(y, wa, wb, wc, vm);
      for (int j = 0; j < 4; ++j) {
        if constexpr (!INVERSE) {
          const auto u = x[j];
          x[j] = simd::add(u, y[j], R); y[j] = simd::sub(u, y[j], R);
        }
        if constexpr (CANON) {
          x[j] = simd::reduce_full(x[j], MOD); y[j] = simd::reduce_full(y[j], MOD);
        }
        lo[j] = Backend::zip_low(x[j], y[j]); hi[j] = Backend::zip_high(x[j], y[j]);
      }
      simd::store4m<2>(a + base, lo[0], lo[1], lo[2], lo[3]);
      simd::store4m<2>(a + base + 16, hi[0], hi[1], hi[2], hi[3]);
    }
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
    if constexpr (W == 2 && COMPACT && simd::backend::split_radix8) {
      if (count >= 32) { radix8_split<false, LAST>(a, s, count); return; }
    }
    const uint32_t* A = fwd_.data() + off_[s];
    const uint32_t* B = A + W;
    const uint32_t* C = B + W;
    alignas(64) uint32_t tw[X86_SHOUP ? 6 : 3][simd::lane];
    for (int i = 0; i < simd::lane; ++i) {
      tw[0][i] = A[i % W];
      tw[1][i] = B[i % W];
      tw[2][i] = C[i % W];
      if constexpr (X86_SHOUP) {
        tw[3][i] = A[3*W + i%W]; tw[4][i] = B[3*W + i%W]; tw[5][i] = C[3*W + i%W];
      }
    }
    const auto wa = tw_load(tw[0], 3*simd::lane);
    const auto wb = tw_load(tw[1], 3*simd::lane), wc = tw_load(tw[2], 3*simd::lane);
    const auto vm = tw_fourth<false>();
    const uint32_t step = 4u * static_cast<uint32_t>(simd::lane);
    uint32_t base = 0;
    for (; base + step <= count; base += step) {
      uint32_t* p = a + base;
      simd_t x0, x1, x2, x3;
      simd::load4m<W>(p, x0, x1, x2, x3);
      simd_t t0 = simd::add(x0, x2, R);
      simd_t t1 = simd::sub(x0, x2, R);
      simd_t t2 = simd::add(x1, x3, R);
      simd_t t3 = submul(x1, x3, vm);
      simd_t y0 = simd::add(t0, t2, R);
      simd_t y1, y2, y3;
      if constexpr (W == 1) {
        // All three twiddles are Montgomery one for this terminal stage.
        y1 = simd::sub(t0, t2, R);
        y2 = simd::add(t1, t3, R);
        y3 = simd::sub(t1, t3, R);
      } else if constexpr (W == 2 && X86_SHOUP) {
        // Every even lane has k=0, hence its twiddle is unity. Multiply only
        // odd lanes; reduce the sum so its even lanes can pass through.
        y1 = simd::mul_unity_even<MOD, NINV>(simd::sub(t0, t2, R), tw_eighth_mont<2>());
        y2 = simd::mul_unity_even<MOD, NINV>(simd::add(t1, t3, R), tw_eighth_mont<1>());
        y3 = simd::mul_unity_even<MOD, NINV>(simd::sub(t1, t3, R), tw_eighth_mont<3>());
      } else {
        y1 = submul(t0, t2, wb);
        y2 = addmul(t1, t3, wa);
        y3 = submul(t1, t3, wc);
      }
      if constexpr (W == 2 && FUSE_RADIX2) {
        y0 = radix2_vec(y0); y1 = radix2_vec(y1);
        y2 = radix2_vec(y2); y3 = radix2_vec(y3);
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
          z1 = smul(z1, B[k]);
          z2 = smul(z2, A[k]);
          z3 = smul(z3, C[k]);
        }
        if constexpr (LAST && LAZY) {
          z0 = z0 >= MOD ? z0 - MOD : z0; z1 = z1 >= MOD ? z1 - MOD : z1;
          z2 = z2 >= MOD ? z2 - MOD : z2; z3 = z3 >= MOD ? z3 - MOD : z3;
        }
        p[k] = z0; p[k + W] = z1;
        p[k + 2 * W] = z2; p[k + 3 * W] = z3;
      }
      if constexpr (W == 2 && FUSE_RADIX2) radix2_fwd<LAST>(p, len);
    }
  }

  /// Inverse counterpart of radix4_small.
  template <int W, bool LAST>
  void radix4_small_inv(uint32_t* a, uint32_t len, size_t s, uint32_t count) const {
    if constexpr (W == 2 && COMPACT && !LAST && simd::backend::split_radix8) {
      if (count >= 32) { radix8_split<true, false>(a, s, count); return; }
    }
    const uint32_t* IA = inv_.data() + off_[s];
    const uint32_t* IB = IA + W;
    const uint32_t* IC = IB + W;
    alignas(64) uint32_t tw[X86_SHOUP ? 6 : 3][simd::lane];
    for (int i = 0; i < simd::lane; ++i) {
      tw[0][i] = IA[i % W];
      tw[1][i] = IB[i % W];
      tw[2][i] = IC[i % W];
      if constexpr (X86_SHOUP) {
        tw[3][i] = IA[3*W + i%W]; tw[4][i] = IB[3*W + i%W]; tw[5][i] = IC[3*W + i%W];
      }
    }
    const auto iw1 = tw_load(tw[0], 3*simd::lane);
    const auto iw2 = tw_load(tw[1], 3*simd::lane), iw3 = tw_load(tw[2], 3*simd::lane);
    const auto vm = tw_fourth<true>();
    const uint32_t step = 4u * static_cast<uint32_t>(simd::lane);
    uint32_t base = 0;
    for (; base + step <= count; base += step) {
      uint32_t* p = a + base;
      simd_t b0, b1, b2, b3;
      simd::load4m<W>(p, b0, b1, b2, b3);
      if constexpr (W == 2 && FUSE_RADIX2) {
        b0 = radix2_vec(b0); b1 = radix2_vec(b1);
        b2 = radix2_vec(b2); b3 = radix2_vec(b3);
      }
      if constexpr (LAST) b0 = bmul(b0, tw_const<false>(inv_n_));
      simd_t c1, c2, c3;
      if constexpr (W == 1 && !LAST) {
        c1 = b1; c2 = b2; c3 = b3;
      } else if constexpr (W == 2 && !LAST && X86_SHOUP) {
        c1 = simd::mul_unity_even<MOD, NINV>(b1, tw_eighth_mont<2, true>());
        c2 = simd::mul_unity_even<MOD, NINV>(b2, tw_eighth_mont<1, true>());
        c3 = simd::mul_unity_even<MOD, NINV>(b3, tw_eighth_mont<3, true>());
      } else {
        c1 = bmul(b1, iw2); c2 = bmul(b2, iw1); c3 = bmul(b3, iw3);
      }
      simd_t d0 = simd::add(b0, c1, R), d1 = simd::sub(b0, c1, R);
      simd_t d2 = simd::add(c2, c3, R);
      simd_t e = submul(c2, c3, vm);
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
      if constexpr (W == 2 && FUSE_RADIX2) radix2_fwd<false>(p, len);
      for (uint32_t k = 0; k < static_cast<uint32_t>(W); ++k) {
        uint32_t b0 = p[k], b1 = p[k + W], b2 = p[k + 2 * W], b3 = p[k + 3 * W];
        if constexpr (LAST) b0 = M::reduce(uint64_t(b0) * inv_n_);
        uint32_t c1 = b1, c2 = b2, c3 = b3;
        if constexpr (W != 1 || LAST) {
          c1 = smul<true>(b1, IB[k]);
          c2 = smul<true>(b2, IA[k]);
          c3 = smul<true>(b3, IC[k]);
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
  template <bool LAST, bool SHOUP>
  void radix4_fwd(uint32_t* a, uint32_t len, size_t s, uint32_t count) const {
    const uint32_t m = len >> 2;
    const uint32_t* A = fwd_.data() + off_[s];
    const uint32_t* B = A + m;
    const uint32_t* C = B + m;
    const auto vm = tw_fourth<false>();
    for (uint32_t base = 0; base < count; base += len) {
      uint32_t* p = a + base;
      // One vector butterfly over lanes of k. Written as a lambda so the main
      // loop can be unrolled: the t3 multiply feeds y2/y3, while y1 is
      // independent. Two butterflies in flight expose further parallelism.
      auto step = [&](uint32_t k) {
        simd_t x0 = simd::load(p + k), x1 = simd::load(p + k + m);
        simd_t x2 = simd::load(p + k + 2 * m), x3 = simd::load(p + k + 3 * m);
        auto wa = tw_load<SHOUP>(A + k, 3*m), wb = tw_load<SHOUP>(B + k, 3*m), wc = tw_load<SHOUP>(C + k, 3*m);
        simd_t t0 = simd::add(x0, x2, R);
        simd_t t1 = simd::sub(x0, x2, R);
        simd_t t2 = simd::add(x1, x3, R);
        simd_t t3 = submul(x1, x3, vm);
        simd_t y0 = simd::add(t0, t2, R);
        simd_t y1 = submul(t0, t2, wb);
        simd_t y2 = addmul(t1, t3, wa);
        simd_t y3 = submul(t1, t3, wc);
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
      if constexpr (simd::lane > 1 && !(SHOUP && simd::lane == 8)) {
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
        p[k + m] = smul<false, SHOUP>(detail::s_sub(t0, t2, R), B[k]);
        p[k + 2 * m] = smul<false, SHOUP>(detail::s_add(t1, t3, R), A[k]);
        p[k + 3 * m] = smul<false, SHOUP>(detail::s_sub(t1, t3, R), C[k]);
      }
    }
  }

  /// One radix-4 DIT butterfly stage (inverse of radix4_fwd at the same `len`).
  template <bool LAST, bool SHOUP>
  void radix4_inv(uint32_t* a, uint32_t len, size_t s, uint32_t count) const {
    const uint32_t m = len >> 2;
    const uint32_t* IA = inv_.data() + off_[s];
    const uint32_t* IB = IA + m;
    const uint32_t* IC = IB + m;
    const auto vm = tw_fourth<true>();
    for (uint32_t base = 0; base < count; base += len) {
      uint32_t* p = a + base;
      auto step = [&](uint32_t k) {
        simd_t b0 = simd::load(p + k), b1 = simd::load(p + k + m);
        simd_t b2 = simd::load(p + k + 2 * m), b3 = simd::load(p + k + 3 * m);
        auto iw1 = tw_load<SHOUP>(IA + k, 3*m), iw2 = tw_load<SHOUP>(IB + k, 3*m), iw3 = tw_load<SHOUP>(IC + k, 3*m);
        if constexpr (LAST) b0 = bmul(b0, tw_const<false>(inv_n_));
        simd_t c1 = bmul(b1, iw2);
        simd_t c2 = bmul(b2, iw1);
        simd_t c3 = bmul(b3, iw3);
        simd_t d0 = simd::add(b0, c1, R), d1 = simd::sub(b0, c1, R);
        simd_t d2 = simd::add(c2, c3, R);
        simd_t e = submul(c2, c3, vm);
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
      if constexpr (simd::lane > 1 && !(SHOUP && simd::lane == 8)) {
        for (; k + 2 * simd::lane <= m; k += 2 * simd::lane) {
          step(k);
          step(k + simd::lane);
        }
      }
      for (; k + simd::lane <= m; k += simd::lane) step(k);
      for (; k < m; ++k) {
        uint32_t b0 = p[k], b1 = p[k + m], b2 = p[k + 2 * m], b3 = p[k + 3 * m];
        if constexpr (LAST) b0 = M::reduce(uint64_t(b0) * inv_n_);
        uint32_t c1 = smul<true, SHOUP>(b1, IB[k]);
        uint32_t c2 = smul<true, SHOUP>(b2, IA[k]);
        uint32_t c3 = smul<true, SHOUP>(b3, IC[k]);
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

  static simd_t radix2_vec(simd_t v) {
    const simd_t w = simd::swap_pairs(v);
    return simd::pick_odd(simd::add(v, w, R), simd::sub(w, v, R));
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
  static constexpr size_t MAX_STAGES = size_t(ntt_max_log<M>()) / 2;
  size_t stage_count_ = 0;
  std::array<uint32_t, MAX_STAGES> lens_{};  // bounded by the modulus, no heap
  detail::AlignedBuffer fwd_, inv_;        // Montgomery/Q31/Q32, 64-byte aligned
  std::array<size_t, MAX_STAGES> off_{};    // stage -> twiddle offset
};

}  // namespace fpx

#endif  // FASTPOLY_NTT_HPP
