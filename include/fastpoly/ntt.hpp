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

  /// Forward DIF transform, in place, on Montgomery limbs.
  void forward(uint32_t* a) const {
    for (size_t s = 0; s < lens_.size(); ++s) radix4_fwd(a, lens_[s], s);
    if (logn_ & 1) radix2_fwd(a);
  }

  /// Inverse DIT transform (already scaled by 1/n), in place.
  void inverse(uint32_t* a) const {
    if (logn_ & 1) radix2_fwd(a);  // len == 2 butterflies are self-transpose
    for (size_t s = lens_.size(); s-- > 0;) radix4_inv(a, lens_[s], s);
    const uint32_t in = inv_n_;
    constexpr uint32_t MOD = M::mod;
    uint32_t k = 0;
    for (; k + simd::lane <= n_; k += simd::lane) {
      simd::store(a + k, simd::mulmod<MOD, M::ninv>(simd::load(a + k), simd::set1(in)));
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

  void build() {
    constexpr uint32_t MOD = M::mod;
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

  /// One radix-4 DIF butterfly stage over blocks of length `len`.
  void radix4_fwd(uint32_t* a, uint32_t len, size_t s) const {
    constexpr uint32_t MOD = M::mod;
    const uint32_t m = len >> 2;
    const uint32_t* A = fwd_.data() + off_[s];
    const uint32_t* B = A + m;
    const uint32_t* C = B + m;
    const simd_t vm = simd::set1(w4_);
    for (uint32_t base = 0; base < n_; base += len) {
      uint32_t* p = a + base;
      uint32_t k = 0;
      for (; k + simd::lane <= m; k += simd::lane) {
        simd_t x0 = simd::load(p + k), x1 = simd::load(p + k + m);
        simd_t x2 = simd::load(p + k + 2 * m), x3 = simd::load(p + k + 3 * m);
        simd_t wa = simd::load(A + k), wb = simd::load(B + k), wc = simd::load(C + k);
        simd_t t0 = simd::add(x0, x2, MOD);
        simd_t t1 = simd::sub(x0, x2, MOD);
        simd_t t2 = simd::add(x1, x3, MOD);
        simd_t t3 = simd::mulmod<MOD, M::ninv>(simd::sub(x1, x3, MOD), vm);
        simd::store(p + k, simd::add(t0, t2, MOD));
        simd::store(p + k + m, simd::mulmod<MOD, M::ninv>(simd::sub(t0, t2, MOD), wb));
        simd::store(p + k + 2 * m, simd::mulmod<MOD, M::ninv>(simd::add(t1, t3, MOD), wa));
        simd::store(p + k + 3 * m, simd::mulmod<MOD, M::ninv>(simd::sub(t1, t3, MOD), wc));
      }
      for (; k < m; ++k) {
        uint32_t x0 = p[k], x1 = p[k + m], x2 = p[k + 2 * m], x3 = p[k + 3 * m];
        uint32_t t0 = detail::s_add(x0, x2, MOD);
        uint32_t t1 = detail::s_sub(x0, x2, MOD);
        uint32_t t2 = detail::s_add(x1, x3, MOD);
        uint32_t t3 = M::reduce(uint64_t(detail::s_sub(x1, x3, MOD)) * w4_);
        p[k] = detail::s_add(t0, t2, MOD);
        p[k + m] = M::reduce(uint64_t(detail::s_sub(t0, t2, MOD)) * B[k]);
        p[k + 2 * m] = M::reduce(uint64_t(detail::s_add(t1, t3, MOD)) * A[k]);
        p[k + 3 * m] = M::reduce(uint64_t(detail::s_sub(t1, t3, MOD)) * C[k]);
      }
    }
  }

  /// One radix-4 DIT butterfly stage (inverse of radix4_fwd at the same `len`).
  void radix4_inv(uint32_t* a, uint32_t len, size_t s) const {
    constexpr uint32_t MOD = M::mod;
    const uint32_t m = len >> 2;
    const uint32_t* IA = inv_.data() + off_[s];
    const uint32_t* IB = IA + m;
    const uint32_t* IC = IB + m;
    const simd_t vm = simd::set1(w4_inv_);
    for (uint32_t base = 0; base < n_; base += len) {
      uint32_t* p = a + base;
      uint32_t k = 0;
      for (; k + simd::lane <= m; k += simd::lane) {
        simd_t b0 = simd::load(p + k), b1 = simd::load(p + k + m);
        simd_t b2 = simd::load(p + k + 2 * m), b3 = simd::load(p + k + 3 * m);
        simd_t iw1 = simd::load(IA + k), iw2 = simd::load(IB + k), iw3 = simd::load(IC + k);
        simd_t c1 = simd::mulmod<MOD, M::ninv>(b1, iw2);
        simd_t c2 = simd::mulmod<MOD, M::ninv>(b2, iw1);
        simd_t c3 = simd::mulmod<MOD, M::ninv>(b3, iw3);
        simd_t d0 = simd::add(b0, c1, MOD), d1 = simd::sub(b0, c1, MOD);
        simd_t d2 = simd::add(c2, c3, MOD), d3 = simd::sub(c2, c3, MOD);
        simd_t e = simd::mulmod<MOD, M::ninv>(d3, vm);
        simd::store(p + k, simd::add(d0, d2, MOD));
        simd::store(p + k + m, simd::add(d1, e, MOD));
        simd::store(p + k + 2 * m, simd::sub(d0, d2, MOD));
        simd::store(p + k + 3 * m, simd::sub(d1, e, MOD));
      }
      for (; k < m; ++k) {
        uint32_t b0 = p[k], b1 = p[k + m], b2 = p[k + 2 * m], b3 = p[k + 3 * m];
        uint32_t c1 = M::reduce(uint64_t(b1) * IB[k]);
        uint32_t c2 = M::reduce(uint64_t(b2) * IA[k]);
        uint32_t c3 = M::reduce(uint64_t(b3) * IC[k]);
        uint32_t d0 = detail::s_add(b0, c1, MOD), d1 = detail::s_sub(b0, c1, MOD);
        uint32_t d2 = detail::s_add(c2, c3, MOD), d3 = detail::s_sub(c2, c3, MOD);
        uint32_t e = M::reduce(uint64_t(d3) * w4_inv_);
        p[k] = detail::s_add(d0, d2, MOD);
        p[k + m] = detail::s_add(d1, e, MOD);
        p[k + 2 * m] = detail::s_sub(d0, d2, MOD);
        p[k + 3 * m] = detail::s_sub(d1, e, MOD);
      }
    }
  }

  /// Trailing radix-2 stage used when log2(n) is odd (len == 2, twiddle == 1).
  /// The butterfly is its own transpose, so this serves both directions.
  void radix2_fwd(uint32_t* a) const {
    constexpr uint32_t MOD = M::mod;
    for (uint32_t k = 0; k < n_; k += 2) {
      uint32_t x = a[k], y = a[k + 1];
      a[k] = detail::s_add(x, y, MOD);
      a[k + 1] = detail::s_sub(x, y, MOD);
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
