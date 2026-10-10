// fastpoly - modular arithmetic for primes in the signed 32-bit range.
//
// Odd-prime values are stored in Montgomery form (a * 2^32 mod p) so that modular
// multiplication becomes mul + shift + conditional subtract: no 64-bit
// division anywhere, and it maps perfectly onto SIMD lanes.
#ifndef FASTPOLY_MODINT_HPP
#define FASTPOLY_MODINT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>

namespace fpx {

/// Multiplicative inverse of an odd `a` modulo 2^32 (Newton iteration).
constexpr uint32_t inv_mod_2_32(uint32_t a) {
  uint32_t x = 1;
  for (int i = 0; i < 5; ++i) x *= 2u - a * x;
  return x;
}

/// Find a primitive root of a prime. Only used when no root is supplied;
/// explicit Mont<Mod, Root> instantiations retain their original constants.
constexpr uint32_t primitive_root(uint32_t p) {
  if (p == 2) return 1;
  std::array<uint32_t, 10> factors{};
  size_t count = 0;
  uint32_t remaining = p - 1;
  for (uint32_t d = 2; uint64_t(d)*d <= remaining; ++d) {
    if (remaining % d != 0) continue;
    factors[count++] = d;
    do { remaining /= d; } while (remaining % d == 0);
  }
  if (remaining > 1) factors[count++] = remaining;
  auto power = [p](uint64_t a, uint32_t e) {
    uint64_t r = 1;
    for (; e != 0; e >>= 1, a = a*a % p)
      if (e & 1) r = r*a % p;
    return r;
  };
  for (uint32_t root = 2; root < p; ++root) {
    bool valid = true;
    for (size_t i = 0; i < count; ++i)
      if (power(root, (p - 1)/factors[i]) == 1) { valid = false; break; }
    if (valid) return root;
  }
  return 0;
}

/// Montgomery arithmetic in GF(Mod).
/// `Mod` must be prime with 2 <= Mod < 2^31. Root defaults to automatic
/// discovery. Characteristic two uses ordinary one-bit residues.
template <uint32_t Mod, uint32_t Root = 0>
class Mont {
 public:
  static_assert(Mod < (1u << 31), "modulus must fit a signed 32-bit lane");
  static_assert(Mod >= 2, "modulus must be prime and at least 2");

  static constexpr uint32_t mod = Mod;
  static constexpr uint32_t primitive_root = Root != 0 ? Root : fpx::primitive_root(Mod);
  static constexpr uint32_t ninv =
      static_cast<uint32_t>(0u - inv_mod_2_32(Mod));  // -Mod^{-1} mod 2^32
  /// Montgomery form of the integer 1, i.e. R mod Mod.
  static constexpr uint32_t one =
      Mod == 2 ? 1u : static_cast<uint32_t>((static_cast<uint64_t>(1) << 32) % Mod);
  /// R^2 mod Mod; `reduce(x * r2)` converts a plain x into Montgomery form.
  static constexpr uint32_t r2 =
      static_cast<uint32_t>(static_cast<uint64_t>(one) * one % Mod);
  static constexpr uint32_t inv2 = (Mod + 1) / 2;

  constexpr Mont() : v_(0) {}

  /// Implicit conversion from a plain integer (enters Montgomery form).
  /// A template so that it never competes with the copy constructor.
  template <class I, class = std::enable_if_t<std::is_integral_v<I> && !std::is_same_v<I, bool>>>
  constexpr Mont(I x) : v_(from_int_impl(static_cast<uint64_t>(x))) {}

  /// Enter Montgomery form from a plain integer.
  static constexpr Mont from_int(uint64_t x) { return wrap(from_int_impl(x)); }
  /// Wrap an already-Montgomery residue (no conversion).
  static constexpr Mont raw(uint32_t x) { return wrap(x); }

  /// Montgomery reduction of a full 64-bit product.
  static constexpr uint32_t reduce(uint64_t t) {
    if constexpr (Mod == 2) return static_cast<uint32_t>(t & 1u);
    uint32_t m = static_cast<uint32_t>(t) * ninv;
    uint64_t u = (t + static_cast<uint64_t>(m) * Mod) >> 32;
    return static_cast<uint32_t>(u >= Mod ? u - Mod : u);
  }

  /// The plain integer represented by this value.
  constexpr uint32_t val() const { return reduce(v_); }
  /// The raw Montgomery limb (what NTT butterflies operate on).
  constexpr uint32_t raw_val() const { return v_; }

  friend constexpr Mont operator+(Mont a, Mont b) {
    uint32_t s = a.v_ + b.v_;
    return wrap(s >= Mod ? s - Mod : s);
  }
  friend constexpr Mont operator-(Mont a, Mont b) {
    return wrap(a.v_ >= b.v_ ? a.v_ - b.v_ : a.v_ - b.v_ + Mod);
  }
  friend constexpr Mont operator-(Mont a) { return wrap(a.v_ == 0 ? 0 : Mod - a.v_); }
  friend constexpr Mont operator*(Mont a, Mont b) {
    return wrap(reduce(static_cast<uint64_t>(a.v_) * b.v_));
  }
  friend constexpr Mont operator/(Mont a, Mont b) { return a * b.inv(); }

  constexpr Mont& operator+=(Mont b) { return *this = *this + b; }
  constexpr Mont& operator-=(Mont b) { return *this = *this - b; }
  constexpr Mont& operator*=(Mont b) { return *this = *this * b; }
  constexpr Mont& operator/=(Mont b) { return *this = *this / b; }
  friend constexpr bool operator==(Mont a, Mont b) { return a.v_ == b.v_; }
  friend constexpr bool operator!=(Mont a, Mont b) { return a.v_ != b.v_; }
  constexpr bool is_zero() const { return v_ == 0; }
  constexpr bool operator!() const { return v_ == 0; }
  constexpr explicit operator bool() const { return v_ != 0; }

  constexpr Mont pow(uint64_t e) const {
    Mont r = from_int(1), b = *this;
    for (; e; e >>= 1) {
      if (e & 1) r *= b;
      b *= b;
    }
    return r;
  }

  /// Multiplicative inverse; 0 maps to 0.
  constexpr Mont inv() const {
    if constexpr (Mod == 2) return *this;
    else return pow(Mod - 2);
  }

  /// Square root (Tonelli-Shanks). Returns false when *this is a non-residue.
  bool sqrt(Mont& out) const {
    if constexpr (Mod == 2) { out = *this; return true; }
    const Mont one_m = from_int(1);
    if (v_ == 0) { out = *this; return true; }
    if (pow((Mod - 1) / 2) != one_m) return false;  // Euler criterion
    if (Mod % 4 == 3) {                             // fast path (most NTT primes)
      out = pow((static_cast<uint64_t>(Mod) + 1) / 4);
      return true;
    }
    uint64_t q = Mod - 1;
    int s = 0;
    while (q % 2 == 0) { q /= 2; ++s; }
    Mont z = from_int(2);
    while (z.pow((Mod - 1) / 2) != -one_m) z += one_m;
    Mont c = z.pow(q), x = pow((q + 1) / 2), t = pow(q);
    int m = s;
    while (t != one_m) {
      int i = 0;
      for (Mont t2 = t; t2 != one_m; t2 *= t2) ++i;
      Mont b = c.pow(static_cast<uint64_t>(1) << (m - i - 1));
      x *= b;
      c = b * b;
      t *= c;
      m = i;
    }
    out = x;
    return true;
  }

  std::string str() const { return std::to_string(val()); }

 private:
  static constexpr uint32_t from_int_impl(uint64_t x) {
    return static_cast<uint32_t>(x % Mod) * static_cast<uint64_t>(one) % Mod;
  }
  static constexpr Mont wrap(uint32_t v) {
    Mont m;
    m.v_ = v;
    return m;
  }
  uint32_t v_;
};

template <uint32_t M, uint32_t R>
std::ostream& operator<<(std::ostream& os, const Mont<M, R>& a) {
  return os << a.val();
}
template <uint32_t M, uint32_t R>
std::istream& operator>>(std::istream& is, Mont<M, R>& a) {
  uint64_t x = 0;
  is >> x;
  a = Mont<M, R>::from_int(x);
  return is;
}

/// Common NTT primes: p - 1 divisible by a large power of two.
using mod998244353 = Mont<998244353u, 3>;    // 119*2^23 + 1
using mod1004535809 = Mont<1004535809u, 3>;  // 479*2^21 + 1
using mod469762049 = Mont<469762049u, 3>;    //   7*2^26 + 1
using mod167772161 = Mont<167772161u, 3>;    //   5*2^25 + 1
using mod754974721 = Mont<754974721u, 11>;   //  45*2^24 + 1
using mod1224736769 = Mont<1224736769u, 3>;  //  73*2^24 + 1

/// Convenient modulus-only spelling, with the same representation and API.
template <uint32_t Mod>
using ModInt = Mont<Mod>;
using mod1000000007 = Mont<1000000007u>;

}  // namespace fpx

#endif  // FASTPOLY_MODINT_HPP
