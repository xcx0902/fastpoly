// fastpoly - polynomial arithmetic modulo x^n over an NTT prime field.
// Implementation headers follow the algebra's dependency graph; this facade
// supplies the complete free-function API and vector-compatible method syntax.
#ifndef FASTPOLY_POLY_HPP
#define FASTPOLY_POLY_HPP

#include "fastpoly/detail/poly/power.hpp"
#include "fastpoly/detail/poly/square_root.hpp"
#include "fastpoly/detail/poly/division.hpp"

namespace fpx {

/// Public polynomial type: std::vector<M> plus method syntax
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
