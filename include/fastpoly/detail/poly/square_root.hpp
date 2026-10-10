// fastpoly - formal square roots and valuation handling.
#ifndef FASTPOLY_DETAIL_POLY_SQUARE_ROOT_HPP
#define FASTPOLY_DETAIL_POLY_SQUARE_ROOT_HPP

#include "fastpoly/detail/poly/inverse.hpp"

namespace fpx::poly {

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

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_SQUARE_ROOT_HPP
