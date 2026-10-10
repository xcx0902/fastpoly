// fastpoly - logarithms and exponentials, including linear scans.
#ifndef FASTPOLY_DETAIL_POLY_EXPONENTIAL_HPP
#define FASTPOLY_DETAIL_POLY_EXPONENTIAL_HPP

#include "fastpoly/detail/poly/inverse.hpp"
#include "fastpoly/detail/poly/calculus.hpp"

namespace fpx::poly {

namespace detail {

// Inclusive four-lane product scan, with Montgomery one in the shifted lanes.
template <class M, class Backend = simd::backend>
simd::native_t prefix_product4(simd::native_t x) {
  const auto one = simd::set1(M::one);
  x = simd::butterfly_mul<M::mod, M::ninv, simd::lazy_ok<M::mod>>(
      x, Backend::template shift_right_lanes<1>(one, x));
  return simd::butterfly_mul<M::mod, M::ninv, simd::lazy_ok<M::mod>>(
      x, Backend::template shift_right_lanes<2>(one, x));
}

// exp(c*x)[i] = c^i/i!. Compute the last coefficient using one factorial
// inversion, then generate the output backwards. Four local scans and a scan
// of their endpoints expose sixteen independent coefficients per block; only
// the scalar endpoint carries a dependency between successive blocks.
template <class M, class Backend = simd::backend>
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
    auto x0 = prefix_product4<M, Backend>(factors[0]), x1 = prefix_product4<M, Backend>(factors[1]);
    auto x2 = prefix_product4<M, Backend>(factors[2]), x3 = prefix_product4<M, Backend>(factors[3]);
    const auto ends = Backend::endpoints(x0, x1, x2, x3);
    const auto joined = prefix_product4<M, Backend>(ends);
    const auto scales = simd::butterfly_mul<M::mod, M::ninv, simd::lazy_ok<M::mod>>(
        simd::set1(previous.raw_val()), Backend::template shift_right_lanes<1>(simd::set1(M::one), joined));
    x0 = simd::mulmod<M::mod, M::ninv>(x0, Backend::template broadcast<0>(scales));
    x1 = simd::mulmod<M::mod, M::ninv>(x1, Backend::template broadcast<1>(scales));
    x2 = simd::mulmod<M::mod, M::ninv>(x2, Backend::template broadcast<2>(scales));
    x3 = simd::mulmod<M::mod, M::ninv>(x3, Backend::template broadcast<3>(scales));
    previous = M::raw(Backend::last_lane(x3));
    simd::store(dst + pos - 4, Backend::reverse(x0)); simd::store(dst + pos - 8, Backend::reverse(x1));
    simd::store(dst + pos - 12, Backend::reverse(x2)); simd::store(dst + pos - 16, Backend::reverse(x3));
    for (auto& factor : factors) factor = simd::sub(factor, decrement, M::mod);
  }
  for (; pos != 0; --pos) {
    previous *= M::from_int(pos)*inverse_c;
    b[pos-1] = previous;
  }
  return b;
}

}  // namespace detail

/// log(a) = integral(a' / a) mod x^n   (requires a(0) == 1)
template <class M>
vec<M> log(const vec<M>& a, size_t n) {
  if (n == 0) return {};
  if (a.empty() || a[0] != M::from_int(1))
    throw domain_error("poly::log: constant term must be 1");
  if (n > M::mod)
    throw domain_error("poly::log: required denominators must be invertible (n <= mod)");
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
  if constexpr (M::mod != 2 && simd::backend::parallel_linear_exp) {
    if (aprime.size() == 1 && n >= 256) return detail::exp_linear(aprime[0], n);
  }
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
    if (m > ntt_max_size<M>()/2) {
      auto product = conv_limbs(aprime, b, m2-1);
      vec<M> residual(hlen);
      for (size_t i = 0; i < hlen && m-1+i < product.size(); ++i)
        residual[i] = product[m-1+i];
      auto correction = conv_limbs(residual, c, hlen);
      correction.resize(hlen);
      for (size_t i = 0; i < hlen; ++i) correction[i] *= denominators[m+i];
      correction = conv_limbs(correction, b, hlen);
      correction.resize(hlen);
      b.insert(b.end(), correction.begin(), correction.end());
      continue;
    }
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

}  // namespace fpx::poly

#endif  // FASTPOLY_DETAIL_POLY_EXPONENTIAL_HPP
