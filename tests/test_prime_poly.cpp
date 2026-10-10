// Arbitrary-prime arithmetic, CRT products and native-NTT overflow fallbacks.
#include <limits>
#include <random>
#include "fastpoly/fastpoly.hpp"
#include "fp_test.hpp"

using namespace fpx;

namespace {

template <class M> using V = std::vector<M>;

template <class M>
V<M> random_poly(size_t n, std::mt19937_64& rng) {
  V<M> a(n);
  for (size_t i = 0; i < n; ++i)
    a[i] = M::from_int(i % 7 == 0 ? M::mod-1 : rng() % M::mod);
  return a;
}

// Ordinary integer arithmetic is independent of both Montgomery and CRT.
template <class M>
V<M> product(const V<M>& a, const V<M>& b, size_t n = SIZE_MAX) {
  if (a.empty() || b.empty() || n == 0) return {};
  const size_t count = std::min(n, a.size()+b.size()-1);
  std::vector<uint64_t> sums(count);
  for (size_t i = 0; i < a.size() && i < count; ++i)
    for (size_t j = 0; j < b.size() && j < count-i; ++j)
      sums[i+j] = (sums[i+j]+uint64_t(a[i].val())*b[j].val()) % M::mod;
  V<M> result(count);
  for (size_t i = 0; i < count; ++i) result[i] = M::from_int(sums[i]);
  return result;
}

template <class M>
V<M> inverse(const V<M>& a, size_t n) {
  V<M> b(n);
  if (n == 0) return b;
  b[0] = a[0].inv();
  for (size_t i = 1; i < n; ++i) {
    M sum;
    for (size_t j = 1; j < a.size() && j <= i; ++j) sum += a[j]*b[i-j];
    b[i] = -sum*b[0];
  }
  return b;
}

template <class M>
V<M> exponential(const V<M>& a, size_t n) {
  V<M> b(n);
  if (n == 0) return b;
  b[0] = 1;
  for (size_t i = 1; i < n; ++i) {
    M sum;
    for (size_t j = 1; j < a.size() && j <= i; ++j) sum += M::from_int(j)*a[j]*b[i-j];
    b[i] = sum/M::from_int(i);
  }
  return b;
}

template <class M>
V<M> power(V<M> a, uint64_t k, size_t n) {
  V<M> r{M(1)};
  for (; k; k >>= 1) {
    if (k & 1) r = product(r, a, n);
    if (k > 1) a = product(a, a, n);
  }
  r.resize(n);
  return r;
}

template <class F>
bool domain_failure(F&& f) {
  try { f(); } catch (const poly::domain_error&) { return true; }
  return false;
}

template <class M>
void arithmetic() {
  std::mt19937_64 rng(9124);
  CHECK_EQ(sizeof(M), sizeof(uint32_t));
  CHECK_EQ(M(0).inv(), M(0));
  for (size_t i = 0; i < 300; ++i) {
    const uint64_t a = rng() % M::mod, b = rng() % M::mod;
    const M x = M::from_int(a), y = M::from_int(b);
    CHECK_EQ(x.val(), a);
    CHECK_EQ(M::raw(x.raw_val()), x);
    CHECK_EQ((x+y).val(), (a+b) % M::mod);
    CHECK_EQ((x-y).val(), (a+M::mod-b) % M::mod);
    CHECK_EQ((x*y).val(), a*b % M::mod);
    if (b) CHECK_EQ(x/y*y, x);
    M root;
    CHECK((x*x).sqrt(root));
    CHECK_EQ(root*root, x*x);
  }
}

template <class M>
void convolutions() {
  std::mt19937_64 rng(4813);
  for (size_t len : {size_t(1), size_t(39), size_t(40), size_t(41), size_t(63),
                    size_t(65), size_t(129), size_t(259)}) {
    const auto a = random_poly<M>(len, rng), b = random_poly<M>(len+13, rng);
    for (size_t out : {size_t(0), size_t(1), len, 2*len+12}) {
      const auto want = product(a, b, out);
      CHECK(poly::conv(a, b, out) == want);
      CHECK(poly::conv(b, a, out) == want);
      // Exercise the exact production block loop across many boundaries,
      // including final short blocks, clipped outputs and large CRT digits.
      CHECK((poly::detail::crt_convolution<M, 64>(a, b, want.size()) == want));
    }
    CHECK(poly::conv(a, a) == product(a, a));
    const auto copy = a;
    CHECK(poly::conv(a, copy) == product(a, a));
    const V<M> extremes(len, M::from_int(M::mod-1));
    CHECK(poly::conv(extremes, extremes) == product(extremes, extremes));
    CHECK((poly::detail::crt_convolution<M, 64>(extremes, extremes, 2*len-1)
           == product(extremes, extremes)));
  }
  CHECK(poly::conv(V<M>{}, V<M>{1}).empty());
}

template <class M>
void series() {
  std::mt19937_64 rng(974);
  for (size_t n : {size_t(1), size_t(31), size_t(32), size_t(33), size_t(40),
                   size_t(41), size_t(64), size_t(65), size_t(73), size_t(127),
                   size_t(128), size_t(129), size_t(257), size_t(513)}) {
    auto a = random_poly<M>(n+11, rng);
    a[0] = M::from_int(M::mod-1);
    for (size_t len : {size_t(1), size_t(2), size_t(8), a.size()}) {
      V<M> input(a.begin(), a.begin()+static_cast<std::ptrdiff_t>(len));
      CHECK(poly::inv(input, n) == inverse(input, n));
    }
    if (n <= M::mod) {
      a[0] = 0;
      auto expected = exponential(a, n);
      CHECK(poly::exp(a, n) == expected);
      auto prefix = a; prefix.resize(n);
      CHECK(poly::log(expected, n) == prefix);
      a[0] = 1; prefix[0] = 1;
      CHECK(poly::exp(poly::log(a, n), n) == prefix);
      for (size_t len : {size_t(1), size_t(2), size_t(7)}) {
        auto input = random_poly<M>(len, rng); input[0] = 0;
        CHECK(poly::exp(input, n) == exponential(input, n));
      }
    }
    a.resize(n); a[0] = M::from_int(M::mod-1);
    const auto square = product(a, a, n);
    const auto root = poly::sqrt(square, n);
    CHECK_EQ(root.size(), n);
    CHECK(product(root, root, n) == square);
    if (n <= 129) {
      for (uint64_t k : {uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(65),
                         uint64_t(M::mod)+1, std::numeric_limits<uint64_t>::max()})
        CHECK(poly::pow(a, k, n) == power(a, k, n));
    }
  }
  // Division exercises the reverse-series fallback and large remainders.
  auto divisor = random_poly<M>(143, rng); divisor.back() = 1;
  auto quotient = random_poly<M>(177, rng); quotient.back() = 1;
  auto remainder = random_poly<M>(141, rng);
  poly::trim(remainder);
  auto dividend = poly::add(product(divisor, quotient), remainder);
  const auto [q, r] = poly::divmod(dividend, divisor);
  CHECK(q == quotient); CHECK(r == remainder);
  CHECK(poly::div(dividend, divisor) == quotient);
  CHECK(poly::mod(dividend, divisor) == remainder);
  CHECK(poly::div(dividend, V<M>{1}) == dividend);
  CHECK(poly::inv(V<M>{}, 0).empty());
  CHECK(poly::exp(V<M>{}, 0).empty());
  CHECK(poly::log(V<M>{}, 0).empty());
  CHECK(poly::sqrt(V<M>{}, 0).empty());
  CHECK(domain_failure([] { (void)poly::inv(V<M>{0, 1}, 8); }));
}

template <class M>
void large_identities() {
  std::mt19937_64 rng(4013);
  const size_t n = 8193;
  auto a = random_poly<M>(n, rng); a[0] = 1;
  auto unit = poly::conv(a, poly::inv(a, n), n);
  CHECK_EQ(unit[0], M(1));
  for (size_t i = 1; i < n; ++i) CHECK_EQ(unit[i], M(0));
  if (n <= M::mod) CHECK(poly::exp(poly::log(a, n), n) == a);
  const auto square = poly::conv(a, a, n), root = poly::sqrt(square, n);
  CHECK(poly::conv(root, root, n) == square);
}

}  // namespace

FP_TEST(prime_modint) {
  arithmetic<ModInt<2>>();
  arithmetic<ModInt<3>>();
  arithmetic<ModInt<641>>();
  arithmetic<mod1000000007>();
  arithmetic<ModInt<2147483647>>();
  static_assert(ModInt<998244353>::primitive_root == 3);
  static_assert(ModInt<1000000007>::primitive_root == 5);
}

FP_TEST(prime_convolution) {
  convolutions<ModInt<2>>();
  convolutions<ModInt<3>>();
  convolutions<ModInt<193>>(); // native NTT below 64, CRT beyond it
  convolutions<mod1000000007>();
  convolutions<ModInt<2147483647>>();
  convolutions<mod998244353>(); // existing path plus forced block coverage
}

FP_TEST(prime_series) {
  series<ModInt<2>>();
  series<ModInt<3>>();
  series<ModInt<193>>();
  series<ModInt<641>>(); // exp/sqrt double from native NTT to CRT
  series<mod1000000007>();
  series<ModInt<2147483647>>();
}

FP_TEST(prime_large_series) {
  large_identities<mod1000000007>();
  large_identities<ModInt<2147483647>>();
  large_identities<ModInt<3>>();
}

FP_TEST(prime_small_characteristic_contracts) {
  using M = ModInt<3>;
  CHECK((poly::exp(V<M>{0, 1}, 3) == V<M>{1, 1, 2}));
  CHECK((poly::log(V<M>{1, 1, 2}, 3) == V<M>{0, 1, 0}));
  CHECK(domain_failure([] { (void)poly::integral(V<M>{1, 1, 1}); }));
  CHECK(domain_failure([] { (void)poly::inv_series<M>(3); }));
  CHECK(domain_failure([] { (void)poly::log(V<M>{1, 1}, 4); }));
  CHECK(domain_failure([] { (void)poly::exp(V<M>{0, 1}, 4); }));
  // (1+x)^p = 1+x^p: reducing the exponent modulo p loses the x^p term.
  CHECK((poly::pow(V<M>{1, 1}, 81, 90) == power(V<M>{1, 1}, 81, 90)));
  using B = ModInt<2>;
  CHECK((poly::exp(V<B>{0, 1}, 2) == V<B>{1, 1}));
  CHECK((poly::log(V<B>{1, 1}, 2) == V<B>{0, 1}));
  CHECK((poly::integral(V<B>{1}) == V<B>{0, 1}));
  CHECK((poly::derivative(V<B>{1, 1, 1, 1}) == V<B>{1, 0, 1}));
  CHECK((poly::sqrt(V<B>{1, 0, 1, 0, 1}, 6) == V<B>{1, 1, 1, 0, 0, 0}));
  CHECK(domain_failure([] { (void)poly::sqrt(V<B>{1, 1}, 6); }));
  CHECK((poly::sqrt(V<B>{1, 1}, 1) == V<B>{1}));
  CHECK((poly::mul_scalar(V<B>{1, 0, 1}, B(1)) == V<B>{1, 0, 1}));
  CHECK((poly::mul_scalar(V<B>{1, 0, 1}, B(0)) == V<B>{0, 0, 0}));
  // Modulus-only syntax also takes the same native NTT as explicit roots.
  using A = ModInt<998244353>;
  V<A> input(256);
  for (size_t i = 0; i < input.size(); ++i) input[i] = A::from_int(i+1);
  auto transformed = input;
  const auto& plan = NttPlan<A>::get_ref(256);
  plan.forward(reinterpret_cast<uint32_t*>(transformed.data()));
  plan.inverse(reinterpret_cast<uint32_t*>(transformed.data()));
  CHECK(transformed == input);
}

FP_TEST(prime_poly_method_api) {
  using M = mod1000000007;
  const Poly<M> a{1, 2, 3}, b{1, 1};
  CHECK(((a+b)-b) == a);
  CHECK((a*(-M(1))) == -a);
  CHECK_EQ((a << 2)[2], a[0]);
  CHECK_EQ(a.eval(M(2)), M(17));
  CHECK(a.integral().derivative() == a);
  CHECK_EQ(a.inv(257).size(), 257u);
  auto padded = a; padded.resize(257);
  CHECK(a.log(257).exp(257) == padded);
  const auto square = a.mul(a, 257);
  CHECK(square.sqrt(257).mul(square.sqrt(257), 257) == a.pow(2, 257));
  CHECK((a*b).div(b) == a);
  CHECK((a*b).mod(b).empty());
}

FP_TEST_MAIN()
