// fastpoly - tests for the Montgomery modint layer and the SIMD kernels.
#include <random>
#include <vector>

#include "fastpoly/modint.hpp"
#include "fastpoly/simd.hpp"
#include "fp_test.hpp"

using namespace fpx;

namespace {

std::mt19937_64 rng(0x5eed1234);

uint64_t ref_add(uint64_t a, uint64_t b, uint64_t p) { return (a + b) % p; }
uint64_t ref_sub(uint64_t a, uint64_t b, uint64_t p) { return (a + p - b) % p; }
uint64_t ref_mul(uint64_t a, uint64_t b, uint64_t p) {
  return static_cast<uint64_t>((__uint128_t)a * b % p);
}
uint64_t ref_pow(uint64_t a, uint64_t e, uint64_t p) {
  uint64_t r = 1, b = a % p;
  for (; e; e >>= 1) {
    if (e & 1) r = ref_mul(r, b, p);
    b = ref_mul(b, b, p);
  }
  return r;
}

template <class M>
void check_field(const char* name) {
  for (int t = 0; t < 3000; ++t) {
    // mix of random values and boundary values
    uint32_t xa, xb;
    switch (t % 5) {
      case 0: xa = static_cast<uint32_t>(rng() % M::mod); xb = static_cast<uint32_t>(rng() % M::mod); break;
      case 1: xa = M::mod - 1; xb = M::mod - 1; break;
      case 2: xa = 0; xb = M::mod - 1; break;
      case 3: xa = 1; xb = static_cast<uint32_t>(rng() % M::mod); break;
      default: xa = M::mod - 2; xb = 2; break;
    }
    M a = M::from_int(xa), b = M::from_int(xb);
    CHECK_EQ((a + b).val(), ref_add(xa, xb, M::mod));
    CHECK_EQ((a - b).val(), ref_sub(xa, xb, M::mod));
    CHECK_EQ((a * b).val(), ref_mul(xa, xb, M::mod));
    CHECK_EQ((-a).val(), (M::mod - xa) % M::mod);
    CHECK_EQ((a * a).val(), ref_mul(xa, xa, M::mod));
    // (a+b)^2 == a^2 + 2ab + b^2  -- an internal consistency law
    CHECK_EQ(((a + b) * (a + b)).val(), (a * a + M::from_int(2) * a * b + b * b).val());
    if (t % 977 == 0) {
      M e = M::from_int(rng() % M::mod);
      CHECK_EQ(e.pow(12345).val(), ref_pow(e.val(), 12345, M::mod));
      if (!e.is_zero()) CHECK_EQ((e * e.inv()).val(), 1u);
      CHECK_EQ(M::from_int(1).val(), 1u);
      CHECK_EQ(M::from_int(M::mod).val(), 0u);
      CHECK_EQ(M::from_int(M::mod + 7).val(), 7u);
    }
  }
  std::printf("  field arithmetic (%s, mod=%u) ok\n", name, M::mod);
}

template <class M>
void check_sqrt() {
  int residues = 0, non_residues = 0;
  for (uint32_t x = 0; x < 200; ++x) {
    M v = M::from_int(x), r;
    if (v.sqrt(r)) {
      ++residues;
      CHECK_EQ((r * r).val(), x);
    } else {
      ++non_residues;
      // Euler criterion must agree
      CHECK_EQ(v.pow((M::mod - 1) / 2).val(), M::mod - 1);
    }
  }
  CHECK(residues > 50);
  CHECK(non_residues > 50);
  for (int t = 0; t < 50; ++t) {  // random squares must have roots
    M x = M::from_int(rng() % M::mod), r;
    M sq = x * x;
    CHECK(sq.sqrt(r));
    CHECK_EQ((r * r).val(), sq.val());
  }
}

/// SIMD kernels must agree with scalar Montgomery arithmetic lane by lane.
template <class M>
void check_simd() {
  constexpr int L = simd::lane;
  std::vector<uint32_t> A(L), B(L), R(L);
  for (int t = 0; t < 20000; ++t) {
    for (int i = 0; i < L; ++i) {
      A[i] = (t % 13 == 0 && i == 0) ? M::mod - 1 : static_cast<uint32_t>(rng() % M::mod);
      B[i] = (t % 13 == 0 && i == 1) ? 0 : static_cast<uint32_t>(rng() % M::mod);
    }
    auto va = simd::load(A.data()), vb = simd::load(B.data());
    simd::store(R.data(), simd::mulmod<M::mod, M::ninv>(va, vb));
    for (int i = 0; i < L; ++i) CHECK_EQ(R[i], M::reduce(uint64_t(A[i]) * B[i]));
    simd::store(R.data(), simd::add(va, vb, M::mod));
    for (int i = 0; i < L; ++i) {
      uint32_t w = A[i] + B[i];
      CHECK_EQ(R[i], w >= M::mod ? w - M::mod : w);
    }
    simd::store(R.data(), simd::sub(va, vb, M::mod));
    for (int i = 0; i < L; ++i) CHECK_EQ(R[i], A[i] >= B[i] ? A[i] - B[i] : A[i] - B[i] + M::mod);
  }
  std::printf("  simd kernels (%s, %d lanes) ok\n", simd::name, L);
}

}  // namespace

FP_TEST(modint_998244353) { check_field<mod998244353>("998244353"); }
FP_TEST(modint_1004535809) { check_field<mod1004535809>("1004535809"); }
FP_TEST(modint_469762049) { check_field<mod469762049>("469762049"); }
FP_TEST(modint_167772161) { check_field<mod167772161>("167772161"); }
FP_TEST(modint_754974721) { check_field<mod754974721>("754974721"); }
FP_TEST(modint_1224736769) { check_field<mod1224736769>("1224736769"); }

FP_TEST(modint_sqrt_tes) { check_sqrt<mod998244353>(); }  // mod % 4 == 1 -> Tonelli-Shanks
FP_TEST(modint_sqrt_p34) { check_sqrt<mod469762049>(); }  // mod % 4 == 1
FP_TEST(modint_sqrt_p34b) { check_sqrt<mod1004535809>(); }

FP_TEST(simd_kernels) { check_simd<mod998244353>(); }
FP_TEST(simd_kernels_big) { check_simd<mod1224736769>(); }

FP_TEST_MAIN()
