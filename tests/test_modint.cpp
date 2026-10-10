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
  constexpr size_t L = static_cast<size_t>(simd::lane);
  std::vector<uint32_t> A(L), B(L), R(L);
  for (int t = 0; t < 20000; ++t) {
    for (size_t i = 0; i < L; ++i) {
      A[i] = (t % 13 == 0 && i == 0) ? M::mod - 1 : static_cast<uint32_t>(rng() % M::mod);
      B[i] = (t % 13 == 0 && i == 1) ? 0 : static_cast<uint32_t>(rng() % M::mod);
    }
    auto va = simd::load(A.data()), vb = simd::load(B.data());
    simd::store(R.data(), simd::mulmod<M::mod, M::ninv>(va, vb));
    for (size_t i = 0; i < L; ++i) CHECK_EQ(R[i], M::reduce(uint64_t(A[i]) * B[i]));
    simd::store(R.data(), simd::add(va, vb, M::mod));
    for (size_t i = 0; i < L; ++i) {
      uint32_t w = A[i] + B[i];
      CHECK_EQ(R[i], w >= M::mod ? w - M::mod : w);
    }
    simd::store(R.data(), simd::sub(va, vb, M::mod));
    for (size_t i = 0; i < L; ++i)
      CHECK_EQ(R[i], A[i] >= B[i] ? A[i] - B[i] : A[i] - B[i] + M::mod);
  }
  std::printf("  simd kernels (%s, %zu lanes) ok\n", simd::name, L);
}

template <class M>
void fixed_kernel_contract() {
  constexpr size_t L = static_cast<size_t>(simd::lane);
  constexpr uint32_t bound = simd::rmod<M::mod>;
  const uint64_t inverse_r = ref_pow(M::one, M::mod - 2, M::mod);
  std::vector<uint32_t> a(L), b(L), mont(L), plain(L), out(L), encoded(L), decoded(L);
  for (uint32_t t = 0; t < 2000; ++t) {
    for (size_t k = 0; k < L; ++k) {
      const uint32_t edges[] = {0, 1, M::mod - 1, M::mod - 2};
      mont[k] = t % 3 == 0 ? edges[(t + k) % 4] : static_cast<uint32_t>(rng() % M::mod);
      plain[k] = static_cast<uint32_t>(ref_mul(mont[k], inverse_r, M::mod));
      a[k] = t % 7 == 0 ? bound - 1 : t % 7 == 1 ? 0 : static_cast<uint32_t>(rng() % bound);
      b[k] = t % 7 == 0 ? 0 : t % 7 == 1 ? bound - 1 : static_cast<uint32_t>(rng() % bound);
    }
    const auto q = simd::encode_twiddle<M::mod, M::ninv, false>(simd::load(mont.data()));
    const auto qr = simd::encode_twiddle<M::mod, M::ninv, true>(simd::load(mont.data()));
    const auto floor = simd::decode_twiddle<M::mod>(q);
    const auto nearest = simd::decode_twiddle<M::mod>(qr);
    simd::store(encoded.data(), q);
    simd::store(decoded.data(), floor.value);
    for (size_t k = 0; k < L; ++k) {
      if constexpr (simd::compact_twiddle<M::mod>) {
        CHECK_EQ(encoded[k], (uint64_t(plain[k]) << 31) / M::mod);
        CHECK_EQ(decoded[k], plain[k]);
      } else {
        CHECK_EQ(encoded[k], mont[k]);
        CHECK_EQ(decoded[k], mont[k]);
      }
    }
    simd::store(encoded.data(), qr);
    simd::store(decoded.data(), nearest.value);
    for (size_t k = 0; k < L; ++k) {
      if constexpr (simd::compact_twiddle<M::mod>) {
        CHECK_EQ(encoded[k], ((uint64_t(plain[k]) << 31) + M::mod / 2) / M::mod);
        CHECK_EQ(decoded[k], plain[k]);
      }
    }
    const auto va = simd::load(a.data()), vb = simd::load(b.data());
    simd::store(out.data(), simd::mul_twiddle<M::mod, M::ninv>(va, floor));
    for (size_t k = 0; k < L; ++k) {
      CHECK(out[k] < bound);
      CHECK_EQ(out[k] % M::mod, ref_mul(a[k], plain[k], M::mod));
    }
    simd::store(out.data(), simd::mul_twiddle_full<M::mod, M::ninv>(va, floor));
    for (size_t k = 0; k < L; ++k) CHECK_EQ(out[k], ref_mul(a[k], plain[k], M::mod));
    simd::store(out.data(), simd::mul_twiddle_diff<M::mod, M::ninv>(va, vb, nearest));
    for (size_t k = 0; k < L; ++k) {
      CHECK(out[k] < bound);
      CHECK_EQ(out[k] % M::mod, ref_mul(uint64_t(a[k]) + 2ull*M::mod - b[k], plain[k], M::mod));
    }
    simd::store(out.data(), simd::mul_twiddle_sum<M::mod, M::ninv>(va, vb, nearest));
    for (size_t k = 0; k < L; ++k) {
      CHECK(out[k] < bound);
      CHECK_EQ(out[k] % M::mod, ref_mul(uint64_t(a[k]) + b[k], plain[k], M::mod));
    }
    const size_t chosen = t % L;
    const uint32_t chosen_mont = mont[chosen];
    const auto fixed = simd::fixed_twiddle<M::mod, M::ninv>(chosen_mont);
    const auto fixed_nearest = simd::fixed_twiddle<M::mod, M::ninv, true>(chosen_mont);
    simd::store(out.data(), simd::mul_twiddle<M::mod, M::ninv>(va, fixed));
    for (size_t k = 0; k < L; ++k) {
      CHECK(out[k] < bound);
      CHECK_EQ(out[k] % M::mod, ref_mul(a[k], plain[chosen], M::mod));
    }
    simd::store(out.data(), simd::mul_twiddle_full<M::mod, M::ninv>(va, fixed));
    for (size_t k = 0; k < L; ++k) CHECK_EQ(out[k], ref_mul(a[k], plain[chosen], M::mod));
    simd::store(out.data(), simd::mul_twiddle_diff<M::mod, M::ninv>(va, vb, fixed_nearest));
    for (size_t k = 0; k < L; ++k) {
      CHECK(out[k] < bound);
      CHECK_EQ(out[k] % M::mod, ref_mul(uint64_t(a[k]) + 2ull*M::mod - b[k], plain[chosen], M::mod));
    }
    simd::store(out.data(), simd::mul_twiddle_sum<M::mod, M::ninv>(va, vb, fixed_nearest));
    for (size_t k = 0; k < L; ++k) {
      CHECK(out[k] < bound);
      CHECK_EQ(out[k] % M::mod, ref_mul(uint64_t(a[k]) + b[k], plain[chosen], M::mod));
    }
#if defined(FPX_HAVE_AVX2_INTRIN)
    simd::store(encoded.data(), fixed.quotient);
    for (size_t k = 0; k < L; ++k)
      CHECK_EQ(encoded[k], (uint64_t(plain[chosen]) << 32) / M::mod);
    const auto packed = simd::fixed_twiddle_vector<M::mod, M::ninv>(simd::load(mont.data()));
    simd::store(encoded.data(), packed.quotient); simd::store(decoded.data(), packed.value);
    for (size_t k = 0; k < L; ++k) {
      CHECK_EQ(encoded[k], (uint64_t(plain[k]) << 32) / M::mod);
      CHECK_EQ(decoded[k], plain[k]);
    }
    for (size_t k = 0; k < L; ++k) mont[k] = k % 2 ? chosen_mont : M::one;
    simd::store(out.data(), simd::mul_unity_even<M::mod, M::ninv>(va, simd::load(mont.data())));
    for (size_t k = 0; k < L; ++k) {
      CHECK(out[k] < bound);
      if (k % 2 == 0) CHECK_EQ(out[k], a[k]);
      else CHECK_EQ(out[k] % M::mod, ref_mul(a[k], plain[chosen], M::mod));
    }
#endif
  }
}

template <int W>
void chunk_layout_contract() {
  constexpr size_t L = simd::lane, count = 4*L;
  for (const size_t offset : {size_t(0), size_t(1), size_t(3), size_t(7), size_t(15)}) {
    std::vector<uint32_t> input(count + offset), result(count + offset);
    uint32_t* p = input.data() + offset;
    for (size_t i = 0; i < count; ++i) p[i] = static_cast<uint32_t>(i);
    simd::native_t x[4];
    simd::load4m<W>(p, x[0], x[1], x[2], x[3]);
    uint32_t columns[4][L];
    bool seen[count]{};
    for (size_t j = 0; j < 4; ++j) simd::store(columns[j], x[j]);
    for (size_t i = 0; i < L; ++i) {
      CHECK_EQ(columns[0][i] % (4*W), i % W);
      for (size_t j = 0; j < 4; ++j) {
        CHECK_EQ(columns[j][i], columns[0][i] + j*W);
        CHECK(columns[j][i] < count);
        CHECK(!seen[columns[j][i]]);
        seen[columns[j][i]] = true;
      }
    }
    for (size_t j = 0; j < 4; ++j) {
      for (size_t i = 0; i < L; ++i) columns[j][i] += static_cast<uint32_t>(1000*(j + 1));
      x[j] = simd::load(columns[j]);
    }
    simd::store4m<W>(result.data() + offset, x[0], x[1], x[2], x[3]);
    for (size_t i = 0; i < count; ++i)
      CHECK_EQ(result[offset + i], i + 1000*(i/W % 4 + 1));
  }
}

template <class M>
void wide_fixed_contract() {
  if constexpr (simd::backend::q32 && simd::lazy_ok<M::mod>) {
    constexpr size_t L = simd::lane;
    constexpr uint32_t P = M::mod;
    const uint32_t edges[] = {0, 1, P - 1, P, P + 1, 2*P - 1,
                              2*P, 3*P - 1, 3*P, 4*P - 1, UINT32_MAX};
    uint32_t a[L], out[L];
    for (const uint32_t w : {0u, 1u, 2u, P/2, P - 1}) {
      const uint32_t mont = M::from_int(w).raw_val();
      const auto fixed = simd::fixed_twiddle<P, M::ninv>(mont);
      for (size_t offset = 0; offset < std::size(edges); ++offset) {
        for (size_t k = 0; k < L; ++k) a[k] = edges[(offset + k) % std::size(edges)];
        const auto v = simd::load(a);
        simd::store(out, simd::mul_twiddle<P, M::ninv>(v, fixed));
        for (size_t k = 0; k < L; ++k) {
          CHECK(out[k] < 2*P);
          CHECK_EQ(out[k] % P, ref_mul(a[k], w, P));
        }
        // A canonical twiddle also permits an arbitrary unsigned Montgomery
        // multiplicand: a*mont < P*2^32. Include 4*P-1 and UINT32_MAX.
        simd::store(out, simd::mulmod_lazy<P, M::ninv>(v, simd::set1(mont)));
        for (size_t k = 0; k < L; ++k) {
          CHECK(out[k] < 2*P);
          CHECK_EQ(out[k] % P, ref_mul(a[k], w, P));
        }
      }
    }
  }
}

template <class Backend = simd::backend>
void split_radix8_layout_contract() {
  if constexpr (Backend::split_radix8) {
    constexpr size_t L = simd::lane;
    uint32_t input[2*L], even[L], odd[L], result[2*L];
    for (size_t i = 0; i < 2*L; ++i) input[i] = static_cast<uint32_t>(i);
    const auto a = simd::load(input), b = simd::load(input + L);
    const auto e = Backend::unzip_even(a, b), o = Backend::unzip_odd(a, b);
    simd::store(even, e); simd::store(odd, o);
    bool seen[L]{};
    for (size_t i = 0; i < L; ++i) {
      CHECK(even[i] < 2*L);
      CHECK_EQ(even[i] % 2, 0u);
      CHECK_EQ(odd[i], even[i] + 1);
      CHECK(!seen[even[i]/2]);
      seen[even[i]/2] = true;
    }
    simd::store(result, Backend::zip_low(e, o));
    simd::store(result + L, Backend::zip_high(e, o));
    for (size_t i = 0; i < 2*L; ++i) CHECK_EQ(result[i], input[i]);
  }
}

template <int L>
void chunk_layout_contract_all() {
  chunk_layout_contract<1>();
  if constexpr (L >= 2) chunk_layout_contract<2>();
  if constexpr (L >= 4) chunk_layout_contract<4>();
  if constexpr (L >= 8) chunk_layout_contract<8>();
  if constexpr (L >= 16) chunk_layout_contract<16>();
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

FP_TEST(fixed_simd_multipliers) {
  fixed_kernel_contract<mod998244353>();
  fixed_kernel_contract<mod1004535809>();
  fixed_kernel_contract<mod469762049>();
  fixed_kernel_contract<mod167772161>();
  fixed_kernel_contract<mod754974721>();
  fixed_kernel_contract<mod1224736769>();
  fixed_kernel_contract<Mont<1073741789u, 2>>();  // prime just below 2^30
  fixed_kernel_contract<Mont<1073479681u, 11>>(); // large NTT just below 2^30
  fixed_kernel_contract<Mont<2013265921u, 31>>(); // full reduction near 2^31
  fixed_kernel_contract<Mont<3u, 2>>();
}

FP_TEST(simd_chunk_layout) {
  chunk_layout_contract_all<simd::lane>();
  split_radix8_layout_contract();
}

FP_TEST(simd_wide_fixed_inputs) {
  wide_fixed_contract<mod998244353>();
  wide_fixed_contract<mod1004535809>();
  wide_fixed_contract<mod469762049>();
  wide_fixed_contract<mod167772161>();
  wide_fixed_contract<mod754974721>();
  wide_fixed_contract<Mont<1073741789u, 2>>();
  wide_fixed_contract<Mont<3u, 2>>();
}

FP_TEST_MAIN()
