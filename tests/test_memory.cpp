// fastpoly - scratch-memory ownership, reuse, bounds and concurrency.
// A small cache budget reaches eviction boundaries without large test RSS.
#ifndef FASTPOLY_SCRATCH_CACHE_BYTES
#define FASTPOLY_SCRATCH_CACHE_BYTES (1024u * 1024u)
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "fastpoly/memory.hpp"
#include "fastpoly/poly.hpp"
#include "fp_test.hpp"

using fpx::detail::ScratchBuffer;

static_assert(!std::is_copy_constructible_v<ScratchBuffer>);
static_assert(!std::is_copy_assignable_v<ScratchBuffer>);
static_assert(std::is_nothrow_move_constructible_v<ScratchBuffer>);
static_assert(std::is_nothrow_move_assignable_v<ScratchBuffer>);

namespace {

bool aligned(const ScratchBuffer& buffer) {
  return reinterpret_cast<std::uintptr_t>(buffer.data()) % 64 == 0;
}

uint32_t value(size_t i, uint32_t salt) {
  return static_cast<uint32_t>(i * 2654435761ull) ^ salt;
}

void write(ScratchBuffer& buffer, uint32_t salt) {
  for (size_t i = 0; i < buffer.size(); ++i) buffer.data()[i] = value(i, salt);
}

bool intact(const ScratchBuffer& buffer, uint32_t salt) {
  for (size_t i = 0; i < buffer.size(); ++i)
    if (buffer.data()[i] != value(i, salt)) return false;
  return true;
}

template <class Field>
std::vector<Field> reference_product(const std::vector<Field>& a,
                                     const std::vector<Field>& b, size_t n);

uint32_t ordinary_multiply(uint32_t a, uint32_t b, uint32_t modulus) {
#if defined(__SIZEOF_INT128__)
  return static_cast<uint32_t>((static_cast<unsigned __int128>(a) * b) % modulus);
#else
  // The supported moduli are < 2^31, so this fallback is also exact.
  return static_cast<uint32_t>((uint64_t(a) * b) % modulus);
#endif
}

uint32_t ordinary_power(uint32_t a, uint64_t exponent, uint32_t modulus) {
  uint32_t out = 1;
  for (; exponent != 0; exponent >>= 1) {
    if (exponent & 1) out = ordinary_multiply(out, a, modulus);
    a = ordinary_multiply(a, a, modulus);
  }
  return out;
}

template <class Field>
std::vector<Field> ordinary_inverse(const std::vector<Field>& a, size_t n) {
  std::vector<uint32_t> coefficients(n);
  const uint32_t inverse0 = ordinary_power(a[0].val(), Field::mod - 2, Field::mod);
  coefficients[0] = inverse0;
  for (size_t i = 1; i < n; ++i) {
    uint32_t sum = 0;
    for (size_t j = 1; j <= i && j < a.size(); ++j)
      sum = static_cast<uint32_t>((uint64_t(sum) + ordinary_multiply(
          a[j].val(), coefficients[i - j], Field::mod)) % Field::mod);
    coefficients[i] = ordinary_multiply(sum == 0 ? 0 : Field::mod - sum, inverse0, Field::mod);
  }
  std::vector<Field> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = Field::from_int(coefficients[i]);
  return out;
}

template <class Field>
std::vector<Field> ordinary_linear_power(uint32_t c, uint32_t d, size_t shift,
                                        uint64_t exponent, size_t n) {
  std::vector<Field> out(n);
  if (shift != 0 && exponent > (n - 1) / shift) return out;
  const size_t offset = shift * static_cast<size_t>(exponent);
  const uint32_t ratio = ordinary_multiply(d,
      ordinary_power(c, Field::mod - 2, Field::mod), Field::mod);
  uint32_t scale = ordinary_power(c, exponent, Field::mod), binomial = 1;
  for (size_t j = 0; j < n - offset; ++j) {
    if (j != 0) {
      if (uint64_t(j) > exponent) binomial = 0;
      else {
        binomial = ordinary_multiply(binomial,
            static_cast<uint32_t>((exponent - uint64_t(j) + 1) % Field::mod), Field::mod);
        binomial = ordinary_multiply(binomial,
            ordinary_power(static_cast<uint32_t>(j), Field::mod - 2, Field::mod), Field::mod);
      }
      scale = ordinary_multiply(scale, ratio, Field::mod);
    }
    out[offset + j] = Field::from_int(ordinary_multiply(binomial, scale, Field::mod));
  }
  return out;
}

template <class Field>
void short_inverse_boundaries() {
  for (const size_t degree : {size_t(1), size_t(2), size_t(4), size_t(8)}) {
    std::vector<Field> a(degree + 1);
    a[0] = Field::from_int(7);  // Exercise nonunit constant normalization.
    for (size_t i = 1; i < a.size(); ++i)
      a[i] = Field::from_int(i % 2 ? Field::mod - 1 : value(i, 23));
    for (const size_t n : {size_t(1), size_t(33), size_t(257), size_t(513),
                           size_t(4095), size_t(4096), size_t(4097)}) {
      const auto want = ordinary_inverse(a, n);
      CHECK_MSG(fpx::poly::inv(a, n) == want,
                "short inverse mod=%u degree=%zu n=%zu", Field::mod, degree, n);
      auto padded = a;
      padded.resize(2 * n + 17);
      CHECK_MSG(fpx::poly::inv(padded, n) == want,
                "padded inverse mod=%u degree=%zu n=%zu", Field::mod, degree, n);
      // Only the prefix below x^n can affect the requested inverse. Nonzero
      // coefficients after that prefix must not change degree selection.
      for (size_t i = n; i < padded.size(); ++i) padded[i] = Field::from_int(value(i, 41));
      CHECK_MSG(fpx::poly::inv(padded, n) == want,
                "inverse ignores suffix mod=%u degree=%zu n=%zu", Field::mod, degree, n);
    }
  }
}

template <class Field>
void linear_power_boundaries() {
  constexpr uint32_t c = 7, d = 11;
  for (const uint64_t exponent : {uint64_t(65), uint64_t(1000000),
                                  std::numeric_limits<uint64_t>::max(),
                                  uint64_t(Field::mod), uint64_t(Field::mod) + 1}) {
    for (const size_t n : {size_t(1), size_t(33), size_t(129)}) {
      const auto want = ordinary_linear_power<Field>(c, d, 0, exponent, n);
      std::vector<Field> a{Field::from_int(c), Field::from_int(d)};
      CHECK_MSG(fpx::poly::pow(a, exponent, n) == want,
                "linear power mod=%u k=%llu n=%zu", Field::mod,
                static_cast<unsigned long long>(exponent), n);
      a.resize(n + 17);
      CHECK_MSG(fpx::poly::pow(a, exponent, n) == want,
                "padded linear power mod=%u k=%llu n=%zu", Field::mod,
                static_cast<unsigned long long>(exponent), n);
      auto prior = fpx::poly::pow(a, exponent - 1, n);
      auto product = reference_product(prior, std::vector<Field>{Field::from_int(c), Field::from_int(d)}, n);
      product.resize(n);
      CHECK_MSG(product == want, "power successor mod=%u k=%llu n=%zu", Field::mod,
                static_cast<unsigned long long>(exponent), n);
    }
  }
  for (const size_t shift : {size_t(1), size_t(3)}) {
    for (const uint64_t exponent : {uint64_t(65), uint64_t(1000000),
                                    std::numeric_limits<uint64_t>::max()}) {
      constexpr size_t n = 257;
      std::vector<Field> a(shift + 2);
      a[shift] = Field::from_int(c);
      a[shift + 1] = Field::from_int(d);
      const auto want = ordinary_linear_power<Field>(c, d, shift, exponent, n);
      CHECK_MSG(fpx::poly::pow(a, exponent, n) == want,
                "valued linear power mod=%u k=%llu shift=%zu", Field::mod,
                static_cast<unsigned long long>(exponent), shift);
      a.resize(513);
      CHECK_MSG(fpx::poly::pow(a, exponent, n) == want,
                "padded valued linear power mod=%u k=%llu shift=%zu", Field::mod,
                static_cast<unsigned long long>(exponent), shift);
    }
  }
}

template <class Field>
void linear_exponential_boundaries() {
  for (const uint32_t coefficient : {uint32_t(0), uint32_t(1), uint32_t(7), Field::mod - 1}) {
    for (const size_t n : {size_t(1), size_t(2), size_t(3), size_t(15), size_t(16),
                           size_t(17), size_t(31), size_t(33), size_t(257), size_t(4097)}) {
      // Ordinary integer arithmetic checks c^j / j!, independently of both
      // Montgomery arithmetic and the library's inv_series table generator.
      std::vector<Field> want(n);
      want[0] = Field::from_int(1);
      uint32_t term = 1;
      for (size_t j = 1; j < n; ++j) {
        term = ordinary_multiply(term, coefficient, Field::mod);
        term = ordinary_multiply(term, ordinary_power(
            static_cast<uint32_t>(j), Field::mod - 2, Field::mod), Field::mod);
        want[j] = Field::from_int(term);
      }
      std::vector<Field> input{Field(), Field::from_int(coefficient)};
      CHECK_MSG(fpx::poly::exp(input, n) == want,
                "linear exp mod=%u coefficient=%u n=%zu", Field::mod, coefficient, n);
      input.resize(n + 17);
      CHECK_MSG(fpx::poly::exp(input, n) == want,
                "padded linear exp mod=%u coefficient=%u n=%zu", Field::mod, coefficient, n);
      if (coefficient == 0) {
        CHECK_MSG(fpx::poly::exp(std::vector<Field>{}, n) == want,
                  "empty exp mod=%u n=%zu", Field::mod, n);
      }
    }
  }
}

struct LateTlsBuffer {
  ScratchBuffer buffer;
  std::atomic<bool>* correct = nullptr;
  ~LateTlsBuffer() {
    const bool ok = buffer.size() == 4097 && intact(buffer, 0xabcddcba);
    buffer.reset();  // The scratch pool has already been destroyed.
    if (correct) correct->store(ok, std::memory_order_relaxed);
  }
};
thread_local LateTlsBuffer late_tls_buffer;
ScratchBuffer late_static_buffer;

template <class Field>
std::vector<Field> reference_product(const std::vector<Field>& a,
                                     const std::vector<Field>& b, size_t n) {
  if (a.empty() || b.empty() || n == 0) return {};
  std::vector<Field> out(std::min(n, a.size() + b.size() - 1));
  for (size_t i = 0; i < std::min(a.size(), out.size()); ++i)
    for (size_t j = 0; j < b.size() && i + j < out.size(); ++j)
      out[i + j] += a[i] * b[j];
  return out;
}

template <class Field>
void reused_algorithm_buffers() {
  // Leave nonzero, noncanonical limbs in every relevant pool bucket. None of
  // these are legal zero padding: an omitted initialization must be detected.
  for (size_t n = 16; n <= 4096; n *= 2) {
    std::array<ScratchBuffer, 4> buffers;
    for (auto& buffer : buffers) {
      buffer.resize_uninitialized(n);
      std::fill_n(buffer.data(), n, 0xffffffffu);
    }
  }
  for (const size_t n : {size_t(257), size_t(33), size_t(513), size_t(65), size_t(129)}) {
    std::vector<Field> a(n + 7), b(n / 2 + 3);
    for (size_t i = 0; i < a.size(); ++i) a[i] = Field::from_int(value(i, 13));
    for (size_t i = 0; i < b.size(); ++i) b[i] = Field::from_int(value(i, 29));
    a[0] = Field::from_int(1);
    CHECK(fpx::poly::conv(a, b, n) == reference_product(a, b, n));
    CHECK(fpx::poly::conv(a, a, n) == reference_product(a, a, n));

    // Independent coefficient recurrences catch shared errors between NTT
    // operations as well as stale scratch limbs at partial Newton steps.
    std::vector<Field> inverse(n);
    inverse[0] = Field::from_int(1);
    for (size_t i = 1; i < n; ++i) {
      Field sum;
      for (size_t j = 1; j <= i; ++j) sum += a[j] * inverse[i - j];
      inverse[i] = -sum;
    }
    CHECK(fpx::poly::inv(a, n) == inverse);
    a[0] = Field();
    std::vector<Field> exponential(n);
    exponential[0] = Field::from_int(1);
    for (size_t i = 1; i < n; ++i) {
      Field sum;
      for (size_t j = 1; j <= i; ++j)
        sum += Field::from_int(j) * a[j] * exponential[i - j];
      exponential[i] = sum / Field::from_int(i);
    }
    CHECK(fpx::poly::exp(a, n) == exponential);
  }
}

}  // namespace

FP_TEST(scratch_alignment_and_size_classes) {
  fpx::release_scratch_memory();
  ScratchBuffer empty;
  CHECK_EQ(empty.data(), nullptr);
  CHECK_EQ(empty.size(), size_t(0));
  CHECK_EQ(empty.capacity(), size_t(0));
  for (const size_t n : {size_t(0), size_t(1), size_t(15), size_t(16), size_t(17),
                         size_t(31), size_t(32), size_t(33), size_t(63), size_t(64),
                         size_t(65), size_t(4095), size_t(4096), size_t(4097)}) {
    ScratchBuffer buffer(n);
    CHECK_EQ(buffer.size(), n);
    CHECK(aligned(buffer));
    if (n == 0) {
      CHECK_EQ(buffer.data(), nullptr);
      CHECK_EQ(buffer.capacity(), size_t(0));
      continue;
    }
    CHECK(buffer.capacity() >= n);
    CHECK(buffer.capacity() >= 16);
    CHECK_EQ(buffer.capacity() & (buffer.capacity() - 1), size_t(0));
    write(buffer, 0x12345678);
    CHECK(intact(buffer, 0x12345678));
    // Last requested limb is writable at every power-of-two boundary.
    CHECK_EQ(buffer.data()[n - 1], value(n - 1, 0x12345678));
  }
}

FP_TEST(persistent_aligned_storage_is_independent_of_scratch_cache) {
  using fpx::detail::AlignedBuffer;
  const auto before = fpx::scratch_memory_stats();
  AlignedBuffer original(4097);
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(original.data()) % 64, uintptr_t(0));
  for (size_t i = 0; i < original.size(); ++i) original[i] = value(i, 31);
  auto* retained = original.data();
  AlignedBuffer moved(std::move(original));
  CHECK_EQ(original.data(), nullptr);
  CHECK_EQ(original.size(), size_t(0));
  CHECK_EQ(moved.data(), retained);
  fpx::release_scratch_memory();
  for (size_t i = 0; i < moved.size(); ++i) CHECK_EQ(moved[i], value(i, 31));
  moved.resize_uninitialized(65);
  CHECK_EQ(moved.data(), retained);
  moved.resize_uninitialized(4098);
  CHECK_EQ(moved.size(), size_t(4098));
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(moved.data()) % 64, uintptr_t(0));
  const auto after = fpx::scratch_memory_stats();
  CHECK_EQ(after.system_allocations, before.system_allocations);
  CHECK_EQ(after.cache_hits, before.cache_hits);
}

FP_TEST(scratch_nested_and_moved_ownership) {
  fpx::release_scratch_memory();
  ScratchBuffer original(73);
  auto* retained = original.data();
  const size_t capacity = original.capacity();
  write(original, 0x55aa55aa);
  {
    std::array<ScratchBuffer, 7> nested;
    for (size_t i = 0; i < nested.size(); ++i) {
      nested[i].resize_uninitialized(73);
      CHECK(nested[i].data() != retained);
      for (size_t j = 0; j < i; ++j) CHECK(nested[j].data() != nested[i].data());
      write(nested[i], static_cast<uint32_t>(i));
    }
    for (size_t i = 0; i < nested.size(); ++i)
      CHECK(intact(nested[i], static_cast<uint32_t>(i)));
    CHECK(intact(original, 0x55aa55aa));
  }
  ScratchBuffer moved(std::move(original));
  CHECK_EQ(original.data(), nullptr);
  CHECK_EQ(original.size(), size_t(0));
  CHECK_EQ(original.capacity(), size_t(0));
  CHECK_EQ(moved.data(), retained);
  CHECK_EQ(moved.capacity(), capacity);
  CHECK(intact(moved, 0x55aa55aa));
  ScratchBuffer assigned(8192);
  assigned = std::move(moved);
  CHECK_EQ(moved.data(), nullptr);
  CHECK_EQ(moved.size(), size_t(0));
  CHECK_EQ(moved.capacity(), size_t(0));
  CHECK_EQ(assigned.data(), retained);
  CHECK(intact(assigned, 0x55aa55aa));
  auto* alias = &assigned;
  assigned = std::move(*alias);  // Self move must retain valid ownership.
  CHECK_EQ(assigned.data(), retained);
  CHECK(intact(assigned, 0x55aa55aa));
  assigned.resize_uninitialized(17);
  CHECK_EQ(assigned.data(), retained);
  CHECK_EQ(assigned.capacity(), capacity);
  assigned.resize_uninitialized(capacity);
  CHECK_EQ(assigned.data(), retained);
  assigned.resize_uninitialized(capacity + 1);
  CHECK_EQ(assigned.size(), capacity + 1);
  CHECK(assigned.capacity() >= capacity + 1);
  CHECK(aligned(assigned));
  write(assigned, 12);
  CHECK(intact(assigned, 12));
  assigned.reset();
  assigned.reset();
  CHECK_EQ(assigned.data(), nullptr);
  CHECK_EQ(assigned.size(), size_t(0));
  CHECK_EQ(assigned.capacity(), size_t(0));
}

FP_TEST(scratch_overflow_has_strong_exception_guarantee) {
  ScratchBuffer buffer(67);
  auto* retained = buffer.data();
  const size_t capacity = buffer.capacity();
  write(buffer, 0x87654321);
  for (const size_t n : {std::numeric_limits<size_t>::max(),
                         std::numeric_limits<size_t>::max() / sizeof(uint32_t) + 1}) {
    bool threw = false;
    try { buffer.resize_uninitialized(n); }
    catch (const std::bad_array_new_length&) { threw = true; }
    CHECK(threw);
    CHECK_EQ(buffer.data(), retained);
    CHECK_EQ(buffer.size(), size_t(67));
    CHECK_EQ(buffer.capacity(), capacity);
    CHECK(intact(buffer, 0x87654321));
  }
}

FP_TEST(scratch_warm_reuse_and_cache_bounds) {
  fpx::release_scratch_memory();
  CHECK_EQ(fpx::scratch_memory_stats().cached_bytes, size_t(0));
  CHECK_EQ(fpx::scratch_memory_stats().cached_blocks, size_t(0));
  {
    ScratchBuffer warm(257);
    write(warm, 1);
  }
  const auto warm = fpx::scratch_memory_stats();
  for (unsigned i = 0; i < 40; ++i) {
    ScratchBuffer buffer(257);
    write(buffer, i);
    CHECK(intact(buffer, i));
  }
  const auto reused = fpx::scratch_memory_stats();
  if constexpr (FASTPOLY_SCRATCH_CACHE_BYTES >= 2048 + 64) {
    CHECK_EQ(reused.system_allocations, warm.system_allocations);
    CHECK_EQ(reused.cache_hits - warm.cache_hits, size_t(40));
  } else if constexpr (FASTPOLY_SCRATCH_CACHE_BYTES == 0) {
    CHECK_EQ(reused.system_allocations - warm.system_allocations, size_t(40));
    CHECK_EQ(reused.cache_hits, warm.cache_hits);
  }

  fpx::release_scratch_memory();
  // Simultaneously active buffers must not alias; returning more than four
  // equally sized blocks also exercises the per-bucket bound.
  {
    std::array<ScratchBuffer, 9> simultaneous;
    for (size_t i = 0; i < simultaneous.size(); ++i) {
      simultaneous[i].resize_uninitialized(4096);
      write(simultaneous[i], static_cast<uint32_t>(i));
      for (size_t j = 0; j < i; ++j) CHECK(simultaneous[i].data() != simultaneous[j].data());
    }
    for (size_t i = 0; i < simultaneous.size(); ++i)
      CHECK(intact(simultaneous[i], static_cast<uint32_t>(i)));
  }
  const auto bucket = fpx::scratch_memory_stats();
  CHECK(bucket.cached_blocks <= 4);
  CHECK(bucket.cached_bytes <= 4 * (size_t(4096) * sizeof(uint32_t) + 64));

  // Allocate twice the budget, then release in many size classes. Oversized
  // blocks and full-cache returns must be freed instead of retained.
  for (size_t n = 16; n <= size_t(1) << 20; n *= 2) {
    std::array<ScratchBuffer, 5> burst;
    for (auto& buffer : burst) buffer.resize_uninitialized(n);
  }
  CHECK(fpx::scratch_memory_stats().cached_bytes <= size_t(FASTPOLY_SCRATCH_CACHE_BYTES));
  fpx::release_scratch_memory();
  CHECK_EQ(fpx::scratch_memory_stats().cached_bytes, size_t(0));
  CHECK_EQ(fpx::scratch_memory_stats().cached_blocks, size_t(0));
}

FP_TEST(scratch_purge_preserves_active_buffers) {
  ScratchBuffer active(1031);
  write(active, 0xdeadbeef);
  auto* retained = active.data();
  { ScratchBuffer cached(4096); write(cached, 15); }
  const auto before = fpx::scratch_memory_stats();
  fpx::release_scratch_memory();
  const auto after = fpx::scratch_memory_stats();
  CHECK_EQ(after.cached_bytes, size_t(0));
  CHECK_EQ(after.cached_blocks, size_t(0));
  CHECK_EQ(after.system_allocations, before.system_allocations);
  CHECK_EQ(after.cache_hits, before.cache_hits);
  CHECK_EQ(active.data(), retained);
  CHECK(intact(active, 0xdeadbeef));
  { ScratchBuffer fresh(1031); CHECK(fresh.data() != retained); }
  CHECK(intact(active, 0xdeadbeef));
}

FP_TEST(scratch_cross_thread_move_outlives_allocating_thread) {
  std::promise<ScratchBuffer> delivery;
  auto receive = delivery.get_future();
  std::thread producer([&] {
    ScratchBuffer buffer(8193);
    write(buffer, 0xfeedbeef);
    delivery.set_value(std::move(buffer));
  });
  producer.join();  // The allocating thread's entire cache is now destroyed.
  ScratchBuffer buffer = receive.get();
  CHECK_EQ(buffer.size(), size_t(8193));
  CHECK(aligned(buffer));
  CHECK(intact(buffer, 0xfeedbeef));
  buffer.reset();  // Return to the receiving thread's pool safely.
}

FP_TEST(scratch_reset_does_not_initialize_a_virgin_thread_cache) {
  ScratchBuffer buffer(4097);
  write(buffer, 0x13572468);
  bool correct = false;
  std::thread consumer([owned = std::move(buffer), &correct]() mutable {
    // This thread has never allocated scratch or queried the pool. Releasing
    // foreign storage must not create a cache, including during static exit.
    correct = intact(owned, 0x13572468);
    owned.reset();
    const auto stats = fpx::scratch_memory_stats();
    correct = correct && stats.cached_bytes == 0 && stats.cached_blocks == 0 &&
              stats.system_allocations == 0 && stats.cache_hits == 0;
  });
  consumer.join();
  CHECK(correct);
}

FP_TEST(scratch_late_tls_and_static_destruction) {
  std::atomic<bool> correct{false};
  std::thread worker([&] {
    // Register this destructor before the first allocation constructs the TLS
    // pool. Destruction order is therefore pool, then this active buffer.
    late_tls_buffer.correct = &correct;
    late_tls_buffer.buffer.resize_uninitialized(4097);
    write(late_tls_buffer.buffer, 0xabcddcba);
    { ScratchBuffer cached(4097); write(cached, 12); }
  });
  worker.join();
  CHECK(correct.load(std::memory_order_relaxed));
  // Main-thread TLS destructors run before global static destructors. The
  // sanitizer pass also checks this active buffer's fallback at program exit.
  late_static_buffer.resize_uninitialized(4097);
  write(late_static_buffer, 17);
}

FP_TEST(scratch_thread_local_isolation) {
  fpx::release_scratch_memory();
  const auto main_before = fpx::scratch_memory_stats();
  constexpr size_t threads = 8;
  std::barrier start(static_cast<std::ptrdiff_t>(threads));
  std::array<bool, threads> correct{};
  std::array<std::thread, threads> workers;
  for (size_t t = 0; t < threads; ++t) {
    workers[t] = std::thread([&, t] {
      bool ok = true;
      start.arrive_and_wait();
      for (size_t iter = 0; iter < 100; ++iter) {
        ScratchBuffer outer(1025 + t), inner(1025 + t);
        ok = ok && aligned(outer) && aligned(inner) && outer.data() != inner.data();
        const auto salt = static_cast<uint32_t>(t * 1000 + iter);
        write(outer, salt);
        write(inner, ~salt);
        if (iter % 7 == 0) fpx::release_scratch_memory();
        ok = ok && intact(outer, salt) && intact(inner, ~salt);
      }
      ok = ok && fpx::scratch_memory_stats().cached_bytes <= size_t(FASTPOLY_SCRATCH_CACHE_BYTES);
      correct[t] = ok;
    });
  }
  for (auto& worker : workers) worker.join();
  for (size_t t = 0; t < threads; ++t) CHECK_MSG(correct[t], "thread-local scratch thread=%zu", t);
  const auto main_after = fpx::scratch_memory_stats();
  CHECK_EQ(main_after.cached_bytes, main_before.cached_bytes);
  CHECK_EQ(main_after.cached_blocks, main_before.cached_blocks);
  CHECK_EQ(main_after.system_allocations, main_before.system_allocations);
  CHECK_EQ(main_after.cache_hits, main_before.cache_hits);
}

FP_TEST(scratch_reused_series_padding_all_moduli) {
  reused_algorithm_buffers<fpx::mod998244353>();
  reused_algorithm_buffers<fpx::mod1004535809>();
  reused_algorithm_buffers<fpx::mod469762049>();
  reused_algorithm_buffers<fpx::mod167772161>();
  reused_algorithm_buffers<fpx::mod754974721>();
  reused_algorithm_buffers<fpx::mod1224736769>();
}

FP_TEST(poly_short_power_large_truncation) {
  using Field = fpx::mod998244353;
  for (const uint64_t k : {uint64_t(2), uint64_t(7), uint64_t(64)}) {
    for (const size_t shift : {size_t(0), size_t(3), size_t(17)}) {
      std::vector<Field> want(4097), input(shift + 2);
      input[shift] = input[shift + 1] = Field::from_int(1);
      const size_t offset = shift * static_cast<size_t>(k);
      Field binomial = Field::from_int(1);
      for (uint64_t i = 0; i <= k && offset + i < want.size(); ++i) {
        want[offset + i] = binomial;
        binomial *= Field::from_int(k - i);
        binomial /= Field::from_int(i + 1);
      }
      CHECK(fpx::poly::pow(input, k, want.size()) == want);
      // Trailing zero storage must not turn a degree-one polynomial into an
      // increasingly dense power during binary exponentiation.
      input.resize(2049);
      CHECK(fpx::poly::pow(input, k, want.size()) == want);
    }
  }
}

FP_TEST(poly_small_divisor_large_dividend) {
  using Field = fpx::mod998244353;
  for (const size_t degree : {size_t(0), size_t(1), size_t(3), size_t(17)}) {
    std::vector<Field> divisor(degree + 1), quotient(4097), remainder(degree);
    for (size_t i = 0; i < divisor.size(); ++i) divisor[i] = Field::from_int(value(i, 7));
    for (size_t i = 0; i < quotient.size(); ++i) quotient[i] = Field::from_int(value(i, 11));
    for (size_t i = 0; i < remainder.size(); ++i) remainder[i] = Field::from_int(value(i, 19));
    divisor.back() = Field::from_int(1);
    auto dividend = reference_product(divisor, quotient, divisor.size() + quotient.size() - 1);
    for (size_t i = 0; i < remainder.size(); ++i) dividend[i] += remainder[i];
    const auto result = fpx::poly::divmod(dividend, divisor);
    CHECK(result.first == quotient);
    CHECK(result.second == remainder);
    divisor.resize(divisor.size() + 11);
    dividend.resize(dividend.size() + 17);
    const auto padded_result = fpx::poly::divmod(dividend, divisor);
    CHECK(padded_result.first == quotient);
    CHECK(padded_result.second == remainder);
  }
}

FP_TEST(poly_short_padded_inverse_all_moduli) {
  short_inverse_boundaries<fpx::mod998244353>();
  short_inverse_boundaries<fpx::mod1004535809>();
  short_inverse_boundaries<fpx::mod469762049>();
  short_inverse_boundaries<fpx::mod167772161>();
  short_inverse_boundaries<fpx::mod754974721>();
  short_inverse_boundaries<fpx::mod1224736769>();
}

FP_TEST(poly_large_linear_power_all_moduli) {
  linear_power_boundaries<fpx::mod998244353>();
  linear_power_boundaries<fpx::mod1004535809>();
  linear_power_boundaries<fpx::mod469762049>();
  linear_power_boundaries<fpx::mod167772161>();
  linear_power_boundaries<fpx::mod754974721>();
  linear_power_boundaries<fpx::mod1224736769>();
}

FP_TEST(poly_linear_power_characteristic_boundary) {
  using Field = fpx::Mont<17, 3>;
  const std::vector<Field> linear{Field::from_int(7), Field::from_int(11)};
  for (const uint64_t exponent : {uint64_t(65), uint64_t(1000000),
                                  std::numeric_limits<uint64_t>::max()}) {
    CHECK(fpx::poly::pow(linear, exponent, 17) ==
          ordinary_linear_power<Field>(7, 11, 0, exponent, 17));
    bool threw = false;
    try { (void)fpx::poly::pow(linear, exponent, 18); }
    catch (const fpx::poly::domain_error&) { threw = true; }
    CHECK(threw);
    // Constants remain valid beyond the characteristic without integrating.
    const auto constant = fpx::poly::pow(std::vector<Field>{Field::from_int(7)}, exponent, 18);
    CHECK_EQ(constant[0].val(), ordinary_power(7, exponent, Field::mod));
    for (size_t i = 1; i < constant.size(); ++i) CHECK(constant[i].is_zero());
  }
}

FP_TEST(poly_linear_padded_exp_all_moduli) {
  linear_exponential_boundaries<fpx::mod998244353>();
  linear_exponential_boundaries<fpx::mod1004535809>();
  linear_exponential_boundaries<fpx::mod469762049>();
  linear_exponential_boundaries<fpx::mod167772161>();
  linear_exponential_boundaries<fpx::mod754974721>();
  linear_exponential_boundaries<fpx::mod1224736769>();
}

FP_TEST(poly_cold_series_scratch_allocations_are_constant) {
  using Field = fpx::mod998244353;
  constexpr uint32_t ratio = 7;
  for (const size_t n : {size_t(4097), size_t(4105)}) {
    std::vector<Field> geometric(n), exponential_input(n), square(n), inverse_want(n);
    uint32_t power = 1;
    for (size_t i = 0; i < n; ++i) {
      geometric[i] = Field::from_int(power);
      // (sum r^j x^j)^2 has coefficient (i+1)r^i, independently of conv.
      square[i] = Field::from_int(ordinary_multiply(
          static_cast<uint32_t>(i + 1), power, Field::mod));
      // exp(sum_{i>=1} r^i x^i/i) = 1/(1-rx), independently of inv_series.
      if (i != 0) exponential_input[i] = Field::from_int(ordinary_multiply(power,
          ordinary_power(static_cast<uint32_t>(i), Field::mod - 2, Field::mod), Field::mod));
      power = ordinary_multiply(power, ratio, Field::mod);
    }
    inverse_want[0] = Field::from_int(1);
    inverse_want[1] = Field::from_int(Field::mod - ratio);
    // A single final coefficient uses the direct tail; nine coefficients need
    // one last transform. Inspect actual cold allocation counts in both cases.
    const size_t max_transform = n == 4097 ? 4096 : 8192;
    auto check_storage = [&](const fpx::ScratchMemoryStats& before, size_t buffers) {
      const auto after = fpx::scratch_memory_stats();
      CHECK_MSG(after.system_allocations - before.system_allocations <= buffers,
                "cold scratch allocations n=%zu max_buffers=%zu actual=%zu", n, buffers,
                after.system_allocations - before.system_allocations);
      CHECK(after.cached_blocks <= buffers);
      CHECK(after.cached_bytes <= buffers * (max_transform * sizeof(uint32_t) + 64));
      CHECK(after.cached_bytes <= size_t(FASTPOLY_SCRATCH_CACHE_BYTES));
    };

    fpx::release_scratch_memory();
    auto before = fpx::scratch_memory_stats();
    const auto inverse = fpx::poly::inv(geometric, n);
    check_storage(before, 2);
    CHECK(inverse == inverse_want);

    fpx::release_scratch_memory();
    before = fpx::scratch_memory_stats();
    const auto exponential = fpx::poly::exp(exponential_input, n);
    check_storage(before, 3);
    CHECK(exponential == geometric);

    fpx::release_scratch_memory();
    before = fpx::scratch_memory_stats();
    const auto root = fpx::poly::sqrt(square, n);
    check_storage(before, 2);
    CHECK_EQ(root.size(), n);
    const Field sign = root[0];
    CHECK(sign == Field::from_int(1) || sign == -Field::from_int(1));
    for (size_t i = 0; i < n; ++i) CHECK_EQ(root[i], geometric[i] * sign);
  }
}

FP_TEST_MAIN()
