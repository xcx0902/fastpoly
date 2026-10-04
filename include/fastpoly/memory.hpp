// fastpoly - aligned, uninitialized limb storage and bounded scratch reuse.
#ifndef FASTPOLY_MEMORY_HPP
#define FASTPOLY_MEMORY_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

// This limit applies to retained (idle) storage per thread, including headers.
// Define consistently in every translation unit. Zero disables retention.
#ifndef FASTPOLY_SCRATCH_CACHE_BYTES
#define FASTPOLY_SCRATCH_CACHE_BYTES (64u * 1024u * 1024u)
#endif

namespace fpx {

struct ScratchMemoryStats {
  size_t cached_bytes = 0;
  size_t cached_blocks = 0;
  size_t system_allocations = 0;
  size_t cache_hits = 0;
};

namespace detail {

inline constexpr size_t limb_alignment = 64;

/// Persistent storage (e.g. twiddle tables); no thread-local lifetime dependency.
/// Contents are uninitialized, and growth discards the old contents.
class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(size_t n) { resize_uninitialized(n); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        capacity_(std::exchange(other.capacity_, 0)) {}
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }
  ~AlignedBuffer() { reset(); }

  uint32_t* data() noexcept { return data_; }
  const uint32_t* data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }
  size_t capacity() const noexcept { return capacity_; }
  uint32_t& operator[](size_t i) noexcept { return data_[i]; }
  const uint32_t& operator[](size_t i) const noexcept { return data_[i]; }

  void resize_uninitialized(size_t n) {
    if (n > capacity_) {
      if (n > std::numeric_limits<size_t>::max() / sizeof(uint32_t))
        throw std::bad_array_new_length();
      auto* p = static_cast<uint32_t*>(::operator new(
          n * sizeof(uint32_t), std::align_val_t(limb_alignment)));
      reset();
      data_ = p;
      capacity_ = n;
    }
    size_ = n;
  }
  void reset() noexcept {
    ::operator delete(data_, std::align_val_t(limb_alignment));
    data_ = nullptr;
    size_ = capacity_ = 0;
  }

 private:
  uint32_t* data_ = nullptr;
  size_t size_ = 0, capacity_ = 0;
};

// A separate cache-line header keeps every returned payload aligned, including
// its first SIMD load. Free-list metadata never aliases a live limb buffer.
struct alignas(limb_alignment) ScratchBlock {
  ScratchBlock* next;
};
static_assert(sizeof(ScratchBlock) == limb_alignment);

inline size_t scratch_capacity(size_t n) {
  constexpr size_t max_words =
      (std::numeric_limits<size_t>::max() - sizeof(ScratchBlock)) / sizeof(uint32_t);
  constexpr size_t max_capacity = std::bit_floor(max_words);
  if (n > max_capacity) throw std::bad_array_new_length();
  return std::bit_ceil(n < 16 ? size_t(16) : n);
}

inline ScratchBlock* new_scratch_block(size_t capacity) {
  void* p = ::operator new(sizeof(ScratchBlock) + capacity * sizeof(uint32_t),
                           std::align_val_t(limb_alignment));
  return ::new (p) ScratchBlock{nullptr};
}
inline void delete_scratch_block(ScratchBlock* block) noexcept {
  ::operator delete(block, std::align_val_t(limb_alignment));
}
inline uint32_t* scratch_payload(ScratchBlock* block) noexcept {
  return reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(block) + sizeof(ScratchBlock));
}
inline ScratchBlock* scratch_header(uint32_t* p) noexcept {
  return reinterpret_cast<ScratchBlock*>(reinterpret_cast<unsigned char*>(p) - sizeof(ScratchBlock));
}

// Trivial TLS state remains accessible during static/TLS destruction. A buffer
// destroyed after the cache is torn down falls back to a direct deallocation.
class ScratchPool;
struct ScratchPoolLifetime { ScratchPool* pool = nullptr; };
inline thread_local ScratchPoolLifetime scratch_pool_lifetime;

class ScratchPool {
 public:
  ScratchPool() noexcept { scratch_pool_lifetime.pool = this; }
  ScratchPool(const ScratchPool&) = delete;
  ScratchPool& operator=(const ScratchPool&) = delete;
  ~ScratchPool() {
    clear();
    scratch_pool_lifetime.pool = nullptr;
  }

  ScratchBlock* acquire(size_t capacity) {
    const size_t slot = static_cast<size_t>(std::countr_zero(capacity));
    Bucket& bucket = buckets_[slot];
    if (bucket.head) {
      ScratchBlock* p = bucket.head;
      bucket.head = p->next;
      --bucket.count;
      stats_.cached_bytes -= sizeof(ScratchBlock) + capacity * sizeof(uint32_t);
      --stats_.cached_blocks;
      ++stats_.cache_hits;
      return p;
    }
    ScratchBlock* p = new_scratch_block(capacity);
    ++stats_.system_allocations;
    return p;
  }

  void release(ScratchBlock* p, size_t capacity) noexcept {
    constexpr size_t limit = FASTPOLY_SCRATCH_CACHE_BYTES;
    const size_t bytes = sizeof(ScratchBlock) + capacity * sizeof(uint32_t);
    const size_t slot = static_cast<size_t>(std::countr_zero(capacity));
    Bucket& bucket = buckets_[slot];
    if (bucket.count < 4 && bytes <= limit && stats_.cached_bytes <= limit - bytes) {
      p->next = bucket.head;
      bucket.head = p;
      ++bucket.count;
      stats_.cached_bytes += bytes;
      ++stats_.cached_blocks;
    } else {
      delete_scratch_block(p);
    }
  }

  void clear() noexcept {
    for (Bucket& bucket : buckets_) {
      while (bucket.head) {
        ScratchBlock* p = bucket.head;
        bucket.head = p->next;
        delete_scratch_block(p);
      }
      bucket.count = 0;
    }
    stats_.cached_bytes = stats_.cached_blocks = 0;
  }
  ScratchMemoryStats stats() const noexcept { return stats_; }

 private:
  struct Bucket { ScratchBlock* head = nullptr; size_t count = 0; };
  std::array<Bucket, std::numeric_limits<size_t>::digits> buckets_{};
  ScratchMemoryStats stats_;
};

inline ScratchPool* current_scratch_pool() noexcept {
  thread_local ScratchPool pool;
  (void)pool;
  return scratch_pool_lifetime.pool;
}

/// A private transform buffer: no initialization, copying, or allocator locks
/// on a cache hit. Independent live buffers never share storage. A moved buffer
/// may cross threads; release uses the receiving thread's cache if initialized,
/// otherwise it frees directly without creating TLS during object destruction.
class ScratchBuffer {
 public:
  ScratchBuffer() = default;
  explicit ScratchBuffer(size_t n) { resize_uninitialized(n); }
  ScratchBuffer(const ScratchBuffer&) = delete;
  ScratchBuffer& operator=(const ScratchBuffer&) = delete;
  ScratchBuffer(ScratchBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        capacity_(std::exchange(other.capacity_, 0)) {}
  ScratchBuffer& operator=(ScratchBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }
  ~ScratchBuffer() { reset(); }

  uint32_t* data() noexcept { return data_; }
  const uint32_t* data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }
  size_t capacity() const noexcept { return capacity_; }
  uint32_t& operator[](size_t i) noexcept { return data_[i]; }
  const uint32_t& operator[](size_t i) const noexcept { return data_[i]; }

  // Growth deliberately discards the prefix: each transform fully overwrites
  // its input and zeroes only its padding. Allocation failure retains *this.
  void resize_uninitialized(size_t n) {
    if (n > capacity_) {
      const size_t capacity = scratch_capacity(n);
      ScratchPool* pool = current_scratch_pool();
      ScratchBlock* p = pool ? pool->acquire(capacity) : new_scratch_block(capacity);
      reset();
      data_ = scratch_payload(p);
      capacity_ = capacity;
    }
    size_ = n;
  }
  void reset() noexcept {
    if (data_) {
      ScratchPool* pool = scratch_pool_lifetime.pool;
      if (pool) pool->release(scratch_header(data_), capacity_);
      else delete_scratch_block(scratch_header(data_));
    }
    data_ = nullptr;
    size_ = capacity_ = 0;
  }

 private:
  uint32_t* data_ = nullptr;
  size_t size_ = 0, capacity_ = 0;
};

}  // namespace detail

/// Current thread only. Active scratch buffers and NTT plans are unaffected.
inline void release_scratch_memory() noexcept {
  if (auto* pool = detail::scratch_pool_lifetime.pool) pool->clear();
}
inline ScratchMemoryStats scratch_memory_stats() noexcept {
  if (auto* pool = detail::scratch_pool_lifetime.pool) return pool->stats();
  return {};
}

}  // namespace fpx

#endif  // FASTPOLY_MEMORY_HPP
