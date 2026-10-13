#include "Aligned.hh"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "Arena.hh"

namespace leanmt {

// Debug lever: LEANMT_POISON_ALLOC=1 fills every fresh allocation (heap and
// arena) with 0xCD bytes. The bergamot allocation strategy relies on
// malloc-top pages being zero in places; poisoning turns any premature read
// of "should be zero" memory into visible garbage in test output instead of
// silently working — which is how allocator-history-dependent decode
// corruption hides in production. Off by default, zero overhead when off.
//
// Non-static: Arena.cc reuses this gate.
bool poison_debug_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("LEANMT_POISON_ALLOC");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

namespace {

constexpr unsigned char kPoisonByte = 0xCD;

}  // namespace

Aligned::Aligned(size_t alignment, size_t size) : size_(size) {
  if (Arena* arena = active_arena()) {
    data_ = arena->allocate(size, alignment);
    from_arena_ = true;
  } else {
    data_ = allocate(alignment, size);
  }
  // Note: arena buffers are poisoned wholesale on Arena::reset(); the per-
  // allocation poison below covers the non-arena path. (Arena raw chunks would
  // otherwise be re-poisoned N times per decode step for nothing.)
  if (data_ != nullptr && !from_arena_ && poison_debug_enabled()) {
    std::memset(data_, kPoisonByte, size);
  }
}

void* Aligned::allocate(size_t alignment, size_t size) {
  size_t aligned_size = (size / alignment) * alignment;
  if (size % alignment != 0) {
    aligned_size += alignment;
  }
  assert(aligned_size >= size);
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
    return nullptr;
  }
  // The tail from `size` up to the aligned allocation is never logically
  // written. SIMD kernels can legally *read* it (vector tail loads read the
  // whole lane), and on freshly mapped malloc pages it happens to be zero.
  // Zero it deterministically so recycled-heap contents can never leak into
  // a kernel's edge lanes.
  if (aligned_size > size && ptr != nullptr) {
    std::memset(static_cast<char*>(ptr) + size, 0, aligned_size - size);
  }
  return ptr;
}

Aligned::~Aligned() { release(); }

void* Aligned::data() const { return data_; }
size_t Aligned::size() const { return size_; }

char* Aligned::begin() const { return reinterpret_cast<char*>(data_); }
char* Aligned::end() const { return reinterpret_cast<char*>(data_) + size_; }

bool Aligned::empty() const { return size_ == 0; }

Aligned::Aligned(Aligned&& from) noexcept {
  if (this != &from) {
    release();
    consume(from);
  }
}

Aligned& Aligned::operator=(Aligned&& from) noexcept {
  if (this != &from) {
    release();
    consume(from);
  }
  return *this;
}

void Aligned::consume(Aligned& from) {
  data_ = from.data_;
  size_ = from.size_;
  from_arena_ = from.from_arena_;

  from.data_ = nullptr;
  from.size_ = 0;
  from.from_arena_ = false;
}

void Aligned::release() {
  if (data_ != nullptr && !from_arena_) {
    free(data_);
  }
  data_ = nullptr;
  size_ = 0;
  from_arena_ = false;
}
}  // namespace leanmt
