#include "alloc_counter.hpp"

#include <cstdlib>
#include <new>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define TRK_NO_ALLOC_COUNT 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define TRK_NO_ALLOC_COUNT 1
#endif
#endif

namespace {
std::atomic<bool> g_on{false};
std::atomic<uint64_t> g_count{0};
}  // namespace

#if !defined(TRK_NO_ALLOC_COUNT)
void* operator new(std::size_t n) {
    if (g_on.load(std::memory_order_relaxed)) g_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
bool alloc_counting_available() { return true; }
#else
bool alloc_counting_available() { return false; }
#endif

void alloc_count_start() {
    g_count.store(0);
    g_on.store(true);
}
uint64_t alloc_count_stop() {
    g_on.store(false);
    return g_count.load();
}
