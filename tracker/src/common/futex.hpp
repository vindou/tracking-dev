// Futex on a 32-bit word (process-shared, works in shared memory). Linux: futex(2). macOS (unit tests only): the
// wait degrades to a sleep of at most 1 ms per call.
#pragma once
#include <atomic>
#include <cstdint>

namespace trk {
void futex_wake_all(std::atomic<uint32_t>* word);
// Sleeps while *word == seen, at most timeout_ns (< 0: no limit). Spurious returns are possible.
void futex_wait(std::atomic<uint32_t>* word, uint32_t seen, int64_t timeout_ns);
}  // namespace trk
