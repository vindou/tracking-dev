// Wakeup primitive between the callbacks (producers) and the matcher (consumer), and futex helpers for the
// shared-memory notification word.
//
// EventSignal: a counter.  notify() increments it and wakes waiters; it never blocks and takes no lock.
// The consumer does:   seen = sig.value();  <consume everything>;  sig.wait(seen, timeout);
// wait() returns at once if the counter is no longer `seen`, so a notify between value() and wait() is not lost.
// Linux: futex.  macOS (unit tests only): dispatch semaphore.
#pragma once

#include <atomic>
#include <cstdint>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace cap {

class EventSignal {
public:
    EventSignal();
    ~EventSignal();
    EventSignal(const EventSignal&) = delete;
    EventSignal& operator=(const EventSignal&) = delete;

    uint32_t value() const { return seq_.load(std::memory_order_acquire); }
    void notify();
    // Sleeps while value() == seen, at most timeout_ns (< 0: no timeout). Spurious returns are possible.
    void wait(uint32_t seen, int64_t timeout_ns);

private:
    std::atomic<uint32_t> seq_{0};
#if defined(__APPLE__)
    dispatch_semaphore_t sem_;
#endif
};

// Futex on a 32-bit word in shared memory (process-shared). On non-Linux systems the wait degrades to a short sleep
// (used only by the unit tests on macOS).
void shared_word_wake_all(std::atomic<uint32_t>* word);
void shared_word_wait(std::atomic<uint32_t>* word, uint32_t seen, int64_t timeout_ns);

}  // namespace cap
