// Lock-free single-producer single-consumer ring of preallocated slots (R3).
//
// head_: next slot to write, written only by the producer.  tail_: next slot to read, written only by the consumer.
// Both are monotonically increasing 64-bit counters; slot index = counter % capacity.
// Producer: claim() -> fill the slot -> publish().  If the ring is full, claim() returns nullptr and counts an overflow;
// the producer never waits.  Consumer: available(), peek(i) (look at the i-th unread slot), pop() (FIFO only).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace cap {

template <typename Slot>
class SpscRing {
public:
    explicit SpscRing(std::size_t capacity) : cap_(capacity), slots_(new Slot[capacity]()) {}
    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    std::size_t capacity() const { return cap_; }

    // ---- producer side ----
    // Returns the slot to fill, or nullptr when the ring is full (the overflow counter is incremented).
    Slot* claim() {
        const uint64_t h = head_.load(std::memory_order_relaxed);
        const uint64_t t = tail_.load(std::memory_order_acquire);
        if (h - t >= cap_) {
            overflows_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        return &slots_[h % cap_];
    }
    // Makes the claimed slot visible to the consumer.
    void publish() {
        const uint64_t h = head_.load(std::memory_order_relaxed) + 1;
        head_.store(h, std::memory_order_release);
        const uint64_t fill = h - tail_.load(std::memory_order_relaxed);
        if (fill > max_fill_.load(std::memory_order_relaxed)) max_fill_.store(fill, std::memory_order_relaxed);
        if (fill > interval_max_fill_.load(std::memory_order_relaxed))
            interval_max_fill_.store(fill, std::memory_order_relaxed);
    }

    // ---- consumer side ----
    std::size_t available() const {
        return std::size_t(head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_relaxed));
    }
    // i < available()
    Slot* peek(std::size_t i) { return &slots_[(tail_.load(std::memory_order_relaxed) + i) % cap_]; }
    void pop() { tail_.store(tail_.load(std::memory_order_relaxed) + 1, std::memory_order_release); }

    // ---- statistics (any thread) ----
    uint64_t overflows() const { return overflows_.load(std::memory_order_relaxed); }
    uint64_t max_fill() const { return max_fill_.load(std::memory_order_relaxed); }
    // Highest fill since the last call (the producer may race with the reset; at most one update is lost).
    uint64_t take_interval_max_fill() { return interval_max_fill_.exchange(0, std::memory_order_relaxed); }
    uint64_t published() const { return head_.load(std::memory_order_relaxed); }

private:
    const std::size_t cap_;
    std::unique_ptr<Slot[]> slots_;
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) std::atomic<uint64_t> overflows_{0};
    std::atomic<uint64_t> max_fill_{0};
    std::atomic<uint64_t> interval_max_fill_{0};
};

}  // namespace cap
