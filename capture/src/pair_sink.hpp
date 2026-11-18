// Output of the matcher (R5).
#pragma once

#include <atomic>
#include <cstdint>

#include "shm_layout.hpp"

namespace cap {

class PairSink {
public:
    virtual ~PairSink() = default;
    // Called by the matcher thread for every pair, in pair_seq order. Must not block for long.
    virtual void on_pair(const PairRecord& rec, const uint16_t* depth_a, const uint16_t* depth_b) = 0;
    // Called once before the first pair, when the matcher knows the measured period.
    virtual void on_period(double /*period_ms*/, double /*tolerance_ms*/) {}
    virtual const char* name() const = 0;
};

// Counts and discards (tests, and runs without a consumer).
class StatsSink : public PairSink {
public:
    void on_pair(const PairRecord& rec, const uint16_t*, const uint16_t*) override {
        if (rec.pair_seq != next_seq_) seq_errors_.fetch_add(1, std::memory_order_relaxed);
        next_seq_ = rec.pair_seq + 1;
        pairs_.fetch_add(1, std::memory_order_relaxed);
    }
    const char* name() const override { return "stats"; }
    uint64_t pairs() const { return pairs_.load(std::memory_order_relaxed); }
    uint64_t seq_errors() const { return seq_errors_.load(std::memory_order_relaxed); }

private:
    uint64_t next_seq_ = 0;
    std::atomic<uint64_t> pairs_{0};
    std::atomic<uint64_t> seq_errors_{0};
};

}  // namespace cap
