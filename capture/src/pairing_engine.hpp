// Pairing engine: pure matching logic (no threads, no clocks of its own, no camera). See DESIGN.md section 4.2.
//
// Every frame given to push() leaves the engine exactly once, through EngineOutput::on_pair or ::on_discard, and per camera
// in the same order as it was pushed. The matcher thread relies on this to pop its ring slots in FIFO order.
#pragma once

#include <cstddef>
#include <cstdint>

#include "common.hpp"

namespace cap {

struct FrameInfo {
    uint64_t id = 0;            // caller-defined (tests: unique id)
    int cam = 0;                // kCamA / kCamB
    uint64_t epoch = 0;         // restart generation of this camera
    double t_ms = 0;            // pairing time from the TimestampSource
    bool time_usable = false;   // TimestampSource accepted this frame (SDK source: Global Time domain)
    int64_t hw_ts_us = -1;      // hardware timestamp, for the frame-period measurement (-1: missing)
    int64_t counter = -1;       // frame counter, for partner-change detection only (-1: missing)
    int64_t arrival_ns = 0;     // CLOCK_MONOTONIC arrival
    const void* ref = nullptr;  // caller data (matcher: ring slot)
    bool settle_ok = false;     // set by the engine at push: camera was settled when this frame arrived
};

enum class Discard : int { Settle = 0, Domain, PartnerSettling, Restart, Unpaired, QueueFull, Count };
const char* discard_name(Discard d);

class EngineOutput {
public:
    virtual ~EngineOutput() = default;
    virtual void on_pair(const FrameInfo& a, const FrameInfo& b, double dt_ms) = 0;  // dt = t(B) - t(A)
    virtual void on_discard(const FrameInfo& f, Discard reason) = 0;
};

struct EngineConfig {
    int64_t settle_ns = 5000000000LL;          // R4: 5 s after stream start
    double timeout_periods = 3.0;              // R4: unpaired after 3 frame periods
    double ambiguity_ms = 0.5;                 // band below half a period where the next frame is needed to decide
    int64_t period_measure_after_ns = 1000000000LL;  // period measurement uses frames from 1 s after stream start
    int64_t min_period_frames = 90;            // at least this many counter steps for a valid measurement
    double fallback_period_ms = kExpectedPeriodMs;
    // While the other camera settles, the newest frames of a settled camera are held (at most this many, at most the
    // timeout) so that the first settled frame of the other camera finds its partner (gap 4).
    std::size_t hold_max_frames = 4;
};

struct EngineCounters {
    uint64_t pushed[2] = {0, 0};
    uint64_t paired[2] = {0, 0};               // frames used in pairs (== pairs on each camera)
    uint64_t discards[2][int(Discard::Count)] = {};
    uint64_t pairs = 0;
    uint64_t timeout_decisions[2] = {0, 0};    // decisions made because a frame waited 3 periods
    uint64_t partner_changes = 0;
    uint64_t epoch_changes[2] = {0, 0};        // restarts seen by the engine
    uint64_t period_fallback[2] = {0, 0};      // settled without a hardware-timestamp period measurement

    uint64_t unpaired(int c) const { return discards[c][int(Discard::Unpaired)]; }
    // Frames that took part in pairing (eligible after settle): paired + unpaired.
    uint64_t eligible(int c) const { return paired[c] + unpaired(c); }
};

class PairingEngine {
public:
    explicit PairingEngine(const EngineConfig& cfg = EngineConfig());

    void push(const FrameInfo& f, EngineOutput& out);
    // Makes every decision that is possible at time now_ns (CLOCK_MONOTONIC).
    void process(int64_t now_ns, EngineOutput& out);
    // Earliest time at which process() can make a timeout decision; INT64_MAX if none.
    int64_t next_deadline_ns() const;

    bool period_known() const { return cams_[0].period_frozen && cams_[1].period_frozen; }
    double period_ms() const;                      // mean of both cameras (fallback before both are measured)
    double camera_period_ms(int cam) const { return cams_[cam].period_ms; }
    double tolerance_ms() const { return period_ms() / 2.0; }
    bool settled(int cam) const { return cams_[cam].settled; }
    std::size_t queued(int cam) const { return cams_[cam].size; }
    const EngineCounters& counters() const { return ctr_; }

    static constexpr std::size_t kQueueCap = 16;

private:
    struct Cam {
        bool has_epoch = false;
        uint64_t epoch = 0;
        int64_t settle_start_ns = 0;
        bool settled = false;
        bool first_valid = false;   // first_t is the time of the first settled, usable frame of this epoch
        double first_t = 0;
        // period measurement (camera clock)
        bool p_first = false;
        int64_t hw0 = 0, c0 = 0, hw1 = 0, c1 = 0;
        bool period_frozen = false;
        double period_ms = 0;
        // FIFO queue
        FrameInfo q[kQueueCap];
        std::size_t head = 0, size = 0;
        const FrameInfo& at(std::size_t i) const { return q[(head + i) % kQueueCap]; }
    };

    bool step(int64_t now_ns, EngineOutput& out);
    bool eligible(const FrameInfo& f) const;
    Discard ineligible_reason(const FrameInfo& f) const;
    void pop_discard(int cam, Discard r, EngineOutput& out);
    void pop_pair(int64_t now_ns, bool by_timeout, int x_cam, EngineOutput& out);
    void flush(int cam, Discard r, EngineOutput& out);
    int64_t timeout_ns() const;

    EngineConfig cfg_;
    Cam cams_[2];
    EngineCounters ctr_;
    bool key_valid_ = false;
    int64_t last_key_ = 0;
};

}  // namespace cap
