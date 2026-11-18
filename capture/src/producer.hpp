// Per-camera producer: everything the SDK callback does after it has read the frame fields (R2, R6).
// deliver() runs in the callback thread: no allocation, no lock, no I/O, never waits.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

#include "common.hpp"
#include "event_signal.hpp"
#include "frame_types.hpp"
#include "histogram.hpp"
#include "spsc_ring.hpp"

namespace cap {

class FrameProducer {
public:
    FrameProducer(int cam, EventSignal& sig)
        : cam_(cam), sig_(sig), ring_(kRingSlots),
          lat_hist_(0.0, 0.1, 400),        // R7: 0.1 ms bins, 0-40 ms
          cb_hist_(0.0, 0.010, 500) {}     // callback time, 10 us bins, 0-5 ms

    // ---- callback thread ----
    // t0_mono_ns / t0_real_ns: clocks read at callback entry. m.arrival_* and m.epoch are filled here.
    void deliver(FrameMeta m, const void* data, std::size_t bytes, int64_t t0_mono_ns, int64_t t0_real_ns) {
        if (stopping_.load(std::memory_order_relaxed)) {
            after_stop_.fetch_add(1, std::memory_order_relaxed);   // shutdown: both cameras stop at the same instant
            return;
        }
        const int64_t ig_from = ignore_from_ns_.load(std::memory_order_relaxed);
        if (ig_from && t0_mono_ns >= ig_from && t0_mono_ns < ignore_until_ns_.load(std::memory_order_relaxed)) {
            ignored_.fetch_add(1, std::memory_order_relaxed);   // test option: behave as if no frame arrived
            return;
        }
        const uint64_t epoch = epoch_.load(std::memory_order_acquire);
        m.epoch = epoch;
        m.arrival_mono_ns = t0_mono_ns;
        m.arrival_real_ns = t0_real_ns;
        frames_.fetch_add(1, std::memory_order_relaxed);
        if (first_arrival_ns_.load(std::memory_order_relaxed) == 0) first_arrival_ns_.store(t0_mono_ns, std::memory_order_relaxed);
        last_arrival_ns_.store(t0_mono_ns, std::memory_order_relaxed);

        // dropped frames from frame-counter gaps (same epoch only; counters restart with the stream)
        if (m.frame_counter >= 0) {
            if (cb_epoch_ == epoch && prev_counter_ >= 0 && m.frame_counter > prev_counter_ + 1)
                dropped_.fetch_add(uint64_t(m.frame_counter - prev_counter_ - 1), std::memory_order_relaxed);
            prev_counter_ = m.frame_counter;
            cb_epoch_ = epoch;
        }
        if (m.domain == kDomainGlobalTime) lat_hist_.add(double(t0_real_ns) / 1e6 - m.ts_ms);

        DepthSlot* s = ring_.claim();                 // nullptr: ring full -> frame discarded, overflow counted
        if (s) {
            const std::size_t n = bytes < kImageBytes ? bytes : kImageBytes;
            std::memcpy(s->depth, data, n);
            m.data_bytes = uint32_t(n);
            s->meta = m;
            ring_.publish();
            if (n != kImageBytes) short_frames_.fetch_add(1, std::memory_order_relaxed);
        }
        sig_.notify();
        const int64_t dt = mono_ns() - t0_mono_ns;
        cb_hist_.add(double(dt) / 1e6);
        atomic_max(cb_max_ns_, dt);
    }

    // ---- supervisor ----
    // Starts a new stream generation (camera restart). Call before the camera is started again.
    void begin_epoch(int64_t now_ns) {
        epoch_.fetch_add(1, std::memory_order_acq_rel);
        first_arrival_ns_.store(0, std::memory_order_relaxed);
        stream_start_ns_.store(now_ns, std::memory_order_relaxed);
    }
    void set_stream_start(int64_t now_ns) { stream_start_ns_.store(now_ns, std::memory_order_relaxed); }
    // Shutdown: from now on deliver() drops frames (counted as after_stop), so the final counters end at one instant.
    void set_stopping() { stopping_.store(true, std::memory_order_relaxed); }
    void set_ignore_window(int64_t from_ns, int64_t until_ns) {
        ignore_until_ns_.store(until_ns, std::memory_order_relaxed);
        ignore_from_ns_.store(from_ns, std::memory_order_relaxed);
    }

    // ---- statistics (any thread) ----
    int cam() const { return cam_; }
    SpscRing<DepthSlot>& ring() { return ring_; }
    const AtomicHistogram& latency_hist() const { return lat_hist_; }
    const AtomicHistogram& callback_hist() const { return cb_hist_; }
    uint64_t frames() const { return frames_.load(std::memory_order_relaxed); }
    uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }
    uint64_t ignored() const { return ignored_.load(std::memory_order_relaxed); }
    uint64_t after_stop() const { return after_stop_.load(std::memory_order_relaxed); }
    uint64_t short_frames() const { return short_frames_.load(std::memory_order_relaxed); }
    int64_t cb_max_ns() const { return cb_max_ns_.load(std::memory_order_relaxed); }
    int64_t last_arrival_ns() const { return last_arrival_ns_.load(std::memory_order_relaxed); }
    int64_t first_arrival_ns() const { return first_arrival_ns_.load(std::memory_order_relaxed); }
    int64_t stream_start_ns() const { return stream_start_ns_.load(std::memory_order_relaxed); }
    uint64_t epoch() const { return epoch_.load(std::memory_order_relaxed); }

private:
    const int cam_;
    EventSignal& sig_;
    SpscRing<DepthSlot> ring_;
    AtomicHistogram lat_hist_;
    AtomicHistogram cb_hist_;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<uint64_t> frames_{0}, dropped_{0}, ignored_{0}, short_frames_{0}, after_stop_{0};
    std::atomic<bool> stopping_{false};
    std::atomic<int64_t> cb_max_ns_{0};
    std::atomic<int64_t> last_arrival_ns_{0}, first_arrival_ns_{0}, stream_start_ns_{0};
    std::atomic<int64_t> ignore_from_ns_{0}, ignore_until_ns_{0};
    // callback-thread only
    int64_t prev_counter_ = -1;
    uint64_t cb_epoch_ = ~0ULL;
};

}  // namespace cap
