// Frame-rate check (R6): warn when the measured rate is outside expected +- tolerance for more than hold_s seconds.
//
// The rate is measured from frame arrival times, not from frame counts per tick: at each tick (about 1 s) the monitor
// takes the number of frames since a tick about window_s earlier and divides it by the time between the last frames of
// those two ticks. Counting whole frames per 1 s window has a resolution of +-1 fps at 90 fps (one frame), more than
// the +-0.5 fps tolerance, and gave false warnings (acceptance run long_1 of the first T2 series).
#pragma once

#include <cstddef>
#include <cstdint>

namespace cap {

class FpsMonitor {
public:
    FpsMonitor(double expected_fps, double tolerance_fps, double window_s = 2.0, double hold_s = 2.0)
        : expected_(expected_fps), tol_(tolerance_fps), window_ns_(int64_t(window_s * 1e9)), hold_ns_(int64_t(hold_s * 1e9)) {}

    // frames: total frames delivered so far; last_arrival_ns: arrival time of the last of them (0 if none).
    // Returns true when a warning must be written now (once per episode). fps_out: rate over the window (0 if no frames).
    bool update(int64_t now_ns, uint64_t frames, int64_t last_arrival_ns, double& fps_out) {
        Sample cur{now_ns, frames, last_arrival_ns};
        // oldest kept sample whose tick is at least window_ns before now
        const Sample* old = nullptr;
        for (std::size_t i = 0; i < n_; i++) {
            const Sample& s = hist_[(head_ + kCap - n_ + i) % kCap];
            if (now_ns - s.tick_ns >= window_ns_) old = &s;
        }
        push(cur);
        if (!old) {                                   // not enough history yet
            fps_out = -1;
            return false;
        }
        if (frames == old->frames || last_arrival_ns <= old->last_arrival_ns) {
            fps_out = 0;                              // no frame in the window
        } else {
            fps_out = double(frames - old->frames) / (double(last_arrival_ns - old->last_arrival_ns) / 1e9);
            // if frames stopped arriving, the time since the last frame also counts
            const int64_t silent = now_ns - last_arrival_ns;
            if (silent > window_ns_ / 2)
                fps_out = double(frames - old->frames) / (double(now_ns - old->last_arrival_ns) / 1e9);
        }
        const bool bad = fps_out < expected_ - tol_ || fps_out > expected_ + tol_;
        if (!bad) {
            bad_since_ = 0;
            warned_ = false;
            return false;
        }
        if (!bad_since_) bad_since_ = now_ns;
        if (!warned_ && now_ns - bad_since_ > hold_ns_) {
            warned_ = true;
            return true;
        }
        return false;
    }

    // After a camera restart: start a new measurement.
    void reset() {
        n_ = 0;
        bad_since_ = 0;
        warned_ = false;
    }

private:
    struct Sample {
        int64_t tick_ns;
        uint64_t frames;
        int64_t last_arrival_ns;
    };
    static constexpr std::size_t kCap = 16;
    void push(const Sample& s) {
        hist_[head_] = s;
        head_ = (head_ + 1) % kCap;
        if (n_ < kCap) n_++;
    }
    double expected_, tol_;
    int64_t window_ns_, hold_ns_;
    Sample hist_[kCap] = {};
    std::size_t head_ = 0, n_ = 0;
    int64_t bad_since_ = 0;
    bool warned_ = false;
};

}  // namespace cap
