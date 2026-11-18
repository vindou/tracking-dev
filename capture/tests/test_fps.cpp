// T1 (extra): frame-rate monitor (R6). No false warnings at the real rate with arrival jitter; a warning after more than
// 2 s at a wrong rate.
#include <cstdint>
#include <random>

#include "../src/fps_monitor.hpp"
#include "test_framework.hpp"

using namespace cap;

namespace {
// Simulates frames at `fps` with uniform arrival jitter, ticks every 1 s +- 50 ms (like the supervisor).
// Returns the number of warnings over `seconds`; first_warning_s = time of the first warning (-1 if none).
int simulate(double fps, double jitter_ms, double seconds, double& first_warning_s, double gap_from_s = -1,
             double gap_len_s = 0) {
    FpsMonitor m(89.87, 0.5);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jit(-jitter_ms, jitter_ms), tickj(-0.05, 0.05);
    const double period_ms = 1000.0 / fps;
    uint64_t frames = 0;
    int64_t last_arrival = 0;
    double next_frame_ms = 0;
    double next_tick_ms = 1000;
    int warnings = 0;
    first_warning_s = -1;
    while (next_tick_ms < seconds * 1000) {
        while (next_frame_ms < next_tick_ms) {
            const double t = next_frame_ms;
            next_frame_ms += period_ms;
            if (gap_from_s >= 0 && t >= gap_from_s * 1000 && t < (gap_from_s + gap_len_s) * 1000) continue;
            frames++;
            last_arrival = int64_t((t + 9.0 + jit(rng)) * 1e6);
        }
        double f = 0;
        if (m.update(int64_t(next_tick_ms * 1e6), frames, last_arrival, f)) {
            if (first_warning_s < 0) first_warning_s = next_tick_ms / 1000;
            warnings++;
        }
        next_tick_ms += 1000 + tickj(rng) * 1000;
    }
    return warnings;
}
}  // namespace

TEST(fps_monitor_no_false_warning_at_real_rate) {
    double first = 0;
    // 89.87 fps, +-3 ms arrival jitter, 1 hour: no warning (the old 1-s frame count gave several per 45 min)
    CHECK_EQ(simulate(89.87, 3.0, 3600, first), 0);
    // still inside the band: 89.5 and 90.25 fps
    CHECK_EQ(simulate(89.5, 3.0, 600, first), 0);
    CHECK_EQ(simulate(90.25, 3.0, 600, first), 0);
}

TEST(fps_monitor_warns_at_wrong_rate) {
    double first = 0;
    // 60 fps (exposure above the frame period): warning after more than 2 s, once per episode
    CHECK_EQ(simulate(60.0, 2.0, 120, first), 1);
    CHECK(first > 2.0 && first < 6.0);
    // 89.0 fps: outside 89.87 +- 0.5
    CHECK_EQ(simulate(89.0, 2.0, 120, first), 1);
    // 91.0 fps
    CHECK_EQ(simulate(91.0, 2.0, 120, first), 1);
}

TEST(fps_monitor_short_gap_and_long_gap) {
    double first = 0;
    // 0.3 s without frames: the 2 s rate stays low for about 2 s -> at most one warning, normal afterwards
    CHECK_LE(simulate(89.87, 2.0, 120, first, 30.0, 0.3), 1);
    // 5 s without frames: one warning
    CHECK_EQ(simulate(89.87, 2.0, 120, first, 30.0, 5.0), 1);
    CHECK(first > 30.0 && first < 36.0);
}
