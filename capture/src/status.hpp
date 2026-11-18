// Supervisor thread (R6, R7): watchdog and camera restart, frame-rate check, status line every N seconds.
// Also writes the final counters, the latency histograms and the optional JSON at shutdown.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <librealsense2/rs.hpp>

#include "camera.hpp"
#include "config.hpp"
#include "fps_monitor.hpp"
#include "matcher.hpp"
#include "producer.hpp"

namespace cap {

class Supervisor {
public:
    Supervisor(const Config& cfg, rs2::context& ctx, Camera* cams[2], FrameProducer* prods[2], Matcher& matcher,
               int64_t start_ns);
    ~Supervisor();
    void start();
    void stop();

    uint64_t restarts(int c) const { return restarts_[c].load(); }
    uint64_t fps_warnings(int c) const { return fps_warnings_[c].load(); }
    static long rss_kb();   // resident set size of this process (Linux /proc/self/statm), 0 if not available

    // Final counters to stdout (and JSON if configured); latency histograms to cfg.hist_out.
    void final_report(int64_t end_ns, uint64_t sink_pairs, const std::string& sink_name);

private:
    struct CamState {
        int64_t epoch_start_ns = 0;
        int64_t retry_at_ns = 0;     // > 0: a restart failed, retry at this time
        int64_t fps_t = 0;
        FpsMonitor fps{kExpectedFps, kFpsWarnTolerance};
        double last_fps = 0;
        // status interval baselines
        std::vector<uint64_t> cb_prev, lat_prev;
        uint64_t frames_prev = 0;
        // condition (gap 2): ok | reset (hardware reset requested) | restarting | absent (not found by serial) |
        // restart_failed | settling (restarted, no pair yet)
        const char* state = "ok";
        int64_t hold_until_ns = 0;   // after a requested hardware reset: no watchdog restart before this time
        bool recovering = false;     // from loss of frames (or reset request) until the first pair of the new epoch
        int64_t lost_ns = 0;
        uint64_t recover_epoch = 0;
        uint64_t recoveries = 0;
    };
    void run();
    void watchdog(int c, int64_t now);
    void fps_check(int c, int64_t now);
    void status_line(int64_t now);
    void do_restart(int c, int64_t now, const char* reason);
    void recovery_check(int c, int64_t now);
    void hw_reset_step(int64_t now);
    void event(int c, const char* what, const std::string& extra = std::string());

    const Config& cfg_;
    rs2::context& ctx_;
    Camera* cams_[2];
    FrameProducer* prods_[2];
    Matcher& matcher_;
    const int64_t start_ns_;
    CamState st_[2];
    std::vector<uint64_t> dt_prev_;
    std::vector<uint64_t> od_prev_[2][2];   // output delay histograms, status interval baselines

    // hardware-reset test (gap 2)
    std::size_t hw_idx_ = 0;
    int64_t hw_next_ns_ = 0;
    int hw_wait_cam_ = -1;        // camera of the last reset, until it delivers pairs again
    int64_t hw_req_ns_ = 0;
    uint64_t hw_requests_ = 0, hw_failures_ = 0, hw_recovery_timeouts_ = 0;
    long rss_after_first_recovery_kb_ = 0;
    int64_t status_prev_ns_ = 0;

    std::atomic<uint64_t> restarts_[2];
    std::atomic<uint64_t> fps_warnings_[2];
    std::atomic<uint64_t> restart_failures_[2];
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

}  // namespace cap
