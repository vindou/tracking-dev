#include "status.hpp"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <unistd.h>

#include "thread_util.hpp"

namespace cap {

namespace {
double ms(int64_t ns) { return double(ns) / 1e6; }
double elapsed_s(int64_t now, int64_t start) { return double(now - start) / 1e9; }
double fin(double v) { return std::isfinite(v) ? v : -1.0; }   // JSON: -1 = no data (empty histogram or overflow)
}  // namespace

Supervisor::Supervisor(const Config& cfg, rs2::context& ctx, Camera* cams[2], FrameProducer* prods[2], Matcher& matcher,
                       int64_t start_ns)
    : cfg_(cfg), ctx_(ctx), cams_{cams[0], cams[1]}, prods_{prods[0], prods[1]}, matcher_(matcher), start_ns_(start_ns) {
    for (int c = 0; c < 2; c++) {
        restarts_[c] = 0;
        fps_warnings_[c] = 0;
        restart_failures_[c] = 0;
        st_[c].epoch_start_ns = start_ns;
        st_[c].fps_t = start_ns;
        prods_[c]->callback_hist().snapshot(st_[c].cb_prev);
        prods_[c]->latency_hist().snapshot(st_[c].lat_prev);
    }
    matcher_.dt_hist().snapshot(dt_prev_);
    for (int k = 0; k < 2; k++)
        for (int w = 0; w < 2; w++) matcher_.out_delay_hist(k, w).snapshot(od_prev_[k][w]);
    status_prev_ns_ = start_ns;
    hw_next_ns_ = start_ns + int64_t(cfg.test_hw_first_s * 1e9);
}

long Supervisor::rss_kb() {
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long size = 0, rss = 0;
    const int n = std::fscanf(f, "%ld %ld", &size, &rss);
    std::fclose(f);
    return n == 2 ? rss * (sysconf(_SC_PAGESIZE) / 1024) : 0;
}

// One machine-readable line per condition change (gap 2 evaluation): times, epochs, the drop and overflow counters of
// both cameras, pairs and RSS at that moment.
void Supervisor::event(int c, const char* what, const std::string& extra) {
    const int64_t now = mono_ns();
    std::printf("[event +%.3fs] cam=%s what=%s mono_ns=%" PRId64 " real_ns=%" PRId64 " epoch=%" PRIu64
                " dropA=%" PRIu64 " ovfA=%" PRIu64 " dropB=%" PRIu64 " ovfB=%" PRIu64 " pairs=%" PRIu64
                " rss_kb=%ld%s%s\n",
                elapsed_s(now, start_ns_), c >= 0 ? cam_name(c) : "-", what, now, real_ns(),
                c >= 0 ? prods_[c]->epoch() : 0, prods_[0]->dropped(), prods_[0]->ring().overflows(),
                prods_[1]->dropped(), prods_[1]->ring().overflows(), matcher_.pairs(), rss_kb(),
                extra.empty() ? "" : " ", extra.c_str());
    std::fflush(stdout);
}

Supervisor::~Supervisor() { stop(); }

void Supervisor::start() { th_ = std::thread([this] { run(); }); }

void Supervisor::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

void Supervisor::run() {
    set_thread_name("supervisor");
    if (cfg_.cpu_supervisor >= 0) set_thread_affinity(cfg_.cpu_supervisor);
    int64_t next_status = start_ns_ + int64_t(cfg_.status_interval_s * 1e9);
    std::unique_lock<std::mutex> lk(mu_);
    while (!stop_) {
        cv_.wait_for(lk, std::chrono::milliseconds(50));
        if (stop_) break;
        lk.unlock();
        const int64_t now = mono_ns();
        hw_reset_step(now);
        for (int c = 0; c < 2; c++) {
            watchdog(c, now);
            recovery_check(c, mono_ns());
            fps_check(c, now);
        }
        if (now >= next_status) {
            status_line(now);
            next_status += int64_t(cfg_.status_interval_s * 1e9);
        }
        lk.lock();
    }
}

void Supervisor::do_restart(int c, int64_t now, const char* reason) {
    std::printf("[watchdog +%.1fs] camera %s: %s -> stop, close, open, start\n", elapsed_s(now, start_ns_), cam_name(c), reason);
    std::fflush(stdout);
    CamState& s = st_[c];
    if (!s.recovering) {
        s.recovering = true;
        s.lost_ns = now;
        event(c, "frames_lost", std::string("reason=\"") + reason + "\"");
    }
    s.state = "restarting";
    std::string err;
    const int64_t t0 = mono_ns();
    const uint64_t full0 = cams_[c]->full_opens();
    bool ok = false;
    try {
        ok = cams_[c]->restart(ctx_, err);
    } catch (const std::exception& e) {   // restart() catches SDK errors itself; this is a second guard
        err = std::string("exception: ") + e.what();
    } catch (...) {
        err = "unknown exception";
    }
    const int64_t t1 = mono_ns();
    restarts_[c]++;
    s.epoch_start_ns = t1;
    s.fps.reset();
    if (ok) {
        s.retry_at_ns = 0;
        s.state = "settling";
        s.recover_epoch = prods_[c]->epoch();
        std::printf("[watchdog +%.1fs] camera %s restarted in %.0f ms (epoch %" PRIu64 "%s)\n", elapsed_s(t1, start_ns_),
                    cam_name(c), ms(t1 - t0), prods_[c]->epoch(),
                    cams_[c]->full_opens() != full0 ? ", found again by serial, all startup checks passed" : "");
        event(c, "restarted", cams_[c]->full_opens() != full0 ? "open=full" : "open=fast");
    } else {
        restart_failures_[c]++;
        s.retry_at_ns = t1 + 1000000000LL;
        s.state = err.find("not found") != std::string::npos ? "absent" : "restart_failed";
        std::printf("[watchdog +%.1fs] camera %s restart FAILED: %s (retry in 1 s)\n", elapsed_s(t1, start_ns_),
                    cam_name(c), err.c_str());
    }
    std::fflush(stdout);
}

// Recovery is complete when a pair with a frame of the new epoch has been published.
void Supervisor::recovery_check(int c, int64_t now) {
    CamState& s = st_[c];
    if (!s.recovering || s.retry_at_ns || s.recover_epoch == 0) return;
    if (std::strcmp(s.state, "settling") != 0) return;
    if (matcher_.last_pair_epoch(c) != s.recover_epoch) return;
    s.recovering = false;
    s.state = "ok";
    s.recoveries++;
    char x[96];
    std::snprintf(x, sizeof x, "lost_for_ms=%.0f", ms(now - s.lost_ns));
    event(c, "pairs_resumed", x);
    if (rss_after_first_recovery_kb_ == 0 && hw_requests_ > 0) rss_after_first_recovery_kb_ = rss_kb();
}

// Test option --test-hw-reset (gap 2): one reset at a time; the next one INTERVAL s after the previous camera
// delivered pairs again (or after 120 s without recovery, which the evaluation counts as a failure).
void Supervisor::hw_reset_step(int64_t now) {
    if (cfg_.test_hw_seq.empty()) return;
    if (hw_wait_cam_ >= 0) {
        const CamState& s = st_[hw_wait_cam_];
        if (!s.recovering) {
            hw_wait_cam_ = -1;
            hw_next_ns_ = now + int64_t(cfg_.test_hw_interval_s * 1e9);
            if (hw_idx_ >= cfg_.test_hw_seq.size()) event(-1, "hw_reset_series_done");
        } else if (now - hw_req_ns_ > 120000000000LL) {
            hw_recovery_timeouts_++;
            event(hw_wait_cam_, "recovery_timeout");
            hw_wait_cam_ = -1;
            hw_next_ns_ = now + int64_t(cfg_.test_hw_interval_s * 1e9);
            if (hw_idx_ >= cfg_.test_hw_seq.size()) event(-1, "hw_reset_series_done");
        }
        return;
    }
    if (hw_idx_ >= cfg_.test_hw_seq.size() || now < hw_next_ns_) return;
    const int c = cfg_.test_hw_seq[hw_idx_] == 'A' ? kCamA : kCamB;
    CamState& s = st_[c];
    if (s.recovering) return;   // that camera is still recovering from something else: wait
    hw_idx_++;
    hw_requests_++;
    char x[64];
    std::snprintf(x, sizeof x, "n=%zu/%zu", hw_idx_, cfg_.test_hw_seq.size());
    event(c, "hw_reset_request", x);
    s.recovering = true;
    s.lost_ns = mono_ns();
    s.state = "reset";
    s.recover_epoch = 0;
    s.hold_until_ns = s.lost_ns + 2000000000LL;   // let the device disconnect before the watchdog restarts it
    hw_req_ns_ = s.lost_ns;
    hw_wait_cam_ = c;
    std::string err;
    if (cams_[c]->hardware_reset(err)) {
        event(c, "hw_reset_sent");
    } else {
        hw_failures_++;
        event(c, "hw_reset_failed", "err=\"" + err + "\"");
    }
}

void Supervisor::watchdog(int c, int64_t now) {
    CamState& s = st_[c];
    if (s.hold_until_ns) {
        if (now < s.hold_until_ns) return;
        s.hold_until_ns = 0;
        do_restart(c, now, "hardware reset requested (device re-enumerates)");
        return;
    }
    if (s.retry_at_ns) {
        if (now >= s.retry_at_ns) do_restart(c, now, "retry after failed restart");
        return;
    }
    const double period = matcher_.snapshot().period_known ? matcher_.snapshot().period_ms : kExpectedPeriodMs;
    const int64_t first = prods_[c]->first_arrival_ns();
    const int64_t last = prods_[c]->last_arrival_ns();
    if (first == 0 || first < s.epoch_start_ns - 1000000000LL) {
        if (now - s.epoch_start_ns > 2000000000LL) do_restart(c, now, "no frame within 2 s after start");
    } else if (double(now - last) > 10.0 * period * 1e6) {
        char why[96];
        std::snprintf(why, sizeof why, "no frame for %.0f ms (> 10 frame periods)", ms(now - last));
        do_restart(c, now, why);
    }
}

void Supervisor::fps_check(int c, int64_t now) {
    CamState& s = st_[c];
    if (now - s.fps_t < 1000000000LL) return;   // one sample per second; the monitor measures over 2 s
    s.fps_t = now;
    double fps = 0;
    if (s.fps.update(now, prods_[c]->frames(), prods_[c]->last_arrival_ns(), fps)) {
        fps_warnings_[c]++;
        std::printf("[warning +%.1fs] camera %s: frame rate %.2f fps is outside %.2f +- %.1f fps for more than 2 s\n",
                    elapsed_s(now, start_ns_), cam_name(c), fps, kExpectedFps, kFpsWarnTolerance);
        std::fflush(stdout);
    }
    s.last_fps = fps;
}

void Supervisor::status_line(int64_t now) {
    const double dt_s = double(now - status_prev_ns_) / 1e9;
    status_prev_ns_ = now;
    char buf[1536];
    int n = std::snprintf(buf, sizeof buf, "[status +%.0fs]", elapsed_s(now, start_ns_));
    for (int c = 0; c < 2; c++) {
        FrameProducer& p = *prods_[c];
        CamState& s = st_[c];
        std::vector<uint64_t> cb, lat;
        p.callback_hist().snapshot(cb);
        p.latency_hist().snapshot(lat);
        const auto dcb = AtomicHistogram::diff(cb, s.cb_prev);
        const auto dlat = AtomicHistogram::diff(lat, s.lat_prev);
        s.cb_prev = cb;
        s.lat_prev = lat;
        const uint64_t f = p.frames();
        const double fps = double(f - s.frames_prev) / dt_s;
        s.frames_prev = f;
        n += std::snprintf(buf + n, sizeof buf - n,
                           " %s[%s]: %.2ffps drop=%" PRIu64 " ovf=%" PRIu64 " fill=%" PRIu64 "/%" PRIu64
                           " cb99=%.2f cbmax=%.2fms lat50=%.1f lat99=%.1fms |",
                           cam_name(c), s.state, fps, p.dropped(), p.ring().overflows(), p.ring().take_interval_max_fill(),
                           p.ring().max_fill(), p.callback_hist().quantile(dcb, 0.99), p.callback_hist().max_value(dcb),
                           p.latency_hist().quantile(dlat, 0.5), p.latency_hist().quantile(dlat, 0.99));
    }
    const MatcherSnapshot m = matcher_.snapshot();
    std::vector<uint64_t> dth;
    matcher_.dt_hist().snapshot(dth);
    const auto ddt = AtomicHistogram::diff(dth, dt_prev_);
    dt_prev_ = dth;
    n += std::snprintf(buf + n, sizeof buf - n,
                       " pairs=%" PRIu64 " unpaired=%" PRIu64 "/%" PRIu64 " dt[min/med/max]=%.2f/%.2f/%.2fms"
                       " partner_changes=%" PRIu64 " restarts=%" PRIu64 "/%" PRIu64 " settled=%d/%d P=%.4fms",
                       m.engine.pairs, m.engine.unpaired(0), m.engine.unpaired(1), matcher_.dt_hist().min_value(ddt),
                       matcher_.dt_hist().quantile(ddt, 0.5), matcher_.dt_hist().max_value(ddt), m.engine.partner_changes,
                       restarts_[0].load(), restarts_[1].load(), m.settled[0] ? 1 : 0, m.settled[1] ? 1 : 0, m.period_ms);
    // output delay (gap 3) over the interval: publish time - global timestamp of the older / newer frame
    std::vector<uint64_t> od[2][2], dod[2][2];
    for (int k = 0; k < 2; k++)
        for (int w = 0; w < 2; w++) {
            matcher_.out_delay_hist(k, w).snapshot(od[k][w]);
            dod[k][w] = AtomicHistogram::diff(od[k][w], od_prev_[k][w]);
            od_prev_[k][w] = od[k][w];
        }
    const AtomicHistogram& oh = matcher_.out_delay_hist(0, 0);
    n += std::snprintf(buf + n, sizeof buf - n,
                       " | out old50/99=%.1f/%.1f new50/99=%.1f/%.1fms near_half n=%" PRIu64 " old50/99=%.1f/%.1fms",
                       oh.quantile(dod[kDelayUsual][kDelayOlder], 0.5), oh.quantile(dod[kDelayUsual][kDelayOlder], 0.99),
                       oh.quantile(dod[kDelayUsual][kDelayNewer], 0.5), oh.quantile(dod[kDelayUsual][kDelayNewer], 0.99),
                       AtomicHistogram::total(dod[kDelayNearHalf][kDelayOlder]),
                       oh.quantile(dod[kDelayNearHalf][kDelayOlder], 0.5),
                       oh.quantile(dod[kDelayNearHalf][kDelayOlder], 0.99));
    std::printf("%s\n", buf);
    std::fflush(stdout);
}

void Supervisor::final_report(int64_t end_ns, uint64_t sink_pairs, const std::string& sink_name) {
    const MatcherSnapshot m = matcher_.snapshot();
    const double run_s = elapsed_s(end_ns, start_ns_);
    std::vector<uint64_t> cb[2], lat[2];
    for (int c = 0; c < 2; c++) {
        prods_[c]->callback_hist().snapshot(cb[c]);
        prods_[c]->latency_hist().snapshot(lat[c]);
    }
    std::vector<uint64_t> dth;
    matcher_.dt_hist().snapshot(dth);

    std::printf("\n=== final counters (run %.1f s, sink %s) ===\n", run_s, sink_name.c_str());
    for (int c = 0; c < 2; c++) {
        FrameProducer& p = *prods_[c];
        const double elig = double(m.engine.eligible(c));
        std::printf("camera %s %s: frames=%" PRIu64 " mean_fps=%.3f dropped=%" PRIu64 " overflows=%" PRIu64
                    " max_fill=%" PRIu64 " ignored(test)=%" PRIu64 " after_stop=%" PRIu64 " short_frames=%" PRIu64 " callback_errors=%" PRIu64 "\n"
                    "   callback p50/p99/max = %.3f/%.3f/%.3f ms  latency p50/p99 = %.1f/%.1f ms  restarts=%" PRIu64
                    " fps_warnings=%" PRIu64 " period=%.5f ms\n"
                    "   matcher: eligible=%" PRIu64 " paired=%" PRIu64 " (%.3f%%) unpaired=%" PRIu64 " settle=%" PRIu64
                    " domain=%" PRIu64 " partner_settling=%" PRIu64 " restart=%" PRIu64 " timeouts=%" PRIu64 "\n",
                    cam_name(c), cams_[c]->serial().c_str(), p.frames(), double(p.frames()) / run_s, p.dropped(),
                    p.ring().overflows(), p.ring().max_fill(), p.ignored(), p.after_stop(), p.short_frames(), cams_[c]->callback_errors(),
                    p.callback_hist().quantile(cb[c], 0.5), p.callback_hist().quantile(cb[c], 0.99), ms(p.cb_max_ns()),
                    p.latency_hist().quantile(lat[c], 0.5), p.latency_hist().quantile(lat[c], 0.99), restarts_[c].load(),
                    fps_warnings_[c].load(), m.camera_period_ms[c], m.engine.eligible(c), m.engine.paired[c],
                    elig > 0 ? 100.0 * double(m.engine.paired[c]) / elig : 0.0, m.engine.unpaired(c),
                    m.engine.discards[c][int(Discard::Settle)], m.engine.discards[c][int(Discard::Domain)],
                    m.engine.discards[c][int(Discard::PartnerSettling)], m.engine.discards[c][int(Discard::Restart)],
                    m.engine.timeout_decisions[c]);
    }
    const bool have_dt = matcher_.pairs() > 0;
    std::printf("matcher: pairs=%" PRIu64 " sink_pairs=%" PRIu64 " partner_changes=%" PRIu64
                " dt min/median/max = %.3f/%.2f/%.3f ms tolerance=%.4f ms order_errors=%" PRIu64 " csv_overflows=%" PRIu64
                "\n",
                m.engine.pairs, sink_pairs, m.engine.partner_changes, have_dt ? ms(matcher_.dt_min_ns()) : 0.0,
                matcher_.dt_hist().quantile(dth, 0.5), have_dt ? ms(matcher_.dt_max_ns()) : 0.0, m.period_ms / 2,
                m.order_errors, m.csv_overflows);
    // output delay (gap 3)
    std::vector<uint64_t> od[2][2];
    for (int k = 0; k < 2; k++)
        for (int w = 0; w < 2; w++) matcher_.out_delay_hist(k, w).snapshot(od[k][w]);
    const AtomicHistogram& oh = matcher_.out_delay_hist(0, 0);
    const char* cat_name[2] = {"usual", "near_half"};
    for (int k = 0; k < 2; k++)
        std::printf("output delay %s (|dt| %s %.1f ms from half a period): pairs=%" PRIu64
                    " older frame p50/p99/max = %.1f/%.1f/%.1f ms, newer frame p50/p99/max = %.1f/%.1f/%.1f ms\n",
                    cat_name[k], k ? "within" : "more than", kNearHalfMarginMs, AtomicHistogram::total(od[k][0]),
                    oh.quantile(od[k][0], 0.5), oh.quantile(od[k][0], 0.99), oh.max_value(od[k][0]),
                    oh.quantile(od[k][1], 0.5), oh.quantile(od[k][1], 0.99), oh.max_value(od[k][1]));
    const long rss_end = rss_kb();
    std::printf("process: rss_kb=%ld", rss_end);
    if (!cfg_.test_hw_seq.empty())
        std::printf(" hw_resets requested=%" PRIu64 " failed=%" PRIu64 " recovery_timeouts=%" PRIu64
                    " rss_kb_after_first_recovery=%ld recoveries A/B=%" PRIu64 "/%" PRIu64 " full_opens A/B=%" PRIu64
                    "/%" PRIu64,
                    hw_requests_, hw_failures_, hw_recovery_timeouts_, rss_after_first_recovery_kb_, st_[0].recoveries,
                    st_[1].recoveries, cams_[0]->full_opens(), cams_[1]->full_opens());
    std::printf("\n");
    std::fflush(stdout);
    if (!cfg_.delay_hist_out.empty()) {
        if (FILE* f = std::fopen(cfg_.delay_hist_out.c_str(), "w")) {
            std::fprintf(f, "# output delay = publish CLOCK_REALTIME - SDK global timestamp of the older / newer frame of a "
                            "pair; 0.1 ms bins; near_half: ||dt| - P/2| <= %.1f ms\n", kNearHalfMarginMs);
            std::fprintf(f, "bin_lo_ms,bin_hi_ms,older_usual,newer_usual,older_near_half,newer_near_half\n");
            auto row = [&](const char* lo, const char* hi, std::size_t i) {
                std::fprintf(f, "%s,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", lo, hi, od[0][0][i], od[0][1][i],
                             od[1][0][i], od[1][1][i]);
            };
            char lo[32], hi[32];
            std::snprintf(hi, sizeof hi, "%.1f", oh.lo());
            row("-inf", hi, 0);
            for (std::size_t i = 0; i < oh.nbins(); i++) {
                std::snprintf(lo, sizeof lo, "%.1f", oh.lo() + double(i) * oh.width());
                std::snprintf(hi, sizeof hi, "%.1f", oh.lo() + double(i + 1) * oh.width());
                row(lo, hi, i + 1);
            }
            std::snprintf(lo, sizeof lo, "%.1f", oh.lo() + double(oh.nbins()) * oh.width());
            row(lo, "inf", oh.nbins() + 1);
            std::fclose(f);
        } else {
            std::fprintf(stderr, "cannot write %s\n", cfg_.delay_hist_out.c_str());
        }
    }

    // latency histograms (R7)
    if (!cfg_.hist_out.empty()) {
        if (FILE* f = std::fopen(cfg_.hist_out.c_str(), "w")) {
            const AtomicHistogram& h = prods_[0]->latency_hist();
            std::fprintf(f, "# delivery latency = arrival CLOCK_REALTIME - SDK global timestamp; 0.1 ms bins\n");
            std::fprintf(f, "bin_lo_ms,bin_hi_ms,count_A,count_B\n");
            std::fprintf(f, "-inf,%.1f,%" PRIu64 ",%" PRIu64 "\n", h.lo(), lat[0][0], lat[1][0]);
            for (std::size_t i = 0; i < h.nbins(); i++)
                std::fprintf(f, "%.1f,%.1f,%" PRIu64 ",%" PRIu64 "\n", h.lo() + double(i) * h.width(),
                             h.lo() + double(i + 1) * h.width(), lat[0][i + 1], lat[1][i + 1]);
            std::fprintf(f, "%.1f,inf,%" PRIu64 ",%" PRIu64 "\n", h.lo() + double(h.nbins()) * h.width(),
                         lat[0][h.nbins() + 1], lat[1][h.nbins() + 1]);
            std::fclose(f);
        } else {
            std::fprintf(stderr, "cannot write %s\n", cfg_.hist_out.c_str());
        }
    }
    if (!cfg_.final_json.empty()) {
        FILE* f = std::fopen(cfg_.final_json.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", cfg_.final_json.c_str());
            return;
        }
        std::fprintf(f, "{\n  \"run_s\": %.3f,\n  \"sink\": \"%s\",\n  \"sink_pairs\": %" PRIu64 ",\n", run_s, sink_name.c_str(),
                     sink_pairs);
        for (int c = 0; c < 2; c++) {
            FrameProducer& p = *prods_[c];
            std::fprintf(f,
                         "  \"%s\": {\"serial\": \"%s\", \"frames\": %" PRIu64 ", \"dropped\": %" PRIu64 ", \"overflows\": %" PRIu64
                         ", \"max_fill\": %" PRIu64 ", \"ignored\": %" PRIu64 ", \"after_stop\": %" PRIu64 ", \"short_frames\": %" PRIu64
                         ", \"callback_errors\": %" PRIu64 ", \"cb_p50_ms\": %.4f, \"cb_p99_ms\": %.4f, \"cb_max_ms\": %.4f"
                         ", \"lat_p50_ms\": %.2f, \"lat_p99_ms\": %.2f, \"restarts\": %" PRIu64 ", \"restart_failures\": %" PRIu64
                         ", \"fps_warnings\": %" PRIu64 ", \"period_ms\": %.6f, \"eligible\": %" PRIu64 ", \"paired\": %" PRIu64
                         ", \"unpaired\": %" PRIu64 ", \"settle_discards\": %" PRIu64 ", \"domain_discards\": %" PRIu64
                         ", \"partner_settling_discards\": %" PRIu64 ", \"restart_discards\": %" PRIu64
                         ", \"timeout_decisions\": %" PRIu64 ", \"period_fallback\": %" PRIu64
                         ", \"recoveries\": %" PRIu64 ", \"full_opens\": %" PRIu64 ", \"state\": \"%s\"},\n",
                         cam_name(c), cams_[c]->serial().c_str(), p.frames(), p.dropped(), p.ring().overflows(),
                         p.ring().max_fill(), p.ignored(), p.after_stop(), p.short_frames(), cams_[c]->callback_errors(),
                         p.callback_hist().quantile(cb[c], 0.5), p.callback_hist().quantile(cb[c], 0.99), ms(p.cb_max_ns()),
                         p.latency_hist().quantile(lat[c], 0.5), p.latency_hist().quantile(lat[c], 0.99), restarts_[c].load(),
                         restart_failures_[c].load(), fps_warnings_[c].load(), m.camera_period_ms[c], m.engine.eligible(c),
                         m.engine.paired[c], m.engine.unpaired(c), m.engine.discards[c][int(Discard::Settle)],
                         m.engine.discards[c][int(Discard::Domain)], m.engine.discards[c][int(Discard::PartnerSettling)],
                         m.engine.discards[c][int(Discard::Restart)], m.engine.timeout_decisions[c],
                         m.engine.period_fallback[c], st_[c].recoveries, cams_[c]->full_opens(), st_[c].state);
        }
        std::fprintf(f, "  \"output_delay\": {");
        for (int k = 0; k < 2; k++)
            std::fprintf(f,
                         "%s\"%s\": {\"pairs\": %" PRIu64 ", \"older_p50_ms\": %.2f, \"older_p99_ms\": %.2f"
                         ", \"older_max_ms\": %.2f, \"newer_p50_ms\": %.2f, \"newer_p99_ms\": %.2f, \"newer_max_ms\": %.2f}",
                         k ? ", " : "", cat_name[k], AtomicHistogram::total(od[k][0]), fin(oh.quantile(od[k][0], 0.5)),
                         fin(oh.quantile(od[k][0], 0.99)), fin(oh.max_value(od[k][0])), fin(oh.quantile(od[k][1], 0.5)),
                         fin(oh.quantile(od[k][1], 0.99)), fin(oh.max_value(od[k][1])));
        std::fprintf(f, "},\n");
        std::fprintf(f,
                     "  \"process\": {\"rss_kb_end\": %ld, \"rss_kb_after_first_recovery\": %ld, \"hw_resets_requested\": %" PRIu64
                     ", \"hw_reset_failures\": %" PRIu64 ", \"hw_recovery_timeouts\": %" PRIu64 "},\n",
                     rss_end, rss_after_first_recovery_kb_, hw_requests_, hw_failures_, hw_recovery_timeouts_);
        std::fprintf(f,
                     "  \"matcher\": {\"pairs\": %" PRIu64 ", \"partner_changes\": %" PRIu64 ", \"dt_min_ms\": %.4f"
                     ", \"dt_median_ms\": %.3f, \"dt_max_ms\": %.4f, \"period_ms\": %.6f, \"tolerance_ms\": %.6f"
                     ", \"order_errors\": %" PRIu64 ", \"csv_overflows\": %" PRIu64 "}\n}\n",
                     m.engine.pairs, m.engine.partner_changes, have_dt ? ms(matcher_.dt_min_ns()) : 0.0,
                     matcher_.dt_hist().quantile(dth, 0.5), have_dt ? ms(matcher_.dt_max_ns()) : 0.0, m.period_ms,
                     m.period_ms / 2, m.order_errors, m.csv_overflows);
        std::fclose(f);
    }
}

}  // namespace cap
