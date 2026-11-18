// d435_capture: two free-running D435i cameras -> per-camera SPSC rings -> matcher -> pair sink (shared memory).
// Architecture and algorithms: DESIGN.md. Build and run: README.md.
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <librealsense2/rs.hpp>

#include "camera.hpp"
#include "config.hpp"
#include "event_signal.hpp"
#include "matcher.hpp"
#include "pair_sink.hpp"
#include "producer.hpp"
#include "shm_sink.hpp"
#include "status.hpp"
#include "thread_util.hpp"
#include "timestamp_source.hpp"

using namespace cap;

namespace {

// Non-time-critical thread: one CSV line per pair from a queue that the matcher fills.
class CsvWriter {
public:
    CsvWriter(const std::string& path, int cpu) : q_(4096), cpu_(cpu) {
        f_ = std::fopen(path.c_str(), "w");
    }
    ~CsvWriter() { stop(); }
    bool ok() const { return f_ != nullptr; }
    SpscRing<PairCsvItem>& queue() { return q_; }
    EventSignal& signal() { return sig_; }
    void start() { th_ = std::thread([this] { run(); }); }
    void stop() {
        if (th_.joinable()) {
            stop_.store(true);
            sig_.notify();
            th_.join();
        }
        if (f_) {
            std::fclose(f_);
            f_ = nullptr;
        }
    }
    uint64_t lines() const { return lines_; }

private:
    void run() {
        set_thread_name("csv");
        if (cpu_ >= 0) set_thread_affinity(cpu_);
        std::fprintf(f_,
                     "pair_seq,dt_ms,"
                     "a_global_ts_ms,a_hw_ts_us,a_sensor_ts_us,a_arrival_mono_ns,a_frame_counter,a_exposure_us,"
                     "b_global_ts_ms,b_hw_ts_us,b_sensor_ts_us,b_arrival_mono_ns,b_frame_counter,b_exposure_us,"
                     "publish_mono_ns,publish_real_ns,a_epoch,b_epoch\n");
        for (;;) {
            const uint32_t seen = sig_.value();
            while (q_.available()) {
                const PairCsvItem& it = *q_.peek(0);
                const PairRecord& r = it.rec;
                std::fprintf(f_, "%" PRIu64 ",%.4f", r.pair_seq, r.dt_ms);
                for (int c = 0; c < 2; c++)
                    std::fprintf(f_, ",%.4f,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%d", r.cam[c].global_ts_ms,
                                 r.cam[c].hw_ts_us, r.cam[c].sensor_ts_us, r.cam[c].arrival_mono_ns, r.cam[c].frame_counter,
                                 r.cam[c].exposure_us);
                std::fprintf(f_, ",%" PRId64 ",%" PRId64 ",%" PRIu64 ",%" PRIu64 "\n", r.publish_mono_ns, r.publish_real_ns,
                             it.epoch[0], it.epoch[1]);
                q_.pop();
                lines_++;
            }
            if (stop_.load()) break;
            sig_.wait(seen, 500000000LL);
        }
        std::fflush(f_);
    }
    SpscRing<PairCsvItem> q_;
    EventSignal sig_;
    int cpu_;
    FILE* f_ = nullptr;
    std::thread th_;
    std::atomic<bool> stop_{false};
    uint64_t lines_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    // R8: SIGINT/SIGTERM are blocked in every thread (also the SDK threads created later) and taken by sigtimedwait.
    sigset_t sigs;
    sigemptyset(&sigs);
    sigaddset(&sigs, SIGINT);
    sigaddset(&sigs, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigs, nullptr);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    Config cfg;
    std::string err;
    bool help = false;
    if (!load_config(argc, argv, cfg, err, help)) {
        std::fprintf(stderr, "error: %s\n%s", err.c_str(), usage_text());
        return 2;
    }
    if (help) {
        std::printf("%s", usage_text());
        return 0;
    }

    EventSignal sig;
    FrameProducer prodA(kCamA, sig), prodB(kCamB, sig);
    FrameProducer* prods[2] = {&prodA, &prodB};

    rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
    std::unique_ptr<rs2::context> ctx;
    try {
        ctx.reset(new rs2::context());
    } catch (const rs2::error& e) {
        std::fprintf(stderr, "error: librealsense context: %s\n", e.what());
        return 2;
    }
    Camera camA(kCamA, cfg.serial[0], prodA, cfg.cpu_callback[0]);
    Camera camB(kCamB, cfg.serial[1], prodB, cfg.cpu_callback[1]);
    Camera* cams[2] = {&camA, &camB};
    rs2::device devs[2];
    Camera::find_devices(*ctx, cfg.serial, devs);   // both, before any camera is opened
    for (int i = 0; i < 2; i++) {
        Camera* c = cams[i];
        if (!c->open(devs[i], err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            camA.stop_close();
            camB.stop_close();
            return 3;
        }
        std::printf("camera %s: serial %s fw %s usb %s | %s| depth_scale %.6g fx %.2f fy %.2f\n",
                    c == &camA ? "A" : "B", c->serial().c_str(), c->firmware().c_str(), c->usb().c_str(),
                    c->option_report().c_str(), double(c->info().depth_scale), double(c->info().fx), double(c->info().fy));
    }

    // Sink
    std::unique_ptr<PairSink> sink;
    StatsSink* stats_sink = nullptr;
    ShmSink* shm_sink = nullptr;
    if (cfg.sink == "shm") {
        CameraInfo ci[2] = {camA.info(), camB.info()};
        std::string note;
        auto s = ShmSink::create(cfg.shm_name, ci, kImageBytes, err, &note);
        if (!s) {
            std::fprintf(stderr, "error: shared memory %s: %s\n", cfg.shm_name.c_str(), err.c_str());
            camA.stop_close();
            camB.stop_close();
            return 4;
        }
        if (!note.empty()) std::printf("note: %s\n", note.c_str());
        std::printf("shared memory %s: %zu bytes, %zu pair slots\n", cfg.shm_name.c_str(), shm_total_size(kImageBytes, kShmSlots), kShmSlots);
        shm_sink = s.get();
        sink = std::move(s);
    } else {
        auto s = std::unique_ptr<StatsSink>(new StatsSink());
        stats_sink = s.get();
        sink = std::move(s);
    }

    SdkGlobalTimestampSource tsrc;
    Matcher matcher(prodA, prodB, tsrc, *sink, sig);
    if (cfg.cpu_matcher >= 0) matcher.set_affinity(cfg.cpu_matcher);
    std::unique_ptr<CsvWriter> csv;
    if (!cfg.pair_csv.empty()) {
        csv.reset(new CsvWriter(cfg.pair_csv, cfg.cpu_csv));
        if (!csv->ok()) {
            std::fprintf(stderr, "error: cannot write %s\n", cfg.pair_csv.c_str());
            return 4;
        }
        matcher.set_csv_queue(&csv->queue(), &csv->signal());
        csv->start();
    }
    matcher.start();

    const int64_t start_ns = mono_ns();
    for (Camera* c : cams) {
        if (!c->start(err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            camA.stop_close();
            camB.stop_close();
            matcher.stop();
            if (csv) csv->stop();
            if (shm_sink) shm_sink->close();
            return 5;
        }
    }
    if (cfg.test_ignore_cam >= 0) {
        const int64_t from = start_ns + int64_t(cfg.test_ignore_at_s * 1e9);
        prods[cfg.test_ignore_cam]->set_ignore_window(from, from + int64_t(cfg.test_ignore_for_s * 1e9));
        std::printf("test: callback %s ignores frames from %.1f s to %.1f s after start\n", cam_name(cfg.test_ignore_cam),
                    cfg.test_ignore_at_s, cfg.test_ignore_at_s + cfg.test_ignore_for_s);
    }
    Supervisor sup(cfg, *ctx, cams, prods, matcher, start_ns);
    sup.start();
    std::printf("running: sink=%s timestamp_source=%s%s\n", sink->name(), tsrc.name(),
                cfg.duration_s > 0 ? (" duration=" + std::to_string(cfg.duration_s) + " s").c_str() : " (stop with Ctrl-C)");

    // Wait for SIGINT/SIGTERM or the run duration.
    int got = 0;
    for (;;) {
        timespec ts{0, 200000000};
        const int s = sigtimedwait(&sigs, nullptr, &ts);
        if (s == SIGINT || s == SIGTERM) {
            got = s;
            break;
        }
        if (cfg.duration_s > 0 && double(mono_ns() - start_ns) / 1e9 >= cfg.duration_s) break;
    }
    const int64_t end_ns = mono_ns();
    std::printf("stopping (%s)\n", got == SIGINT ? "SIGINT" : got == SIGTERM ? "SIGTERM" : "duration reached");

    // Shutdown order (R8): no more restarts -> both callbacks stop taking frames at the same instant -> matcher (frames
    // still waiting in it stay pending, they are not counted unpaired) -> streams -> CSV -> sink.
    sup.stop();
    prodA.set_stopping();
    prodB.set_stopping();
    matcher.stop();
    camA.stop_close();
    camB.stop_close();
    if (csv) csv->stop();
    const uint64_t sink_pairs = stats_sink ? stats_sink->pairs() : matcher.pairs();
    sup.final_report(end_ns, sink_pairs, sink->name());
    if (csv) std::printf("csv: %" PRIu64 " lines written to %s\n", csv->lines(), cfg.pair_csv.c_str());
    if (shm_sink) {
        shm_sink->close();
        std::printf("shared memory %s removed\n", cfg.shm_name.c_str());
    }
    return 0;
}
