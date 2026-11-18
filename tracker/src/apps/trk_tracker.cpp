// trk_tracker: pose tracking of the test object from the capture pair buffer.
//
//   trk_tracker [--config config/tracker.json] [--duration S] [--final-json FILE] [--quiet]
//               [--initial-pose x,y,z,qw,qx,qy,qz]   (test option: start in track state at this pose)
//
// Threads: loop (wait for the pair notification, newest pair, stages 1-6, publish pose every cycle and cloud at
// cloud_rate_hz), status (one line every status_interval_s), two heartbeat threads (pose and cloud buffers), main
// (setup, SIGINT/SIGTERM, ordered stop). All loop memory is allocated before the loop starts; the loop does no file
// output and no network I/O.
#include <atomic>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <unistd.h>

#include "common/clock.hpp"
#include "core/argutil.hpp"
#include "core/config.hpp"
#include "core/pair_source.hpp"
#include "core/tracker_core.hpp"
#include "histogram.hpp"   // capture: AtomicHistogram (lock-free)
#include "records/records.hpp"
#include "shmring/shm_ring.hpp"

using namespace trk;

namespace {

std::atomic<bool> g_stop{false};

struct Stats {
    std::atomic<uint64_t> cycles{0}, valid{0}, state_count[3] = {{0}, {0}, {0}}, reattaches{0}, clouds{0};
    std::atomic<uint32_t> state{kStateSearch};
    std::atomic<uint64_t> last_pose_seq{0};
    std::atomic<double> last_rmse{0}, last_inlier{0};
    std::atomic<uint32_t> last_points{0};
    cap::AtomicHistogram total{0.0, 0.01, 3000};    // ms, 0-30 ms, 10 us bins (stages 1-6 + publish)
    cap::AtomicHistogram solve{0.0, 0.01, 3000};
    cap::AtomicHistogram read{0.0, 0.01, 3000};
    cap::AtomicHistogram delay{0.0, 0.05, 2000};    // capture -> publish, ms, 0-100 ms
    std::atomic<uint64_t> pairs_skipped{0}, pairs_torn{0}, pairs_used{0};
    std::atomic<uint64_t> capture_published_start{0}, capture_published_now{0};
};

long rss_kb() {
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long size = 0, rss = 0;
    const int n = std::fscanf(f, "%ld %ld", &size, &rss);
    std::fclose(f);
    return n == 2 ? rss * (sysconf(_SC_PAGESIZE) / 1024) : 0;
}

const char* state_name(uint32_t s) { return s == kStateTrack ? "track" : s == kStateLost ? "lost" : "search"; }

}  // namespace

int main(int argc, char** argv) {
    sigset_t sigs;
    sigemptyset(&sigs);
    sigaddset(&sigs, SIGINT);
    sigaddset(&sigs, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigs, nullptr);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string cfg_path = "config/tracker.json", final_json;
    double duration = 0;
    bool quiet = false;
    std::vector<double> init_pose;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) cfg_path = argv[++i];
        else if (a == "--duration" && i + 1 < argc) duration = std::atof(argv[++i]);
        else if (a == "--final-json" && i + 1 < argc) final_json = argv[++i];
        else if (a == "--quiet") quiet = true;
        else if (a == "--initial-pose" && i + 1 < argc) init_pose = parse_list(argv[++i]);
        else { std::fprintf(stderr, "usage: %s [--config FILE] [--duration S] [--final-json FILE] [--quiet] [--initial-pose x,y,z,qw,qx,qy,qz]\n", argv[0]); return 2; }
    }
    TrackerConfig cfg;
    std::string err;
    if (!load_tracker_config(cfg_path, cfg, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 2; }
    std::unique_ptr<TrackerCore> core(new TrackerCore());
    if (!core->init(cfg, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 3; }
    if (init_pose.size() == 7) {
        const double q[4] = {init_pose[3], init_pose[4], init_pose[5], init_pose[6]};
        core->force_track(make_pose(R_from_quat(q), Vec3(init_pose[0], init_pose[1], init_pose[2])));
        std::printf("test option: start in track state at the given pose\n");
    }

    std::string note;
    auto pose_w = RingWriter::create(cfg.pose_buffer, kPoseRecordType, kPoseRecordVersion, sizeof(PoseRecord), kPoseSlots, err, &note);
    if (!pose_w) { std::fprintf(stderr, "error: pose buffer %s: %s\n", cfg.pose_buffer.c_str(), err.c_str()); return 4; }
    if (!note.empty()) std::printf("note: %s\n", note.c_str());
    note.clear();
    auto cloud_w = RingWriter::create(cfg.cloud_buffer, kCloudRecordType, kCloudRecordVersion, sizeof(CloudRecord), kCloudSlots, err, &note);
    if (!cloud_w) { std::fprintf(stderr, "error: cloud buffer %s: %s\n", cfg.cloud_buffer.c_str(), err.c_str()); return 4; }
    if (!note.empty()) std::printf("note: %s\n", note.c_str());
    std::printf("tracker: instance %016" PRIx64 ", pose buffer %s (%u slots), cloud buffer %s (%u slots), config %s\n",
                pose_w->instance_id(), cfg.pose_buffer.c_str(), kPoseSlots, cfg.cloud_buffer.c_str(), kCloudSlots, cfg_path.c_str());

    // preallocated loop memory
    std::vector<uint16_t> imgA(std::size_t(cap::kWidth) * cap::kHeight), imgB(imgA.size());
    uint16_t* img[2] = {imgA.data(), imgB.data()};
    std::unique_ptr<PoseRecord> pose(new PoseRecord());
    std::unique_ptr<CloudRecord> cloud(new CloudRecord());
    std::memset(cloud.get(), 0, sizeof(CloudRecord));
    Stats st;
    PairSource src;
    const int64_t t_start = mono_ns();

    std::thread loop([&] {
        pthread_setname_np(pthread_self(), "trk-loop");
        bool attached = false;
        int64_t last_cloud_ns = 0;
        const int64_t cloud_period = int64_t(1e9 / std::max(0.1, cfg.cloud_rate_hz));
        uint64_t pose_seq = 0;
        float publish_us = 0;
        while (!g_stop.load(std::memory_order_relaxed)) {
            if (!attached) {
                std::string e;
                attached = src.attached() ? src.reattach(e) : src.attach(cfg.pair_buffer, e);
                if (!attached) {   // no capture writer: retry every 100 ms (only while there is none)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                st.reattaches.fetch_add(1);
                cap::CameraInfo ci[2] = {src.header().cam[0], src.header().cam[1]};
                std::string n;
                if (!core->set_cameras(ci, e, &n)) {
                    std::printf("error: %s\n", e.c_str());
                    g_stop.store(true);
                    break;
                }
                st.capture_published_start.store(src.header().pairs_published.load());
                std::printf("[attach +%.1fs] pair buffer %s writer instance %016" PRIx64 ": %s\n",
                            double(mono_ns() - t_start) / 1e9, cfg.pair_buffer.c_str(), src.instance_id(), n.c_str());
            }
            CropRect crops[2];
            core->crops(crops);
            cap::PairRecord rec;
            int64_t copy_ns = 0;
            const PairSource::Status s = src.read_newest(crops, rec, img, 200000000, copy_ns);
            if (s == PairSource::Status::WriterGone) {
                std::printf("[detach +%.1fs] pair buffer writer %s\n", double(mono_ns() - t_start) / 1e9, src.gone_reason());
                attached = false;
                continue;
            }
            if (s == PairSource::Status::Timeout) continue;
            const uint16_t* cimg[2] = {img[0], img[1]};
            const int64_t now = mono_ns();
            const bool want_cloud = now - last_cloud_ns >= cloud_period;
            core->process(rec, cimg, *pose, want_cloud ? cloud.get() : nullptr);
            const int64_t t_pub = mono_ns();
            pose->pose_seq = pose_seq++;
            pose->writer_instance_id = pose_w->instance_id();
            pose->capture_instance_id = src.instance_id();
            pose->pairs_skipped = src.skipped() + src.torn();
            pose->stage_us[kStageRead] = float(double(copy_ns) / 1000.0);
            pose->stage_us[kStagePublish] = publish_us;
            pose->stage_us[kStageTotal] += pose->stage_us[kStageRead];
            pose->publish_time_mono_ns = mono_ns();
            pose_w->publish(pose.get());
            if (want_cloud) {
                cloud->pose_seq = pose->pose_seq;
                cloud_w->publish(cloud.get());
                last_cloud_ns = now;
                st.clouds.fetch_add(1, std::memory_order_relaxed);
            }
            publish_us = float(double(mono_ns() - t_pub) / 1000.0);
            // statistics (atomics only)
            st.cycles.fetch_add(1, std::memory_order_relaxed);
            if (pose->valid) st.valid.fetch_add(1, std::memory_order_relaxed);
            st.state_count[pose->track_state].fetch_add(1, std::memory_order_relaxed);
            st.state.store(pose->track_state, std::memory_order_relaxed);
            st.last_pose_seq.store(pose->pose_seq, std::memory_order_relaxed);
            st.last_rmse.store(pose->fit_rmse, std::memory_order_relaxed);
            st.last_inlier.store(pose->inlier_ratio, std::memory_order_relaxed);
            st.last_points.store(pose->point_count, std::memory_order_relaxed);
            st.total.add(double(pose->stage_us[kStageTotal] + publish_us) / 1000.0);
            st.solve.add(double(pose->stage_us[kStageSolve]) / 1000.0);
            st.read.add(double(pose->stage_us[kStageRead]) / 1000.0);
            st.delay.add(double(pose->publish_time_mono_ns - pose->meas_time_mono_ns) / 1e6);
            st.pairs_skipped.store(src.skipped(), std::memory_order_relaxed);
            st.pairs_torn.store(src.torn(), std::memory_order_relaxed);
            st.pairs_used.store(src.used(), std::memory_order_relaxed);
            st.capture_published_now.store(src.header().pairs_published.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
    });

    // status thread
    std::thread status([&] {
        pthread_setname_np(pthread_self(), "trk-status");
        std::vector<uint64_t> tot_prev, sol_prev, rd_prev, dl_prev;
        st.total.snapshot(tot_prev);
        st.solve.snapshot(sol_prev);
        st.read.snapshot(rd_prev);
        st.delay.snapshot(dl_prev);
        uint64_t cyc_prev = 0, valid_prev = 0;
        int64_t t_prev = mono_ns();
        int64_t next = t_prev + int64_t(cfg.status_interval_s * 1e9);
        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const int64_t now = mono_ns();
            if (now < next) continue;
            next += int64_t(cfg.status_interval_s * 1e9);
            std::vector<uint64_t> tot, sol, rd, dl;
            st.total.snapshot(tot);
            st.solve.snapshot(sol);
            st.read.snapshot(rd);
            st.delay.snapshot(dl);
            const auto dt = cap::AtomicHistogram::diff(tot, tot_prev), ds = cap::AtomicHistogram::diff(sol, sol_prev),
                       dr = cap::AtomicHistogram::diff(rd, rd_prev), dd = cap::AtomicHistogram::diff(dl, dl_prev);
            tot_prev = tot; sol_prev = sol; rd_prev = rd; dl_prev = dl;
            const uint64_t cyc = st.cycles.load(), val = st.valid.load();
            const double secs = double(now - t_prev) / 1e9;
            t_prev = now;
            if (!quiet)
                std::printf("[status +%.0fs] state=%s rate=%.1fHz valid=%.1f%% pts=%u rmse=%.2fmm inl=%.3f | cycle p50/p99=%.2f/%.2fms "
                            "solve p50/p99=%.2f/%.2fms read p99=%.2fms delay p50/p99=%.1f/%.1fms | pairs used=%" PRIu64
                            " skipped=%" PRIu64 " torn=%" PRIu64 " attaches=%" PRIu64 " rss=%ldkB\n",
                            double(now - t_start) / 1e9, state_name(st.state.load()), double(cyc - cyc_prev) / secs,
                            cyc > cyc_prev ? 100.0 * double(val - valid_prev) / double(cyc - cyc_prev) : 0.0,
                            st.last_points.load(), st.last_rmse.load() * 1000, st.last_inlier.load(),
                            st.total.quantile(dt, 0.5), st.total.quantile(dt, 0.99), st.solve.quantile(ds, 0.5),
                            st.solve.quantile(ds, 0.99), st.read.quantile(dr, 0.99), st.delay.quantile(dd, 0.5),
                            st.delay.quantile(dd, 0.99), st.pairs_used.load(), st.pairs_skipped.load(), st.pairs_torn.load(),
                            st.reattaches.load(), rss_kb());
            cyc_prev = cyc;
            valid_prev = val;
        }
    });

    std::printf("running%s\n", duration > 0 ? (" for " + std::to_string(duration) + " s").c_str() : " (stop with Ctrl-C)");
    int got = 0;
    for (;;) {
        timespec ts{0, 100000000};
        const int s = sigtimedwait(&sigs, nullptr, &ts);
        if (s == SIGINT || s == SIGTERM) { got = s; break; }
        if (g_stop.load()) break;
        if (duration > 0 && double(mono_ns() - t_start) / 1e9 >= duration) break;
    }
    g_stop.store(true);
    loop.join();
    status.join();
    const double run_s = double(mono_ns() - t_start) / 1e9;
    std::vector<uint64_t> tot, sol, rd, dl;
    st.total.snapshot(tot);
    st.solve.snapshot(sol);
    st.read.snapshot(rd);
    st.delay.snapshot(dl);
    const uint64_t cap_pub = st.capture_published_now.load() - st.capture_published_start.load();
    std::printf("stopping (%s) after %.1f s\n", got == SIGINT ? "SIGINT" : got == SIGTERM ? "SIGTERM" : "duration", run_s);
    std::printf("final: cycles=%" PRIu64 " valid=%" PRIu64 " search/track/lost=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                " pairs used=%" PRIu64 " skipped=%" PRIu64 " torn=%" PRIu64 " capture published (since attach)=%" PRIu64
                " attaches=%" PRIu64 " clouds=%" PRIu64 "\n"
                "       cycle p50/p99/max=%.3f/%.3f/%.3f ms <=2ms %.2f%% <=5ms %.2f%%; solve p50/p99=%.3f/%.3f ms; read p50/p99=%.3f/%.3f ms;"
                " delay p50/p99=%.2f/%.2f ms; rss=%ld kB\n",
                st.cycles.load(), st.valid.load(), st.state_count[0].load(), st.state_count[1].load(), st.state_count[2].load(),
                st.pairs_used.load(), st.pairs_skipped.load(), st.pairs_torn.load(), cap_pub, st.reattaches.load(),
                st.clouds.load(), st.total.quantile(tot, 0.5), st.total.quantile(tot, 0.99), st.total.max_value(tot),
                [&] { uint64_t n = 0, a = 0; for (std::size_t i = 0; i < tot.size(); i++) { a += tot[i]; if (i <= 200) n += tot[i]; } return a ? 100.0 * double(n) / double(a) : 0.0; }(),
                [&] { uint64_t n = 0, a = 0; for (std::size_t i = 0; i < tot.size(); i++) { a += tot[i]; if (i <= 500) n += tot[i]; } return a ? 100.0 * double(n) / double(a) : 0.0; }(),
                st.solve.quantile(sol, 0.5), st.solve.quantile(sol, 0.99), st.read.quantile(rd, 0.5), st.read.quantile(rd, 0.99),
                st.delay.quantile(dl, 0.5), st.delay.quantile(dl, 0.99), rss_kb());
    if (!final_json.empty()) {
        if (FILE* f = std::fopen(final_json.c_str(), "w")) {
            uint64_t le2 = 0, le5 = 0, all = 0;
            for (std::size_t i = 0; i < tot.size(); i++) { all += tot[i]; if (i <= 200) le2 += tot[i]; if (i <= 500) le5 += tot[i]; }
            std::fprintf(f, "{\"run_s\": %.3f, \"cycles\": %" PRIu64 ", \"valid\": %" PRIu64 ", \"search\": %" PRIu64 ", \"track\": %" PRIu64
                         ", \"lost\": %" PRIu64 ", \"pairs_used\": %" PRIu64 ", \"pairs_skipped\": %" PRIu64 ", \"pairs_torn\": %" PRIu64
                         ", \"capture_published\": %" PRIu64 ", \"attaches\": %" PRIu64 ", \"clouds\": %" PRIu64
                         ", \"cycle_p50_ms\": %.4f, \"cycle_p99_ms\": %.4f, \"cycle_max_ms\": %.4f, \"cycle_le_2ms_pct\": %.3f"
                         ", \"cycle_le_5ms_pct\": %.3f, \"solve_p50_ms\": %.4f, \"solve_p99_ms\": %.4f, \"read_p99_ms\": %.4f"
                         ", \"delay_p50_ms\": %.3f, \"delay_p99_ms\": %.3f, \"rss_kb\": %ld, \"pose_instance\": \"%016" PRIx64 "\"}\n",
                         run_s, st.cycles.load(), st.valid.load(), st.state_count[0].load(), st.state_count[1].load(),
                         st.state_count[2].load(), st.pairs_used.load(), st.pairs_skipped.load(), st.pairs_torn.load(), cap_pub,
                         st.reattaches.load(), st.clouds.load(), st.total.quantile(tot, 0.5), st.total.quantile(tot, 0.99),
                         st.total.max_value(tot), all ? 100.0 * double(le2) / double(all) : 0.0, all ? 100.0 * double(le5) / double(all) : 0.0,
                         st.solve.quantile(sol, 0.5), st.solve.quantile(sol, 0.99), st.read.quantile(rd, 0.99),
                         st.delay.quantile(dl, 0.5), st.delay.quantile(dl, 0.99), rss_kb(), pose_w->instance_id());
            std::fclose(f);
        }
    }
    cloud_w->close();
    pose_w->close();
    std::printf("pose and cloud buffers removed\n");
    return 0;
}
