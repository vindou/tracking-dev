// rs_probe: measurement tool for the RealSense D435i investigation.
//
//   rs_probe inventory                      -> device info, all options (range/default/current), all profiles
//   rs_probe capture [options]              -> per-frame CSV from one or more cameras via sensor callbacks
//
// capture options:
//   --cam NAME:SERIAL      camera to use (repeatable). NAME is used in logs (A, B).
//   --w W --h H --fps F    depth profile (Z16)
//   --seconds S            run duration
//   --global-time 0|1      RS2_OPTION_GLOBAL_TIME_ENABLED (default: leave SDK default)
//   --exposure auto|US     depth auto-exposure, or manual exposure in microseconds
//   --sync NAME:MODE       inter_cam_sync_mode per camera (repeatable)
//   --cb log|memcpy|keep   callback type (exp 8). Default log.
//   --out FILE.csv         per-frame CSV
//   --ir 0|1               also stream IR1+IR2 Y8 at the same size/fps (bandwidth proxy for a 2nd camera)
//
// The callback stores one fixed-size record per frame in a preallocated array (no I/O, no allocation).
// The CSV is written after streaming stops, so file I/O does not perturb timing.
#include <librealsense2/rs.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static inline int64_t now_ns(clockid_t c) {
    timespec t; clock_gettime(c, &t);
    return int64_t(t.tv_sec) * 1000000000LL + t.tv_nsec;
}

// ---------------------------------------------------------------- inventory
static void print_options(rs2::sensor s) {
    for (auto opt : s.get_supported_options()) {
        const char* name = rs2_option_to_string(opt);
        try {
            auto r = s.get_option_range(opt);
            float cur = 0; bool ro = s.is_option_read_only(opt);
            try { cur = s.get_option(opt); } catch (...) { cur = NAN; }
            printf("    option %-40s min=%g max=%g step=%g default=%g current=%g%s\n",
                   name, r.min, r.max, r.step, r.def, cur, ro ? " (read-only)" : "");
        } catch (const rs2::error& e) {
            printf("    option %-40s <error: %s>\n", name, e.what());
        }
    }
}

static int cmd_inventory() {
    rs2::context ctx;
    auto devs = ctx.query_devices();
    printf("librealsense API version: %d.%d.%d\n", RS2_API_MAJOR_VERSION, RS2_API_MINOR_VERSION, RS2_API_PATCH_VERSION);
    printf("device count: %u\n", devs.size());
    for (auto&& d : devs) {
        printf("=== DEVICE\n");
        for (int i = 0; i < RS2_CAMERA_INFO_COUNT; i++) {
            auto info = rs2_camera_info(i);
            if (d.supports(info)) printf("  info %-32s %s\n", rs2_camera_info_to_string(info), d.get_info(info));
        }
        for (auto&& s : d.query_sensors()) {
            printf("  --- SENSOR %s\n", s.get_info(RS2_CAMERA_INFO_NAME));
            print_options(s);
            for (auto&& p : s.get_stream_profiles()) {
                if (auto vp = p.as<rs2::video_stream_profile>())
                    printf("    profile %s idx=%d %dx%d %s %dfps%s\n", rs2_stream_to_string(vp.stream_type()),
                           vp.stream_index(), vp.width(), vp.height(), rs2_format_to_string(vp.format()), vp.fps(),
                           vp.is_default() ? " (default)" : "");
                else
                    printf("    profile %s idx=%d %s %dfps\n", rs2_stream_to_string(p.stream_type()), p.stream_index(),
                           rs2_format_to_string(p.format()), p.fps());
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------- capture
struct Rec {
    int64_t mono_ns, real_ns, cb_ns;   // arrival (callback entry) and callback duration
    double ts_ms;                      // frame.get_timestamp()
    int64_t md[RS2_FRAME_METADATA_COUNT];
    uint64_t md_mask;                  // bit i set if metadata i supported on this frame
    uint64_t frame_number;
    int domain;
    int push_ok;                       // keep-mode: 1 if handle pushed, 0 if queue full; -1 otherwise
    int stream;                        // 0 depth, 1 IR1, 2 IR2
    int overlap;                       // other callbacks of this camera running at entry
};

static const rs2_frame_metadata_value MD_FIELDS[] = {
    RS2_FRAME_METADATA_FRAME_COUNTER, RS2_FRAME_METADATA_FRAME_TIMESTAMP, RS2_FRAME_METADATA_SENSOR_TIMESTAMP,
    RS2_FRAME_METADATA_ACTUAL_EXPOSURE, RS2_FRAME_METADATA_ACTUAL_FPS, RS2_FRAME_METADATA_TIME_OF_ARRIVAL,
    RS2_FRAME_METADATA_BACKEND_TIMESTAMP, RS2_FRAME_METADATA_AUTO_EXPOSURE, RS2_FRAME_METADATA_GAIN_LEVEL};

// Single-producer single-consumer ring of frame handles (exp 8c).
struct SpscFrames {
    static constexpr size_t N = 16;
    rs2::frame slot[N];
    std::atomic<size_t> head{0}, tail{0};
    bool push(rs2::frame&& f) {
        size_t h = head.load(std::memory_order_relaxed);
        if (h - tail.load(std::memory_order_acquire) == N) return false;
        slot[h % N] = std::move(f);
        head.store(h + 1, std::memory_order_release);
        return true;
    }
    bool pop(rs2::frame& out) {
        size_t t = tail.load(std::memory_order_relaxed);
        if (t == head.load(std::memory_order_acquire)) return false;
        out = std::move(slot[t % N]);
        tail.store(t + 1, std::memory_order_release);
        return true;
    }
};

struct Cam {
    std::string name, serial;
    rs2::device dev;
    rs2::sensor depth;
    std::vector<Rec> recs;
    std::atomic<size_t> n{0};
    std::atomic<uint64_t> overflow{0};
    std::vector<uint8_t> copybuf;      // memcpy target (preallocated)
    SpscFrames q;
    std::thread consumer;
    std::atomic<bool> run{true};
    std::atomic<uint64_t> consumed{0};
    std::atomic<int> in_cb{0};
    std::atomic<uint64_t> overlaps{0};
};

static void usage() {
    fprintf(stderr, "usage: rs_probe inventory | rs_probe capture --cam A:SERIAL [--cam B:SERIAL] --w 848 --h 480 --fps 90 "
                    "--seconds 60 [--global-time 0|1] [--exposure auto|US] [--sync A:1] [--cb log|memcpy|keep] [--ir 0|1] --out f.csv\n");
}

static int cmd_capture(int argc, char** argv) {
    std::vector<std::pair<std::string, std::string>> cams;
    std::map<std::string, float> sync;
    int W = 848, H = 480, FPS = 90, gt = -1, ir = 0;
    double secs = 10;
    std::string exposure, cb = "log", out;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
        if (a == "--cam") { auto v = next(); auto c = v.find(':'); cams.push_back({v.substr(0, c), v.substr(c + 1)}); }
        else if (a == "--w") W = std::stoi(next());
        else if (a == "--h") H = std::stoi(next());
        else if (a == "--fps") FPS = std::stoi(next());
        else if (a == "--seconds") secs = std::stod(next());
        else if (a == "--global-time") gt = std::stoi(next());
        else if (a == "--exposure") exposure = next();
        else if (a == "--sync") { auto v = next(); auto c = v.find(':'); sync[v.substr(0, c)] = std::stof(v.substr(c + 1)); }
        else if (a == "--cb") cb = next();
        else if (a == "--out") out = next();
        else if (a == "--ir") ir = std::stoi(next());
        else { usage(); return 2; }
    }
    if (cams.empty() || out.empty()) { usage(); return 2; }

    rs2::context ctx;
    std::vector<std::unique_ptr<Cam>> cs;
    for (auto& [name, serial] : cams) {
        auto c = std::make_unique<Cam>();
        c->name = name; c->serial = serial;
        for (auto&& d : ctx.query_devices())
            if (serial == d.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) c->dev = d;
        if (!c->dev) { fprintf(stderr, "camera %s serial %s not found\n", name.c_str(), serial.c_str()); return 1; }
        c->depth = c->dev.first<rs2::depth_sensor>();
        size_t cap = (size_t(FPS * secs * 1.3) + 2000) * (ir ? 3 : 1);
        c->recs.resize(cap);
        c->copybuf.resize(size_t(W) * H * 2 * 4);
        cs.push_back(std::move(c));
    }

    // Configure options; print what was requested and what the device reports back.
    for (auto& c : cs) {
        auto& s = c->depth;
        auto setopt = [&](rs2_option o, float v) {
            try { s.set_option(o, v); printf("[%s] set %s=%g -> readback %g\n", c->name.c_str(), rs2_option_to_string(o), v, s.get_option(o)); }
            catch (const rs2::error& e) { printf("[%s] set %s=%g FAILED: %s\n", c->name.c_str(), rs2_option_to_string(o), v, e.what()); }
        };
        if (gt >= 0) setopt(RS2_OPTION_GLOBAL_TIME_ENABLED, float(gt));
        if (exposure == "auto") setopt(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
        else if (!exposure.empty()) { setopt(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0); setopt(RS2_OPTION_EXPOSURE, std::stof(exposure)); }
        if (sync.count(c->name)) setopt(RS2_OPTION_INTER_CAM_SYNC_MODE, sync[c->name]);
        for (auto o : {RS2_OPTION_GLOBAL_TIME_ENABLED, RS2_OPTION_ENABLE_AUTO_EXPOSURE, RS2_OPTION_EXPOSURE,
                       RS2_OPTION_INTER_CAM_SYNC_MODE, RS2_OPTION_FRAMES_QUEUE_SIZE})
            if (s.supports(o)) printf("[%s] option %s = %g\n", c->name.c_str(), rs2_option_to_string(o), s.get_option(o));
        printf("[%s] serial=%s fw=%s usb=%s port=%s\n", c->name.c_str(), c->serial.c_str(),
               c->dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION),
               c->dev.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) ? c->dev.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) : "?",
               c->dev.get_info(RS2_CAMERA_INFO_PHYSICAL_PORT));

        rs2::stream_profile prof;
        for (auto&& p : s.get_stream_profiles()) {
            auto vp = p.as<rs2::video_stream_profile>();
            if (vp && vp.stream_type() == RS2_STREAM_DEPTH && vp.format() == RS2_FORMAT_Z16 &&
                vp.width() == W && vp.height() == H && vp.fps() == FPS) { prof = p; break; }
        }
        if (!prof) { fprintf(stderr, "[%s] profile %dx%d Z16 %d fps NOT AVAILABLE\n", c->name.c_str(), W, H, FPS); return 3; }
        std::vector<rs2::stream_profile> profs{prof};
        for (int idx = 1; ir && idx <= 2; idx++) {
            rs2::stream_profile q;
            for (auto&& p : s.get_stream_profiles()) {
                auto vp = p.as<rs2::video_stream_profile>();
                if (vp && vp.stream_type() == RS2_STREAM_INFRARED && vp.stream_index() == idx && vp.format() == RS2_FORMAT_Y8 &&
                    vp.width() == W && vp.height() == H && vp.fps() == FPS) { q = p; break; }
            }
            if (!q) { fprintf(stderr, "[%s] IR%d %dx%d Y8 %d fps NOT AVAILABLE\n", c->name.c_str(), idx, W, H, FPS); return 3; }
            profs.push_back(q);
        }
        printf("[%s] opening %zu stream(s)\n", c->name.c_str(), profs.size());
        s.open(profs);
    }

    // Start all cameras back-to-back.
    for (auto& c : cs) {
        Cam* cp = c.get();
        if (cb == "keep") {
            cp->consumer = std::thread([cp] {
                rs2::frame f;
                while (cp->run.load(std::memory_order_relaxed)) {
                    if (cp->q.pop(f)) { f = rs2::frame(); cp->consumed++; }
                    else std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
                while (cp->q.pop(f)) { f = rs2::frame(); cp->consumed++; }
            });
        }
        int mode = cb == "memcpy" ? 1 : cb == "keep" ? 2 : 0;
        size_t slot_bytes = size_t(W) * H * 2;
        cp->depth.start([cp, mode, slot_bytes](rs2::frame f) {
            int64_t t0 = now_ns(CLOCK_MONOTONIC);
            int ov = cp->in_cb.fetch_add(1, std::memory_order_acq_rel);
            if (ov) cp->overlaps++;
            int64_t tr = now_ns(CLOCK_REALTIME);
            size_t i = cp->n.fetch_add(1, std::memory_order_relaxed);
            if (i >= cp->recs.size()) { cp->overflow++; cp->in_cb--; return; }
            Rec& r = cp->recs[i];
            r.mono_ns = t0; r.real_ns = tr;
            r.ts_ms = f.get_timestamp();
            r.domain = int(f.get_frame_timestamp_domain());
            r.frame_number = f.get_frame_number();
            r.md_mask = 0;
            for (auto m : MD_FIELDS)
                if (f.supports_frame_metadata(m)) { r.md_mask |= 1ULL << m; r.md[m] = f.get_frame_metadata(m); }
            r.push_ok = -1;
            r.overlap = ov;
            auto sp = f.get_profile();
            r.stream = sp.stream_type() == RS2_STREAM_INFRARED ? sp.stream_index() : 0;
            if (mode == 1) {
                auto vf = f.as<rs2::video_frame>();
                size_t sz = std::min<size_t>(vf.get_data_size(), slot_bytes);
                memcpy(cp->copybuf.data() + size_t(r.stream) * slot_bytes, vf.get_data(), sz);
            } else if (mode == 2) {
                f.keep();
                r.push_ok = cp->q.push(std::move(f)) ? 1 : 0;
            }
            r.cb_ns = now_ns(CLOCK_MONOTONIC) - t0;
            cp->in_cb.fetch_sub(1, std::memory_order_acq_rel);
        });
        printf("[%s] started at mono_ns=%lld\n", c->name.c_str(), (long long)now_ns(CLOCK_MONOTONIC));
    }
    fflush(stdout);
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    for (auto& c : cs) { c->depth.stop(); }
    for (auto& c : cs) {
        c->depth.close();
        c->run = false;
        if (c->consumer.joinable()) c->consumer.join();
    }

    FILE* fo = fopen(out.c_str(), "w");
    if (!fo) { perror(out.c_str()); return 1; }
    fprintf(fo, "cam,idx,frame_number,ts_ms,domain,arrival_mono_ns,arrival_real_ns,cb_ns,push_ok,overlap");
    for (auto m : MD_FIELDS) fprintf(fo, ",md_%s", rs2_frame_metadata_to_string(m));
    fprintf(fo, "\n");
    for (auto& c : cs) {
        size_t n = std::min(c->n.load(), c->recs.size());
        for (size_t i = 0; i < n; i++) {
            auto& r = c->recs[i];
            static const char* SUF[] = {"", "_ir1", "_ir2"};
            fprintf(fo, "%s%s,%zu,%llu,%.6f,%s,%lld,%lld,%lld,%d,%d", c->name.c_str(), SUF[r.stream], i,
                    (unsigned long long)r.frame_number, r.ts_ms, rs2_timestamp_domain_to_string(rs2_timestamp_domain(r.domain)),
                    (long long)r.mono_ns, (long long)r.real_ns, (long long)r.cb_ns, r.push_ok, r.overlap);
            for (auto m : MD_FIELDS) {
                if (r.md_mask & (1ULL << m)) fprintf(fo, ",%lld", (long long)r.md[m]);
                else fprintf(fo, ",");
            }
            fprintf(fo, "\n");
        }
        printf("[%s] frames=%zu overflow=%llu consumed=%llu callback_overlaps=%llu\n", c->name.c_str(), n,
               (unsigned long long)c->overflow.load(), (unsigned long long)c->consumed.load(),
               (unsigned long long)c->overlaps.load());
    }
    fclose(fo);
    for (auto& c : cs)
        if (c->n.load() == 0) { fprintf(stderr, "[%s] STALL: 0 frames received\n", c->name.c_str()); return 4; }
    return 0;
}

int main(int argc, char** argv) try {
    if (argc < 2) { usage(); return 2; }
    std::string cmd = argv[1];
    if (cmd == "inventory") return cmd_inventory();
    if (cmd == "capture") return cmd_capture(argc, argv);
    usage();
    return 2;
} catch (const rs2::error& e) {
    fprintf(stderr, "rs2 error in %s(%s): %s\n", e.get_failed_function().c_str(), e.get_failed_args().c_str(), e.what());
    return 1;
} catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
