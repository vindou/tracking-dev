// Test source for the live path with external pose hypotheses (DESIGN.md 13): shows the static scenes of a scenes
// file (format: trk_hint_eval.cpp) one after another in a pair buffer with the capture layout, like trk_replay.
// Each scene: --blank seconds of empty images (the tracker loses the object and goes to search), then the pair of
// the scene for --hold seconds, at the pair period of the cameras. The log has one line per scene with the times of
// the first and the last pair of the scene and the true pose, for flow/live_eval.py.
//   trk_scenes --scenes FILE --log CSV [--first 0] [--step 1] [--count 0 (all)] [--hold 2.0] [--blank 0.7]
//              [--shm-name /d435_pairs] [--start-delay 1.0]
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "common/clock.hpp"
#include "core/synth.hpp"
#include "shm_sink.hpp"

using namespace trk;

namespace { volatile std::sig_atomic_t g_stop = 0; void on_sig(int) { g_stop = 1; } }

int main(int argc, char** argv) {
    std::string scenes, log, name = "/d435_pairs";
    long first = 0, step = 1, count = 0;
    double hold = 2.0, blank = 0.7, start_delay = 1.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--scenes") scenes = argv[i + 1];
        else if (a == "--log") log = argv[i + 1];
        else if (a == "--first") first = std::atol(argv[i + 1]);
        else if (a == "--step") step = std::max(1L, std::atol(argv[i + 1]));
        else if (a == "--count") count = std::atol(argv[i + 1]);
        else if (a == "--hold") hold = std::atof(argv[i + 1]);
        else if (a == "--blank") blank = std::atof(argv[i + 1]);
        else if (a == "--shm-name") name = argv[i + 1];
        else if (a == "--start-delay") start_delay = std::atof(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (scenes.empty() || log.empty()) { std::fprintf(stderr, "usage: trk_scenes --scenes FILE --log CSV [--first N] [--step N] [--count N] [--hold S] [--blank S]\n"); return 2; }
    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);
    Camera cams[2];
    default_cameras(cams);
    cap::CameraInfo ci[2];
    camera_infos(cams, ci);
    FILE* fs = std::fopen(scenes.c_str(), "rb");
    char magic[8];
    uint32_t h[4];
    if (!fs || std::fread(magic, 1, 8, fs) != 8 || std::memcmp(magic, "TRKSCENE", 8) != 0 || std::fread(h, 4, 4, fs) != 4 || h[0] != 1 ||
        int(h[2]) != cams[0].width || int(h[3]) != cams[0].height) {
        std::fprintf(stderr, "%s: not a scenes file (version 1, %d x %d)\n", scenes.c_str(), cams[0].width, cams[0].height);
        return 3;
    }
    const std::size_t n = std::size_t(h[2]) * h[3];
    const long scene_bytes = long(16 * 8 + 4 * n);
    std::vector<uint16_t> a(n), b(n), zero(n, 0);
    FILE* fl = std::fopen(log.c_str(), "w");
    if (!fl) { std::fprintf(stderr, "cannot write %s\n", log.c_str()); return 3; }
    std::fprintf(fl, "scene,first_pair_seq,last_pair_seq,first_capture_mono_ns,last_capture_mono_ns");
    for (int i = 0; i < 16; i++) std::fprintf(fl, ",T%d", i);
    std::fprintf(fl, "\n");
    std::string err, note;
    auto sink = cap::ShmSink::create(name, ci, n * 2, err, &note);
    if (!sink) { std::fprintf(stderr, "shared memory %s: %s\n", name.c_str(), err.c_str()); return 4; }
    if (!note.empty()) std::printf("note: %s\n", note.c_str());
    const double period_ms = 11.1283;
    const int64_t period = int64_t(period_ms * 1e6);
    sink->on_period(period_ms, 1.0);
    timespec ts{time_t(start_delay), long((start_delay - double(time_t(start_delay))) * 1e9)};
    nanosleep(&ts, nullptr);
    const int64_t real_minus_mono = real_ns() - mono_ns();
    int64_t when = mono_ns() + period;      // publish time of the next pair
    uint64_t seq = 0;
    int64_t late_max = 0;
    // one pair: captured 10.6 ms before it is published, arrival 9.6 ms after capture (as the capture program measures)
    auto publish = [&](const uint16_t* ia, const uint16_t* ib, int64_t& cap_mono) {
        timespec t{time_t(when / 1000000000LL), long(when % 1000000000LL)};
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, nullptr);
        cap::PairRecord rec;
        std::memset(&rec, 0, sizeof rec);
        cap_mono = when - 10600000;
        rec.pair_seq = seq;
        for (int c = 0; c < 2; c++) {
            rec.cam[c].global_ts_ms = double(cap_mono + real_minus_mono) / 1e6;
            rec.cam[c].hw_ts_us = cap_mono / 1000;
            rec.cam[c].sensor_ts_us = cap_mono / 1000;
            rec.cam[c].arrival_mono_ns = cap_mono + 9600000;
            rec.cam[c].frame_counter = seq;
            rec.cam[c].exposure_us = 1000;
            rec.cam[c].domain = 2;
        }
        rec.publish_mono_ns = when;
        rec.publish_real_ns = when + real_minus_mono;
        late_max = std::max(late_max, mono_ns() - when);
        sink->on_pair(rec, ia, ib);
        when += period;
        return seq++;
    };
    long shown = 0;
    for (long s = first; s < long(h[1]) && !g_stop && (count <= 0 || shown < count); s += step, shown++) {
        double truth[16];
        if (std::fseek(fs, 24 + s * scene_bytes, SEEK_SET) != 0 || std::fread(truth, 8, 16, fs) != 16 ||
            std::fread(a.data(), 2, n, fs) != n || std::fread(b.data(), 2, n, fs) != n) {
            std::fprintf(stderr, "%s: short file at scene %ld\n", scenes.c_str(), s);
            break;
        }
        when = std::max(when, mono_ns() + period);   // the file read may have taken longer than a period
        int64_t cm = 0, cm_first = 0;
        for (int k = 0; k < int(blank * 1e9 / double(period)) && !g_stop; k++) publish(zero.data(), zero.data(), cm);
        uint64_t s_first = 0, s_last = 0;
        const int frames = std::max(1, int(hold * 1e9 / double(period)));
        for (int k = 0; k < frames && !g_stop; k++) {
            s_last = publish(a.data(), b.data(), cm);
            if (k == 0) { s_first = s_last; cm_first = cm; }
        }
        std::fprintf(fl, "%ld,%" PRIu64 ",%" PRIu64 ",%" PRId64 ",%" PRId64, s, s_first, s_last, cm_first, cm);
        for (double v : truth) std::fprintf(fl, ",%.9f", v);
        std::fprintf(fl, "\n");
        std::fflush(fl);
    }
    int64_t cm = 0;
    for (int k = 0; k < 20 && !g_stop; k++) publish(zero.data(), zero.data(), cm);
    sink->close();
    std::fclose(fl);
    std::printf("trk_scenes: %ld scenes, %" PRIu64 " pairs published, max lateness %.2f ms\n", shown, seq, double(late_max) / 1e6);
    return 0;
}
