// T3 replay: writes a pair file into a pair buffer with the capture layout (version 2), using the capture program's
// own writer (cap::ShmSink), at the original timing. All time fields are shifted to the current clocks (monotonic and
// realtime by the same offsets that the original pairs had), so the tracker cannot tell replay from capture.
//   trk_replay --in FILE [--shm-name /d435_pairs] [--loops N] [--start-delay S] [--speed X]
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "common/clock.hpp"
#include "records/pairfile.hpp"
#include "shm_sink.hpp"

using namespace trk;

namespace { volatile std::sig_atomic_t g_stop = 0; void on_sig(int) { g_stop = 1; } }

int main(int argc, char** argv) {
    std::string in, name = "/d435_pairs";
    int loops = 1;
    double start_delay = 1.0, speed = 1.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--in") in = argv[i + 1];
        else if (a == "--shm-name") name = argv[i + 1];
        else if (a == "--loops") loops = std::atoi(argv[i + 1]);
        else if (a == "--start-delay") start_delay = std::atof(argv[i + 1]);
        else if (a == "--speed") speed = std::atof(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (in.empty()) { std::fprintf(stderr, "usage: trk_replay --in FILE [--shm-name NAME] [--loops N] [--start-delay S]\n"); return 2; }
    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);
    std::string err;
    PairFileReader f;
    if (!f.open(in, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    const PairFileHeader& fh = f.header();
    const std::size_t n = std::size_t(fh.width) * fh.height;
    // preload (decoding during playback could disturb the timing)
    std::vector<cap::PairRecord> recs;
    std::vector<uint16_t> imgs;
    PairChunkHeader ch;
    std::vector<uint16_t> a(n), b(n);
    while (f.next(ch, a.data(), b.data(), err)) {
        recs.push_back(ch.rec);
        imgs.insert(imgs.end(), a.begin(), a.end());
        imgs.insert(imgs.end(), b.begin(), b.end());
    }
    if (!err.empty() || recs.empty()) { std::fprintf(stderr, "%s: %s\n", in.c_str(), err.empty() ? "no pairs" : err.c_str()); return 3; }
    cap::CameraInfo ci[2] = {fh.cam[0], fh.cam[1]};
    std::string note;
    auto sink = cap::ShmSink::create(name, ci, n * 2, err, &note);
    if (!sink) { std::fprintf(stderr, "shared memory %s: %s\n", name.c_str(), err.c_str()); return 4; }
    if (!note.empty()) std::printf("note: %s\n", note.c_str());
    std::printf("replay %s: %zu pairs x %d loops into %s (instance %016" PRIx64 ")\n", in.c_str(), recs.size(), loops,
                name.c_str(), sink->instance_id());
    timespec ts{time_t(start_delay), long((start_delay - std::floor(start_delay)) * 1e9)};
    nanosleep(&ts, nullptr);
    const double span_ns = double(recs.back().publish_mono_ns - recs.front().publish_mono_ns) + 1e9 * fh.frame_period_ms / 1000.0;
    uint64_t seq = 0;
    int64_t late_max = 0;
    for (int l = 0; l < loops && !g_stop; l++) {
        const int64_t base_mono = mono_ns(), base_real = real_ns();
        bool period_sent = false;
        for (std::size_t k = 0; k < recs.size() && !g_stop; k++) {
            cap::PairRecord r = recs[k];
            const int64_t rel = int64_t(double(r.publish_mono_ns - recs[0].publish_mono_ns) / speed);
            const int64_t when = base_mono + rel;
            timespec t{time_t(when / 1000000000LL), long(when % 1000000000LL)};
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, nullptr);
            // shift both clocks: the offsets between capture, arrival and publish times stay as recorded
            const int64_t d_mono = base_mono - recs[0].publish_mono_ns + (rel - (r.publish_mono_ns - recs[0].publish_mono_ns));
            const int64_t d_real = base_real - recs[0].publish_real_ns + (rel - (r.publish_mono_ns - recs[0].publish_mono_ns));
            for (int c = 0; c < 2; c++) {
                r.cam[c].global_ts_ms += double(d_real) / 1e6;
                r.cam[c].arrival_mono_ns += d_mono;
            }
            r.publish_mono_ns += d_mono;
            r.publish_real_ns += d_real;
            r.pair_seq = seq++;
            if (!period_sent) { sink->on_period(fh.frame_period_ms, fh.tolerance_ms); period_sent = true; }
            late_max = std::max(late_max, mono_ns() - when);
            sink->on_pair(r, &imgs[k * 2 * n], &imgs[(k * 2 + 1) * n]);
        }
        (void)span_ns;
    }
    sink->close();
    std::printf("replay done: %" PRIu64 " pairs published, max lateness %.2f ms\n", seq, double(late_max) / 1e6);
    return 0;
}
