// Reads the pose buffer and writes every pose record to CSV (tests, analysis). Follows tracker restarts.
//   trk_posedump --out CSV [--shm-name /trk_pose] [--duration S] [--wait S]
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>

#include "common/clock.hpp"
#include "records/pose_csv.hpp"
#include "shmring/shm_ring.hpp"

using namespace trk;

namespace { volatile std::sig_atomic_t g_stop = 0; void on_sig(int) { g_stop = 1; } }

int main(int argc, char** argv) {
    std::string out, name = "/trk_pose";
    double duration = 0, wait_s = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--out") out = argv[i + 1];
        else if (a == "--shm-name") name = argv[i + 1];
        else if (a == "--duration") duration = std::atof(argv[i + 1]);
        else if (a == "--wait") wait_s = std::atof(argv[i + 1]);
    }
    if (out.empty()) { std::fprintf(stderr, "usage: trk_posedump --out CSV [--shm-name NAME] [--duration S]\n"); return 2; }
    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);
    std::string err;
    std::unique_ptr<RingReader> r;
    const int64_t t0 = mono_ns();
    while (!g_stop && !(r = RingReader::attach(name, kPoseRecordType, kPoseRecordVersion, sizeof(PoseRecord), err))) {
        if (double(mono_ns() - t0) / 1e9 > wait_s) { std::fprintf(stderr, "cannot attach %s: %s\n", name.c_str(), err.c_str()); return 3; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!r) return 3;
    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) return 3;
    write_pose_csv_header(f, false);
    PoseRecord p;
    uint64_t n = 0;
    while (!g_stop && (duration <= 0 || double(mono_ns() - t0) / 1e9 < duration)) {
        const auto res = r->next(&p, 500000000LL);
        if (res == RingReader::Result::Ok) { write_pose_csv_row(f, p, nullptr); n++; continue; }
        if (res == RingReader::Result::Timeout) continue;
        if (res == RingReader::Result::WriterStopped) break;
        while (!g_stop && !r->reattach(err)) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::fclose(f);
    std::printf("%s: %" PRIu64 " poses, missing %" PRIu64 ", attaches %" PRIu64 "\n", out.c_str(), n, r->missing(), r->attaches());
    return 0;
}
