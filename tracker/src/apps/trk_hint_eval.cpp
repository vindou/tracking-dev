// Acquisition on independent static scenes, with and without external pose hypotheses (flow module, ../flow).
// Each scene is one depth pair with a true pose. For each scene the tracker core starts in search state and gets the
// same pair every cycle (pair period 11.1283 ms) until it accepts a pose or --max-cycles is reached.
//   baseline: the search of the tracker (start positions x orientation grid, time-sliced).
//   hints:    the hypotheses of the scene are given before the first cycle (TrackerCore::set_hints); if none passes
//             the fit check the same search follows, so "cycles == 1 and by_hint" is the hint path alone.
// Both use the same ICP and the same fit check. One CSV line per scene and mode.
//   trk_hint_eval --scenes FILE --out CSV [--hints FILE] [--baseline 0|1] [--hint-order score|rank] [--config config/tracker.json] [--max-cycles 180]
//                 [--corr kdtree|bounded|table] [--hint-threads N] (override icp.correspondence_search, hints.threads)
//                 [--step N] (every N-th scene only) [--cycles CSV] (one line per cycle: elapsed time, hypotheses
//                 tried, CPU time of the loop thread)
// A profiling build (cmake -DTRK_PROFILE=ON) prints the time per section at the end.
// Scenes file: "TRKSCENE", u32 version 1, u32 count, u32 width, u32 height, then per scene f64 T_rig_from_object[16]
// (row-major), u16 image A[w*h], u16 image B[w*h] (depth units of 1 mm, 0 = no data).
// Hints file: "TRKHINTS", u32 version 1, u32 count, u32 K, then per scene u32 n, f64 T_rig_from_object[K][16].
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "common/clock.hpp"
#include "common/prof.hpp"
#include "core/config.hpp"
#include "core/geometry.hpp"
#include "core/synth.hpp"
#include "core/tracker_core.hpp"

using namespace trk;

namespace {
// CPU time of the calling thread, ms (not disturbed by other load on the machine, unlike the elapsed time)
double thread_cpu_ms() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}
cap::PairRecord make_record(uint64_t k) {
    cap::PairRecord rec;
    std::memset(&rec, 0, sizeof rec);
    const double t = double(k) * 11.1283e-3;
    const int64_t base = 1000000000LL, real_minus_mono = 1700000000LL * 1000000000LL;
    const int64_t cap_mono = base + int64_t(t * 1e9);
    rec.pair_seq = k;
    for (int c = 0; c < 2; c++) {
        rec.cam[c].global_ts_ms = double(cap_mono + real_minus_mono) / 1e6;
        rec.cam[c].hw_ts_us = int64_t(t * 1e6);
        rec.cam[c].sensor_ts_us = int64_t(t * 1e6);
        rec.cam[c].arrival_mono_ns = cap_mono + 9600000;
        rec.cam[c].frame_counter = k;
        rec.cam[c].exposure_us = 1000;
        rec.cam[c].domain = 2;
    }
    rec.publish_mono_ns = cap_mono + 10600000;
    rec.publish_real_ns = rec.publish_mono_ns + real_minus_mono;
    return rec;
}
}  // namespace

int main(int argc, char** argv) {
    std::string scenes, hints, out, cycles_csv, cfgp = "config/tracker.json";
    std::string corr;
    int max_cycles = 180, scene_step = 1, hint_threads = 0;
    bool baseline = true, in_order = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--scenes") scenes = argv[i + 1];
        else if (a == "--hints") hints = argv[i + 1];
        else if (a == "--out") out = argv[i + 1];
        else if (a == "--config") cfgp = argv[i + 1];
        else if (a == "--max-cycles") max_cycles = std::atoi(argv[i + 1]);
        else if (a == "--baseline") baseline = std::atoi(argv[i + 1]) != 0;
        else if (a == "--step") scene_step = std::max(1, std::atoi(argv[i + 1]));
        else if (a == "--corr") corr = argv[i + 1];
        else if (a == "--hint-threads") hint_threads = std::atoi(argv[i + 1]);
        else if (a == "--cycles") cycles_csv = argv[i + 1];
        else if (a == "--hint-order") in_order = std::string(argv[i + 1]) == "rank";
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (scenes.empty() || out.empty()) {
        std::fprintf(stderr, "usage: trk_hint_eval --scenes FILE --out CSV [--hints FILE] [--config F] [--max-cycles N]\n");
        return 2;
    }
    TrackerConfig cfg;
    std::string err;
    if (!load_tracker_config(cfgp, cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    if (!corr.empty()) cfg.corr_search = corr;             // overrides of the configuration (benchmarks)
    if (hint_threads > 0) cfg.hint_threads = hint_threads;
    TrackerCore core;
    if (!core.init(cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    Camera cams[2];
    default_cameras(cams);
    cap::CameraInfo ci[2];
    camera_infos(cams, ci);
    if (!core.set_cameras(ci, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }

    FILE* fs = std::fopen(scenes.c_str(), "rb");
    if (!fs) { std::fprintf(stderr, "cannot open %s\n", scenes.c_str()); return 3; }
    char magic[8];
    uint32_t h[4];
    if (std::fread(magic, 1, 8, fs) != 8 || std::memcmp(magic, "TRKSCENE", 8) != 0 || std::fread(h, 4, 4, fs) != 4 || h[0] != 1 ||
        int(h[2]) != cams[0].width || int(h[3]) != cams[0].height) {
        std::fprintf(stderr, "%s: not a scenes file (version 1, %d x %d)\n", scenes.c_str(), cams[0].width, cams[0].height);
        return 3;
    }
    const uint32_t count = h[1];
    FILE* fh = nullptr;
    uint32_t K = 0;
    if (!hints.empty()) {
        fh = std::fopen(hints.c_str(), "rb");
        uint32_t g[3];
        if (!fh || std::fread(magic, 1, 8, fh) != 8 || std::memcmp(magic, "TRKHINTS", 8) != 0 || std::fread(g, 4, 3, fh) != 3 ||
            g[0] != 1 || g[1] != count) {
            std::fprintf(stderr, "%s: not a hints file for %u scenes\n", hints.c_str(), count);
            return 3;
        }
        K = g[2];
    }
    FILE* o = std::fopen(out.c_str(), "w");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 3; }
    std::fprintf(o, "scene,mode,valid,cycles,by_hint,hints_tried,compute_ms,first_cycle_ms,px,py,pz,qw,qx,qy,qz,rmse_mm,"
                    "inlier_ratio,violation,err_mm,err_deg\n");
    FILE* oc = cycles_csv.empty() ? nullptr : std::fopen(cycles_csv.c_str(), "w");
    if (oc) std::fprintf(oc, "scene,mode,cycle,ms,hints_tried,by_hint,valid,solve_ms,cpu_ms\n");
    const std::size_t n = std::size_t(h[2]) * h[3];
    std::vector<uint16_t> a(n), b(n);
    std::vector<double> hbuf(std::size_t(K) * 16);
    std::vector<Mat4> hint(K);
    uint64_t k = 0;
    for (uint32_t s = 0; s < count; s++) {
        double truth[16];
        if (std::fread(truth, 8, 16, fs) != 16 || std::fread(a.data(), 2, n, fs) != n || std::fread(b.data(), 2, n, fs) != n) {
            std::fprintf(stderr, "%s: short file at scene %u\n", scenes.c_str(), s);
            return 3;
        }
        Mat4 Tt;
        for (int i = 0; i < 16; i++) Tt(i / 4, i % 4) = truth[i];
        uint32_t nh = 0;
        if (fh) {
            if (std::fread(&nh, 4, 1, fh) != 1 || std::fread(hbuf.data(), 8, hbuf.size(), fh) != hbuf.size() || nh > K) {
                std::fprintf(stderr, "%s: short file at scene %u\n", hints.c_str(), s);
                return 3;
            }
            for (uint32_t j = 0; j < nh; j++)
                for (int i = 0; i < 16; i++) hint[j](i / 4, i % 4) = hbuf[std::size_t(j) * 16 + std::size_t(i)];
        }
        const uint16_t* img[2] = {a.data(), b.data()};
        if (s % uint32_t(scene_step) != 0) continue;
        for (int mode = baseline ? 0 : 1; mode < (fh ? 2 : 1); mode++) {
            core.reset_to_search();
            if (mode == 1) core.set_hints(hint.data(), int(nh), in_order);
            PoseRecord p;
            int cycles = 0, by_hint = 0, tried = 0;
            double compute_ms = 0, first_ms = 0;
            bool valid = false;
            while (cycles < max_cycles && !valid) {
                CropRect cr[2];
                core.crops(cr);
                const double c0 = thread_cpu_ms();
                const int64_t t0 = mono_ns();
                core.process(make_record(k++), img, p, nullptr);
                const double ms = double(mono_ns() - t0) / 1e6, cpu_ms = thread_cpu_ms() - c0;
                compute_ms += ms;
                if (cycles == 0) {
                    first_ms = ms;
                    tried = core.hints_tried();
                }
                cycles++;
                valid = p.valid != 0 && core.state() == kStateTrack;
                if (oc)
                    std::fprintf(oc, "%u,%s,%d,%.4f,%d,%d,%d,%.4f,%.4f\n", s, mode ? "hints" : "baseline", cycles - 1, ms, core.hints_tried(),
                                 int(p.by_hint), valid ? 1 : 0, double(p.stage_us[kStageSolve]) / 1e3, cpu_ms);
                if (valid && cycles == 1 && tried > 0) by_hint = 1;
            }
            double e_mm = -1, e_deg = -1;
            if (valid) {
                const Mat3 R = R_from_quat(p.icp_orientation);
                e_mm = 1000.0 * (Vec3(p.icp_position[0], p.icp_position[1], p.icp_position[2]) - Tt.block<3, 1>(0, 3)).norm();
                e_deg = rotation_angle(R * Tt.block<3, 3>(0, 0).transpose()) * 180.0 / 3.14159265358979323846;
            }
            std::fprintf(o, "%u,%s,%d,%d,%d,%d,%.3f,%.3f,%.6f,%.6f,%.6f,%.7f,%.7f,%.7f,%.7f,%.3f,%.4f,%.4f,%.3f,%.3f\n", s,
                         mode ? "hints" : "baseline", valid ? 1 : 0, cycles, by_hint, tried, compute_ms, first_ms,
                         p.icp_position[0], p.icp_position[1], p.icp_position[2], p.icp_orientation[0], p.icp_orientation[1],
                         p.icp_orientation[2], p.icp_orientation[3], p.fit_rmse * 1000, p.inlier_ratio,
                         double(p.free_space_violation), e_mm, e_deg);
        }
    }
    std::fclose(o);
    if (oc) std::fclose(oc);
#ifdef TRK_PROFILE
    prof::report(stdout);
#endif
    std::fclose(fs);
    if (fh) std::fclose(fh);
    std::printf("%s: %u scenes -> %s\n", scenes.c_str(), count, out.c_str());
    return 0;
}
