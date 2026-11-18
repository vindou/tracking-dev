// Offline run of the tracker core on a pair file (no shared memory): one cycle per pair, as the live loop with no
// skipped pairs. Writes one CSV line per pose. Used as the offline reference of the replay test and for analysis.
//   trk_offline --in FILE --out CSV [--config config/tracker.json] [--calib FILE] [--camera-mask 1|2|3] [--force-track x,y,z,qw,qx,qy,qz]
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include "core/argutil.hpp"
#include "core/config.hpp"
#include "core/tracker_core.hpp"
#include "records/pairfile.hpp"
#include "records/pose_csv.hpp"

using namespace trk;

int main(int argc, char** argv) {
    std::string in, out, cfgp = "config/tracker.json", calib;
    int mask = 3;
    std::vector<double> force;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--in") in = argv[i + 1];
        else if (a == "--out") out = argv[i + 1];
        else if (a == "--config") cfgp = argv[i + 1];
        else if (a == "--calib") calib = argv[i + 1];
        else if (a == "--camera-mask") mask = std::atoi(argv[i + 1]);
        else if (a == "--force-track") force = parse_list(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (in.empty() || out.empty()) { std::fprintf(stderr, "usage: trk_offline --in FILE --out CSV [--config F] [--calib F] [--camera-mask M]\n"); return 2; }
    TrackerConfig cfg;
    std::string err;
    if (!load_tracker_config(cfgp, cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    if (!calib.empty()) {
        Camera tmp[2];
        tmp[0].name = cfg.cam_name[0];
        tmp[1].name = cfg.cam_name[1];
        if (!apply_calibration_file(calib, tmp, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        cfg.cam_T[0] = tmp[0].T_rig_cam;
        cfg.cam_T[1] = tmp[1].T_rig_cam;
    }
    TrackerCore core;
    if (!core.init(cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    PairFileReader f;
    if (!f.open(in, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    cap::CameraInfo ci[2] = {f.header().cam[0], f.header().cam[1]};
    std::string note;
    if (!core.set_cameras(ci, err, &note)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    core.set_camera_mask(mask);
    if (force.size() == 7) {
        const double q[4] = {force[3], force[4], force[5], force[6]};
        core.force_track(make_pose(R_from_quat(q), Vec3(force[0], force[1], force[2])));
    }
    FILE* o = std::fopen(out.c_str(), "w");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 3; }
    write_pose_csv_header(o, true);
    const std::size_t n = std::size_t(f.header().width) * f.header().height;
    std::vector<uint16_t> a(n), b(n);
    PairChunkHeader ch;
    PoseRecord p;
    uint64_t k = 0, valid = 0;
    while (f.next(ch, a.data(), b.data(), err)) {
        CropRect cr[2];
        core.crops(cr);
        const uint16_t* img[2] = {a.data(), b.data()};
        core.process(ch.rec, img, p, nullptr);
        p.pose_seq = k++;
        if (p.valid) valid++;
        write_pose_csv_row(o, p, (ch.flags & 1) ? ch.truth : nullptr);
        if (std::getenv("TRK_DEBUG") && p.track_state == kStateSearch && p.search_done == 0 && k > 1) {
            // a search round ended in this cycle: start positions and the best coarse candidates
            Vec3 c[8];
            const int nc = core.search_centers(c);
            std::printf("pair %llu: round ended; %d start positions:", (unsigned long long)k - 1, nc);
            for (int i = 0; i < nc; i++) std::printf(" (%.3f %.3f %.3f)", c[i].x(), c[i].y(), c[i].z());
            std::printf("\n");
            const auto& cd = core.candidates();
            for (int i = 0; i < 5 && i < int(cd.size()); i++) {
                const Mat4 T = inverse_pose(cd[std::size_t(i)].T_obj_rig);
                std::printf("   cand %d: centre %d fitness %.3f rmse %.2f mm violation %.3f  p = %.3f %.3f %.3f\n", i,
                            cd[std::size_t(i)].center, cd[std::size_t(i)].fitness, cd[std::size_t(i)].rmse * 1000,
                            cd[std::size_t(i)].violation, T(0, 3), T(1, 3), T(2, 3));
            }
            std::printf("   this cycle: reject %u points %u rmse %.2f mm inliers %.3f eig %.4f violation %.3f\n", p.reject_reasons,
                        p.point_count, p.fit_rmse * 1000, p.inlier_ratio, p.min_eigenvalue, double(p.free_space_violation));
        }
    }
    std::fclose(o);
    std::printf("%s: %" PRIu64 " pairs, %" PRIu64 " valid poses, final state %u; cameras: %s\n", in.c_str(), k, valid,
                unsigned(core.state()), note.c_str());
    return err.empty() ? 0 : 1;
}
