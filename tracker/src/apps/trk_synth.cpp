// T2: synthetic depth pairs for a given object pose and motion, written as a pair file (record/replay format).
//
//   trk_synth --out FILE [--frames N] [--pose x,y,z,qw,qx,qy,qz] [--random-pose SEED] [--vel vx,vy,vz]
//             [--angvel wx,wy,wz (deg/s)] [--offset-ms MS] [--noise-mm SIGMA] [--plane-z Z] [--absent FIRST:LAST]
//             [--period-ms P] [--seed S] [--stl models/test_object.stl] [--calib FILE] [--blank A|B:FIRST:LAST]
//
// Cameras: intrinsics and serials of the real cameras; transforms nominal (D435i_1 = A, D435i_2 = B) or from a
// calibration file (T5). Each pair also stores the true pose at the measurement time (mean capture time).
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/argutil.hpp"
#include "core/calib.hpp"
#include "core/synth.hpp"
#include "common/clock.hpp"
#include "records/pairfile.hpp"

using namespace trk;

int main(int argc, char** argv) {
    std::string out, stl = "models/test_object.stl", calib;
    int frames = 90;
    SynthScenario s;
    s.T0 = make_pose(Mat3::Identity(), Vec3(0, 0, 0.35));
    long random_seed = -1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--out") out = val();
        else if (a == "--frames") frames = std::atoi(val().c_str());
        else if (a == "--pose") {
            const auto v = parse_list(val());
            if (v.size() != 7) { std::fprintf(stderr, "--pose needs x,y,z,qw,qx,qy,qz\n"); return 2; }
            const double q[4] = {v[3], v[4], v[5], v[6]};
            s.T0 = make_pose(R_from_quat(q), Vec3(v[0], v[1], v[2]));
        } else if (a == "--random-pose") random_seed = std::atol(val().c_str());
        else if (a == "--vel") { const auto v = parse_list(val()); s.velocity = Vec3(v.at(0), v.at(1), v.at(2)); }
        else if (a == "--angvel") { const auto v = parse_list(val()); s.angular_velocity = Vec3(v.at(0), v.at(1), v.at(2)) * M_PI / 180.0; }
        else if (a == "--offset-ms") s.offset_ms = std::atof(val().c_str());
        else if (a == "--noise-mm") s.render.noise_sigma_m = std::atof(val().c_str()) / 1000.0;
        else if (a == "--plane-z") { s.render.plane = true; s.render.plane_z = std::atof(val().c_str()); }
        else if (a == "--absent") { const std::string v = val(); const auto c = v.find(':'); s.absent.emplace_back(std::atoi(v.substr(0, c).c_str()), std::atoi(v.substr(c + 1).c_str())); }
        else if (a == "--period-ms") s.period_ms = std::atof(val().c_str());
        else if (a == "--blank") {   // CAM:FIRST:LAST  (covered camera A or B)
            const std::string v = val();
            s.blank_cam = v[0] == 'B' ? 1 : 0;
            const auto c1 = v.find(':'), c2 = v.find(':', c1 + 1);
            s.blank_from = std::atoi(v.substr(c1 + 1, c2 - c1 - 1).c_str());
            s.blank_to = std::atoi(v.substr(c2 + 1).c_str());
        }
        else if (a == "--seed") s.seed = uint64_t(std::atoll(val().c_str()));
        else if (a == "--stl") stl = val();
        else if (a == "--calib") calib = val();
        else { std::fprintf(stderr, "unknown option %s (see the header of trk_synth.cpp)\n", a.c_str()); return 2; }
    }
    if (out.empty()) { std::fprintf(stderr, "--out FILE is required\n"); return 2; }
    std::string err;
    Mesh mesh;
    if (!load_stl(stl, mesh, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    Camera cams[2];
    default_cameras(cams);
    if (!calib.empty() && !apply_calibration_file(calib, cams, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    if (random_seed >= 0) {
        std::mt19937_64 rng{uint64_t(random_seed)};
        s.T0 = random_pose_in_view(cams, mesh, rng);
    }
    s.mono_base_ns = mono_ns();
    s.real_minus_mono_ns = real_ns() - mono_ns();
    cap::CameraInfo ci[2];
    camera_infos(cams, ci);
    PairFileWriter w;
    if (!w.open(out, ci, 1, s.period_ms, s.period_ms / 2, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    SynthGenerator gen(mesh, cams, s);
    std::vector<uint16_t> a(std::size_t(cams[0].width) * cams[0].height), b(a.size());
    for (int k = 0; k < frames; k++) {
        cap::PairRecord rec;
        Mat4 truth;
        bool present;
        gen.make(k, a.data(), b.data(), rec, truth, present);
        double t16[16];
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) t16[r * 4 + c] = truth(r, c);
        w.append(rec, a.data(), b.data(), t16);
    }
    w.close();
    double q[4];
    quat_from_R(s.T0.block<3, 3>(0, 0), q);
    std::printf("%s: %d pairs, %.1f MB; pose at t=0: p = %.4f %.4f %.4f m, q = %.5f %.5f %.5f %.5f; offset %.2f ms, noise %.2f mm\n",
                out.c_str(), frames, double(w.bytes()) / 1e6, s.T0(0, 3), s.T0(1, 3), s.T0(2, 3), q[0], q[1], q[2], q[3],
                s.offset_ms, s.render.noise_sigma_m * 1000);
    return 0;
}
