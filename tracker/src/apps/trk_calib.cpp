// T5, calibration: with the object stationary, solves the object pose with camera A alone and with camera B alone
// (tracker core with a camera mask, search from scratch), reports the difference, and writes a calibration file that
// keeps camera A at its configured (nominal) transform and corrects camera B so that both single-camera poses agree.
// If the two poses differ by more than 20 mm or 5 degrees, the camera-frame assumption is checked: the difference is
// recomputed for other axis conventions of the camera frame, swapped names and other translation units.
//
//   trk_calib --out calib.json [--config config/tracker.json] [--source shm:/d435_pairs | FILE] [--pairs 300]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "common/json.hpp"
#include "core/config.hpp"
#include "core/pairinput.hpp"
#include "core/tracker_core.hpp"

using namespace trk;

namespace {
struct Solve {
    int valid = 0, cycles = 0;
    uint32_t reasons = 0;
    std::vector<Mat4> poses;   // valid ICP poses (rig frame, configured transforms)
};

Mat4 mean_pose(const std::vector<Mat4>& v) {
    Vec3 p = Vec3::Zero();
    Eigen::Vector4d q = Eigen::Vector4d::Zero();
    Eigen::Quaterniond q0(v[0].block<3, 3>(0, 0));
    for (const Mat4& T : v) {
        p += T.block<3, 1>(0, 3);
        Eigen::Quaterniond qi(T.block<3, 3>(0, 0));
        if (qi.dot(q0) < 0) qi.coeffs() = -qi.coeffs();
        q += Eigen::Vector4d(qi.w(), qi.x(), qi.y(), qi.z());
    }
    q.normalize();
    return make_pose(Eigen::Quaterniond(q(0), q(1), q(2), q(3)).toRotationMatrix(), p / double(v.size()));
}

void diff(const Mat4& a, const Mat4& b, double& mm, double& deg) {
    mm = (a.block<3, 1>(0, 3) - b.block<3, 1>(0, 3)).norm() * 1000;
    deg = rotation_angle(a.block<3, 3>(0, 0).transpose() * b.block<3, 3>(0, 0)) * 180 / M_PI;
}

Json mat_json(const Mat4& T) {
    Json m = Json::array();
    for (int r = 0; r < 4; r++) {
        Json row = Json::array();
        for (int c = 0; c < 4; c++) row.push(Json::number(T(r, c)));
        m.push(row);
    }
    return m;
}
}  // namespace

int main(int argc, char** argv) {
    std::string out, cfgp = "config/tracker.json", source = "shm:/d435_pairs";
    int pairs = 300;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--out") out = argv[i + 1];
        else if (a == "--config") cfgp = argv[i + 1];
        else if (a == "--source") source = argv[i + 1];
        else if (a == "--pairs") pairs = std::atoi(argv[i + 1]);
    }
    if (out.empty()) { std::fprintf(stderr, "usage: trk_calib --out FILE [--config F] [--source shm:NAME|FILE] [--pairs N]\n"); return 2; }
    TrackerConfig cfg;
    std::string err;
    if (!load_tracker_config(cfgp, cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    PairInput in;
    if (!in.open(source, 20, err)) { std::fprintf(stderr, "cannot open %s: %s\n", source.c_str(), err.c_str()); return 3; }
    TrackerCore core[2];
    for (int c = 0; c < 2; c++) {
        if (!core[c].init(cfg, err) || !core[c].set_cameras(in.cameras(), err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
        core[c].set_camera_mask(1 << c);
    }
    std::vector<uint16_t> a(in.pixels()), b(in.pixels());
    const uint16_t* img[2] = {a.data(), b.data()};
    cap::PairRecord rec;
    Solve s[2];
    PoseRecord p;
    int k = 0;
    while (k < pairs && in.next(rec, a.data(), b.data())) {
        k++;
        for (int c = 0; c < 2; c++) {
            CropRect cr[2];
            core[c].crops(cr);
            core[c].process(rec, img, p, nullptr);
            s[c].cycles++;
            if (p.valid && p.track_state == kStateTrack) {
                s[c].valid++;
                s[c].poses.push_back(make_pose(R_from_quat(p.icp_orientation), Vec3(p.icp_position[0], p.icp_position[1], p.icp_position[2])));
            } else {
                s[c].reasons |= p.reject_reasons;
            }
        }
    }
    const char* cn[2] = {"A", "B"};
    std::printf("%d pairs from %s\n", k, source.c_str());
    for (int c = 0; c < 2; c++) {
        const uint32_t r = s[c].reasons;
        std::printf("camera %s (%s, serial %s) alone: %d valid poses of %d cycles; reject reasons seen:%s%s%s%s%s%s%s\n", cn[c],
                    core[c].camera(c).name.c_str(), core[c].camera(c).serial.c_str(), s[c].valid, s[c].cycles,
                    (r & kRejectFewPoints) ? " points" : "", (r & kRejectRmse) ? " rmse" : "", (r & kRejectInliers) ? " inliers" : "",
                    (r & kRejectEigen) ? " eigenvalue" : "", (r & kRejectJump) ? " jump" : "",
                    (r & kRejectVisibility) ? " visibility" : "", (r & kRejectNoSolve) ? " no-solve(search)" : "");
    }
    if (s[0].poses.size() < 10 || s[1].poses.size() < 10) {
        std::printf("RESULT: single-camera solve FAILED (fewer than 10 valid poses for camera %s); no calibration written\n",
                    s[0].poses.size() < 10 ? "A" : "B");
        return 1;
    }
    const Mat4 PA = mean_pose(s[0].poses), PB = mean_pose(s[1].poses);
    double dmm, ddeg;
    diff(PA, PB, dmm, ddeg);
    double qa[4], qb[4];
    quat_from_R(PA.block<3, 3>(0, 0), qa);
    quat_from_R(PB.block<3, 3>(0, 0), qb);
    std::printf("object pose from camera A: p = %.4f %.4f %.4f m, q = %.5f %.5f %.5f %.5f\n", PA(0, 3), PA(1, 3), PA(2, 3), qa[0], qa[1], qa[2], qa[3]);
    std::printf("object pose from camera B: p = %.4f %.4f %.4f m, q = %.5f %.5f %.5f %.5f\n", PB(0, 3), PB(1, 3), PB(2, 3), qb[0], qb[1], qb[2], qb[3]);
    std::printf("difference A - B: %.2f mm, %.3f deg (limits 20 mm, 5 deg)\n", dmm, ddeg);
    const Camera& camA = core[0].camera(0);
    const Camera& camB = core[1].camera(1);
    // camera-frame poses (independent of the rig transforms)
    const Mat4 TcA = camA.T_cam_rig * PA, TcB = camB.T_cam_rig * PB;
    // correction of B: T_rig_camB_new * T_camB_obj = pose from A
    const Mat4 TB_new = PA * inverse_pose(TcB);
    double cmm, cdeg;
    diff(TB_new, camB.T_rig_cam, cmm, cdeg);
    std::printf("correction of camera B (%s): %.2f mm, %.3f deg from its configured transform\n", camB.name.c_str(), cmm, cdeg);
    Json hyp = Json::array();
    if (dmm > 20 || ddeg > 5) {
        std::printf("camera-frame assumption check (difference over the limits): difference for other assumptions\n");
        struct H { const char* name; Mat3 F; bool swap; double tscale; };
        const H hs[] = {
            {"nominal (x right, y down, z forward)", Mat3::Identity(), false, 1.0},
            {"camera frame rotated 180 deg about z (x left, y up)", Eigen::Vector3d(-1, -1, 1).asDiagonal(), false, 1.0},
            {"camera frame rotated 180 deg about x (y up, z backward)", Eigen::Vector3d(1, -1, -1).asDiagonal(), false, 1.0},
            {"camera frame rotated 180 deg about y (x left, z backward)", Eigen::Vector3d(-1, 1, -1).asDiagonal(), false, 1.0},
            {"names swapped (D435i_1 <-> D435i_2)", Mat3::Identity(), true, 1.0},
            {"translations in cm instead of mm", Mat3::Identity(), false, 10.0},
            {"translations in inches instead of mm", Mat3::Identity(), false, 25.4},
        };
        double best = 1e9;
        const char* best_name = "";
        for (const H& h : hs) {
            Mat4 TA = camA.T_rig_cam, TB = camB.T_rig_cam;
            if (h.swap) std::swap(TA, TB);
            TA.block<3, 1>(0, 3) *= h.tscale;
            TB.block<3, 1>(0, 3) *= h.tscale;
            Mat4 F = Mat4::Identity();
            F.block<3, 3>(0, 0) = h.F;
            double m, d;
            diff(TA * F * TcA, TB * F * TcB, m, d);
            std::printf("   %-58s %9.2f mm %8.3f deg%s\n", h.name, m, d, (m <= 20 && d <= 5) ? "   <- within the limits" : "");
            hyp.push(Json::object().set("assumption", Json::string(h.name)).set("difference_mm", Json::number(m)).set("difference_deg", Json::number(d)));
            if (m / 20 + d / 5 < best) { best = m / 20 + d / 5; best_name = h.name; }
        }
        std::printf("   (a name swap rotates the whole scene by 180 deg about the rig Z axis: both poses move together, so T5 cannot detect it; T4 does)\n");
        std::printf("RESULT: single-camera poses differ by more than the limits; closest assumption: %s\n", best_name);
    } else {
        std::printf("RESULT: camera-frame assumption confirmed (difference within 20 mm and 5 deg)\n");
    }
    Json j = Json::object();
    j.set("version", Json::number(1));
    Json cams = Json::object();
    cams.set(camA.name, Json::object().set("T_rig_from_camera", mat_json(camA.T_rig_cam)).set("serial", Json::string(camA.serial)).set("note", Json::string("configured transform (kept)")));
    cams.set(camB.name, Json::object().set("T_rig_from_camera", mat_json(TB_new)).set("serial", Json::string(camB.serial)).set("note", Json::string("corrected by trk_calib")));
    j.set("cameras", cams);
    j.set("pairs", Json::number(k));
    j.set("difference_before_mm", Json::number(dmm)).set("difference_before_deg", Json::number(ddeg));
    j.set("correction_of_B_mm", Json::number(cmm)).set("correction_of_B_deg", Json::number(cdeg));
    j.set("assumption_check", hyp);
    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 3; }
    std::fprintf(f, "%s\n", j.dump().c_str());
    std::fclose(f);
    std::printf("calibration written to %s (set cameras.calibration_file in the configuration to use it)\n", out.c_str());
    return 0;
}
