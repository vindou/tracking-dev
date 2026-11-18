#include "calib.hpp"

#include <cmath>

#include "../common/json.hpp"

namespace trk {

bool apply_calibration_file(const std::string& path, Camera cams[2], std::string& err) {
    const Json j = Json::parse_file(path, err);
    if (!err.empty()) return false;
    if (j["version"].num() != 1) { err = path + ": calibration file version must be 1"; return false; }
    for (int c = 0; c < 2; c++) {
        const Json& m = j["cameras"][cams[c].name]["T_rig_from_camera"];
        if (m.size() != 4) { err = path + ": no T_rig_from_camera for " + cams[c].name; return false; }
        Mat4 T;
        for (int r = 0; r < 4; r++)
            for (int k = 0; k < 4; k++) T(r, k) = m[std::size_t(r)][std::size_t(k)].num();
        cams[c].set_pose(T);
    }
    return true;
}

bool seen_by_both(const Camera cams[2], const Vec3& p, double min_depth, double max_depth, double margin_px) {
    for (int c = 0; c < 2; c++) {
        const Vec3 pc = xform(cams[c].T_cam_rig, p);
        double u, v;
        if (pc.z() < min_depth || pc.z() > max_depth || !cams[c].project(pc, u, v)) return false;
        if (u < margin_px || v < margin_px || u > cams[c].width - 1 - margin_px || v > cams[c].height - 1 - margin_px)
            return false;
    }
    return true;
}

Box compute_work_volume(const Camera cams[2], double min_depth, double max_depth, double z_min, double* seen_fraction) {
    Box b;
    b.min = Vec3(1e9, 1e9, 1e9);
    b.max = -b.min;
    const double step = 0.005;
    long seen = 0;
    for (double x = -1.0; x <= 1.0; x += step)
        for (double y = -1.0; y <= 1.0; y += step)
            for (double z = z_min; z <= 1.2; z += step) {
                const Vec3 p(x, y, z);
                if (!seen_by_both(cams, p, min_depth, max_depth, 0.0)) continue;
                seen++;
                b.min = b.min.cwiseMin(p);
                b.max = b.max.cwiseMax(p);
            }
    if (seen_fraction) {
        const Vec3 e = (b.max - b.min) / step + Vec3::Ones();
        *seen_fraction = double(seen) / (e.x() * e.y() * e.z());
    }
    return b;
}

Mat3 random_rotation(std::mt19937_64& rng) {
    std::normal_distribution<double> nd;
    Eigen::Quaterniond q(nd(rng), nd(rng), nd(rng), nd(rng));
    q.normalize();
    return q.toRotationMatrix();
}

Mat4 random_pose_in_view(const Camera cams[2], const Mesh& mesh, std::mt19937_64& rng, double z_min, double max_depth) {
    std::uniform_real_distribution<double> ux(-0.25, 0.25), uy(-0.25, 0.25), uz(z_min, 0.9);
    const Vec3 lo = mesh.bmin, hi = mesh.bmax;
    for (int attempt = 0; attempt < 100000; attempt++) {
        const Mat3 R = random_rotation(rng);
        const Vec3 p(ux(rng), uy(rng), uz(rng));
        const Mat4 T = make_pose(R, p);
        bool ok = true;
        for (int k = 0; k < 8 && ok; k++) {
            const Vec3 corner((k & 1) ? hi.x() : lo.x(), (k & 2) ? hi.y() : lo.y(), (k & 4) ? hi.z() : lo.z());
            const Vec3 w = xform(T, corner);
            ok = w.z() >= z_min && seen_by_both(cams, w, 0.25, max_depth, 10.0);
        }
        if (ok) return T;
    }
    return make_pose(Mat3::Identity(), Vec3(0, 0, 0.4));
}

}  // namespace trk
