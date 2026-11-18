#include "render.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace trk {

void render_depth(const Mesh& mesh, const Camera& cam, const Mat4* T_rig_obj, const RenderOptions& opt,
                  std::mt19937_64& rng, uint16_t* out) {
    const int W = cam.width, H = cam.height;
    std::vector<double> z(std::size_t(W) * H, std::numeric_limits<double>::infinity());
    // background plane
    if (opt.plane) {
        const Vec3 o = cam.T_rig_cam.block<3, 1>(0, 3);
        const Mat3 R = cam.T_rig_cam.block<3, 3>(0, 0);
        for (int v = 0; v < H; v++)
            for (int u = 0; u < W; u++) {
                const Vec3 d((u - cam.ppx) / cam.fx, (v - cam.ppy) / cam.fy, 1.0);   // camera frame, d.z = 1
                const Vec3 dr = R * d;
                if (std::fabs(dr.z()) < 1e-12) continue;
                const double t = (opt.plane_z - o.z()) / dr.z();
                if (t > 0) z[std::size_t(v) * W + u] = t;
            }
    }
    if (T_rig_obj) {
        const Mat4 T = cam.T_cam_rig * (*T_rig_obj);
        for (std::size_t i = 0; i < mesh.size(); i++) {
            const Vec3 a = xform(T, mesh.v0[i]), b = xform(T, mesh.v1[i]), c = xform(T, mesh.v2[i]);
            int u0 = 0, u1 = W - 1, v0 = 0, v1 = H - 1;
            if (a.z() > 1e-6 && b.z() > 1e-6 && c.z() > 1e-6) {
                double us[3], vs[3];
                cam.project(a, us[0], vs[0]);
                cam.project(b, us[1], vs[1]);
                cam.project(c, us[2], vs[2]);
                u0 = std::max(0, int(std::floor(std::min({us[0], us[1], us[2]}))));
                u1 = std::min(W - 1, int(std::ceil(std::max({us[0], us[1], us[2]}))));
                v0 = std::max(0, int(std::floor(std::min({vs[0], vs[1], vs[2]}))));
                v1 = std::min(H - 1, int(std::ceil(std::max({vs[0], vs[1], vs[2]}))));
            } else if (a.z() <= 0 && b.z() <= 0 && c.z() <= 0) {
                continue;
            }
            const Vec3 e1 = b - a, e2 = c - a;
            for (int v = v0; v <= v1; v++)
                for (int u = u0; u <= u1; u++) {
                    // Moeller-Trumbore, ray from the camera origin with direction d (d.z = 1, so t = z)
                    const Vec3 d((u - cam.ppx) / cam.fx, (v - cam.ppy) / cam.fy, 1.0);
                    const Vec3 p = d.cross(e2);
                    const double det = e1.dot(p);
                    if (std::fabs(det) < 1e-15) continue;
                    const double inv = 1.0 / det;
                    const Vec3 s = -a;
                    const double bu = s.dot(p) * inv;
                    if (bu < 0 || bu > 1) continue;
                    const Vec3 q = s.cross(e1);
                    const double bv = d.dot(q) * inv;
                    if (bv < 0 || bu + bv > 1) continue;
                    const double t = e2.dot(q) * inv;
                    double& zz = z[std::size_t(v) * W + u];
                    if (t > 1e-6 && t < zz) zz = t;
                }
        }
    }
    std::normal_distribution<double> nd(0.0, opt.noise_sigma_m > 0 ? opt.noise_sigma_m : 1.0);
    for (std::size_t k = 0; k < z.size(); k++) {
        double zz = z[k];
        if (!std::isfinite(zz) || zz > opt.max_depth_m) { out[k] = 0; continue; }
        if (opt.noise_sigma_m > 0) zz += nd(rng);
        const double units = std::round(zz / cam.depth_scale);
        out[k] = units < 1 || units > 65535 ? 0 : uint16_t(units);
    }
}

}  // namespace trk
