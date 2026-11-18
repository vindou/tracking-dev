#include "config.hpp"

#include "../common/json.hpp"

namespace trk {

std::string config_relative(const TrackerConfig& c, const std::string& p) {
    if (p.empty() || p[0] == '/') return p;
    const auto s = c.path.find_last_of('/');
    return s == std::string::npos ? p : c.path.substr(0, s + 1) + p;
}

namespace {
void num(const Json& j, const char* k, double& v) { if (j[k].type() == Json::Number) v = j[k].num(); }
void inum(const Json& j, const char* k, int& v) { if (j[k].type() == Json::Number) v = int(j[k].num()); }
void str(const Json& j, const char* k, std::string& v) { if (j[k].type() == Json::String) v = j[k].str(); }
bool vec3(const Json& j, Vec3& v) {
    if (j.size() != 3) return false;
    v = Vec3(j[std::size_t(0)].num(), j[std::size_t(1)].num(), j[std::size_t(2)].num());
    return true;
}
}  // namespace

bool load_tracker_config(const std::string& path, TrackerConfig& c, std::string& err) {
    const Json j = Json::parse_file(path, err);
    if (!err.empty()) return false;
    if (j["version"].num() != 1) { err = path + ": configuration version must be 1"; return false; }
    c.path = path;
    str(j, "pair_buffer", c.pair_buffer);
    str(j, "pose_buffer", c.pose_buffer);
    str(j, "cloud_buffer", c.cloud_buffer);
    str(j, "model_file", c.model_file);
    str(j, "mesh_file", c.mesh_file);
    c.model_file = config_relative(c, c.model_file);
    c.mesh_file = config_relative(c, c.mesh_file);
    const Json& cams = j["cameras"];
    for (int k = 0; k < 2; k++) {
        const Json& e = cams[c.cam_name[k]];
        str(e, "serial", c.cam_serial[k]);
        const Json& m = e["T_rig_from_camera_mm"];
        if (m.size() == 4) {
            for (int r = 0; r < 4; r++)
                for (int q = 0; q < 4; q++) c.cam_T[k](r, q) = m[std::size_t(r)][std::size_t(q)].num();
            c.cam_T[k].block<3, 1>(0, 3) /= 1000.0;
        } else {
            c.cam_T[k] = nominal_T_rig_cam(k + 1);
        }
    }
    str(cams, "calibration_file", c.calibration_file);
    if (!c.calibration_file.empty()) {
        c.calibration_file = config_relative(c, c.calibration_file);
        Camera tmp[2];
        tmp[0].name = c.cam_name[0];
        tmp[1].name = c.cam_name[1];
        if (!apply_calibration_file(c.calibration_file, tmp, err)) return false;
        c.cam_T[0] = tmp[0].T_rig_cam;
        c.cam_T[1] = tmp[1].T_rig_cam;
    }
    const Json& wv = j["work_volume"];
    if (!vec3(wv["min"], c.work_volume.min) || !vec3(wv["max"], c.work_volume.max)) {
        err = path + ": work_volume.min / max required";
        return false;
    }
    num(wv, "min_camera_distance", c.min_depth);
    num(wv, "max_camera_distance", c.max_depth);
    str(j, "background_map", c.background_map);
    c.background_map = config_relative(c, c.background_map);
    const Json& cl = j["cloud"];
    inum(cl, "max_points_per_camera", c.max_points_per_camera);
    inum(cl, "max_points_per_camera_search", c.max_points_per_camera_search);
    inum(cl, "solver_points_track", c.solver_points_track);
    inum(cl, "solver_points_search", c.solver_points_search);
    num(cl, "box_margin_m", c.box_margin);
    num(cl, "lost_box_factor", c.lost_box_factor);
    inum(cl, "pixel_margin", c.pixel_margin);
    const Json& icp = j["icp"];
    if (icp["distances_m"].size() > 0) {
        c.distances.clear();
        for (const auto& d : icp["distances_m"].elements()) c.distances.push_back(d.num());
    }
    str(icp, "kernel", c.kernel);
    num(icp, "tukey_k_m", c.tukey_k);
    num(icp, "tukey_k_ratio", c.tukey_k_ratio);
    inum(icp, "max_iterations", c.max_iterations);
    num(icp, "relative_fitness", c.rel_fitness);
    num(icp, "relative_rmse", c.rel_rmse);
    const Json& fc = j["fit_check"];
    inum(fc, "min_points", c.min_points);
    num(fc, "max_rmse_m", c.max_rmse);
    num(fc, "min_inlier_ratio", c.min_inlier_ratio);
    num(fc, "min_eigenvalue", c.min_eigenvalue);
    num(fc, "max_jump_m", c.max_jump_m);
    num(fc, "max_jump_rad", c.max_jump_rad);
    num(fc, "max_free_space_violation", c.max_violation);
    num(fc, "visibility_tolerance_m", c.visibility_tol);
    inum(fc, "visibility_points", c.visibility_points);
    inum(fc, "min_visibility_checked", c.min_visibility_checked);
    const Json& fl = j["filter"];
    num(fl, "accel_noise", c.accel_noise);
    num(fl, "min_meas_sigma_m", c.min_meas_sigma);
    num(fl, "orientation_alpha", c.ori_alpha);
    num(fl, "orientation_beta", c.ori_beta);
    const Json& st = j["states"];
    inum(st, "lost_after_rejected", c.lost_after);
    inum(st, "search_after_rejected", c.search_after);
    num(st, "search_time_limit_ms", c.search_time_ms);
    inum(st, "search_orientations", c.search_orientations);
    inum(st, "search_refine_best", c.search_refine_best);
    inum(st, "search_coarse_iterations", c.search_coarse_iterations);
    if (st["search_early_accept"].type() == Json::Bool) c.search_early_accept = st["search_early_accept"].boolean();
    num(st, "early_accept_min_fitness", c.early_min_fitness);
    num(st, "early_accept_max_violation", c.early_max_violation);
    num(j, "cloud_rate_hz", c.cloud_rate_hz);
    if (j["cloud_content"].type() == Json::String) c.cloud_scene = j["cloud_content"].str() != "solver";
    const Json& br = j["bridge"];
    str(br, "listen_address", c.bridge_address);
    inum(br, "port", c.bridge_port);
    if (br["queue_bytes"].type() == Json::Number) c.bridge_queue_bytes = long(br["queue_bytes"].num());
    num(br, "status_interval_s", c.bridge_status_s);
    num(j, "status_interval_s", c.status_interval_s);
    if (c.kernel != "tukey" && c.kernel != "l2") { err = path + ": icp.kernel must be tukey or l2"; return false; }
    if (c.distances.empty() || c.distances.size() > 8) { err = path + ": icp.distances_m needs 1 to 8 values"; return false; }
    return true;
}

}  // namespace trk
