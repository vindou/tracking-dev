// Tracker configuration (one JSON file, config/tracker.json). Relative paths are relative to the file's directory.
#pragma once
#include <string>
#include <vector>

#include "calib.hpp"

namespace trk {

struct TrackerConfig {
    std::string path;
    std::string pair_buffer = "/d435_pairs", pose_buffer = "/trk_pose", cloud_buffer = "/trk_cloud";
    std::string model_file, mesh_file;
    std::string cam_name[2] = {"D435i_1", "D435i_2"};
    std::string cam_serial[2];
    Mat4 cam_T[2] = {Mat4::Identity(), Mat4::Identity()};   // T_rig_from_camera (m), nominal or calibrated
    std::string calibration_file;
    Box work_volume;
    double min_depth = 0.2, max_depth = 0.8;
    std::string background_map;
    // cloud
    int max_points_per_camera = 2500, max_points_per_camera_search = 45000, solver_points_track = 800, solver_points_search = 400;
    double box_margin = 0.015, lost_box_factor = 2.5;
    int pixel_margin = 12;
    // icp
    std::vector<double> distances = {0.05, 0.02, 0.01};
    std::string kernel = "tukey";
    double tukey_k = 0.005, tukey_k_ratio = 0.5;   // Tukey k per stage = max(tukey_k, ratio * distance)
    int max_iterations = 30;
    double rel_fitness = 1e-6, rel_rmse = 1e-6;
    // fit check
    int min_points = 150;
    double max_rmse = 0.004, min_inlier_ratio = 0.6, min_eigenvalue = 0.01, max_jump_m = 0.03, max_jump_rad = 0.35;
    double max_violation = 0.15, visibility_tol = 0.01;
    int visibility_points = 600, min_visibility_checked = 50;
    // filter
    double accel_noise = 4.0, min_meas_sigma = 0.0003, ori_alpha = 0.6, ori_beta = 0.1;
    // states
    int lost_after = 3, search_after = 45;
    double search_time_ms = 8.0;
    int search_orientations = 72, search_refine_best = 4, search_coarse_iterations = 12;
    bool search_early_accept = false;
    double early_min_fitness = 0.95, early_max_violation = 0.05;
    double cloud_rate_hz = 30, status_interval_s = 10;
    // external pose hypotheses (DESIGN.md 13): read from this ring while in search state; empty name = off
    std::string hint_buffer = "/trk_hints";
    double hint_max_age_ms = 400;      // hypotheses computed from a pair older than this are not used
    bool cloud_scene = true;   // cloud buffer: the filtered cloud (true) or only the solver's points (false)
    // telemetry bridge
    std::string bridge_address = "192.168.55.1";
    int bridge_port = 9870;
    long bridge_queue_bytes = 1048576;
    double bridge_status_s = 1.0;
};

// Loads the file (missing keys keep the defaults above). Applies the calibration file if one is named.
bool load_tracker_config(const std::string& path, TrackerConfig& c, std::string& err);
// Resolves a path relative to the configuration file's directory.
std::string config_relative(const TrackerConfig& c, const std::string& p);

}  // namespace trk
