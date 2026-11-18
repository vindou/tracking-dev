// Camera transforms (nominal or calibration file), the work volume (region both cameras see), random poses in it.
#pragma once
#include <random>
#include <string>

#include "camera.hpp"
#include "mesh.hpp"

namespace trk {

struct Box {
    Vec3 min = Vec3::Zero(), max = Vec3::Zero();
    bool contains(const Vec3& p) const {
        return p.x() >= min.x() && p.y() >= min.y() && p.z() >= min.z() && p.x() <= max.x() && p.y() <= max.y() &&
               p.z() <= max.z();
    }
};

// Calibration file (T5): {"version": 1, "cameras": {"D435i_1": {"T_rig_from_camera": [[4x4]] }, "D435i_2": ...}}.
// cams[i].name selects the entry.
bool apply_calibration_file(const std::string& path, Camera cams[2], std::string& err);

// True if p (rig frame) is in both images (margin in pixels) at depth in [min_depth, max_depth] of both cameras.
bool seen_by_both(const Camera cams[2], const Vec3& p, double min_depth, double max_depth, double margin_px);

// Work volume: axis-aligned box of the grid points (5 mm) seen by both cameras with depth in [min_depth, max_depth]
// and above z_min (rig plate). Also returns the fraction of the box that both cameras see.
Box compute_work_volume(const Camera cams[2], double min_depth, double max_depth, double z_min, double* seen_fraction);

// Random pose with the whole object bounding box seen by both cameras (depth >= 0.25 m) and above z_min.
Mat4 random_pose_in_view(const Camera cams[2], const Mesh& mesh, std::mt19937_64& rng, double z_min = 0.08,
                         double max_depth = 0.8);
Mat3 random_rotation(std::mt19937_64& rng);

}  // namespace trk
