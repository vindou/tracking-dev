// Camera model: pinhole intrinsics of the depth stream (from the pair-buffer header), depth unit, and the transform
// T_rig_from_camera (camera frame = RealSense depth frame: x right, y down, z forward).
#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <string>

#include "geometry.hpp"

namespace trk {

struct Camera {
    std::string name;            // D435i_1 / D435i_2
    std::string serial;
    int width = 848, height = 480;
    double fx = 431.0, fy = 431.0, ppx = 424.0, ppy = 240.0;
    double depth_scale = 0.001;  // m per depth unit
    Mat4 T_rig_cam = Mat4::Identity();
    Mat4 T_cam_rig = Mat4::Identity();
    void set_pose(const Mat4& T) { T_rig_cam = T; T_cam_rig = inverse_pose(T); }
    // Project a camera-frame point; false if behind the camera.
    bool project(const Vec3& pc, double& u, double& v) const {
        if (pc.z() <= 1e-6) return false;
        u = fx * pc.x() / pc.z() + ppx;
        v = fy * pc.y() / pc.z() + ppy;
        return true;
    }
    Vec3 backproject(double u, double v, double z) const { return Vec3((u - ppx) / fx * z, (v - ppy) / fy * z, z); }
};

// Nominal transforms from the task (CAD, meters).
Mat4 nominal_T_rig_cam(int index /* 1 or 2 */);

}  // namespace trk
