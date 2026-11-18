// Rigid-body helpers: quaternions (w, x, y, z), SO(3) exp/log, pose composition.
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace trk {

using Mat4 = Eigen::Matrix4d;
using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

inline Mat4 make_pose(const Mat3& R, const Vec3& t) {
    Mat4 T = Mat4::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}
inline Mat4 inverse_pose(const Mat4& T) {
    const Mat3 R = T.block<3, 3>(0, 0);
    return make_pose(R.transpose(), -R.transpose() * T.block<3, 1>(0, 3));
}
inline Vec3 xform(const Mat4& T, const Vec3& p) { return T.block<3, 3>(0, 0) * p + T.block<3, 1>(0, 3); }

// Rotation vector (axis * angle) <-> rotation matrix.
inline Mat3 so3_exp(const Vec3& w) {
    const double a = w.norm();
    if (a < 1e-12) return Mat3::Identity() + (Mat3() << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0).finished();
    return Eigen::AngleAxisd(a, w / a).toRotationMatrix();
}
inline Vec3 so3_log(const Mat3& R) {
    const Eigen::AngleAxisd aa(R);
    return aa.axis() * aa.angle();
}
inline double rotation_angle(const Mat3& R) {
    const double c = std::max(-1.0, std::min(1.0, (R.trace() - 1.0) / 2.0));
    return std::acos(c);
}
// Quaternion w, x, y, z (w >= 0).
inline void quat_from_R(const Mat3& R, double q[4]) {
    Eigen::Quaterniond e(R);
    e.normalize();
    if (e.w() < 0) e.coeffs() = -e.coeffs();
    q[0] = e.w(); q[1] = e.x(); q[2] = e.y(); q[3] = e.z();
}
inline Mat3 R_from_quat(const double q[4]) {
    return Eigen::Quaterniond(q[0], q[1], q[2], q[3]).normalized().toRotationMatrix();
}

}  // namespace trk
