// Tracking filter: constant-velocity Kalman filter for the position, alpha-beta filter on SO(3) for the orientation.
// Equations: DESIGN.md section 8.
#pragma once
#include <Eigen/Core>

#include "geometry.hpp"

namespace trk {

class PositionKF {
public:
    void init(const Vec3& p, const Mat3& R_meas, double vel_sigma) {
        x_.setZero();
        x_.head<3>() = p;
        P_.setZero();
        P_.block<3, 3>(0, 0) = R_meas;
        P_.block<3, 3>(3, 3) = Mat3::Identity() * vel_sigma * vel_sigma;
        ok_ = true;
    }
    void predict(double dt, double q) {
        if (dt <= 0) return;
        Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
        F.block<3, 3>(0, 3) = Mat3::Identity() * dt;
        Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
        Q.block<3, 3>(0, 0) = Mat3::Identity() * (q * dt * dt * dt / 3.0);
        Q.block<3, 3>(0, 3) = Mat3::Identity() * (q * dt * dt / 2.0);
        Q.block<3, 3>(3, 0) = Q.block<3, 3>(0, 3);
        Q.block<3, 3>(3, 3) = Mat3::Identity() * (q * dt);
        x_ = F * x_;
        P_ = F * P_ * F.transpose() + Q;
    }
    void update(const Vec3& z, const Mat3& R) {
        const Mat3 S = P_.block<3, 3>(0, 0) + R;
        const Eigen::Matrix<double, 6, 3> K = P_.block<6, 3>(0, 0) * S.inverse();
        x_ += K * (z - x_.head<3>());
        P_ = (Eigen::Matrix<double, 6, 6>::Identity() - K * (Eigen::Matrix<double, 3, 6>() << Mat3::Identity(), Mat3::Zero()).finished()) * P_;
        P_ = 0.5 * (P_ + P_.transpose());
    }
    void zero_velocity() { x_.tail<3>().setZero(); }
    Vec3 p() const { return x_.head<3>(); }
    Vec3 v() const { return x_.tail<3>(); }
    bool ok() const { return ok_; }
    void reset() { ok_ = false; }
private:
    Eigen::Matrix<double, 6, 1> x_ = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> P_ = Eigen::Matrix<double, 6, 6>::Identity();
    bool ok_ = false;
};

class OrientationFilter {
public:
    void init(const Mat3& R) { R_ = R; w_.setZero(); ok_ = true; }
    void predict(double dt) { if (dt > 0) R_ = so3_exp(w_ * dt) * R_; }
    // R_meas: measured orientation; the prediction for this time must have been applied.
    void update(const Mat3& R_meas, double dt, double alpha, double beta) {
        const Vec3 e = so3_log(R_meas * R_.transpose());
        R_ = so3_exp(alpha * e) * R_;
        if (dt > 1e-6) w_ += (beta / dt) * e;
    }
    void zero_rate() { w_.setZero(); }
    const Mat3& R() const { return R_; }
    const Vec3& w() const { return w_; }
    bool ok() const { return ok_; }
    void reset() { ok_ = false; }
private:
    Mat3 R_ = Mat3::Identity();
    Vec3 w_ = Vec3::Zero();
    bool ok_ = false;
};

// K orientations spread evenly over SO(3): farthest-point selection (quaternion distance) from 20000 random
// rotations with a fixed seed.
std::vector<Mat3> search_orientations(int k);

}  // namespace trk
