// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Source: cpp/open3d/utility/Eigen.cpp. The TBB parallel_reduce of ComputeJTJandJTr is a sequential loop here
// (same sums in index order).

#include "Eigen.h"

#include <Eigen/Dense>

namespace open3d {
namespace utility {

std::tuple<bool, Eigen::VectorXd> SolveLinearSystemPSD(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        bool /*prefer_sparse = false */,
        bool check_symmetric /* = false */,
        bool check_det /* = false */,
        bool check_psd /* = false */) {
    // PSD implies symmetric
    check_symmetric = check_symmetric || check_psd;
    if (check_symmetric && !A.isApprox(A.transpose())) {
        return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
    }

    if (check_det) {
        double det = A.determinant();
        if (fabs(det) < 1e-6 || std::isnan(det) || std::isinf(det)) {
            return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
        }
    }

    // Check PSD: https://stackoverflow.com/a/54569657/1255535
    if (check_psd) {
        Eigen::LLT<Eigen::MatrixXd> A_llt(A);
        if (A_llt.info() == Eigen::NumericalIssue) {
            return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
        }
    }

    Eigen::VectorXd x(b.size());
    x = A.ldlt().solve(b);
    return std::make_tuple(true, std::move(x));
}

Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d &input) {
    Eigen::Matrix4d output;
    output.setIdentity();
    output.block<3, 3>(0, 0) =
            (Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX()))
                    .matrix();
    output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
    return output;
}

std::tuple<bool, Eigen::Matrix4d> SolveJacobianSystemAndObtainExtrinsicMatrix(
        const Eigen::Matrix6d &JTJ, const Eigen::Vector6d &JTr) {
    bool solution_exist;
    Eigen::Vector6d x;
    std::tie(solution_exist, x) = SolveLinearSystemPSD(JTJ, -JTr);

    if (solution_exist) {
        Eigen::Matrix4d extrinsic = TransformVector6dToMatrix4d(x);
        return std::make_tuple(solution_exist, std::move(extrinsic));
    }
    return std::make_tuple(false, Eigen::Matrix4d::Identity());
}

template <typename MatType, typename VecType>
std::tuple<MatType, VecType, double> ComputeJTJandJTr(
        std::function<void(int, VecType &, double &, double &)> f,
        int iteration_num,
        bool /*verbose = true*/) {
    MatType JTJ = MatType::Zero();
    VecType JTr = VecType::Zero();
    double r2_sum = 0.0;
    VecType J_r = VecType::Zero();
    double r = 0.0;
    double w = 0.0;
    for (int i = 0; i < iteration_num; ++i) {
        f(i, J_r, r, w);
        JTJ.noalias() += J_r * w * J_r.transpose();
        JTr.noalias() += J_r * w * r;
        r2_sum += r * r;
    }
    return std::make_tuple(std::move(JTJ), std::move(JTr), r2_sum);
}

template std::tuple<Eigen::Matrix6d, Eigen::Vector6d, double> ComputeJTJandJTr(
        std::function<void(int, Eigen::Vector6d &, double &, double &)> f,
        int iteration_num, bool verbose);

}  // namespace utility
}  // namespace open3d
