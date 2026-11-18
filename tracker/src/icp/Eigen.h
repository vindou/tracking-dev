// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Source: cpp/open3d/utility/Eigen.h (the functions used by the point-to-plane ICP).
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <functional>
#include <tuple>

namespace Eigen {
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 4, 4, Eigen::DontAlign> Matrix4d_u;
}  // namespace Eigen

namespace open3d {
namespace utility {

/// Function to solve Ax=b (dense path of the original; the sparse path is not ported).
std::tuple<bool, Eigen::VectorXd> SolveLinearSystemPSD(const Eigen::MatrixXd &A,
                                                       const Eigen::VectorXd &b,
                                                       bool prefer_sparse = false,
                                                       bool check_symmetric = false,
                                                       bool check_det = false,
                                                       bool check_psd = false);

/// Function to transform 6D motion vector to 4D motion matrix
/// Reference:
/// https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html#TutorialGeoTransform
Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d &input);

/// Function to solve Jacobian system
/// Input: 6x6 Jacobian matrix and 6-dim residual vector.
/// Output: tuple of is_success, 4x4 extrinsic matrices.
std::tuple<bool, Eigen::Matrix4d> SolveJacobianSystemAndObtainExtrinsicMatrix(
        const Eigen::Matrix6d &JTJ, const Eigen::Vector6d &JTr);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.
template <typename MatType, typename VecType>
std::tuple<MatType, VecType, double> ComputeJTJandJTr(
        std::function<void(int, VecType &, double &, double &)> f,
        int iteration_num,
        bool verbose = true);

}  // namespace utility
}  // namespace open3d
