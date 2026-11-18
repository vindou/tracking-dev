// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Source: cpp/open3d/pipelines/registration/Registration.h (ICP part).
#pragma once

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "Eigen.h"
#include "KDTreeFlann.h"
#include "PointCloud.h"
#include "TransformationEstimation.h"

namespace open3d {
namespace pipelines {
namespace registration {

/// Class that defines the convergence criteria of ICP: ICP stops if the relative change of fitness and rmse hit
/// relative_fitness_ and relative_rmse_ individually, or the iteration number exceeds max_iteration_.
class ICPConvergenceCriteria {
public:
    ICPConvergenceCriteria(double relative_fitness = 1e-6,
                           double relative_rmse = 1e-6,
                           int max_iteration = 30)
        : relative_fitness_(relative_fitness),
          relative_rmse_(relative_rmse),
          max_iteration_(max_iteration) {}
    ~ICPConvergenceCriteria() {}

public:
    double relative_fitness_;
    double relative_rmse_;
    int max_iteration_;
};

/// Class that contains the registration results.
class RegistrationResult {
public:
    RegistrationResult(const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity())
        : transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0) {}
    ~RegistrationResult() {}

public:
    Eigen::Matrix4d_u transformation_;
    CorrespondenceSet correspondence_set_;
    double inlier_rmse_;
    double fitness_;
    /// Tracker addition (not in Open3D): number of transformation updates (ComputeTransformation calls).
    int num_iteration_ = 0;
};

/// Function for evaluating registration between point clouds.
RegistrationResult EvaluateRegistration(const geometry::PointCloud &source,
                                        const geometry::PointCloud &target,
                                        double max_correspondence_distance,
                                        const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity());

/// Function for ICP registration.
RegistrationResult RegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init,
        const TransformationEstimation &estimation,
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
