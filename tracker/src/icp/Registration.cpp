// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Source: cpp/open3d/pipelines/registration/Registration.cpp (ICP part). The TBB parallel_reduce of the
// correspondence search is a sequential loop (correspondences in source index order).

#include "Registration.h"

#include <cmath>

namespace open3d {
namespace pipelines {
namespace registration {

static RegistrationResult GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const geometry::KDTreeFlann &target_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    (void)target;
    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    double error2 = 0.0;
    for (std::size_t i = 0; i < source.points_.size(); ++i) {
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        const auto &point = source.points_[i];
        if (target_kdtree.SearchHybrid(point, max_correspondence_distance, 1, indices, dists) > 0) {
            error2 += dists[0];
            result.correspondence_set_.emplace_back(int(i), indices[0]);
        }
    }

    if (result.correspondence_set_.empty()) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        size_t corres_number = result.correspondence_set_.size();
        result.fitness_ = (double)corres_number / (double)source.points_.size();
        result.inlier_rmse_ = std::sqrt(error2 / (double)corres_number);
    }
    return result;
}

RegistrationResult EvaluateRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation /* = Eigen::Matrix4d::Identity()*/) {
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    geometry::PointCloud pcd = source;
    if (!transformation.isIdentity()) {
        pcd.Transform(transformation);
    }
    return GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
}

RegistrationResult RegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimation &estimation,
        const ICPConvergenceCriteria &criteria /* = ICPConvergenceCriteria()*/) {
    if (max_correspondence_distance <= 0.0) {
        return RegistrationResult(init);   // original: utility::LogError (throws)
    }

    // InitializePointCloudsForTransformation of the point-to-plane estimation returns the inputs unchanged.
    Eigen::Matrix4d transformation = init;
    geometry::KDTreeFlann kdtree;
    const geometry::PointCloud &target_initialized = target;
    kdtree.SetGeometry(target_initialized);
    geometry::PointCloud pcd(source);

    if (!init.isIdentity()) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target_initialized, kdtree, max_correspondence_distance,
            transformation);
    int iterations = 0;
    for (int i = 0; i < criteria.max_iteration_; i++) {
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target_initialized, result.correspondence_set_);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target_initialized, kdtree, max_correspondence_distance,
                transformation);
        iterations = i + 1;
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }
    }
    result.num_iteration_ = iterations;
    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
