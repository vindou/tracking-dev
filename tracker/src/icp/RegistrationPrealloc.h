// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Tracker optimization of RegistrationICP (Registration.cpp) with the point-to-plane step of
// TransformationEstimation.cpp, kept in the same structure and the same arithmetic order:
//  - the target tree is built once by the caller (KDTreeFlann) and passed in;
//  - no memory allocation in the call: the transformed source and the correspondences live in a preallocated
//    workspace, the JTJ reduction is a template loop instead of std::function;
//  - a sequence of correspondence distances (each stage starts from the previous result);
//  - the 6x6 matrix (JTJ, JTr) of the last iteration is returned, with the weight sum and the centroid of the matched
//    source points (normalization of the 6x6 matrix).
// With a single distance the result equals RegistrationICP (reference test tests/test_icp.cpp).
#pragma once

#include <Eigen/Core>
#include <vector>

#include "Eigen.h"
#include "KDTreeFlann.h"
#include "PointCloud.h"
#include "Registration.h"
#include "RobustKernel.h"

namespace open3d {
namespace pipelines {
namespace registration {

/// Preallocated work arrays (sized once with Reserve()).
struct ICPWorkspace {
    std::vector<Eigen::Vector3d> pcd;         ///< source transformed by the current estimate
    std::vector<Eigen::Vector2i> corres;      ///< correspondence set (source index, target index)
    std::vector<uint8_t> inlier;              ///< per source point: 1 if it has a correspondence (last evaluation)
    /// Allocates and writes the arrays once (resident from the start; the capacity stays after clear()).
    void Reserve(std::size_t max_points) {
        pcd.assign(max_points, Eigen::Vector3d::Zero());
        corres.assign(max_points, Eigen::Vector2i::Zero());
        inlier.assign(max_points, 0);
        pcd.clear();
        corres.clear();
        inlier.clear();
    }
};

/// Result of RegistrationICPPrealloc (RegistrationResult without the correspondence vector, plus the last 6x6 system).
struct RegistrationResultPrealloc {
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    double inlier_rmse = 0.0;
    double fitness = 0.0;
    std::size_t num_corres = 0;
    int num_iteration = 0;                    ///< sum over the distance stages
    bool has_system = false;                  ///< JTJ / JTr valid (at least one iteration)
    Eigen::Matrix6d JTJ = Eigen::Matrix6d::Zero();   ///< last iteration, parameters (rotation, translation)
    Eigen::Vector6d JTr = Eigen::Vector6d::Zero();
    double weight_sum = 0.0;                  ///< sum of the kernel weights of the last iteration
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();  ///< mean of the matched source points (target frame), last iteration
};

/// ICP over the distance sequence distances[0..num_distances-1] (each > 0). source: n points. ws must be reserved for
/// at least n points. Does not allocate. The same kernel is used in every stage.
RegistrationResultPrealloc RegistrationICPPrealloc(const Eigen::Vector3d *source,
                                                   std::size_t n,
                                                   const geometry::PointCloud &target,
                                                   const geometry::KDTreeFlann &target_kdtree,
                                                   const double *distances,
                                                   int num_distances,
                                                   const Eigen::Matrix4d &init,
                                                   const RobustKernel &kernel,
                                                   const ICPConvergenceCriteria &criteria,
                                                   ICPWorkspace &ws);

/// Same, with one kernel per distance stage (kernels[s] for distances[s]); tracker extension.
RegistrationResultPrealloc RegistrationICPPrealloc(const Eigen::Vector3d *source,
                                                   std::size_t n,
                                                   const geometry::PointCloud &target,
                                                   const geometry::KDTreeFlann &target_kdtree,
                                                   const double *distances,
                                                   int num_distances,
                                                   const Eigen::Matrix4d &init,
                                                   const RobustKernel *const *kernels,
                                                   const ICPConvergenceCriteria &criteria,
                                                   ICPWorkspace &ws);

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
