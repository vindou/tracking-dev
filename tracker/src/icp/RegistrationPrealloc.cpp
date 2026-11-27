// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).

#include "RegistrationPrealloc.h"

#include <algorithm>
#include <cmath>

#include "../common/prof.hpp"

namespace open3d {
namespace pipelines {
namespace registration {

namespace {

// PointCloud::Transform for the points only (same arithmetic).
inline void TransformPoints(const Eigen::Matrix4d &transformation, std::vector<Eigen::Vector3d> &points) {
    TRK_PROF(kTransform);
    for (auto &point : points) {
        Eigen::Vector4d new_point = transformation * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.head<3>() / new_point(3);
    }
}

// GetRegistrationResultAndCorrespondences into the workspace.
inline void GetResultAndCorrespondences(const std::vector<Eigen::Vector3d> &pcd,
                                        const geometry::KDTreeFlann &target_kdtree,
                                        double max_correspondence_distance,
                                        ICPWorkspace &ws,
                                        double &fitness,
                                        double &inlier_rmse) {
    TRK_PROF(kCorres);
    ws.corres.clear();
    ws.inlier.assign(pcd.size(), 0);
    double error2 = 0.0;
    if (ws.search == ICPWorkspace::kBounded) {
        for (std::size_t i = 0; i < pcd.size(); ++i) {
            int index = 0;
            double dist2 = 0.0;
            if (target_kdtree.SearchBounded1(pcd[i], max_correspondence_distance, ws.prev[i], index, dist2) > 0) {
                error2 += dist2;
                ws.corres.emplace_back(int(i), index);
                ws.inlier[i] = 1;
                ws.prev[i] = index;
            } else {
                ws.prev[i] = -1;
            }
        }
    } else if (ws.search == ICPWorkspace::kTableExact && ws.table) {
        for (std::size_t i = 0; i < pcd.size(); ++i) {
            int index = 0;
            double dist2 = 0.0;
            if (ws.table->SearchExact1(target_kdtree, pcd[i], max_correspondence_distance, index, dist2) > 0) {
                error2 += dist2;
                ws.corres.emplace_back(int(i), index);
                ws.inlier[i] = 1;
            }
        }
    } else if (ws.search == ICPWorkspace::kTable && ws.table) {
        for (std::size_t i = 0; i < pcd.size(); ++i) {
            int index = 0;
            double dist2 = 0.0;
            if (ws.table->Search1(pcd[i], max_correspondence_distance, index, dist2) > 0) {
                error2 += dist2;
                ws.corres.emplace_back(int(i), index);
                ws.inlier[i] = 1;
            }
        }
    } else {
    for (std::size_t i = 0; i < pcd.size(); ++i) {
        int index = 0;
        double dist2 = 0.0;
        if (target_kdtree.SearchHybrid1(pcd[i], max_correspondence_distance, index, dist2) > 0) {
            error2 += dist2;
            ws.corres.emplace_back(int(i), index);
            ws.inlier[i] = 1;
        }
    }
    }
    if (ws.corres.empty()) {
        fitness = 0.0;
        inlier_rmse = 0.0;
    } else {
        std::size_t corres_number = ws.corres.size();
        fitness = (double)corres_number / (double)pcd.size();
        inlier_rmse = std::sqrt(error2 / (double)corres_number);
    }
}

// TransformationEstimationPointToPlane::ComputeTransformation with utility::ComputeJTJandJTr inlined (same order).
inline Eigen::Matrix4d ComputeTransformationPointToPlane(const std::vector<Eigen::Vector3d> &source,
                                                         const geometry::PointCloud &target,
                                                         const std::vector<Eigen::Vector2i> &corres,
                                                         const RobustKernel &kernel,
                                                         RegistrationResultPrealloc &out) {
    if (corres.empty() || !target.HasNormals()) return Eigen::Matrix4d::Identity();
    Eigen::Matrix6d JTJ = Eigen::Matrix6d::Zero();
    Eigen::Vector6d JTr = Eigen::Vector6d::Zero();
    Eigen::Vector6d J_r = Eigen::Vector6d::Zero();
    double r = 0.0, w = 0.0, wsum = 0.0;
    Eigen::Vector3d csum = Eigen::Vector3d::Zero();
    {
    TRK_PROF(kAccumulate);
    for (int i = 0; i < (int)corres.size(); ++i) {
        const Eigen::Vector3d &vs = source[corres[i][0]];
        const Eigen::Vector3d &vt = target.points_[corres[i][1]];
        const Eigen::Vector3d &nt = target.normals_[corres[i][1]];
        r = (vs - vt).dot(nt);
        w = kernel.Weight(r);
        J_r.block<3, 1>(0, 0) = vs.cross(nt);
        J_r.block<3, 1>(3, 0) = nt;
        JTJ.noalias() += J_r * w * J_r.transpose();
        JTr.noalias() += J_r * w * r;
        wsum += w;
        csum += vs;
    }
    }
    TRK_PROF(kSolve);
    out.JTJ = JTJ;
    out.JTr = JTr;
    out.weight_sum = wsum;
    out.centroid = csum / double(corres.size());
    out.has_system = true;
    // utility::SolveJacobianSystemAndObtainExtrinsicMatrix -> SolveLinearSystemPSD (dense path: ldlt) with fixed-size
    // matrices (the original converts to dynamic matrices, which allocates).
    const Eigen::Vector6d x = JTJ.ldlt().solve(-JTr);
    return utility::TransformVector6dToMatrix4d(x);
}

}  // namespace

RegistrationResultPrealloc RegistrationICPPrealloc(const Eigen::Vector3d *source,
                                                   std::size_t n,
                                                   const geometry::PointCloud &target,
                                                   const geometry::KDTreeFlann &target_kdtree,
                                                   const double *distances,
                                                   int num_distances,
                                                   const Eigen::Matrix4d &init,
                                                   const RobustKernel &kernel,
                                                   const ICPConvergenceCriteria &criteria,
                                                   ICPWorkspace &ws) {
    const RobustKernel *kernels[8];
    for (int s = 0; s < num_distances && s < 8; s++) kernels[s] = &kernel;
    return RegistrationICPPrealloc(source, n, target, target_kdtree, distances, std::min(num_distances, 8), init,
                                   kernels, criteria, ws);
}

RegistrationResultPrealloc RegistrationICPPrealloc(const Eigen::Vector3d *source,
                                                   std::size_t n,
                                                   const geometry::PointCloud &target,
                                                   const geometry::KDTreeFlann &target_kdtree,
                                                   const double *distances,
                                                   int num_distances,
                                                   const Eigen::Matrix4d &init,
                                                   const RobustKernel *const *kernels,
                                                   const ICPConvergenceCriteria &criteria,
                                                   ICPWorkspace &ws) {
    RegistrationResultPrealloc result;
    result.transformation = init;
    ws.pcd.assign(source, source + n);
    if (ws.search == ICPWorkspace::kBounded) ws.prev.assign(n, -1);
    if (!init.isIdentity()) {
        TransformPoints(init, ws.pcd);
    }
    Eigen::Matrix4d transformation = init;
    for (int s = 0; s < num_distances; s++) {
        const double max_correspondence_distance = distances[s];
        if (max_correspondence_distance <= 0.0) break;
        double fitness = 0.0, rmse = 0.0;
        GetResultAndCorrespondences(ws.pcd, target_kdtree, max_correspondence_distance, ws, fitness, rmse);
        for (int i = 0; i < criteria.max_iteration_; i++) {
            Eigen::Matrix4d update = ComputeTransformationPointToPlane(ws.pcd, target, ws.corres, *kernels[s], result);
            transformation = update * transformation;
            TransformPoints(update, ws.pcd);
            const double backup_fitness = fitness, backup_rmse = rmse;
            GetResultAndCorrespondences(ws.pcd, target_kdtree, max_correspondence_distance, ws, fitness, rmse);
            result.num_iteration++;
            if (std::abs(backup_fitness - fitness) < criteria.relative_fitness_ &&
                std::abs(backup_rmse - rmse) < criteria.relative_rmse_) {
                break;
            }
        }
        result.fitness = fitness;
        result.inlier_rmse = rmse;
        result.num_corres = ws.corres.size();
    }
    result.transformation = transformation;
    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
