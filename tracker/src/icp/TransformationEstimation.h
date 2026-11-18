// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Source: cpp/open3d/pipelines/registration/TransformationEstimation.h (point-to-plane only).
#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "PointCloud.h"
#include "RobustKernel.h"

namespace open3d {
namespace pipelines {
namespace registration {

typedef std::vector<Eigen::Vector2i> CorrespondenceSet;

enum class TransformationEstimationType {
    Unspecified = 0,
    PointToPoint = 1,
    PointToPlane = 2,
};

/// Base class that estimates a transformation between two point clouds.
class TransformationEstimation {
public:
    TransformationEstimation() {}
    virtual ~TransformationEstimation() {}
    virtual TransformationEstimationType GetTransformationEstimationType() const = 0;
    virtual double ComputeRMSE(const geometry::PointCloud &source,
                               const geometry::PointCloud &target,
                               const CorrespondenceSet &corres) const = 0;
    virtual Eigen::Matrix4d ComputeTransformation(const geometry::PointCloud &source,
                                                  const geometry::PointCloud &target,
                                                  const CorrespondenceSet &corres) const = 0;
};

/// Class to estimate a transformation for point to plane distance.
class TransformationEstimationPointToPlane : public TransformationEstimation {
public:
    TransformationEstimationPointToPlane() {}
    ~TransformationEstimationPointToPlane() override {}

    explicit TransformationEstimationPointToPlane(std::shared_ptr<RobustKernel> kernel)
        : kernel_(std::move(kernel)) {}

public:
    TransformationEstimationType GetTransformationEstimationType() const override { return type_; };
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(const geometry::PointCloud &source,
                                          const geometry::PointCloud &target,
                                          const CorrespondenceSet &corres) const override;

public:
    /// shared_ptr to an Abstract RobustKernel that could mutate at runtime.
    std::shared_ptr<RobustKernel> kernel_ = std::make_shared<L2Loss>();

private:
    const TransformationEstimationType type_ = TransformationEstimationType::PointToPlane;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
