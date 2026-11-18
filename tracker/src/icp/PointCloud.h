// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Minimal replacement of cpp/open3d/geometry/PointCloud.h: points, normals, Transform (TransformPoints and
// TransformNormals of cpp/open3d/geometry/Geometry3D.cpp).
#pragma once

#include <Eigen/Core>
#include <vector>

namespace open3d {
namespace geometry {

class PointCloud {
public:
    PointCloud() = default;
    bool HasNormals() const { return !points_.empty() && normals_.size() == points_.size(); }
    PointCloud &Transform(const Eigen::Matrix4d &transformation) {
        for (auto &point : points_) {
            Eigen::Vector4d new_point = transformation * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
            point = new_point.head<3>() / new_point(3);
        }
        for (auto &normal : normals_) {
            Eigen::Vector4d new_normal = transformation * Eigen::Vector4d(normal(0), normal(1), normal(2), 0.0);
            normal = new_normal.head<3>();
        }
        return *this;
    }

public:
    std::vector<Eigen::Vector3d> points_;
    std::vector<Eigen::Vector3d> normals_;
};

}  // namespace geometry
}  // namespace open3d
