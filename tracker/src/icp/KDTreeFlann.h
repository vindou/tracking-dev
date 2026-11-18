// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Source: cpp/open3d/geometry/KDTreeFlann.h/.cpp: SetGeometry (points of a PointCloud), SearchHybrid; same nanoflann
// adaptor (KDTreeEigenMatrixAdaptor, column-major 3 x N, metric L2, leaf size 15) as the original.
#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "PointCloud.h"

namespace nanoflann {
struct metric_L2;
template <class MatrixType, int32_t DIM, class Distance, bool row_major>
struct KDTreeEigenMatrixAdaptor;
}  // namespace nanoflann

namespace open3d {
namespace geometry {

class KDTreeFlann {
public:
    KDTreeFlann();
    explicit KDTreeFlann(const PointCloud &geometry);
    ~KDTreeFlann();
    KDTreeFlann(const KDTreeFlann &) = delete;
    KDTreeFlann &operator=(const KDTreeFlann &) = delete;

    bool SetGeometry(const PointCloud &geometry);

    int SearchHybrid(const Eigen::Vector3d &query,
                     double radius,
                     int max_nn,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const;

    /// Allocation-free form for max_nn = 1 (tracker optimization; same result as SearchHybrid(query, radius, 1)).
    int SearchHybrid1(const Eigen::Vector3d &query, double radius, int &index, double &distance2) const;

protected:
    bool SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data);

    using KDTree_t = nanoflann::KDTreeEigenMatrixAdaptor<const Eigen::MatrixXd,
                                                         -1,
                                                         nanoflann::metric_L2,
                                                         false>;

    /// Tracker optimization: the same tree with the dimension fixed at compile time (3). nanoflann allocates a
    /// per-query vector when the dimension is dynamic (-1); with 3 the search is the same and allocation-free.
    /// (nanoflann takes the index dimension from the matrix type's compile-time rows, so the data is 3 x N fixed-row.)
    using Matrix3Xd = Eigen::Matrix<double, 3, Eigen::Dynamic>;
    using KDTree3_t = nanoflann::KDTreeEigenMatrixAdaptor<const Matrix3Xd, 3, nanoflann::metric_L2, false>;

    Eigen::MatrixXd data_;
    Matrix3Xd data3_;
    std::unique_ptr<KDTree_t> nanoflann_index_;
    std::unique_ptr<KDTree3_t> nanoflann_index3_;
};

}  // namespace geometry
}  // namespace open3d
