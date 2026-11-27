// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).

#include "KDTreeFlann.h"

#include <algorithm>
#include <nanoflann.hpp>

namespace open3d {
namespace geometry {

KDTreeFlann::KDTreeFlann() {}

KDTreeFlann::KDTreeFlann(const PointCloud &geometry) { SetGeometry(geometry); }

KDTreeFlann::~KDTreeFlann() {}

bool KDTreeFlann::SetGeometry(const PointCloud &geometry) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
            (const double *)((const PointCloud &)geometry).points_.data(), 3,
            ((const PointCloud &)geometry).points_.size()));
}

bool KDTreeFlann::SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data) {
    if (data.size() == 0) {
        return false;
    }
    data_ = data;
    nanoflann_index_ = std::make_unique<KDTree_t>(data_.rows(), data_, 15);
    nanoflann_index_->index_->buildIndex();
    data3_ = data_;
    nanoflann_index3_ = std::make_unique<KDTree3_t>(3, data3_, 15);
    nanoflann_index3_->index_->buildIndex();
    return true;
}

int KDTreeFlann::SearchHybrid(const Eigen::Vector3d &query,
                              double radius,
                              int max_nn,
                              std::vector<int> &indices,
                              std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // It is also the recommended setting for search.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory allocation/deallocation.
    if (data_.size() == 0 || query.rows() != data_.rows() || max_nn < 0) {
        return -1;
    }
    distance2.resize(max_nn);
    std::vector<Eigen::Index> indices_eigen(max_nn);
    int k = nanoflann_index_->index_->knnSearch(
            query.data(), max_nn, indices_eigen.data(), distance2.data());
    k = std::distance(distance2.begin(),
                      std::lower_bound(distance2.begin(), distance2.begin() + k,
                                       radius * radius));
    indices.resize(k);
    distance2.resize(k);
    std::copy_n(indices_eigen.begin(), k, indices.begin());
    return k;
}

int KDTreeFlann::SearchHybrid1(const Eigen::Vector3d &query, double radius, int &index, double &distance2) const {
    if (data_.size() == 0) return -1;
    Eigen::Index idx = 0;
    double d2 = 0;
    const int k = int(nanoflann_index3_->index_->knnSearch(query.data(), 1, &idx, &d2));
    if (k < 1 || !(d2 < radius * radius)) return 0;   // lower_bound(..., r^2): kept only if d2 < r^2
    index = int(idx);
    distance2 = d2;
    return 1;
}

namespace {
// Result set for one neighbour with an initial bound (interface of nanoflann's result sets).
struct BoundedNearest {
    using DistanceType = double;
    using IndexType = Eigen::Index;
    double best;
    Eigen::Index idx;
    std::size_t size() const { return idx >= 0 ? 1 : 0; }
    bool full() const { return true; }
    bool addPoint(double dist, Eigen::Index index) {
        if (dist < best) {
            best = dist;
            idx = index;
        }
        return true;
    }
    double worstDist() const { return best; }
};
}  // namespace

int KDTreeFlann::SearchBounded1(const Eigen::Vector3d &query, double radius, int seed, int &index, double &distance2) const {
    if (data_.size() == 0) return -1;
    BoundedNearest rs{radius * radius, -1};
    if (seed >= 0) {   // the same distance function as the search uses, so that the values compare exactly
        const double ds = nanoflann_index3_->index_->distance_.evalMetric(query.data(), Eigen::Index(seed), 3);
        if (ds < rs.best) {
            rs.best = ds;
            rs.idx = seed;
        }
    }
    nanoflann_index3_->index_->findNeighbors(rs, query.data());
    if (rs.idx < 0) return 0;
    index = int(rs.idx);
    distance2 = rs.best;
    return 1;
}

}  // namespace geometry
}  // namespace open3d
