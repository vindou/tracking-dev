// Tracker extension (not part of the Open3D port): correspondences from a table instead of the KD-tree.
// A regular grid around the target cloud. Cells from which no target point can be within max_distance hold -1.
// Two forms (icp.correspondence_search; the KD-tree stays the default):
//   "table_exact"  Per cell, the list of every target point that can be the closest one for some query in the cell:
//                  with d the distance from the cell centre c to the cloud and h the half diagonal, the closest point
//                  p of a query q in the cell has |p - c| <= |p - q| + h <= (d + h) + h, so the list is all points
//                  within d + 2 h of c. A query takes the minimum over the list: the closest point, as the tree
//                  finds it. Cells whose list would be longer than kMaxList (far from the surface) are answered by
//                  the bounded tree search instead. Same correspondences as the KD-tree path.
//   "table"        Per cell only the point closest to the centre; a query takes the best among that point and its
//                  kNeighbours nearest neighbours in the cloud. Faster and smaller, but the answer is not always the
//                  closest point, so poses can differ slightly.
#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <vector>

#include "KDTreeFlann.h"
#include "PointCloud.h"

namespace open3d {
namespace geometry {

class ClosestTable {
public:
    static constexpr int kNeighbours = 8;
    static constexpr int kMaxList = 24;
    /// Builds the table (allocates; call once). cell: edge length; max_distance: largest correspondence distance;
    /// exact: also the lists of the exact form.
    void Build(const PointCloud &target, const KDTreeFlann &tree, double cell, double max_distance, bool exact = false);
    bool empty() const { return cells_.empty(); }
    std::size_t bytes() const {
        return (cells_.size() + near_.size() + list_.size() + first_.size()) * sizeof(int32_t) + count_.size();
    }
    double fallback_share() const { return cells_.empty() ? 0.0 : double(n_fallback_) / double(cells_.size()); }

    /// The exact form. As KDTreeFlann::SearchHybrid1 (radius <= max_distance of Build).
    inline int SearchExact1(const KDTreeFlann &tree, const Eigen::Vector3d &q, double radius, int &index, double &distance2) const {
        const Eigen::Vector3d u = (q - lo_) * inv_cell_;
        if (!(u.x() >= 0.0 && u.y() >= 0.0 && u.z() >= 0.0)) return 0;
        const int ix = int(u.x()), iy = int(u.y()), iz = int(u.z());
        if (ix >= nx_ || iy >= ny_ || iz >= nz_) return 0;
        const std::size_t cell = (std::size_t(ix) * std::size_t(ny_) + std::size_t(iy)) * std::size_t(nz_) + std::size_t(iz);
        if (cells_[cell] < 0) return 0;
        const int n = count_[cell];
        if (n == 0) return tree.SearchBounded1(q, radius, -1, index, distance2);
        const Eigen::Vector3d *P = points_;
        const int32_t *l = &list_[std::size_t(first_[cell])];
        int best = l[0];
        double d2 = 1e300;
        for (int k = 0; k < n; k++) {
            const Eigen::Vector3d &p = P[l[k]];
            const double dx = q.x() - p.x(), dy = q.y() - p.y(), dz = q.z() - p.z();
            const double d = dx * dx + dy * dy + dz * dz;
            if (d < d2) { d2 = d; best = l[k]; }
        }
        if (!(d2 < radius * radius)) return 0;
        index = best;
        distance2 = d2;
        return 1;
    }

    /// As KDTreeFlann::SearchHybrid1: 1 and (index, squared distance) if a point lies within radius, else 0.
    inline int Search1(const Eigen::Vector3d &q, double radius, int &index, double &distance2) const {
        const Eigen::Vector3d u = (q - lo_) * inv_cell_;
        if (!(u.x() >= 0.0 && u.y() >= 0.0 && u.z() >= 0.0)) return 0;
        const int ix = int(u.x()), iy = int(u.y()), iz = int(u.z());
        if (ix >= nx_ || iy >= ny_ || iz >= nz_) return 0;
        const int32_t seed = cells_[(std::size_t(ix) * std::size_t(ny_) + std::size_t(iy)) * std::size_t(nz_) + std::size_t(iz)];
        if (seed < 0) return 0;
        const Eigen::Vector3d *P = points_;
        int best = seed;
        double d2 = (P[seed] - q).squaredNorm();
        const int32_t *nb = &near_[std::size_t(seed) * kNeighbours];
        for (int k = 0; k < kNeighbours; k++) {
            const double d = (P[nb[k]] - q).squaredNorm();
            if (d < d2) { d2 = d; best = nb[k]; }
        }
        if (!(d2 < radius * radius)) return 0;
        index = best;
        distance2 = d2;
        return 1;
    }

private:
    const Eigen::Vector3d *points_ = nullptr;   // the target's points (the target must outlive the table)
    Eigen::Vector3d lo_ = Eigen::Vector3d::Zero();
    double inv_cell_ = 0.0;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    std::vector<int32_t> cells_;
    std::vector<int32_t> near_;                 // per target point: its kNeighbours nearest other points
    // exact form: per cell the first entry and the length of its list in list_ (length 0: use the tree)
    std::vector<int32_t> first_, list_;
    std::vector<uint8_t> count_;
    std::size_t n_fallback_ = 0;
};

}  // namespace geometry
}  // namespace open3d
