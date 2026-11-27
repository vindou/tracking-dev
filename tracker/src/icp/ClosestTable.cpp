#include "ClosestTable.h"

#include <algorithm>
#include <cmath>

namespace open3d {
namespace geometry {

void ClosestTable::Build(const PointCloud &target, const KDTreeFlann &tree, double cell, double max_distance, bool exact) {
    const std::vector<Eigen::Vector3d> &P = target.points_;
    points_ = P.data();
    cells_.clear();
    near_.clear();
    first_.clear();
    list_.clear();
    count_.clear();
    n_fallback_ = 0;
    if (P.empty() || !(cell > 0.0)) return;
    Eigen::Vector3d lo = P[0], hi = P[0];
    for (const auto &p : P) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    lo_ = lo - Eigen::Vector3d::Constant(max_distance + cell);
    inv_cell_ = 1.0 / cell;
    const Eigen::Vector3d ext = hi - lo + Eigen::Vector3d::Constant(2.0 * (max_distance + cell));
    nx_ = int(std::ceil(ext.x() / cell));
    ny_ = int(std::ceil(ext.y() / cell));
    nz_ = int(std::ceil(ext.z() / cell));
    cells_.assign(std::size_t(nx_) * std::size_t(ny_) * std::size_t(nz_), -1);
    const double half_diag = 0.5 * cell * std::sqrt(3.0);
    const double reach = max_distance + half_diag;   // a cell whose centre is farther than this from every point has no inlier
    if (exact) {
        first_.assign(cells_.size(), 0);
        count_.assign(cells_.size(), 0);
    }
    std::vector<int> li;
    std::vector<double> ld;
    std::size_t k = 0;
    for (int ix = 0; ix < nx_; ix++)
        for (int iy = 0; iy < ny_; iy++)
            for (int iz = 0; iz < nz_; iz++, k++) {
                const Eigen::Vector3d c = lo_ + Eigen::Vector3d(ix + 0.5, iy + 0.5, iz + 0.5) * cell;
                int index = 0;
                double d2 = 0.0;
                if (tree.SearchHybrid1(c, reach, index, d2) <= 0) continue;
                cells_[k] = index;
                if (!exact) continue;
                // every point within d + 2 h of the centre (a little more, for rounding); too many: the tree answers
                const double rad = std::sqrt(d2) + 2.0 * half_diag + 1e-9;
                const int n = tree.SearchHybrid(c, rad, kMaxList + 1, li, ld);
                if (n > kMaxList) {
                    n_fallback_++;
                    continue;
                }
                first_[k] = int32_t(list_.size());
                count_[k] = uint8_t(n);
                list_.insert(list_.end(), li.begin(), li.begin() + n);
            }
    // neighbours of every target point (the point itself excluded); fewer points than kNeighbours + 1: repeats
    const int nn = std::min<int>(kNeighbours + 1, int(P.size()));
    std::vector<int> idx;
    std::vector<double> dist;
    near_.assign(P.size() * kNeighbours, 0);
    for (std::size_t i = 0; i < P.size(); i++) {
        tree.SearchHybrid(P[i], 1e9, nn, idx, dist);
        int m = 0;
        for (int j : idx)
            if (j != int(i) && m < kNeighbours) near_[i * kNeighbours + std::size_t(m++)] = j;
        for (; m < kNeighbours; m++) near_[i * kNeighbours + std::size_t(m)] = int(i);
    }
}

}  // namespace geometry
}  // namespace open3d
