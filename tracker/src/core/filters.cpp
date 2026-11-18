#include "filters.hpp"

#include <random>
#include <vector>

namespace trk {

std::vector<Mat3> search_orientations(int k) {
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> nd;
    std::vector<Eigen::Quaterniond> c(20000);
    for (auto& q : c) {
        q = Eigen::Quaterniond(nd(rng), nd(rng), nd(rng), nd(rng));
        q.normalize();
    }
    c[0] = Eigen::Quaterniond::Identity();
    std::vector<double> d(c.size(), 1e9);
    std::vector<Mat3> out;
    std::size_t cur = 0;
    for (int i = 0; i < k; i++) {
        out.push_back(c[cur].toRotationMatrix());
        std::size_t best = 0;
        for (std::size_t j = 0; j < c.size(); j++) {
            const double dist = 1.0 - std::fabs(c[j].dot(c[cur]));   // 0 for the same rotation
            if (dist < d[j]) d[j] = dist;
            if (d[j] > d[best]) best = j;
        }
        cur = best;
    }
    return out;
}

}  // namespace trk
