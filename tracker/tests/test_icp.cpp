// Phase 3: reference test of the Open3D ICP port against the Open3D library (tools/icp_reference.py).
// Pass: transform difference < 1e-6 (matrix elements), the same iteration count, fitness and RMSE.
#include <cmath>
#include <cstdio>
#include <memory>

#include "../src/icp/Registration.h"
#include "icp_ref_data.hpp"
#include "test_framework.hpp"

using namespace open3d;
using namespace open3d::pipelines::registration;

namespace {
const IcpRefData& ref() {
    static IcpRefData d;
    static bool loaded = load_icp_ref(std::string(TRK_SOURCE_DIR) + "/tests/data/icp_ref.bin", d);
    if (!loaded) std::printf("  cannot read tests/data/icp_ref.bin\n");
    return d;
}
}  // namespace

TEST(icp_port_matches_open3d_reference) {
    const IcpRefData& d = ref();
    CHECK_EQ(d.cases.size(), 80u);
    geometry::PointCloud target;
    target.points_ = d.tp;
    target.normals_ = d.tn;
    double max_dT = 0, max_dfit = 0, max_drmse = 0;
    int iter_mismatch = 0, failures = 0;
    for (std::size_t c = 0; c < d.cases.size(); c++) {
        const IcpRefCase& k = d.cases[c];
        geometry::PointCloud source;
        source.points_ = k.src;
        std::shared_ptr<RobustKernel> kernel = k.kernel == 0 ? std::shared_ptr<RobustKernel>(std::make_shared<L2Loss>())
                                                             : std::make_shared<TukeyLoss>(k.k);
        TransformationEstimationPointToPlane est(kernel);
        const RegistrationResult r = RegistrationICP(source, target, k.max_dist, k.init, est,
                                                     ICPConvergenceCriteria(k.rel_fit, k.rel_rmse, int(k.max_iter)));
        const Eigen::Matrix4d T = r.transformation_;
        const double dT = (T - k.T).cwiseAbs().maxCoeff();
        const double dfit = std::fabs(r.fitness_ - k.fitness), drmse = std::fabs(r.inlier_rmse_ - k.rmse);
        max_dT = std::max(max_dT, dT);
        max_dfit = std::max(max_dfit, dfit);
        max_drmse = std::max(max_drmse, drmse);
        const bool it_ok = uint32_t(r.num_iteration_) == k.iters;
        if (!it_ok) iter_mismatch++;
        const bool ok = dT < 1e-6 && it_ok && dfit < 1e-9 && drmse < 1e-9 && r.correspondence_set_.size() == k.ncorr;
        if (!ok) failures++;
        std::printf("  case %2zu kernel %s noise %u: |dT|max %.2e  iterations %d/%u  fitness %.6f (d %.1e)  rmse %.6f mm (d %.1e)%s\n",
                    c, k.kernel ? "Tukey" : "L2   ", k.noise, dT, r.num_iteration_, k.iters, r.fitness_, dfit,
                    r.inlier_rmse_ * 1000, drmse, ok ? "" : "  <-- MISMATCH");
    }
    std::printf("  80 cases: max |dT| %.3e, max |d fitness| %.3e, max |d rmse| %.3e m, iteration mismatches %d, failed %d\n",
                max_dT, max_dfit, max_drmse, iter_mismatch, failures);
    CHECK_EQ(failures, 0);
    CHECK_LE(max_dT, 1e-6);
    CHECK_EQ(iter_mismatch, 0);
}

#include "../src/icp/RegistrationPrealloc.h"
#include "alloc_counter.hpp"

// The optimized ICP with one distance gives the reference result (same criteria as the port), and its call does not
// allocate memory.
TEST(icp_prealloc_matches_open3d_reference_and_does_not_allocate) {
    const IcpRefData& d = ref();
    geometry::PointCloud target;
    target.points_ = d.tp;
    target.normals_ = d.tn;
    geometry::KDTreeFlann tree(target);   // built once
    ICPWorkspace ws;
    ws.Reserve(8192);
    L2Loss l2;
    double max_dT = 0;
    int failures = 0;
    uint64_t allocs = 0;
    for (std::size_t c = 0; c < d.cases.size(); c++) {
        const IcpRefCase& k = d.cases[c];
        TukeyLoss tukey(k.k);
        const RobustKernel& kernel = k.kernel == 0 ? static_cast<const RobustKernel&>(l2) : tukey;
        const ICPConvergenceCriteria crit(k.rel_fit, k.rel_rmse, int(k.max_iter));
        const double dist = k.max_dist;
        alloc_count_start();
        const RegistrationResultPrealloc r =
                RegistrationICPPrealloc(k.src.data(), k.src.size(), target, tree, &dist, 1, k.init, kernel, crit, ws);
        allocs += alloc_count_stop();
        const double dT = (r.transformation - k.T).cwiseAbs().maxCoeff();
        max_dT = std::max(max_dT, dT);
        const bool ok = dT < 1e-6 && uint32_t(r.num_iteration) == k.iters && std::fabs(r.fitness - k.fitness) < 1e-9 &&
                        std::fabs(r.inlier_rmse - k.rmse) < 1e-9 && r.num_corres == k.ncorr && r.has_system;
        if (!ok) {
            failures++;
            std::printf("  case %zu MISMATCH: |dT| %.2e iterations %d/%u\n", c, dT, r.num_iteration, k.iters);
        }
    }
    std::printf("  80 cases: max |dT| %.3e, failed %d, heap allocations in the ICP calls: %llu%s\n", max_dT, failures,
                (unsigned long long)allocs, alloc_counting_available() ? "" : " (not counted in sanitizer builds)");
    CHECK_EQ(failures, 0);
    if (alloc_counting_available()) CHECK_EQ(allocs, 0u);
}

// Distance sequence 50 / 20 / 10 mm from a start up to 20 degrees / 30 mm away: converges to the true pose.
TEST(icp_prealloc_distance_sequence_converges) {
    const IcpRefData& d = ref();
    geometry::PointCloud target;
    target.points_ = d.tp;
    target.normals_ = d.tn;
    geometry::KDTreeFlann tree(target);
    ICPWorkspace ws;
    ws.Reserve(8192);
    TukeyLoss tukey(0.005);
    const double dist[3] = {0.05, 0.02, 0.01};
    double max_t = 0, max_r = 0;
    for (std::size_t c = 0; c < d.cases.size(); c += 4) {
        const IcpRefCase& k = d.cases[c];
        Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
        P.block<3, 3>(0, 0) = Eigen::AngleAxisd(0.35, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
        P.block<3, 1>(0, 3) = Eigen::Vector3d(0.02, -0.015, 0.015);
        const RegistrationResultPrealloc r = RegistrationICPPrealloc(k.src.data(), k.src.size(), target, tree, dist, 3,
                                                                     P * k.truth, tukey, ICPConvergenceCriteria(), ws);
        const Eigen::Matrix4d E = r.transformation * k.truth.inverse();
        const double et = E.block<3, 1>(0, 3).norm();
        const double er = std::acos(std::min(1.0, (E.block<3, 3>(0, 0).trace() - 1) / 2));
        max_t = std::max(max_t, et);
        max_r = std::max(max_r, er);
    }
    std::printf("  20 cases from 20 deg / 30 mm: max error %.3f mm, %.3f deg\n", max_t * 1000, max_r * 180 / M_PI);
    CHECK_LE(max_t, 0.002);
    CHECK_LE(max_r * 180 / M_PI, 1.0);
}
