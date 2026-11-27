// The switchable CPU paths (DESIGN.md section 14): the bounded KD-tree search gives the same correspondences and the
// same ICP result as the port; the closest-point table agrees with the tree except for a small share of queries;
// hint trials on the pool give the same pose record as one after another and allocate nothing more.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "../src/icp/RegistrationPrealloc.h"
#include "alloc_counter.hpp"
#include "icp_ref_data.hpp"
#include "synth_harness.hpp"
#include "test_framework.hpp"

using namespace open3d;
using namespace open3d::pipelines::registration;
using namespace trkt;

namespace {
const IcpRefData& ref2() {
    static IcpRefData d;
    static bool loaded = load_icp_ref(std::string(TRK_SOURCE_DIR) + "/tests/data/icp_ref.bin", d);
    (void)loaded;
    return d;
}
}  // namespace

TEST(bounded_search_equals_kdtree_search) {
    const IcpRefData& d = ref2();
    geometry::PointCloud target;
    target.points_ = d.tp;
    target.normals_ = d.tn;
    geometry::KDTreeFlann tree(target);
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> u(-1, 1);
    std::uniform_int_distribution<int> pick(0, int(d.tp.size()) - 1), seed(-1, int(d.tp.size()) - 1);
    const double radii[4] = {0.05, 0.02, 0.01, 0.002};
    int diff = 0, found = 0;
    const int N = 200000;
    for (int k = 0; k < N; k++) {
        const double r = radii[k % 4];
        const Eigen::Vector3d q = d.tp[std::size_t(pick(rng))] + Eigen::Vector3d(u(rng), u(rng), u(rng)) * (k % 3 == 0 ? 0.08 : 0.01);
        int ia = -1, ib = -1;
        double da = -1, db = -1;
        const int na = tree.SearchHybrid1(q, r, ia, da), nb = tree.SearchBounded1(q, r, seed(rng), ib, db);
        found += na;
        if (na != nb || (na == 1 && (ia != ib || da != db))) diff++;
    }
    std::printf("  %d queries (%d with a point in range): %d differ\n", N, found, diff);
    CHECK_EQ(diff, 0);
    CHECK_GE(found, N / 4);
}

TEST(icp_bounded_search_equals_port) {
    const IcpRefData& d = ref2();
    geometry::PointCloud target;
    target.points_ = d.tp;
    target.normals_ = d.tn;
    geometry::KDTreeFlann tree(target);
    ICPWorkspace wa, wb;
    std::size_t maxn = 0;
    for (const auto& k : d.cases) maxn = std::max(maxn, k.src.size());
    wa.Reserve(maxn);
    wb.Reserve(maxn);
    wb.search = ICPWorkspace::kBounded;
    const double dist[3] = {0.05, 0.02, 0.01};
    int diff = 0;
    for (const IcpRefCase& k : d.cases) {
        const TukeyLoss kernel(0.005);
        const ICPConvergenceCriteria crit(1e-5, 1e-5, 30);
        const RegistrationResultPrealloc a = RegistrationICPPrealloc(k.src.data(), k.src.size(), target, tree, dist, 3, k.init, kernel, crit, wa);
        const RegistrationResultPrealloc b = RegistrationICPPrealloc(k.src.data(), k.src.size(), target, tree, dist, 3, k.init, kernel, crit, wb);
        if (!(a.transformation == b.transformation) || a.num_iteration != b.num_iteration || a.fitness != b.fitness ||
            a.inlier_rmse != b.inlier_rmse || !(a.JTJ == b.JTJ) || wa.inlier != wb.inlier)
            diff++;
    }
    std::printf("  %zu cases, 3 distance stages: %d differ from the KD-tree path (bit for bit)\n", d.cases.size(), diff);
    CHECK_EQ(diff, 0);
}

TEST(exact_table_equals_kdtree_search) {
    const IcpRefData& d = ref2();
    geometry::PointCloud target;
    target.points_ = d.tp;
    target.normals_ = d.tn;
    geometry::KDTreeFlann tree(target);
    geometry::ClosestTable table;
    table.Build(target, tree, 0.002, 0.05, true);
    std::mt19937_64 rng(13);
    std::uniform_real_distribution<double> u(-1, 1);
    std::uniform_int_distribution<int> pick(0, int(d.tp.size()) - 1);
    const double radii[3] = {0.05, 0.02, 0.01};
    int diff = 0, found = 0;
    double worst = 0;
    const int N = 400000;
    for (int k = 0; k < N; k++) {
        const double r = radii[k % 3];
        const Eigen::Vector3d q = d.tp[std::size_t(pick(rng))] + Eigen::Vector3d(u(rng), u(rng), u(rng)) * (k % 4 == 0 ? 0.09 : k % 4 == 1 ? 0.03 : 0.006);
        int ia = -1, ib = -1;
        double da = 0, db = 0;
        const int na = tree.SearchHybrid1(q, r, ia, da), nb = table.SearchExact1(tree, q, r, ib, db);
        found += na;
        if (na != nb || (na == 1 && ia != ib)) diff++;
        if (na == 1 && nb == 1) worst = std::max(worst, std::fabs(da - db) / da);
    }
    std::printf("  table %.1f MB, %.1f %% of the cells answered by the tree; %d queries (%d with a point in range): %d differ; largest "
                "relative difference of the squared distance %.1e\n", double(table.bytes()) / 1e6, 100.0 * table.fallback_share(), N, found, diff, worst);
    CHECK_EQ(diff, 0);
    CHECK_LE(worst, 1e-12);
}

TEST(closest_table_agrees_with_kdtree) {
    const IcpRefData& d = ref2();
    geometry::PointCloud target;
    target.points_ = d.tp;
    target.normals_ = d.tn;
    geometry::KDTreeFlann tree(target);
    geometry::ClosestTable table;
    table.Build(target, tree, 0.002, 0.05);
    std::mt19937_64 rng(12);
    std::uniform_real_distribution<double> u(-1, 1);
    std::uniform_int_distribution<int> pick(0, int(d.tp.size()) - 1);
    int both = 0, same = 0, only_tree = 0, only_table = 0;
    double worst = 0;
    const int N = 200000;
    for (int k = 0; k < N; k++) {
        const double r = k % 2 ? 0.05 : 0.01;
        const Eigen::Vector3d q = d.tp[std::size_t(pick(rng))] + Eigen::Vector3d(u(rng), u(rng), u(rng)) * (k % 3 == 0 ? 0.06 : 0.008);
        int ia = -1, ib = -1;
        double da = 0, db = 0;
        const int na = tree.SearchHybrid1(q, r, ia, da), nb = table.Search1(q, r, ib, db);
        if (na && nb) {
            both++;
            same += ia == ib;
            worst = std::max(worst, std::sqrt(db) - std::sqrt(da));
        } else if (na) only_tree++;
        else if (nb) only_table++;
    }
    std::printf("  table %.1f MB; %d queries in range of both: same point %.2f %%, largest extra distance %.2f mm; in range only of the tree %d, "
                "only of the table %d\n", double(table.bytes()) / 1e6, both, 100.0 * same / std::max(1, both), worst * 1e3, only_tree, only_table);
    CHECK_GE(double(same), 0.97 * both);
    CHECK_EQ(only_table, 0);            // the table never reports a point the tree would not
    CHECK_LE(worst, 0.004);
    CHECK_LE(double(only_tree), 0.01 * N);
}

// Hint trials on the pool against one after another: the same pose record (all fields that do not hold times), and
// no additional allocation.
TEST(hint_trials_in_parallel_equal_sequential) {
    std::mt19937_64 rng(505);
    Camera cams[2];
    default_cameras(cams);
    int differ = 0, accepted = 0;
    uint64_t allocs[2] = {0, 0};   // in the step (rendering of the pair included), 1 thread and 4
    const int N = 12;
    for (int k = 0; k < N; k++) {
        SynthScenario s;
        s.T0 = random_pose_in_view(cams, object_mesh(), rng);
        s.render.noise_sigma_m = 0.001;
        s.seed = uint64_t(5000 + k);
        Mat4 hint[12];
        std::uniform_real_distribution<double> u(-1, 1);
        for (int j = 0; j < 12; j++) {   // near, far and useless hypotheses in a fixed random order
            const Eigen::AngleAxisd aa((j % 3 == 0 ? 10 : 70) * M_PI / 180 * u(rng), Vec3(u(rng), u(rng), u(rng) + 1.5).normalized());
            hint[j] = s.T0 * make_pose(aa.toRotationMatrix(), Vec3(u(rng), u(rng), u(rng)) * (j % 4 == 3 ? 0.2 : 0.012));
        }
        PoseRecord rec[2];
        for (int threads = 1; threads <= 4; threads += 3) {
            TrackerConfig cfg = test_config();
            cfg.hint_threads = threads;
            Harness h(s, cfg);
            h.core().set_hints(hint, 12, (k % 2) == 1);
            alloc_count_start();
            rec[threads > 1] = h.step(0);
            allocs[threads > 1] += alloc_count_stop();
        }
        for (PoseRecord& p : rec) std::memset(p.stage_us, 0, sizeof p.stage_us);
        if (std::memcmp(&rec[0], &rec[1], sizeof(PoseRecord)) != 0) differ++;
        accepted += rec[0].by_hint;
    }
    std::printf("  %d scenes x 12 hypotheses, 1 thread against 4: pose records that differ %d; accepted from a hypothesis %d; "
                "allocations in the step: %llu with 1 thread, %llu with the pool%s\n", N, differ, accepted, (unsigned long long)allocs[0],
                (unsigned long long)allocs[1],
                alloc_counting_available() ? "" : " (not counted in sanitizer builds)");
    CHECK_EQ(differ, 0);
    CHECK_GE(accepted, N / 2);
    if (alloc_counting_available()) CHECK_EQ(allocs[0], allocs[1]);   // the pool adds none (the renderer of the test allocates)
}
