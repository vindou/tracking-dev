// Phase 5 synthetic tests (no camera): static accuracy, degenerate view, acquisition, moving object, object removed
// and put back, fit-check value distributions (used to select the default limits). Pass criteria: TEST_REPORT.md.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "../src/common/clock.hpp"
#include "synth_harness.hpp"
#include "test_framework.hpp"

using namespace trkt;

namespace {
int env_int(const char* n, int def) { const char* e = std::getenv(n); return e ? std::atoi(e) : def; }
double q(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[std::size_t(double(v.size() - 1) * p)];
}
Mat4 perturb(const Mat4& T, double dp, double dr_rad, std::mt19937_64& rng) {
    std::normal_distribution<double> nd;
    Vec3 a(nd(rng), nd(rng), nd(rng)), t(nd(rng), nd(rng), nd(rng));
    return make_pose(so3_exp(a.normalized() * dr_rad) * T.block<3, 3>(0, 0), T.block<3, 1>(0, 3) + t.normalized() * dp);
}
}  // namespace

// 100 random poses in the work volume, depth noise 1 mm. The tracker starts in track state 5 mm / 3 degrees from the
// true pose and runs 5 cycles on the static object; the ICP pose of the last cycle is compared with the truth.
// Pass: RMS position error <= 1 mm and RMS orientation error <= 1 degree over the poses that pass the fit check.
TEST(synth_static_accuracy_100_poses) {
    std::mt19937_64 rng(101);
    Camera cams[2];
    default_cameras(cams);
    const int N = env_int("TRK_N_POSES", 100);
    double se_p = 0, se_r = 0, max_p = 0, max_r = 0;
    int passed = 0, rejected = 0;
    uint32_t reasons_or = 0;
    std::vector<double> rmse, inl, eig, solve_us;
    for (int k = 0; k < N; k++) {
        SynthScenario s;
        s.T0 = random_pose_in_view(cams, object_mesh(), rng);
        s.render.noise_sigma_m = 0.001;
        s.seed = uint64_t(1000 + k);
        Harness h(s);
        CHECK(h.ok());
        h.core().force_track(perturb(s.T0, 0.005, 3.0 * M_PI / 180, rng));
        PoseRecord last{};
        for (int i = 0; i < 5; i++) {
            last = h.step(i);
            solve_us.push_back(last.stage_us[kStageSolve]);
        }
        rmse.push_back(last.fit_rmse * 1000);
        inl.push_back(last.inlier_ratio);
        eig.push_back(last.min_eigenvalue);
        if (last.valid) {
            const PoseError e = pose_error(icp_pose(last), h.truth());
            se_p += e.pos_m * e.pos_m;
            se_r += e.rot_rad * e.rot_rad;
            max_p = std::max(max_p, e.pos_m);
            max_r = std::max(max_r, e.rot_rad);
            passed++;
        } else {
            rejected++;
            reasons_or |= last.reject_reasons;
            std::printf("  pose %d rejected: %s (points %u, rmse %.2f mm, inliers %.3f, eig %.4f)\n", k,
                        reasons_str(last.reject_reasons), last.point_count, last.fit_rmse * 1000, last.inlier_ratio,
                        last.min_eigenvalue);
        }
    }
    const double rms_p = passed ? std::sqrt(se_p / passed) : NAN, rms_r = passed ? std::sqrt(se_r / passed) : NAN;
    std::printf("  %d poses: %d passed the fit check, %d rejected (reasons: %s)\n", N, passed, rejected, reasons_str(reasons_or));
    std::printf("  position error RMS %.3f mm (max %.3f), orientation error RMS %.3f deg (max %.3f)\n", rms_p * 1000,
                max_p * 1000, rms_r * 180 / M_PI, max_r * 180 / M_PI);
    std::printf("  fit values: rmse p50/max %.2f/%.2f mm, inlier ratio min/p50 %.3f/%.3f, min eigenvalue min/p50 %.4f/%.4f\n",
                q(rmse, .5), q(rmse, 1), q(inl, 0), q(inl, .5), q(eig, 0), q(eig, .5));
    std::printf("  solver time p50/p99/max %.0f/%.0f/%.0f us\n", q(solve_us, .5), q(solve_us, .99), q(solve_us, 1));
    CHECK_GE(passed, 1);
    CHECK_LE(rms_p, 0.001);
    CHECK_LE(rms_r * 180 / M_PI, 1.0);
}

// Only one flat face visible: the measured scene is the largest face of the object alone. Pass: every cycle rejected.
TEST(synth_degenerate_single_face_rejected) {
    const Mesh& m = object_mesh();
    // group triangles by normal; take the group with the largest area
    std::vector<Vec3> normals;
    std::vector<double> areas;
    std::vector<int> group(m.size());
    for (std::size_t i = 0; i < m.size(); i++) {
        const Vec3 n = (m.v1[i] - m.v0[i]).cross(m.v2[i] - m.v0[i]);
        const double a = 0.5 * n.norm();
        const Vec3 u = n.normalized();
        int g = -1;
        for (std::size_t j = 0; j < normals.size(); j++)
            if (normals[j].dot(u) > 0.9999 && std::fabs(normals[j].dot(m.v0[i]) - normals[j].dot(m.v0[std::size_t(group[0])])) >= 0) {
                // same normal and same plane offset
                bool same_plane = false;
                for (std::size_t t = 0; t < i; t++)
                    if (group[t] == int(j)) { same_plane = std::fabs(u.dot(m.v0[i] - m.v0[t])) < 1e-6; break; }
                if (same_plane) { g = int(j); break; }
            }
        if (g < 0) { normals.push_back(u); areas.push_back(0); g = int(normals.size()) - 1; }
        group[i] = g;
        areas[std::size_t(g)] += a;
    }
    const int big = int(std::max_element(areas.begin(), areas.end()) - areas.begin());
    Mesh face;
    for (std::size_t i = 0; i < m.size(); i++)
        if (group[i] == big) { face.v0.push_back(m.v0[i]); face.v1.push_back(m.v1[i]); face.v2.push_back(m.v2[i]); }
    face.bmin = m.bmin;
    face.bmax = m.bmax;
    std::printf("  largest flat face: %zu triangles, %.1f cm2, normal (%.2f %.2f %.2f)\n", face.size(), areas[std::size_t(big)] * 1e4,
                normals[std::size_t(big)].x(), normals[std::size_t(big)].y(), normals[std::size_t(big)].z());
    std::mt19937_64 rng(202);
    Camera cams[2];
    default_cameras(cams);
    int cycles = 0, valid = 0;
    std::vector<double> eig;
    for (int k = 0; k < 20; k++) {
        SynthScenario s;
        // the face turned toward the cameras (its normal along +z, toward both cameras' viewing direction)
        const Vec3 n = normals[std::size_t(big)];
        const Mat3 R = Eigen::Quaterniond::FromTwoVectors(n, Vec3(0, 0, -1)).toRotationMatrix();
        Mat4 T = random_pose_in_view(cams, m, rng);
        T.block<3, 3>(0, 0) = so3_exp(Vec3(0, 0, 1) * std::uniform_real_distribution<double>(0, 2 * M_PI)(rng)) * R;
        s.T0 = T;
        s.render.noise_sigma_m = 0.001;
        s.seed = uint64_t(2000 + k);
        Harness h(s, test_config(), &face);
        h.core().force_track(perturb(T, 0.003, 2.0 * M_PI / 180, rng));
        for (int i = 0; i < 5; i++) {
            const PoseRecord& p = h.step(i);
            cycles++;
            if (p.valid) valid++;
            eig.push_back(p.min_eigenvalue);
            if (h.core().state() == kStateSearch) break;
        }
    }
    std::printf("  %d cycles with one face only: %d valid; min eigenvalue p50/max %.5f/%.5f (limit %.4f)\n", cycles, valid,
                q(eig, .5), q(eig, 1), test_config().min_eigenvalue);
    CHECK_EQ(valid, 0);
}

// 100 random poses, unknown start (search). Pass: track state in 1 s or less (90 pairs at 90 Hz) for >= 90 %.
// Reports each failure and each pose with a wrong orientation (> 5 degrees or > 5 mm) that passed the fit check.
TEST(synth_acquisition_100_poses) {
    std::mt19937_64 rng(303);
    Camera cams[2];
    default_cameras(cams);
    const int N = env_int("TRK_N_POSES", 100);
    int ok = 0, wrong = 0;
    std::vector<double> t_acq;
    double max_cycle_us = 0;
    std::vector<double> cyc;
    for (int k = 0; k < N; k++) {
        SynthScenario s;
        s.T0 = random_pose_in_view(cams, object_mesh(), rng);
        s.render.noise_sigma_m = 0.001;
        s.seed = uint64_t(3000 + k);
        Harness h(s);
        int got = -1;
        for (int i = 0; i < 180; i++) {
            const PoseRecord& p = h.step(i);
            cyc.push_back(p.stage_us[kStageTotal]);
            max_cycle_us = std::max(max_cycle_us, double(p.stage_us[kStageTotal]));
            if (p.track_state == kStateTrack && p.valid) { got = i; break; }
        }
        if (got >= 0) {
            const double t = (got + 1) * 11.1283e-3;
            const PoseError e = pose_error(filt_pose(h.step(got + 1)), h.truth());
            const bool bad = e.pos_m > 0.005 || e.rot_rad > 5 * M_PI / 180;
            if (bad) {
                wrong++;
                std::printf("  pose %d: WRONG pose passed the fit check after %.2f s: error %.1f mm, %.1f deg\n", k, t,
                            e.pos_m * 1000, e.rot_rad * 180 / M_PI);
            }
            if (t <= 1.0 && !bad) ok++;
            else if (!bad) std::printf("  pose %d: track after %.2f s (> 1 s)\n", k, t);
            t_acq.push_back(t);
        } else {
            std::printf("  pose %d: no track within 2 s\n", k);
        }
    }
    std::printf("  %d poses: track within 1 s with the right pose: %d (%.0f %%); wrong poses accepted: %d; time to track p50/p90/max %.2f/%.2f/%.2f s\n",
                N, ok, 100.0 * ok / N, wrong, q(t_acq, .5), q(t_acq, .9), q(t_acq, 1));
    std::printf("  cycle time p50/p99/max %.0f/%.0f/%.0f us (search slice limit %.1f ms)\n", q(cyc, .5), q(cyc, .99),
                max_cycle_us, test_config().search_time_ms);
    CHECK_GE(double(ok), 0.9 * N);
}

// Moving object: 0.2 m/s and 30 deg/s, camera time offsets 0 to 5.6 ms. Pass: the track stays valid (track state in
// every cycle) and the RMS position error of the filtered pose <= 3 mm.
TEST(synth_moving_object) {
    const double offsets[] = {0.0, 1.4, 2.8, 4.2, 5.6};
    std::mt19937_64 rng(404);
    for (double off : offsets) {
        SynthScenario s;
        Vec3 dir(1, 0, 0);
        s.T0 = make_pose(random_rotation(rng), Vec3(-0.15, 0.0, 0.42));
        s.velocity = 0.2 * dir;
        s.angular_velocity = Vec3(0.3, -0.5, 0.8).normalized() * (30.0 * M_PI / 180);
        s.offset_ms = off;
        s.render.noise_sigma_m = 0.001;
        s.seed = uint64_t(4000 + int(off * 10));
        Harness h(s);
        h.core().force_track(s.T0);
        const int frames = 135;   // 1.5 s, 0.3 m
        int not_track = 0, invalid = 0;
        double se = 0, se_r = 0, mx = 0;
        int n = 0;
        for (int i = 0; i < frames; i++) {
            const PoseRecord& p = h.step(i);
            if (p.track_state != kStateTrack) not_track++;
            if (!p.valid) invalid++;
            if (i >= 10) {   // after the filter's start-up
                const PoseError e = pose_error(filt_pose(p), h.truth());
                se += e.pos_m * e.pos_m;
                se_r += e.rot_rad * e.rot_rad;
                mx = std::max(mx, e.pos_m);
                n++;
            }
        }
        const double rms = std::sqrt(se / n);
        std::printf("  offset %.1f ms: cycles not in track %d, invalid %d of %d; filtered position error RMS %.2f mm (max %.2f), orientation RMS %.2f deg\n",
                    off, not_track, invalid, frames, rms * 1000, mx * 1000, std::sqrt(se_r / n) * 180 / M_PI);
        CHECK_EQ(not_track, 0);
        CHECK_LE(rms, 0.003);
    }
}

// Object removed, then put back: track -> lost -> search -> track, and no invalid pose marked valid (a valid pose
// while the object is absent, or more than 5 mm / 5 degrees from the truth, counts as invalid).
TEST(synth_object_removed_and_put_back) {
    std::mt19937_64 rng(505);
    Camera cams[2];
    default_cameras(cams);
    int sequences_ok = 0;
    const int runs = 5;
    for (int run = 0; run < runs; run++) {
        SynthScenario s;
        s.T0 = random_pose_in_view(cams, object_mesh(), rng);
        s.render.noise_sigma_m = 0.001;
        s.absent.emplace_back(60, 240);   // removed for 2 s
        s.seed = uint64_t(5000 + run);
        Harness h(s);
        h.core().force_track(s.T0);
        std::string seq;
        uint32_t last = 99;
        int bad_valid = 0;
        for (int i = 0; i < 420; i++) {
            const PoseRecord& p = h.step(i);
            if (p.track_state != last) {
                seq += p.track_state == kStateTrack ? "T" : p.track_state == kStateLost ? "L" : "S";
                last = p.track_state;
            }
            if (p.valid) {
                const PoseError e = pose_error(icp_pose(p), h.truth());
                if (!h.present() || e.pos_m > 0.005 || e.rot_rad > 5 * M_PI / 180) bad_valid++;
            }
        }
        const bool ok = seq == "TLST" && bad_valid == 0 && h.core().state() == kStateTrack;
        std::printf("  run %d: states %s, invalid poses marked valid %d -> %s\n", run, seq.c_str(), bad_valid, ok ? "ok" : "FAIL");
        if (ok) sequences_ok++;
    }
    CHECK_EQ(sequences_ok, runs);
}
