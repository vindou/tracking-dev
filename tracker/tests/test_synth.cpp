// Phase 4: T2 synthetic depth images. The back-projected object pixels lie on the mesh surface (within the depth
// quantization); a moving object with a camera time offset is rendered at each camera's own capture time; the pair
// file round trip is exact.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>

#include "../src/core/calib.hpp"
#include "../src/core/synth.hpp"
#include "../src/records/pairfile.hpp"
#include "test_framework.hpp"

using namespace trk;

namespace {
const Mesh& mesh() {
    static Mesh m;
    static bool ok = [] { std::string e; return load_stl(std::string(TRK_SOURCE_DIR) + "/../models/test_object.stl", m, e); }();
    if (!ok) std::printf("  cannot read models/test_object.stl\n");
    return m;
}
// Distance of p to the closest point of triangle (a, b, c) (Ericson, Real-Time Collision Detection 5.1.5).
double tri_dist(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const double d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) return (p - a).norm();
    const Vec3 bp = p - b;
    const double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) return (p - b).norm();
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return (p - (a + d1 / (d1 - d3) * ab)).norm();
    const Vec3 cp = p - c;
    const double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) return (p - c).norm();
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return (p - (a + d2 / (d2 - d6) * ac)).norm();
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return (p - (b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b))).norm();
    const double den = 1.0 / (va + vb + vc);
    return (p - (a + ab * (vb * den) + ac * (vc * den))).norm();
}
double mesh_dist(const Mesh& m, const Vec3& p) {
    double d = 1e9;
    for (std::size_t i = 0; i < m.size(); i++) d = std::min(d, tri_dist(p, m.v0[i], m.v1[i], m.v2[i]));
    return d;
}
}  // namespace

TEST(synth_pixels_lie_on_the_mesh) {
    Camera cams[2];
    default_cameras(cams);
    std::mt19937_64 rng(3);
    double worst = 0;
    long pixels = 0;
    for (int k = 0; k < 10; k++) {
        const Mat4 T = random_pose_in_view(cams, mesh(), rng);
        for (int c = 0; c < 2; c++) {
            std::vector<uint16_t> img(std::size_t(cams[c].width) * cams[c].height);
            RenderOptions opt;
            render_depth(mesh(), cams[c], &T, opt, rng, img.data());
            const Mat4 T_obj_cam = inverse_pose(T) * cams[c].T_rig_cam;
            long n = 0;
            for (int v = 0; v < cams[c].height; v++)
                for (int u = 0; u < cams[c].width; u++) {
                    const uint16_t d = img[std::size_t(v) * cams[c].width + u];
                    if (!d) continue;
                    n++;
                    const Vec3 pc = cams[c].backproject(u, v, d * cams[c].depth_scale);
                    // the depth is quantized to 1 mm: the exact point is within 0.5 mm along the viewing ray
                    const double dist = mesh_dist(mesh(), xform(T_obj_cam, pc));
                    worst = std::max(worst, dist / (0.5e-3 * pc.norm() / pc.z()));
                }
            pixels += n;
            CHECK_GE(n, 500);   // the object is in view of both cameras
        }
    }
    std::printf("  20 images, %ld object pixels; largest distance to the mesh / quantization bound: %.3f\n", pixels, worst);
    CHECK_LE(worst, 1.001);
}

TEST(synth_moving_object_with_time_offset_and_file_roundtrip) {
    Camera cams[2];
    default_cameras(cams);
    SynthScenario s;
    s.T0 = make_pose(Mat3::Identity(), Vec3(0, 0, 0.35));
    s.velocity = Vec3(0.2, 0, 0);
    s.offset_ms = 5.0;
    s.render.noise_sigma_m = 0.001;
    s.render.plane = true;
    s.render.plane_z = 0.07;
    s.absent.emplace_back(3, 4);
    s.mono_base_ns = 1000000000000LL;
    s.real_minus_mono_ns = 1700000000000000000LL;
    SynthGenerator gen(mesh(), cams, s);
    const std::size_t n = std::size_t(cams[0].width) * cams[0].height;
    std::vector<std::vector<uint16_t>> A, B;
    std::vector<cap::PairRecord> recs;
    std::vector<Mat4> truths;
    std::vector<bool> present;
    for (int k = 0; k < 5; k++) {
        std::vector<uint16_t> a(n), b(n);
        cap::PairRecord rec;
        Mat4 t;
        bool p;
        gen.make(k, a.data(), b.data(), rec, t, p);
        A.push_back(a); B.push_back(b); recs.push_back(rec); truths.push_back(t); present.push_back(p);
    }
    // truth of pair k = pose at the mean capture time; B was captured offset later
    const double tmean = (2 * 11.1283 + 5.0 / 2) / 1000.0 + 0 * 0;
    CHECK_LE(std::fabs(truths[2](0, 3) - 0.2 * (2 * 0.0111283 + 0.0025)), 1e-9);
    (void)tmean;
    CHECK_LE(std::fabs((recs[2].cam[1].global_ts_ms - recs[2].cam[0].global_ts_ms) - 5.0), 1e-6);
    CHECK(!present[3] && present[2]);
    // capture time from the record (tracker method) equals the synthetic capture time
    const double cap_mono = recs[2].cam[0].global_ts_ms * 1e6 - double(recs[2].publish_real_ns - recs[2].publish_mono_ns);
    CHECK_LE(std::fabs(cap_mono - (s.mono_base_ns + 2 * 11.1283e6)), 2000.0);   // double precision of ms timestamps
    // file round trip
    const std::string path = "/tmp/trk_synth_test_" + std::to_string(getpid()) + ".trk";
    cap::CameraInfo ci[2];
    camera_infos(cams, ci);
    {
        PairFileWriter w;
        std::string err;
        CHECK(w.open(path, ci, 1, 11.1283, 5.564, err));
        for (int k = 0; k < 5; k++) {
            double t16[16];
            for (int r = 0; r < 16; r++) t16[r] = truths[k](r / 4, r % 4);
            CHECK(w.append(recs[k], A[k].data(), B[k].data(), t16));
        }
        CHECK(w.close());
        std::printf("  5 pairs with background plane and noise: %.2f MB (raw %.2f MB)\n", double(w.bytes()) / 1e6, 5 * 2 * n * 2 / 1e6);
    }
    PairFileReader r;
    std::string err;
    CHECK(r.open(path, err));
    CHECK_EQ(r.header().pair_count, 5u);
    CHECK(std::string(r.header().cam[1].serial) == "234322070144");
    std::vector<uint16_t> a(n), b(n);
    PairChunkHeader ch;
    int k = 0;
    while (r.next(ch, a.data(), b.data(), err)) {
        CHECK(a == A[k] && b == B[k]);
        CHECK(std::memcmp(&ch.rec, &recs[k], sizeof ch.rec) == 0);
        CHECK_EQ(ch.truth[3], truths[k](0, 3));
        k++;
    }
    CHECK_EQ(k, 5);
    CHECK(err.empty());
    unlink(path.c_str());
}
