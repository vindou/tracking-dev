// Drives TrackerCore with synthetic pairs (T2 renderer), one pair per cycle, as the live loop does.
#pragma once
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../src/core/calib.hpp"
#include "../src/core/synth.hpp"
#include "../src/core/tracker_core.hpp"

namespace trkt {
using namespace trk;

inline const Mesh& object_mesh() {
    static Mesh m;
    static bool ok = [] { std::string e; return load_stl(std::string(TRK_SOURCE_DIR) + "/../models/test_object.stl", m, e); }();
    (void)ok;
    return m;
}

inline TrackerConfig test_config() {
    TrackerConfig c;
    std::string err;
    if (!load_tracker_config(std::string(TRK_SOURCE_DIR) + "/config/tracker.json", c, err)) std::printf("  config: %s\n", err.c_str());
    return c;
}

struct PoseError { double pos_m, rot_rad; };
inline PoseError pose_error(const Mat4& est, const Mat4& truth) {
    return {(est.block<3, 1>(0, 3) - truth.block<3, 1>(0, 3)).norm(),
            rotation_angle(est.block<3, 3>(0, 0) * truth.block<3, 3>(0, 0).transpose())};
}
inline Mat4 icp_pose(const PoseRecord& p) {
    return make_pose(R_from_quat(p.icp_orientation), Vec3(p.icp_position[0], p.icp_position[1], p.icp_position[2]));
}
inline Mat4 filt_pose(const PoseRecord& p) {
    return make_pose(R_from_quat(p.orientation), Vec3(p.position[0], p.position[1], p.position[2]));
}

class Harness {
public:
    explicit Harness(const SynthScenario& s, const TrackerConfig& cfg = test_config(), const Mesh* render_mesh = nullptr)
        : cfg_(cfg) {
        default_cameras(cams_);
        std::string err;
        ok_ = core_.init(cfg_, err);
        if (!ok_) std::printf("  core init: %s\n", err.c_str());
        cap::CameraInfo ci[2];
        camera_infos(cams_, ci);
        ok_ = ok_ && core_.set_cameras(ci, err);
        gen_.reset(new SynthGenerator(render_mesh ? *render_mesh : object_mesh(), cams_, s));
        a_.resize(std::size_t(cams_[0].width) * cams_[0].height);
        b_.resize(a_.size());
    }
    bool ok() const { return ok_; }
    TrackerCore& core() { return core_; }
    // Renders pair k and runs one cycle.
    const PoseRecord& step(int k) {
        CropRect cr[2];
        core_.crops(cr);
        cap::PairRecord rec;
        gen_->make(k, a_.data(), b_.data(), rec, truth_, present_);
        const uint16_t* img[2] = {a_.data(), b_.data()};
        core_.process(rec, img, pose_, nullptr);
        return pose_;
    }
    const Mat4& truth() const { return truth_; }
    bool present() const { return present_; }
private:
    TrackerConfig cfg_;
    Camera cams_[2];
    TrackerCore core_;
    std::unique_ptr<SynthGenerator> gen_;
    std::vector<uint16_t> a_, b_;
    PoseRecord pose_{};
    Mat4 truth_ = Mat4::Identity();
    bool present_ = true;
    bool ok_ = false;
};

inline const char* reasons_str(uint32_t r) {
    static char b[96];
    b[0] = 0;
    if (r & kRejectFewPoints) std::strcat(b, "points ");
    if (r & kRejectRmse) std::strcat(b, "rmse ");
    if (r & kRejectInliers) std::strcat(b, "inliers ");
    if (r & kRejectEigen) std::strcat(b, "eigenvalue ");
    if (r & kRejectJump) std::strcat(b, "jump ");
    if (r & kRejectNoSolve) std::strcat(b, "no-solve ");
    if (r & kRejectVisibility) std::strcat(b, "visibility ");
    return b;
}

}  // namespace trkt
