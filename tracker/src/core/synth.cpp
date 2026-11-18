#include "synth.hpp"

#include <algorithm>
#include <cstring>

namespace trk {

void default_cameras(Camera cams[2]) {
    cams[0].name = "D435i_1";
    cams[0].serial = "234322070408";
    cams[0].fx = cams[0].fy = 431.372;
    cams[0].ppx = 424.663;
    cams[0].ppy = 234.116;
    cams[0].set_pose(nominal_T_rig_cam(1));
    cams[1].name = "D435i_2";
    cams[1].serial = "234322070144";
    cams[1].fx = cams[1].fy = 430.382;
    cams[1].ppx = 424.649;
    cams[1].ppy = 235.569;
    cams[1].set_pose(nominal_T_rig_cam(2));
}

void camera_infos(const Camera cams[2], cap::CameraInfo out[2]) {
    for (int c = 0; c < 2; c++) {
        std::memset(&out[c], 0, sizeof out[c]);
        std::snprintf(out[c].serial, sizeof out[c].serial, "%s", cams[c].serial.c_str());
        out[c].depth_scale = float(cams[c].depth_scale);
        out[c].width = uint32_t(cams[c].width);
        out[c].height = uint32_t(cams[c].height);
        out[c].ppx = float(cams[c].ppx);
        out[c].ppy = float(cams[c].ppy);
        out[c].fx = float(cams[c].fx);
        out[c].fy = float(cams[c].fy);
        out[c].distortion_model = 0;
    }
}

Mat4 synth_pose_at(const SynthScenario& s, double t) {
    const Mat3 R = so3_exp(s.angular_velocity * t) * s.T0.block<3, 3>(0, 0);
    const Vec3 p = s.T0.block<3, 1>(0, 3) + s.velocity * t;
    return make_pose(R, p);
}

SynthGenerator::SynthGenerator(const Mesh& mesh, const Camera cams[2], const SynthScenario& s)
    : mesh_(mesh), cams_{cams[0], cams[1]}, s_(s), rng_(s.seed) {}

void SynthGenerator::make(int k, uint16_t* a, uint16_t* b, cap::PairRecord& rec, Mat4& truth, bool& present) {
    present = true;
    for (const auto& r : s_.absent)
        if (k >= r.first && k < r.second) present = false;
    const double tA = k * s_.period_ms / 1000.0, tB = tA + s_.offset_ms / 1000.0;
    const Mat4 TA = synth_pose_at(s_, tA), TB = synth_pose_at(s_, tB);
    truth = synth_pose_at(s_, 0.5 * (tA + tB));
    render_depth(mesh_, cams_[0], present ? &TA : nullptr, s_.render, rng_, a);
    render_depth(mesh_, cams_[1], present ? &TB : nullptr, s_.render, rng_, b);
    if (s_.blank_cam >= 0 && k >= s_.blank_from && k < s_.blank_to)
        std::fill(s_.blank_cam ? b : a, (s_.blank_cam ? b : a) + std::size_t(cams_[0].width) * cams_[0].height, uint16_t(0));
    std::memset(&rec, 0, sizeof rec);
    rec.pair_seq = uint64_t(k);
    rec.dt_ms = s_.offset_ms;
    const double t[2] = {tA, tB};
    int64_t last_arrival = 0;
    for (int c = 0; c < 2; c++) {
        const int64_t cap_mono = s_.mono_base_ns + int64_t(t[c] * 1e9);
        rec.cam[c].global_ts_ms = double(cap_mono + s_.real_minus_mono_ns) / 1e6;
        rec.cam[c].hw_ts_us = int64_t(t[c] * 1e6);
        rec.cam[c].sensor_ts_us = int64_t(t[c] * 1e6);
        rec.cam[c].arrival_mono_ns = cap_mono + 9600000;
        rec.cam[c].frame_counter = k;
        rec.cam[c].exposure_us = 1000;
        rec.cam[c].domain = 2;
        last_arrival = std::max(last_arrival, rec.cam[c].arrival_mono_ns);
    }
    rec.publish_mono_ns = last_arrival + 1000000;
    rec.publish_real_ns = rec.publish_mono_ns + s_.real_minus_mono_ns;
}

}  // namespace trk
