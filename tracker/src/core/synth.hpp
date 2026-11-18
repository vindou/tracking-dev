// Synthetic pairs (T2 and the synthetic tests): the object moves with constant linear and angular velocity; frame A
// is captured at t_k = k * period, frame B at t_k + offset; each frame is rendered at the object pose of its own
// capture time. The pair record is filled like the capture program does (global timestamps in a CLOCK_REALTIME base,
// arrival 9.6 ms after capture, publish times in both clocks), so the time conversion of the tracker applies.
#pragma once
#include <cstdint>
#include <random>
#include <vector>

#include "camera.hpp"
#include "mesh.hpp"
#include "render.hpp"
#include "shm_layout.hpp"

namespace trk {

struct SynthScenario {
    Mat4 T0 = Mat4::Identity();         // T_rig_from_object at t = 0
    Vec3 velocity = Vec3::Zero();       // m/s, rig frame
    Vec3 angular_velocity = Vec3::Zero();  // rad/s, rig frame (about the object origin)
    double period_ms = 11.1283;
    double offset_ms = 0.0;             // capture time B - A
    RenderOptions render;
    std::vector<std::pair<int, int>> absent;   // frame ranges [first, last) without the object
    int blank_cam = -1, blank_from = 0, blank_to = 0;   // camera whose images are empty in [from, to) (covered camera)
    int64_t mono_base_ns = 0;           // CLOCK_MONOTONIC of capture time 0 of camera A
    int64_t real_minus_mono_ns = 0;     // clock offset used for the global timestamps
    uint64_t seed = 1;
};

// Default cameras: serials, intrinsics of the real cameras (camera A / B), nominal transforms D435i_1 = A, D435i_2 = B.
void default_cameras(Camera cams[2]);
void camera_infos(const Camera cams[2], cap::CameraInfo out[2]);

Mat4 synth_pose_at(const SynthScenario& s, double t_s);

class SynthGenerator {
public:
    SynthGenerator(const Mesh& mesh, const Camera cams[2], const SynthScenario& s);
    // Pair k: images (width * height each), record, and the true pose at the measurement time (mean capture time).
    // present = false if the object is absent in this pair.
    void make(int k, uint16_t* a, uint16_t* b, cap::PairRecord& rec, Mat4& truth, bool& present);
    const SynthScenario& scenario() const { return s_; }
private:
    const Mesh& mesh_;
    Camera cams_[2];
    SynthScenario s_;
    std::mt19937_64 rng_;
};

}  // namespace trk
