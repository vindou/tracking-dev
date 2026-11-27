// Tracker core: stages 2-6 of one cycle (cloud builder, scene filter, pose solver, fit check, tracking filter) and the
// track states. No I/O, no allocation after init(). Used by the live loop (trk_tracker), the replay test and the
// synthetic tests. Stage 1 (pair reader) is outside: the caller reads the crop rows given by crops().
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../icp/KDTreeFlann.h"
#include "../icp/RegistrationPrealloc.h"
#include "../records/records.hpp"
#include "bgmap.hpp"
#include "camera.hpp"
#include "config.hpp"
#include "filters.hpp"
#include "mesh.hpp"
#include "shm_layout.hpp"
#include "work_pool.hpp"

namespace trk {

struct CropRect {
    int u0 = 0, v0 = 0, u1 = -1, v1 = -1;   // inclusive pixel rectangle; empty if u1 < u0
    bool empty() const { return u1 < u0 || v1 < v0; }
};

class TrackerCore {
public:
    TrackerCore();
    ~TrackerCore();
    bool init(const TrackerConfig& cfg, std::string& err);
    // Intrinsics, serials and depth unit from the pair-buffer header; assigns D435i_1 / D435i_2 by serial.
    // note: describes the assignment (or that the serials were not found and the default was used).
    bool set_cameras(const cap::CameraInfo info[2], std::string& err, std::string* note = nullptr);
    const Camera& camera(int c) const { return cams_[c]; }
    void set_camera_pose(int c, const Mat4& T_rig_cam) { cams_[c].set_pose(T_rig_cam); }

    // Pixel rectangles to read for the next cycle (prediction box in track / lost, work volume in search).
    void crops(CropRect out[2]);
    // Runs stages 2-6 on a pair whose crop rows (as returned by the last crops() call) are filled. Fills the pose
    // record (all fields except writer/capture instance ids, pose_seq, publish time and the read / publish stage
    // times) and, if cloud != nullptr, the cloud record.
    void process(const cap::PairRecord& rec, const uint16_t* const img[2], PoseRecord& pose, CloudRecord* cloud);

    // Test helpers
    void force_track(const Mat4& T_rig_obj);        // start in track state at this pose (filter initialized)
    void reset_to_search();
    // Pose hypotheses T_rig_from_object from an external source (the flow module, ../flow). They are tried once, at
    // the start of the next cycle in search state, before the orientation grid: full ICP from each, the usual fit
    // check, the best passing one is accepted. At most kMaxHints are kept; nothing is allocated.
    static constexpr int kMaxHints = 32;
    // in_order: accept the first hypothesis that passes (they are ranked by the source) instead of the best passing one.
    void set_hints(const Mat4* T_rig_obj, int n, bool in_order = false);
    int hints_tried() const { return hints_tried_; }   // of the last cycle
    void set_camera_mask(int mask) { cam_mask_ = mask; }   // bit c: use camera c (T5 single-camera solves)
    TrackState state() const { return state_; }
    Mat4 filtered_pose() const;
    const Model& model() const { return model_; }
    const TrackerConfig& config() const { return cfg_; }
    static int64_t capture_mono_ns(const cap::PairRecord& rec, int cam);   // time conversion (DESIGN.md 4)

    // Diagnostics (tests): search candidates of the last completed coarse stage, visibility of a pose (current images).
    struct Candidate { Mat4 T_obj_rig; double fitness, rmse, violation; int center; };
    const std::vector<Candidate>& candidates() const { return cand_; }
    int search_centers(Vec3* out) const { for (int i = 0; i < n_centers_; i++) out[i] = centers_[i]; return n_centers_; }
    double visibility_of(const Mat4& T_rig_obj, int& checked) const { return visibility(T_rig_obj, checked); }

private:
    int build_cloud(const uint16_t* const img[2]);
    int filter_cloud(const Mat4* T_obj_rig_pred, double margin);
    int select_solver_points(int max_points);
    int select_points_near(const Vec3& c, double radius, int max_points);   // filtered points within radius of c
    int find_search_centers();   // object-sized clusters of the filtered cloud (connected 4 cm cells)
    void evaluate(const open3d::pipelines::registration::RegistrationResultPrealloc& r, const Mat4* pred, PoseRecord& pose,
                  bool check_jump);
    bool search_step(int64_t t_start, PoseRecord& pose);
    bool try_hints(PoseRecord& pose);
    bool try_hints_parallel(PoseRecord& pose);
    void hint_job(int i, int worker);
    int select_near_into(const Vec3& c, double radius, int max_points, Eigen::Vector3d* out) const;
    void evaluate_into(const open3d::pipelines::registration::RegistrationResultPrealloc& r, const Mat4* pred, PoseRecord& pose,
                       bool check_jump, int ns, Mat4& T_out) const;
    void fill_cloud(CloudRecord& cloud, const PoseRecord& pose);
    // Free-space check: fraction of the model points facing a camera whose pixel shows a surface farther away than
    // the model surface (the camera sees through the object). checked = points with a usable measurement.
    double visibility(const Mat4& T_rig_obj, int& checked) const;
    Mat4 predicted_pose(double dt) const;

    TrackerConfig cfg_;
    Camera cams_[2];
    Model model_;
    Mesh mesh_;
    open3d::geometry::PointCloud target_;
    std::unique_ptr<open3d::geometry::KDTreeFlann> tree_;
    std::vector<std::unique_ptr<open3d::pipelines::registration::RobustKernel>> kernels_;   // one per distance stage
    const open3d::pipelines::registration::RobustKernel* kptr_[8] = {};
    open3d::pipelines::registration::ICPWorkspace ws_;
    BackgroundMap bg_;
    std::vector<Mat3> orientations_;

    // per-cycle arrays (allocated in init)
    std::vector<Eigen::Vector3d> pts_;      // built points (rig frame)
    std::vector<uint8_t> pcam_;
    std::vector<Eigen::Vector3d> fpts_;     // filtered
    std::vector<uint8_t> fcam_;
    std::vector<Eigen::Vector3d> spts_;     // given to the solver
    std::vector<uint8_t> scam_;
    std::vector<int> sidx_;                 // index in fpts_ of each solver point
    std::vector<uint8_t> fflag_;            // per filtered point: cloud flag
    std::vector<Candidate> cand_;
    int npts_ = 0, nf_ = 0, ns_ = 0;
    CropRect crop_[2];
    const uint16_t* img_[2] = {nullptr, nullptr};   // images of the current cycle (crop rows valid)
    std::vector<Eigen::Vector3d> vis_pts_, vis_nrm_; // model subset for the visibility check
    int cam_mask_ = 3;

    // state
    TrackState state_ = kStateSearch;
    PositionKF kf_;
    OrientationFilter of_;
    int64_t last_meas_ns_ = 0;
    uint32_t rejected_in_row_ = 0;
    double period_s_ = 0.0111283;
    // search
    int search_idx_ = 0;
    static constexpr int kMaxCenters = 4;
    Vec3 centers_[kMaxCenters];
    int n_centers_ = 0, cur_center_ = -1;
    // grid for the cluster search (allocated in init)
    int gx_ = 0, gy_ = 0, gz_ = 0;
    std::vector<uint16_t> gcount_;
    std::vector<int32_t> glabel_, gqueue_;
    Eigen::Vector3d search_center_ = Eigen::Vector3d::Zero();
    open3d::pipelines::registration::RegistrationResultPrealloc best_;   // best refined candidate of the round
    bool have_best_ = false;
    int best_center_ = 0;
    double est_coarse_ns_ = 2e6, est_refine_ns_ = 3e6;   // running estimates of the step durations (time slicing)
    double best_fit_ = -1, best_rmse_ = 1e9;
    Mat4 last_icp_T_rig_obj_ = Mat4::Identity();
    double max_extent_ = 0.1;
    Mat4 hints_[kMaxHints];
    int n_hints_ = 0, hints_tried_ = 0;
    bool hints_in_order_ = false;
    // hint trials in parallel (cfg.hint_threads > 1); all memory allocated in init()
    struct HintWorker {
        open3d::pipelines::registration::ICPWorkspace ws;
        std::vector<Eigen::Vector3d> pts;
    };
    struct HintResult {
        open3d::pipelines::registration::RegistrationResultPrealloc r;
        Mat4 T = Mat4::Identity();
        Vec3 c = Vec3::Zero();
        double score = -1;
        int ns = 0;
        bool ran = false, valid = false;
    };
    std::vector<HintWorker> hw_;
    HintResult hres_[kMaxHints];
    std::vector<uint8_t> hint_inlier_[kMaxHints];
    open3d::geometry::ClosestTable table_;
    WorkPool pool_;   // last member: its threads stop before the rest is destroyed
};

}  // namespace trk
