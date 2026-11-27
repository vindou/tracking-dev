#include "tracker_core.hpp"

#include "../common/prof.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <Eigen/Eigenvalues>

#include "../common/clock.hpp"

namespace trk {

using open3d::pipelines::registration::ICPConvergenceCriteria;
using open3d::pipelines::registration::ICPWorkspace;
using open3d::pipelines::registration::RegistrationICPPrealloc;
using open3d::pipelines::registration::RegistrationResultPrealloc;

TrackerCore::TrackerCore() = default;
TrackerCore::~TrackerCore() = default;

int64_t TrackerCore::capture_mono_ns(const cap::PairRecord& rec, int cam) {
    // global timestamp (CLOCK_REALTIME base, ms) -> CLOCK_MONOTONIC with the clock offset of the publish times, which
    // the capture program reads at the same instant (DESIGN.md section 4)
    return int64_t(std::llround(rec.cam[cam].global_ts_ms * 1e6)) - (rec.publish_real_ns - rec.publish_mono_ns);
}

bool TrackerCore::init(const TrackerConfig& cfg, std::string& err) {
    cfg_ = cfg;
    if (!load_model(cfg.model_file, model_, err)) return false;
    if (!cfg.mesh_file.empty() && !load_stl(cfg.mesh_file, mesh_, err)) return false;
    target_.points_ = model_.points;
    target_.normals_ = model_.normals;
    tree_.reset(new open3d::geometry::KDTreeFlann(target_));
    kernels_.clear();
    for (std::size_t s = 0; s < cfg.distances.size() && s < 8; s++) {
        if (cfg.kernel == "tukey")
            kernels_.emplace_back(new open3d::pipelines::registration::TukeyLoss(
                    std::max(cfg.tukey_k, cfg.tukey_k_ratio * cfg.distances[s])));
        else
            kernels_.emplace_back(new open3d::pipelines::registration::L2Loss());
        kptr_[s] = kernels_.back().get();
    }
    const int maxp = std::min<int>(2 * std::max(cfg.max_points_per_camera, cfg.max_points_per_camera_search), 131072);
    // assign (not resize): Eigen vectors are not initialized by resize, so the pages would become resident only when
    // the loop first writes them (page faults in the loop, memory growth during the run). Written once here instead.
    pts_.assign(std::size_t(maxp), Eigen::Vector3d::Zero());
    pcam_.assign(std::size_t(maxp), 0);
    fpts_.assign(std::size_t(maxp), Eigen::Vector3d::Zero());
    fcam_.assign(std::size_t(maxp), 0);
    const int maxs = std::min<int>(std::max(cfg.solver_points_track, cfg.solver_points_search), int(kCloudMaxPoints));
    spts_.assign(std::size_t(maxs), Eigen::Vector3d::Zero());
    scam_.resize(std::size_t(maxs));
    sidx_.resize(std::size_t(maxs));
    fflag_.resize(std::size_t(maxp));
    ws_.Reserve(std::size_t(maxs));
    // correspondence search of every ICP run (config icp.correspondence_search)
    const int search = cfg.corr_search == "bounded" ? ICPWorkspace::kBounded : cfg.corr_search == "table" ? ICPWorkspace::kTable :
                       cfg.corr_search == "table_exact" ? ICPWorkspace::kTableExact : ICPWorkspace::kKDTree;
    if (search == ICPWorkspace::kTable || search == ICPWorkspace::kTableExact)
        table_.Build(target_, *tree_, cfg.table_cell, *std::max_element(cfg.distances.begin(), cfg.distances.end()),
                     search == ICPWorkspace::kTableExact);
    ws_.search = search;
    ws_.table = &table_;
    // hint trials in parallel (config hints.threads): one workspace and one point buffer per worker
    const int ht = std::max(1, std::min(cfg.hint_threads, 8));
    hw_.clear();
    hw_.resize(ht > 1 ? std::size_t(ht) : 0);
    for (auto& w : hw_) {
        w.ws.Reserve(std::size_t(maxs));
        w.ws.search = search;
        w.ws.table = &table_;
        w.pts.assign(std::size_t(maxs), Eigen::Vector3d::Zero());
    }
    for (auto& v : hint_inlier_) v.assign(ht > 1 ? std::size_t(maxs) : 0, 0);
    pool_.start(ht);
    orientations_ = search_orientations(cfg.search_orientations);
    cand_.resize(orientations_.size() * kMaxCenters);
    const Vec3 ext = cfg.work_volume.max - cfg.work_volume.min;
    gx_ = int(ext.x() / 0.04) + 1;
    gy_ = int(ext.y() / 0.04) + 1;
    gz_ = int(ext.z() / 0.04) + 1;
    gcount_.assign(std::size_t(gx_) * gy_ * gz_, 0);
    glabel_.assign(gcount_.size(), -1);
    gqueue_.assign(gcount_.size(), 0);
    if (!cfg.background_map.empty() && !bg_.load(cfg.background_map, err)) return false;
    for (int c = 0; c < 2; c++) {
        cams_[c].name = cfg.cam_name[c];
        cams_[c].serial = cfg.cam_serial[c];
        cams_[c].set_pose(cfg.cam_T[c]);
    }
    max_extent_ = (model_.bmax - model_.bmin).maxCoeff();
    const std::size_t step = std::max<std::size_t>(1, model_.points.size() / std::size_t(std::max(1, cfg.visibility_points)));
    for (std::size_t i = 0; i < model_.points.size(); i += step) {
        vis_pts_.push_back(model_.points[i]);
        vis_nrm_.push_back(model_.normals[i]);
    }
    reset_to_search();
    return true;
}

bool TrackerCore::set_cameras(const cap::CameraInfo info[2], std::string& err, std::string* note) {
    int k_of[2] = {-1, -1};
    for (int c = 0; c < 2; c++)
        for (int k = 0; k < 2; k++)
            if (cfg_.cam_serial[k] == info[c].serial) k_of[c] = k;
    std::string n;
    if (k_of[0] < 0 || k_of[1] < 0 || k_of[0] == k_of[1]) {
        k_of[0] = 0;
        k_of[1] = 1;
        n = std::string("camera serials ") + info[0].serial + " / " + info[1].serial +
            " not in the configuration: default assignment D435i_1 = A, D435i_2 = B";
    } else {
        n = std::string("A ") + info[0].serial + " = " + cfg_.cam_name[k_of[0]] + ", B " + info[1].serial + " = " +
            cfg_.cam_name[k_of[1]];
    }
    for (int c = 0; c < 2; c++) {
        Camera& cam = cams_[c];
        const int k = k_of[c];
        cam.name = cfg_.cam_name[k];
        cam.serial = info[c].serial;
        cam.set_pose(cfg_.cam_T[k]);
        cam.width = int(info[c].width ? info[c].width : 848);
        cam.height = int(info[c].height ? info[c].height : 480);
        cam.fx = info[c].fx;
        cam.fy = info[c].fy;
        cam.ppx = info[c].ppx;
        cam.ppy = info[c].ppy;
        cam.depth_scale = info[c].depth_scale > 0 ? info[c].depth_scale : 0.001;
        if (cam.fx <= 0 || cam.fy <= 0) { err = "camera intrinsics missing in the pair header"; return false; }
    }
    if (note) *note = n;
    return true;
}

void TrackerCore::reset_to_search() {
    state_ = kStateSearch;
    kf_.reset();
    of_.reset();
    rejected_in_row_ = 0;
    search_idx_ = 0;
}

void TrackerCore::force_track(const Mat4& T) {
    kf_.init(T.block<3, 1>(0, 3), Mat3::Identity() * 1e-6, 0.01);
    of_.init(T.block<3, 3>(0, 0));
    state_ = kStateTrack;
    rejected_in_row_ = 0;
}

Mat4 TrackerCore::filtered_pose() const {
    if (!kf_.ok()) return Mat4::Identity();
    return make_pose(of_.R(), kf_.p());
}

Mat4 TrackerCore::predicted_pose(double dt) const {
    return make_pose(so3_exp(of_.w() * dt) * of_.R(), kf_.p() + kf_.v() * dt);
}

void TrackerCore::crops(CropRect out[2]) {
    for (int c = 0; c < 2; c++) {
        const Camera& cam = cams_[c];
        CropRect r;
        if (!(cam_mask_ & (1 << c))) {
            out[c] = crop_[c] = r;
            continue;
        }
        Vec3 lo, hi;
        Mat4 T = Mat4::Identity();
        if (state_ == kStateSearch || !kf_.ok()) {
            lo = cfg_.work_volume.min;
            hi = cfg_.work_volume.max;
        } else {
            const double m = state_ == kStateTrack ? cfg_.box_margin
                                                   : cfg_.box_margin + (cfg_.lost_box_factor - 1.0) * 0.5 * max_extent_;
            lo = model_.bmin - Vec3::Constant(m);
            hi = model_.bmax + Vec3::Constant(m);
            T = predicted_pose(period_s_);
        }
        double umin = 1e9, umax = -1e9, vmin = 1e9, vmax = -1e9;
        bool full = false;
        for (int k = 0; k < 8; k++) {
            const Vec3 corner((k & 1) ? hi.x() : lo.x(), (k & 2) ? hi.y() : lo.y(), (k & 4) ? hi.z() : lo.z());
            const Vec3 pc = xform(cam.T_cam_rig, xform(T, corner));
            double u, v;
            if (pc.z() < 0.05 || !cam.project(pc, u, v)) { full = true; break; }
            umin = std::min(umin, u); umax = std::max(umax, u);
            vmin = std::min(vmin, v); vmax = std::max(vmax, v);
        }
        if (full) {
            r.u0 = 0; r.v0 = 0; r.u1 = cam.width - 1; r.v1 = cam.height - 1;
        } else {
            const double pm = cfg_.pixel_margin;
            r.u0 = std::max(0, int(std::floor(umin - pm)));
            r.v0 = std::max(0, int(std::floor(vmin - pm)));
            r.u1 = std::min(cam.width - 1, int(std::ceil(umax + pm)));
            r.v1 = std::min(cam.height - 1, int(std::ceil(vmax + pm)));
        }
        out[c] = crop_[c] = r;
    }
}

int TrackerCore::build_cloud(const uint16_t* const img[2]) {
    TRK_PROF(kBuild);
    npts_ = 0;
    const int cap = int(pts_.size());
    for (int c = 0; c < 2; c++) {
        const CropRect& r = crop_[c];
        if (r.empty() || !img[c]) continue;
        const Camera& cam = cams_[c];
        const double area = double(r.u1 - r.u0 + 1) * double(r.v1 - r.v0 + 1);
        const int per_cam = state_ == kStateSearch ? cfg_.max_points_per_camera_search : cfg_.max_points_per_camera;
        const int n = std::max(1, int(std::ceil(std::sqrt(area / double(per_cam)))));
        const Mat3 R = cam.T_rig_cam.block<3, 3>(0, 0);
        const Vec3 t = cam.T_rig_cam.block<3, 1>(0, 3);
        const int start = npts_;
        for (int v = r.v0; v <= r.v1 && npts_ < cap; v += n) {
            const uint16_t* row = img[c] + std::size_t(v) * std::size_t(cam.width);
            for (int u = r.u0; u <= r.u1; u += n) {
                const uint16_t d = row[u];
                if (!d) continue;
                const double z = d * cam.depth_scale;
                if (z < cfg_.min_depth || z > cfg_.max_depth) continue;
                pts_[std::size_t(npts_)] = R * cam.backproject(u, v, z) + t;
                pcam_[std::size_t(npts_)] = uint8_t(c);
                if (++npts_ >= cap || npts_ - start >= per_cam) break;
            }
            if (npts_ - start >= per_cam) break;
        }
    }
    return npts_;
}

int TrackerCore::filter_cloud(const Mat4* T_obj_rig, double margin) {
    TRK_PROF(kFilter);
    nf_ = 0;
    Vec3 lo = model_.bmin - Vec3::Constant(margin), hi = model_.bmax + Vec3::Constant(margin);
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();
    if (T_obj_rig) {
        R = T_obj_rig->block<3, 3>(0, 0);
        t = T_obj_rig->block<3, 1>(0, 3);
    }
    for (int i = 0; i < npts_; i++) {
        const Vec3& p = pts_[std::size_t(i)];
        if (!cfg_.work_volume.contains(p)) continue;
        if (bg_.occupied(p)) continue;
        if (T_obj_rig) {
            const Vec3 q = R * p + t;
            if ((q.array() < lo.array()).any() || (q.array() > hi.array()).any()) continue;
        }
        fpts_[std::size_t(nf_)] = p;
        fcam_[std::size_t(nf_)] = pcam_[std::size_t(i)];
        nf_++;
    }
    return nf_;
}

int TrackerCore::select_solver_points(int max_points) {
    max_points = std::min<int>(max_points, int(spts_.size()));
    if (nf_ <= max_points) {
        for (int i = 0; i < nf_; i++) {
            spts_[std::size_t(i)] = fpts_[std::size_t(i)];
            scam_[std::size_t(i)] = fcam_[std::size_t(i)];
            sidx_[std::size_t(i)] = i;
        }
        ns_ = nf_;
    } else {
        const double step = double(nf_) / double(max_points);
        for (int i = 0; i < max_points; i++) {
            const std::size_t j = std::size_t(double(i) * step);
            spts_[std::size_t(i)] = fpts_[j];
            scam_[std::size_t(i)] = fcam_[j];
            sidx_[std::size_t(i)] = int(j);
        }
        ns_ = max_points;
    }
    return ns_;
}

double TrackerCore::visibility(const Mat4& T_rig_obj, int& checked) const {
    TRK_PROF(kVisibility);
    int seen_through = 0;
    checked = 0;
    for (int c = 0; c < 2; c++) {
        const CropRect& r = crop_[c];
        if (r.empty() || !img_[c]) continue;
        const Camera& cam = cams_[c];
        const Mat4 T = cam.T_cam_rig * T_rig_obj;   // object -> camera
        const Mat3 R = T.block<3, 3>(0, 0);
        const Vec3 t = T.block<3, 1>(0, 3);
        for (std::size_t i = 0; i < vis_pts_.size(); i++) {
            const Vec3 p = R * vis_pts_[i] + t;
            const Vec3 n = R * vis_nrm_[i];
            if (p.z() < cfg_.min_depth || n.dot(-p) < 0.3 * p.norm()) continue;   // not facing the camera
            const int u = int(std::lround(cam.fx * p.x() / p.z() + cam.ppx)), v = int(std::lround(cam.fy * p.y() / p.z() + cam.ppy));
            if (u < r.u0 || u > r.u1 || v < r.v0 || v > r.v1) continue;
            const uint16_t d = img_[c][std::size_t(v) * std::size_t(cam.width) + std::size_t(u)];
            if (!d) continue;                                   // no measurement: unknown
            const double z = d * cam.depth_scale;
            if (z < p.z() - cfg_.visibility_tol) continue;      // something in front (occlusion): unknown
            checked++;
            if (z > p.z() + cfg_.visibility_tol) seen_through++;
        }
    }
    return checked ? double(seen_through) / double(checked) : 1.0;
}

int TrackerCore::select_points_near(const Vec3& c, double radius, int max_points) {
    TRK_PROF(kSelect);
    const double r2 = radius * radius;
    int cnt = 0;
    for (int i = 0; i < nf_; i++) cnt += (fpts_[std::size_t(i)] - c).squaredNorm() <= r2;
    max_points = std::min<int>(max_points, int(spts_.size()));
    const double step = cnt > max_points ? double(cnt) / double(max_points) : 1.0;
    int k = 0, out = 0;
    double next = 0;
    for (int i = 0; i < nf_ && out < max_points; i++) {
        if ((fpts_[std::size_t(i)] - c).squaredNorm() > r2) continue;
        if (double(k++) + 1e-9 >= next) {
            spts_[std::size_t(out)] = fpts_[std::size_t(i)];
            scam_[std::size_t(out)] = fcam_[std::size_t(i)];
            sidx_[std::size_t(out)] = i;
            out++;
            next += step;
        }
    }
    ns_ = out;
    return ns_;
}

// Search start positions: connected components of occupied 4 cm cells; components whose size fits the object
// (largest extent <= 2 x the object's largest extent) and with enough points, largest first (at most kMaxCenters).
// Without such a component: the centroid of the whole cloud.
int TrackerCore::find_search_centers() {
    TRK_PROF(kCenters);
    std::fill(gcount_.begin(), gcount_.end(), uint16_t(0));
    std::fill(glabel_.begin(), glabel_.end(), -1);
    const Vec3 o = cfg_.work_volume.min;
    auto cell = [&](const Vec3& p) {
        const int ix = std::min(gx_ - 1, std::max(0, int((p.x() - o.x()) / 0.04)));
        const int iy = std::min(gy_ - 1, std::max(0, int((p.y() - o.y()) / 0.04)));
        const int iz = std::min(gz_ - 1, std::max(0, int((p.z() - o.z()) / 0.04)));
        return (ix * gy_ + iy) * gz_ + iz;
    };
    for (int i = 0; i < nf_; i++) {
        uint16_t& c = gcount_[std::size_t(cell(fpts_[std::size_t(i)]))];
        if (c < 65535) c++;
    }
    struct Comp { int cells; long points; Vec3 lo, hi; };
    Comp comps[64];
    int nc = 0;
    for (int start = 0; start < int(gcount_.size()) && nc < 64; start++) {
        if (gcount_[std::size_t(start)] < 3 || glabel_[std::size_t(start)] >= 0) continue;
        int head = 0, tail = 0;
        gqueue_[std::size_t(tail++)] = start;
        glabel_[std::size_t(start)] = nc;
        Comp cp{0, 0, Vec3::Constant(1e9), Vec3::Constant(-1e9)};
        while (head < tail) {
            const int q = gqueue_[std::size_t(head++)];
            const int ix = q / (gy_ * gz_), iy = (q / gz_) % gy_, iz = q % gz_;
            cp.cells++;
            cp.points += gcount_[std::size_t(q)];
            const Vec3 lo = o + Vec3(ix, iy, iz) * 0.04;
            cp.lo = cp.lo.cwiseMin(lo);
            cp.hi = cp.hi.cwiseMax(lo + Vec3::Constant(0.04));
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++) {
                        const int jx = ix + dx, jy = iy + dy, jz = iz + dz;
                        if (jx < 0 || jy < 0 || jz < 0 || jx >= gx_ || jy >= gy_ || jz >= gz_) continue;
                        const int j = (jx * gy_ + jy) * gz_ + jz;
                        if (gcount_[std::size_t(j)] < 3 || glabel_[std::size_t(j)] >= 0) continue;
                        glabel_[std::size_t(j)] = nc;
                        gqueue_[std::size_t(tail++)] = j;
                    }
        }
        comps[nc++] = cp;
    }
    // Candidate start positions: object-sized components (largest first) and density peaks (occupied cells with the
    // most points in their 12 cm neighbourhood, at least 12 cm apart; they find the object also when it is connected
    // to a support or a hand). Up to 16 candidates.
    constexpr int kMaxCand = 16;
    Vec3 cand[kMaxCand];
    int ncand = 0;
    const double max_ext = 2.0 * max_extent_ + 0.08;
    bool used[64] = {false};
    while (ncand < kMaxCand / 2) {
        int best = -1;
        for (int i = 0; i < nc; i++) {
            if (used[i] || comps[i].points < cfg_.min_points) continue;
            if ((comps[i].hi - comps[i].lo).maxCoeff() > max_ext) continue;
            if (best < 0 || comps[i].points > comps[best].points) best = i;
        }
        if (best < 0) break;
        used[best] = true;
        Vec3 c = Vec3::Zero();
        long n = 0;
        for (int i = 0; i < nf_; i++)
            if (glabel_[std::size_t(cell(fpts_[std::size_t(i)]))] == best) { c += fpts_[std::size_t(i)]; n++; }
        cand[ncand++] = c / double(std::max<long>(1, n));
    }
    while (ncand < kMaxCand) {
        int best = -1;
        long best_sum = 0;
        for (int q = 0; q < int(gcount_.size()); q++) {
            if (gcount_[std::size_t(q)] < 3) continue;
            const int ix = q / (gy_ * gz_), iy = (q / gz_) % gy_, iz = q % gz_;
            const Vec3 cc = o + (Vec3(ix, iy, iz) + Vec3::Constant(0.5)) * 0.04;
            bool near = false;
            for (int k = 0; k < ncand && !near; k++) near = (cand[k] - cc).cwiseAbs().maxCoeff() < 0.12;
            if (near) continue;
            long sum = 0;
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++) {
                        const int jx = ix + dx, jy = iy + dy, jz = iz + dz;
                        if (jx < 0 || jy < 0 || jz < 0 || jx >= gx_ || jy >= gy_ || jz >= gz_) continue;
                        sum += gcount_[std::size_t((jx * gy_ + jy) * gz_ + jz)];
                    }
            if (sum > best_sum) { best_sum = sum; best = q; }
        }
        if (best < 0 || best_sum < cfg_.min_points) break;
        const int ix = best / (gy_ * gz_), iy = (best / gz_) % gy_, iz = best % gz_;
        const Vec3 cc = o + (Vec3(ix, iy, iz) + Vec3::Constant(0.5)) * 0.04;
        Vec3 c = Vec3::Zero();
        long n = 0;
        for (int i = 0; i < nf_; i++)
            if ((fpts_[std::size_t(i)] - cc).cwiseAbs().maxCoeff() <= 0.06) { c += fpts_[std::size_t(i)]; n++; }
        cand[ncand++] = n ? Vec3(c / double(n)) : cc;
    }
    // Rank: the object is in view of both cameras, so score = min(points of A, points of B) within reach of the
    // position (model radius + 20 mm); clutter seen by one camera only scores 0. Keep the best kMaxCenters with a
    // score of at least min_points / 4; if none qualifies, the candidate with the most points.
    const double r2 = (model_.radius + 0.02) * (model_.radius + 0.02);
    long score[kMaxCand], total[kMaxCand];
    for (int k = 0; k < ncand; k++) {
        long na = 0, nb = 0;
        for (int i = 0; i < nf_; i++)
            if ((fpts_[std::size_t(i)] - cand[k]).squaredNorm() <= r2) (fcam_[std::size_t(i)] ? nb : na)++;
        score[k] = std::min(na, nb);
        total[k] = na + nb;
    }
    bool taken[kMaxCand] = {false};
    n_centers_ = 0;
    while (n_centers_ < kMaxCenters) {
        int best = -1;
        for (int k = 0; k < ncand; k++)
            if (!taken[k] && score[k] >= cfg_.min_points / 4 && (best < 0 || score[k] > score[best])) best = k;
        if (best < 0) break;
        taken[best] = true;
        centers_[n_centers_++] = cand[best];
    }
    if (n_centers_ == 0 && ncand > 0) {
        int best = 0;
        for (int k = 1; k < ncand; k++)
            if (total[k] > total[best]) best = k;
        centers_[n_centers_++] = cand[best];
    }
    if (n_centers_ == 0 && nf_ > 0) {
        Vec3 c = Vec3::Zero();
        for (int i = 0; i < nf_; i++) c += fpts_[std::size_t(i)];
        centers_[n_centers_++] = c / double(nf_);
    }
    return n_centers_;
}

// Fit check and pose fields of an ICP result (DESIGN.md section 5).
void TrackerCore::evaluate_into(const RegistrationResultPrealloc& r, const Mat4* pred, PoseRecord& pose, bool check_jump, int ns,
                                Mat4& T_out) const {
    const Mat4 T_obj_rig = r.transformation;
    const Mat4 T = inverse_pose(T_obj_rig);
    const Mat3 R = T.block<3, 3>(0, 0);
    for (int k = 0; k < 3; k++) pose.icp_position[k] = T(k, 3);
    quat_from_R(R, pose.icp_orientation);
    pose.fit_rmse = r.inlier_rmse;
    pose.inlier_ratio = r.fitness;
    pose.point_count = uint32_t(ns);
    pose.icp_iterations = uint32_t(r.num_iteration);
    double min_eig = 0;
    Eigen::Matrix<double, 6, 6> C = Eigen::Matrix<double, 6, 6>::Zero();
    if (r.has_system && r.weight_sum > 0) {
        TRK_PROF(kFitCheck);
        const double L = model_.radius > 0 ? model_.radius : 0.05;
        const Vec3 c = r.centroid;
        Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Identity();
        A.block<3, 3>(0, 0) = Mat3::Identity() / L;
        A.block<3, 3>(0, 3) = -(Mat3() << 0, -c.z(), c.y(), c.z(), 0, -c.x(), -c.y(), c.x(), 0).finished() / L;
        const Eigen::Matrix<double, 6, 6> Hn = A * r.JTJ * A.transpose() / r.weight_sum;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Hn, Eigen::EigenvaluesOnly);
        min_eig = es.eigenvalues()(0);
        // covariance of the increment (rotation, translation of T_object_from_rig), then of the output pose
        const Eigen::Matrix<double, 6, 6> Cx = r.inlier_rmse * r.inlier_rmse * r.JTJ.ldlt().solve(
                Eigen::Matrix<double, 6, 6>::Identity());
        const Mat3 Rt = T_obj_rig.block<3, 3>(0, 0).transpose();   // R^T of T_object_from_rig = rotation of T
        C.block<3, 3>(0, 0) = Rt * Cx.block<3, 3>(3, 3) * Rt.transpose();   // position
        C.block<3, 3>(3, 3) = Rt * Cx.block<3, 3>(0, 0) * Rt.transpose();   // rotation (rig frame)
        C.block<3, 3>(0, 3) = Rt * Cx.block<3, 3>(3, 0) * Rt.transpose();
        C.block<3, 3>(3, 0) = C.block<3, 3>(0, 3).transpose();
    }
    for (int i = 0; i < 36; i++) pose.covariance[i] = C(i / 6, i % 6);
    pose.min_eigenvalue = min_eig;
    uint32_t why = 0;
    if (ns < cfg_.min_points) why |= kRejectFewPoints;
    if (!(r.inlier_rmse <= cfg_.max_rmse) || r.num_corres == 0) why |= kRejectRmse;
    if (!(r.fitness >= cfg_.min_inlier_ratio)) why |= kRejectInliers;
    if (!(min_eig >= cfg_.min_eigenvalue)) why |= kRejectEigen;
    if (check_jump && pred) {
        const double dp = (T.block<3, 1>(0, 3) - pred->block<3, 1>(0, 3)).norm();
        const double dr = rotation_angle(R * pred->block<3, 3>(0, 0).transpose());
        if (dp > cfg_.max_jump_m || dr > cfg_.max_jump_rad) why |= kRejectJump;
    }
    int checked = 0;
    const double viol = visibility(T, checked);
    pose.free_space_violation = float(viol);
    if (checked < cfg_.min_visibility_checked || viol > cfg_.max_violation) why |= kRejectVisibility;
    pose.reject_reasons = why;
    pose.valid = why == 0 ? 1 : 0;
    T_out = T;
}

void TrackerCore::evaluate(const RegistrationResultPrealloc& r, const Mat4* pred, PoseRecord& pose, bool check_jump) {
    evaluate_into(r, pred, pose, check_jump, ns_, last_icp_T_rig_obj_);
}

// One time slice of the search (DESIGN.md section 7). Returns true when a candidate was accepted (pose valid).
bool TrackerCore::search_step(int64_t t_start, PoseRecord& pose) {
    TRK_PROF_CTX(kSearch);
    if (search_idx_ == 0) {   // new round: start positions from the current cloud
        find_search_centers();
        have_best_ = false;
        best_fit_ = -1;
        best_rmse_ = 1e9;
    }
    // Steps per start position: K coarse candidates, then the B best refined; a passing candidate of a start position
    // is accepted at once (positions are ranked, the most likely first); otherwise the next position follows.
    const int K = int(orientations_.size()), C = std::max(1, n_centers_);
    const int B = std::min(cfg_.search_refine_best, K), S = K + B;
    pose.search_total = uint32_t(C * S);
    if (nf_ < cfg_.min_points || n_centers_ == 0) {
        pose.reject_reasons = kRejectFewPoints | kRejectNoSolve;
        pose.point_count = uint32_t(nf_);
        pose.search_done = uint32_t(search_idx_);
        search_idx_ = 0;
        return false;
    }
    const double radius = model_.radius + 0.02;
    cur_center_ = -1;
    auto use_center = [&](int c) {
        if (c != cur_center_) {
            select_points_near(centers_[c], radius, cfg_.solver_points_search);
            cur_center_ = c;
        }
    };
    const double coarse[2] = {cfg_.distances[0], cfg_.distances.size() > 1 ? cfg_.distances[1] : cfg_.distances[0]};
    const ICPConvergenceCriteria crit_coarse(cfg_.rel_fitness, cfg_.rel_rmse, cfg_.search_coarse_iterations);
    const ICPConvergenceCriteria crit(cfg_.rel_fitness, cfg_.rel_rmse, cfg_.max_iterations);
    const int64_t limit = int64_t(cfg_.search_time_ms * 1e6);
    bool accepted = false;
    pose.reject_reasons = kRejectNoSolve;
    auto refine = [&](const Candidate& cd) {
        use_center(cd.center);
        if (ns_ < cfg_.min_points) return -1.0;
        const RegistrationResultPrealloc r = RegistrationICPPrealloc(
                spts_.data(), std::size_t(ns_), target_, *tree_, cfg_.distances.data(), int(cfg_.distances.size()),
                cd.T_obj_rig, kptr_, crit, ws_);
        PoseRecord tmp;
        std::memset(&tmp, 0, sizeof tmp);
        evaluate(r, nullptr, tmp, false);
        const double score = r.fitness - double(tmp.free_space_violation);
        if (tmp.valid && (score > best_fit_ || (score == best_fit_ && r.inlier_rmse < best_rmse_))) {
            best_ = r;
            have_best_ = true;
            best_fit_ = score;
            best_rmse_ = r.inlier_rmse;
            best_center_ = cd.center;
        }
        return tmp.valid ? score : -1.0;
    };
    int steps = 0;
    while (search_idx_ < C * S) {
        const int c = search_idx_ / S, j = search_idx_ % S;
        const int64_t now = mono_ns();
        // time slice: stop before a step that would not end within the limit (estimated from earlier steps); at least
        // one step per cycle, so that the search always progresses
        const double est = std::min(j < K ? est_coarse_ns_ : est_refine_ns_, double(limit));
        if (steps > 0 && double(now - t_start) + est > double(limit)) break;
        steps++;
        Candidate* seg = &cand_[std::size_t(c * K)];
        if (j < K) {
            use_center(c);
            const Mat3& R = orientations_[std::size_t(j)];
            const Mat4 T_obj_rig = inverse_pose(make_pose(R, centers_[c] - R * model_.centroid));
            seg[j] = {T_obj_rig, 0.0, 1e9, 1.0, c};
            if (ns_ >= cfg_.min_points) {
                const RegistrationResultPrealloc r =
                        RegistrationICPPrealloc(spts_.data(), std::size_t(ns_), target_, *tree_, coarse, 2, T_obj_rig,
                                                kptr_, crit_coarse, ws_);
                int checked = 0;
                const double viol = visibility(inverse_pose(r.transformation), checked);
                seg[j] = {r.transformation, r.fitness, r.inlier_rmse, checked ? viol : 1.0, c};
            }
            search_idx_++;
            est_coarse_ns_ = 0.8 * est_coarse_ns_ + 0.2 * double(mono_ns() - now);
            if (j == K - 1)   // rank this position's candidates: fitness minus free-space violations, then RMSE
                std::sort(seg, seg + K, [](const Candidate& x, const Candidate& y) {
                    const double sx = x.fitness - x.violation, sy = y.fitness - y.violation;
                    return sx > sy || (sx == sy && x.rmse < y.rmse);
                });
        } else {
            refine(seg[j - K]);
            search_idx_++;
            est_refine_ns_ = 0.8 * est_refine_ns_ + 0.2 * double(mono_ns() - now);
            if (j == S - 1) {   // this start position is done
                if (have_best_) {
                    use_center(best_center_);
                    evaluate(best_, nullptr, pose, false);
                    accepted = pose.valid != 0;
                }
                if (accepted || c == C - 1) {
                    search_idx_ = 0;   // accepted, or the round is over: new start positions next time
                    if (!accepted) pose.reject_reasons |= kRejectNoSolve;
                    break;
                }
                have_best_ = false;
                best_fit_ = -1;
                best_rmse_ = 1e9;
            }
        }
    }
    pose.search_done = uint32_t(search_idx_);
    pose.point_count = uint32_t(ns_);
    return accepted;
}

void TrackerCore::set_hints(const Mat4* T_rig_obj, int n, bool in_order) {
    hints_in_order_ = in_order;
    n_hints_ = std::max(0, std::min(n, int(kMaxHints)));
    for (int i = 0; i < n_hints_; i++) hints_[i] = T_rig_obj[i];
}

// External pose hypotheses (flow module): full ICP from each on the points near its position, the fit check of the
// search (no jump test); the best passing one (fitness minus free-space violations, then RMSE) is accepted.
// The points select_points_near() would put into spts_, written to out (no other state touched).
int TrackerCore::select_near_into(const Vec3& c, double radius, int max_points, Eigen::Vector3d* out) const {
    const double r2 = radius * radius;
    int cnt = 0;
    for (int i = 0; i < nf_; i++) cnt += (fpts_[std::size_t(i)] - c).squaredNorm() <= r2;
    max_points = std::min<int>(max_points, int(spts_.size()));
    const double step = cnt > max_points ? double(cnt) / double(max_points) : 1.0;
    int k = 0, n = 0;
    double next = 0;
    for (int i = 0; i < nf_ && n < max_points; i++) {
        if ((fpts_[std::size_t(i)] - c).squaredNorm() > r2) continue;
        if (double(k++) + 1e-9 >= next) {
            out[n++] = fpts_[std::size_t(i)];
            next += step;
        }
    }
    return n;
}

// One hint on one worker: what the loop body of try_hints() does, into hres_[i].
void TrackerCore::hint_job(int i, int worker) {
    HintWorker& w = hw_[std::size_t(worker)];
    HintResult& out = hres_[i];
    const Vec3 c = hints_[i].block<3, 3>(0, 0) * model_.centroid + hints_[i].block<3, 1>(0, 3);
    out.c = c;
    out.ran = false;
    out.valid = false;
    out.ns = select_near_into(c, model_.radius + 0.02, cfg_.solver_points_search, w.pts.data());
    if (out.ns < cfg_.min_points) return;
    const ICPConvergenceCriteria crit(cfg_.rel_fitness, cfg_.rel_rmse, cfg_.max_iterations);
    out.r = RegistrationICPPrealloc(w.pts.data(), std::size_t(out.ns), target_, *tree_, cfg_.distances.data(), int(cfg_.distances.size()),
                                    inverse_pose(hints_[i]), kptr_, crit, w.ws);
    PoseRecord tmp;
    std::memset(&tmp, 0, sizeof tmp);
    evaluate_into(out.r, nullptr, tmp, false, out.ns, out.T);
    out.score = out.r.fitness - double(tmp.free_space_violation);
    out.valid = tmp.valid != 0;
    out.ran = true;
    std::copy(w.ws.inlier.begin(), w.ws.inlier.end(), hint_inlier_[i].begin());
}

// try_hints() with the ICP runs of the hints on the pool. The runs are independent; the choice afterwards goes through
// the hints in their order with the rule of the sequential code, so the result does not depend on the thread timing
// (equal scores and RMSE: the lower hint index).
bool TrackerCore::try_hints_parallel(PoseRecord& pose) {
    TRK_PROF_CTX(kHint);
    const int n = n_hints_;
    n_hints_ = 0;
    hints_tried_ = 0;
    if (nf_ < cfg_.min_points) return false;
    pool_.run(n, [](void* self, int job, int worker) { static_cast<TrackerCore*>(self)->hint_job(job, worker); }, this);
    bool have = false;
    double best_score = -1, best_rmse = 1e9;
    Vec3 best_c = Vec3::Zero();
    int last_ran = -1;
    for (int i = 0; i < n; i++) {
        const HintResult& h = hres_[i];
        hints_tried_++;
        if (!h.ran) continue;
        last_ran = i;
        if (h.valid && (h.score > best_score || (h.score == best_score && h.r.inlier_rmse < best_rmse))) {
            best_ = h.r;
            have = true;
            best_score = h.score;
            best_rmse = h.r.inlier_rmse;
            best_c = h.c;
            if (hints_in_order_) break;
        }
    }
    if (last_ran >= 0) {   // the state the sequential code leaves: flags and pose of the last run
        ws_.inlier.assign(hint_inlier_[last_ran].begin(), hint_inlier_[last_ran].begin() + hres_[last_ran].ns);
        last_icp_T_rig_obj_ = hres_[last_ran].T;
    }
    if (!have) return false;
    select_points_near(best_c, model_.radius + 0.02, cfg_.solver_points_search);
    evaluate(best_, nullptr, pose, false);
    return pose.valid != 0;
}

bool TrackerCore::try_hints(PoseRecord& pose) {
    if (!hw_.empty()) return try_hints_parallel(pose);
    TRK_PROF_CTX(kHint);
    const int n = n_hints_;
    n_hints_ = 0;
    hints_tried_ = 0;
    if (nf_ < cfg_.min_points) return false;
    const ICPConvergenceCriteria crit(cfg_.rel_fitness, cfg_.rel_rmse, cfg_.max_iterations);
    const double radius = model_.radius + 0.02;
    bool have = false;
    double best_score = -1, best_rmse = 1e9;
    Vec3 best_c = Vec3::Zero();
    for (int i = 0; i < n; i++) {
        const Vec3 c = hints_[i].block<3, 3>(0, 0) * model_.centroid + hints_[i].block<3, 1>(0, 3);
        select_points_near(c, radius, cfg_.solver_points_search);
        hints_tried_++;
        if (ns_ < cfg_.min_points) continue;
        const RegistrationResultPrealloc r = RegistrationICPPrealloc(
                spts_.data(), std::size_t(ns_), target_, *tree_, cfg_.distances.data(), int(cfg_.distances.size()),
                inverse_pose(hints_[i]), kptr_, crit, ws_);
        PoseRecord tmp;
        std::memset(&tmp, 0, sizeof tmp);
        evaluate(r, nullptr, tmp, false);
        const double score = r.fitness - double(tmp.free_space_violation);
        if (tmp.valid && (score > best_score || (score == best_score && r.inlier_rmse < best_rmse))) {
            best_ = r;
            have = true;
            best_score = score;
            best_rmse = r.inlier_rmse;
            best_c = c;
            if (hints_in_order_) break;
        }
    }
    if (!have) return false;
    select_points_near(best_c, radius, cfg_.solver_points_search);
    evaluate(best_, nullptr, pose, false);
    return pose.valid != 0;
}

void TrackerCore::process(const cap::PairRecord& rec, const uint16_t* const img[2], PoseRecord& pose, CloudRecord* cloud) {
    const int64_t t0 = mono_ns();
    img_[0] = img[0];
    img_[1] = img[1];
    std::memset(&pose, 0, sizeof pose);
    pose.version = kPoseRecordVersion;
    pose.pair_seq = rec.pair_seq;
    pose.pair_dt_ms = rec.dt_ms;
    for (int c = 0; c < 2; c++) pose.capture_time_mono_ns[c] = capture_mono_ns(rec, c);
    const int64_t meas = (pose.capture_time_mono_ns[0] + pose.capture_time_mono_ns[1]) / 2;
    pose.meas_time_mono_ns = meas;
    const double dt = last_meas_ns_ ? double(meas - last_meas_ns_) / 1e9 : 0.0;
    if (dt > 0.001 && dt < 0.5) period_s_ = 0.9 * period_s_ + 0.1 * std::min(dt, 0.05);
    last_meas_ns_ = meas;
    // prediction to this measurement time
    if (kf_.ok()) {
        kf_.predict(dt, cfg_.accel_noise);
        of_.predict(dt);
    }
    const TrackState state_in = state_;
    const int64_t t1 = mono_ns();
    build_cloud(img);
    const int64_t t2 = mono_ns();
    Mat4 pred = Mat4::Identity(), pred_obj_rig = Mat4::Identity();
    const bool have_pred = state_ != kStateSearch && kf_.ok();
    if (have_pred) {
        pred = filtered_pose();
        pred_obj_rig = inverse_pose(pred);
    }
    const double margin = state_ == kStateTrack ? cfg_.box_margin
                                                : cfg_.box_margin + (cfg_.lost_box_factor - 1.0) * 0.5 * max_extent_;
    filter_cloud(have_pred ? &pred_obj_rig : nullptr, margin);
    const int64_t t3 = mono_ns();
    bool icp_ran = false;
    int64_t t4 = t3;
    if (state_ == kStateSearch) {
        hints_tried_ = 0;
        bool ok = n_hints_ > 0 && try_hints(pose);
        const bool by_hint = ok;
        if (ok) search_idx_ = 0;
        else ok = search_step(t3, pose);
        pose.hints_tried = uint8_t(hints_tried_);
        pose.by_hint = by_hint ? 1 : 0;
        icp_ran = true;
        t4 = mono_ns();
        if (ok) {
            const Mat4 T = last_icp_T_rig_obj_;
            Mat3 Rp;
            for (int i = 0; i < 9; i++) Rp(i / 3, i % 3) = pose.covariance[(i / 3) * 6 + i % 3];
            kf_.init(T.block<3, 1>(0, 3), Rp + Mat3::Identity() * cfg_.min_meas_sigma * cfg_.min_meas_sigma, 0.5);
            of_.init(T.block<3, 3>(0, 0));
            state_ = kStateTrack;
            rejected_in_row_ = 0;
        }
    } else {
        select_solver_points(cfg_.solver_points_track);
        if (ns_ >= cfg_.min_points) {
            const RegistrationResultPrealloc r = RegistrationICPPrealloc(
                    spts_.data(), std::size_t(ns_), target_, *tree_, cfg_.distances.data(), int(cfg_.distances.size()),
                    pred_obj_rig, kptr_,
                    ICPConvergenceCriteria(cfg_.rel_fitness, cfg_.rel_rmse, cfg_.max_iterations), ws_);
            icp_ran = true;
            t4 = mono_ns();
            evaluate(r, &pred, pose, true);
        } else {
            t4 = mono_ns();
            pose.point_count = uint32_t(ns_);
            pose.reject_reasons = kRejectFewPoints | kRejectNoSolve;
            pose.valid = 0;
        }
    }
    const int64_t t5 = mono_ns();
    // tracking filter and state
    if (state_in != kStateSearch) {
        if (pose.valid) {
            const Mat4 T = last_icp_T_rig_obj_;
            Mat3 Rp;
            for (int i = 0; i < 9; i++) Rp(i / 3, i % 3) = pose.covariance[(i / 3) * 6 + i % 3];
            kf_.update(T.block<3, 1>(0, 3), Rp + Mat3::Identity() * cfg_.min_meas_sigma * cfg_.min_meas_sigma);
            of_.update(T.block<3, 3>(0, 0), dt, cfg_.ori_alpha, cfg_.ori_beta);
            rejected_in_row_ = 0;
            state_ = kStateTrack;
        } else {
            rejected_in_row_++;
            if (state_ == kStateTrack && int(rejected_in_row_) >= cfg_.lost_after) {
                state_ = kStateLost;
                kf_.zero_velocity();
                of_.zero_rate();
            }
            if (state_ == kStateLost && int(rejected_in_row_) >= cfg_.search_after) reset_to_search();
        }
    }
    const int64_t t6 = mono_ns();
    pose.track_state = state_;
    pose.rejected_in_row = rejected_in_row_;
    if (kf_.ok()) {
        const Mat4 F = filtered_pose();
        for (int k = 0; k < 3; k++) {
            pose.position[k] = F(k, 3);
            pose.velocity[k] = kf_.v()(k);
            pose.angular_velocity[k] = of_.w()(k);
        }
        quat_from_R(F.block<3, 3>(0, 0), pose.orientation);
    } else {
        pose.orientation[0] = 1;
    }
    if (!icp_ran) {
        pose.icp_orientation[0] = 1;
        std::fill(ws_.inlier.begin(), ws_.inlier.end(), uint8_t(0));
    }
    pose.stage_us[kStageBuild] = float(double(t2 - t1) / 1000.0);
    pose.stage_us[kStageFilter] = float(double(t3 - t2) / 1000.0);
    pose.stage_us[kStageSolve] = float(double(t4 - t3) / 1000.0);
    pose.stage_us[kStageCheck] = float(double(t5 - t4) / 1000.0);
    pose.stage_us[kStageTrack] = float(double(t6 - t5) / 1000.0);
    pose.stage_us[kStageTotal] = float(double(t6 - t0) / 1000.0);
    if (cloud) fill_cloud(*cloud, pose);
}

void TrackerCore::fill_cloud(CloudRecord& cloud, const PoseRecord& pose) {
    cloud.version = kCloudRecordVersion;
    cloud.pair_seq = pose.pair_seq;
    cloud.meas_time_mono_ns = pose.meas_time_mono_ns;
    cloud.track_state = pose.track_state;
    const bool flags = int(ws_.inlier.size()) == ns_;
    if (!cfg_.cloud_scene) {   // only the points given to the solver
        const int n = std::min<int>(ns_, int(kCloudMaxPoints));
        for (int i = 0; i < n; i++) {
            CloudPoint& p = cloud.points[i];
            const Vec3& s = spts_[std::size_t(i)];
            p.x = float(s.x());
            p.y = float(s.y());
            p.z = float(s.z());
            p.inlier = flags ? ws_.inlier[std::size_t(i)] : 0;
            p.camera = scam_[std::size_t(i)];
            p.reserved = 0;
        }
        cloud.point_count = uint32_t(n);
        return;
    }
    // the filtered cloud (crop box in track / lost, work volume in search), evenly thinned to at most 8192 points;
    // flag 1 / 0: inlier / outlier of the last ICP iteration, 2: not given to the solver
    for (int i = 0; i < nf_; i++) fflag_[std::size_t(i)] = 2;
    for (int i = 0; i < ns_; i++) fflag_[std::size_t(sidx_[std::size_t(i)])] = flags ? ws_.inlier[std::size_t(i)] : 0;
    const double step = nf_ > int(kCloudMaxPoints) ? double(nf_) / double(kCloudMaxPoints) : 1.0;
    uint32_t n = 0;
    for (double x = 0; x < double(nf_) && n < kCloudMaxPoints; x += step) {
        const std::size_t i = std::size_t(x);
        CloudPoint& p = cloud.points[n++];
        const Vec3& s = fpts_[i];
        p.x = float(s.x());
        p.y = float(s.y());
        p.z = float(s.z());
        p.inlier = fflag_[i];
        p.camera = fcam_[i];
        p.reserved = 0;
    }
    cloud.point_count = n;
}

}  // namespace trk
