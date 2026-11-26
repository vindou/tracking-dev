// C interface for the inference process of the flow module (../flow/live.py, Python with ctypes): everything that
// process needs from the tracker side, built from the same classes the tracker uses.
//   - configuration, calibration and model (centroid) as the tracker loads them
//   - the newest pair of the capture pair buffer (PairSource) and its cloud for the network: every stride-th pixel of
//     both images, pinhole back-projection, rig frame, points inside the work volume (as flow/scene.py Rig.cloud)
//   - the state of the tracker (newest record of the pose buffer)
//   - the writer of the hint buffer (HintRecord, DESIGN.md 13)
// Built as the shared library trk_flowio. Not used by the tracker itself. All functions return 0 on success unless
// stated; err (if not null) receives a message.
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "flowio.h"

#include "common/clock.hpp"
#include "core/config.hpp"
#include "core/pair_source.hpp"
#include "core/synth.hpp"
#include "core/tracker_core.hpp"
#include "records/records.hpp"
#include "shmring/shm_ring.hpp"

using namespace trk;

namespace {
struct Fio {
    TrackerConfig cfg;
    std::unique_ptr<TrackerCore> core;
    PairSource src;
    bool attached = false;
    std::vector<uint16_t> img[2];
    std::unique_ptr<RingReader> pose_r;
    int64_t pose_attach_ns = 0;
    int state = -1;
    std::unique_ptr<RingWriter> hint_w;
    std::unique_ptr<PoseRecord> pose{new PoseRecord()};
    uint64_t hint_seq = 0;
};
void set_err(char* err, int n, const std::string& s) {
    if (err && n > 0) std::snprintf(err, std::size_t(n), "%s", s.c_str());
}
}  // namespace

extern "C" {

void* fio_open(const char* config_path, char* err, int errlen) {
    std::unique_ptr<Fio> f(new Fio());
    std::string e;
    if (!load_tracker_config(config_path, f->cfg, e)) { set_err(err, errlen, e); return nullptr; }
    f->core.reset(new TrackerCore());
    if (!f->core->init(f->cfg, e)) { set_err(err, errlen, e); return nullptr; }
    for (auto& i : f->img) i.assign(std::size_t(cap::kWidth) * cap::kHeight, 0);
    return f.release();
}

void fio_close(void* h) {
    Fio* f = static_cast<Fio*>(h);
    if (!f) return;
    if (f->hint_w) f->hint_w->close();
    delete f;
}

void fio_info(void* h, FioInfo* out) {
    Fio* f = static_cast<Fio*>(h);
    std::memset(out, 0, sizeof *out);
    for (int i = 0; i < 3; i++) {
        out->centroid[i] = f->core->model().centroid[i];
        out->wv_min[i] = f->cfg.work_volume.min[i];
        out->wv_max[i] = f->cfg.work_volume.max[i];
    }
    out->radius = f->core->model().radius;
    std::snprintf(out->pair_buffer, sizeof out->pair_buffer, "%s", f->cfg.pair_buffer.c_str());
    std::snprintf(out->pose_buffer, sizeof out->pose_buffer, "%s", f->cfg.pose_buffer.c_str());
    std::snprintf(out->hint_buffer, sizeof out->hint_buffer, "%s", f->cfg.hint_buffer.c_str());
    out->width = cap::kWidth;
    out->height = cap::kHeight;
}

// Cameras without a pair buffer (files of scenes): the default cameras of the synthetic tests.
int fio_default_cameras(void* h, char* err, int errlen) {
    Fio* f = static_cast<Fio*>(h);
    Camera cams[2];
    default_cameras(cams);
    cap::CameraInfo ci[2];
    camera_infos(cams, ci);
    std::string e;
    if (!f->core->set_cameras(ci, e)) { set_err(err, errlen, e); return 1; }
    return 0;
}

// Attach to the pair buffer (or attach again after the writer has gone). 1: no writer now.
int fio_attach_pairs(void* h, char* err, int errlen) {
    Fio* f = static_cast<Fio*>(h);
    std::string e;
    f->attached = f->src.attached() ? f->src.reattach(e) : f->src.attach(f->cfg.pair_buffer, e);
    if (!f->attached) { set_err(err, errlen, e); return 1; }
    cap::CameraInfo ci[2] = {f->src.header().cam[0], f->src.header().cam[1]};
    std::string note;
    if (!f->core->set_cameras(ci, e, &note)) { set_err(err, errlen, e); f->attached = false; return 2; }
    set_err(err, errlen, note);
    return 0;
}

// The cloud of a pair of full images. P: cap x 3 floats (rig frame, m), C: cap bytes (0: D435i_1, 1: D435i_2).
// Returns the number of points.
int fio_cloud(void* h, const uint16_t* a, const uint16_t* b, int stride, float* P, uint8_t* C, int cap_points) {
    Fio* f = static_cast<Fio*>(h);
    const uint16_t* img[2] = {a, b};
    int n = 0;
    for (int c = 0; c < 2; c++) {
        const Camera& cam = f->core->camera(c);
        const uint8_t label = cam.name == f->cfg.cam_name[1] ? 1 : 0;
        const Mat3 R = cam.T_rig_cam.block<3, 3>(0, 0);
        const Vec3 t = cam.T_rig_cam.block<3, 1>(0, 3);
        for (int v = 0; v < cam.height; v += stride) {
            const uint16_t* row = img[c] + std::size_t(v) * std::size_t(cam.width);
            for (int u = 0; u < cam.width; u += stride) {
                const uint16_t d = row[u];
                if (!d) continue;
                const Vec3 p = R * cam.backproject(u, v, d * cam.depth_scale) + t;
                if (!f->cfg.work_volume.contains(p)) continue;
                if (n >= cap_points) return n;
                P[3 * n] = float(p.x());
                P[3 * n + 1] = float(p.y());
                P[3 * n + 2] = float(p.z());
                C[n++] = label;
            }
        }
    }
    return n;
}

// Waits for a pair newer than the last one read, takes the newest and builds its cloud.
// 0: ok, 1: timeout, 2: the writer has gone (call fio_attach_pairs).
int fio_read_cloud(void* h, int timeout_ms, int stride, float* P, uint8_t* C, int cap_points, FioPair* out) {
    Fio* f = static_cast<Fio*>(h);
    if (!f->attached) return 2;
    CropRect full[2];
    for (int c = 0; c < 2; c++) full[c] = CropRect{0, 0, cap::kWidth - 1, cap::kHeight - 1};
    cap::PairRecord rec;
    uint16_t* img[2] = {f->img[0].data(), f->img[1].data()};
    int64_t copy_ns = 0;
    const PairSource::Status s = f->src.read_newest(full, rec, img, int64_t(timeout_ms) * 1000000, copy_ns);
    if (s == PairSource::Status::WriterGone) { f->attached = false; return 2; }
    if (s == PairSource::Status::Timeout) return 1;
    out->pair_seq = rec.pair_seq;
    out->capture_instance_id = f->src.instance_id();
    out->meas_time_mono_ns = (TrackerCore::capture_mono_ns(rec, 0) + TrackerCore::capture_mono_ns(rec, 1)) / 2;
    out->read_done_mono_ns = mono_ns();
    out->points = fio_cloud(h, img[0], img[1], stride, P, C, cap_points);
    out->reserved = 0;
    return 0;
}

// State of the tracker from the newest pose record: 0 search, 1 track, 2 lost; -1: no live tracker.
int fio_tracker_state(void* h) {
    Fio* f = static_cast<Fio*>(h);
    const int64_t now = mono_ns();
    if (!f->pose_r) {
        if (now - f->pose_attach_ns < 200000000) return -1;
        f->pose_attach_ns = now;
        std::string e;
        f->pose_r = RingReader::attach(f->cfg.pose_buffer, kPoseRecordType, kPoseRecordVersion, sizeof(PoseRecord), e);
        f->state = -1;
        if (!f->pose_r) return -1;
    }
    const RingReader::Result r = f->pose_r->latest(f->pose.get());
    if (r == RingReader::Result::Ok) f->state = int(f->pose->track_state);
    else if (r != RingReader::Result::Timeout) {
        f->pose_r.reset();
        f->pose_attach_ns = now;
        f->state = -1;
    }
    return f->state;
}

int fio_hints_open(void* h, char* err, int errlen) {
    Fio* f = static_cast<Fio*>(h);
    std::string e, note;
    f->hint_w = RingWriter::create(f->cfg.hint_buffer, kHintRecordType, kHintRecordVersion, sizeof(HintRecord), kHintSlots, e, &note);
    if (!f->hint_w) { set_err(err, errlen, e); return 1; }
    set_err(err, errlen, note);
    return 0;
}

// Publishes count hypotheses (T: count x 16 doubles, row-major T_rig_from_object; score: count floats), best first.
int fio_hints_publish(void* h, const FioPair* pair, int count, const double* T, const float* score, float prep_ms,
                      float net_ms, float post_ms, int hypotheses) {
    Fio* f = static_cast<Fio*>(h);
    if (!f->hint_w) return 1;
    HintRecord* r = static_cast<HintRecord*>(f->hint_w->begin());
    std::memset(r, 0, sizeof *r);
    r->version = kHintRecordVersion;
    r->count = uint32_t(std::max(0, std::min(count, int(kHintMax))));
    r->hint_seq = f->hint_seq++;
    r->pair_seq = pair->pair_seq;
    r->capture_instance_id = pair->capture_instance_id;
    r->meas_time_mono_ns = pair->meas_time_mono_ns;
    r->prep_ms = prep_ms;
    r->net_ms = net_ms;
    r->post_ms = post_ms;
    r->hypotheses = uint32_t(hypotheses);
    for (uint32_t k = 0; k < r->count; k++) {
        r->score[k] = score[k];
        std::memcpy(r->T_rig_from_object[k], T + 16 * k, 16 * sizeof(double));
    }
    r->publish_time_mono_ns = mono_ns();
    f->hint_w->commit();
    return 0;
}

int64_t fio_mono_ns(void) { return mono_ns(); }

}  // extern "C"
