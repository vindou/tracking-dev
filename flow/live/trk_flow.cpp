// trk_flow: the inference process of the flow module on the Jetson. Depth pairs -> pose hypotheses for the tracker in
// search state (hint buffer, tracker/DESIGN.md section 13).
//
//   trk_flow --config ~/tracker/config/tracker.json --engines DIR [--tag fp16] [--always] [--status 10]      live
//   trk_flow --config ... --engines DIR --scenes scenes.bin --out hints.bin [--limit N]                       a scenes file
//            [--step K] [--gap-ms G] [--probe]     every K-th scene, a pause after each, and per scene the GPU and
//                                                  CPU clock and the core before and after (flow/jetson_contention.sh)
//
// Live: while the tracker is in search state (newest pose record; --always: regardless of the tracker), take the
// newest pair, compute the hypotheses and publish up to 16, best first, with the capture time of the pair. The
// computation is flow/propose.py with the deployed settings:
//   1. cloud: every 2nd pixel of both images, rig frame, inside the work volume (tracker library, flowio.h)
//   2. seeds: farthest-point selection of 32 among the centres of the occupied 3 cm cells (cells with >= 4 points)
//   3. position stage (engine stage1e): per seed 512 random points within 0.16 m and a random start rotation; the
//      end positions are grouped (30 mm) and the 4 groups with the most object points become the positions
//   4. pose stage (engines stage2e, refine, score): per position 32 start orientations; flow, 6 point-to-plane steps,
//      verifier. The three engines share their buffers on the device. With a single position (a scene with nothing
//      but the object) the engines stage2e1, refine1, score1 for 32 hypotheses run instead, if DIR has them
//      (flow/jetson_deploy.sh); the result is the same, the 96 copies are not computed.
//   5. hypotheses with a score of at least 0.85, grouped (15 mm, 15 degrees), the best of each group, highest first
// Scenes file: the same for every scene of the file (format: tracker/src/apps/trk_hint_eval.cpp), written as a
// hints file for trk_hint_eval, with the times per scene in <out>.times.csv.
//
// Libraries: TensorRT and the CUDA runtime (JetPack), the tracker library through its C interface (trk_flowio).
// A proposal allocates no memory. One line of statistics every --status seconds.
#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include "flowio/flowio.h"

namespace {

constexpr int K1 = 32, CENTERS = 4, ORIENT = 32, K2 = CENTERS * ORIENT, NP = 512, NHINT = 16;
constexpr float SET_RADIUS = 0.16f, CELL = 0.03f;
constexpr double MIN_SCORE = 0.85, GROUP_M = 0.015, GROUP_DEG = 15.0, CENTER_M = 0.03;
constexpr int STRIDE = 2, MAX_CLOUD = 2 * (848 / STRIDE) * (480 / STRIDE), MIN_CELL_POINTS = 4;
const float kOrient[ORIENT][4] = {
#include "orientations.inc"
};

volatile std::sig_atomic_t g_stop = 0;
void on_sig(int) { g_stop = 1; }

double now_ms() { return double(fio_mono_ns()) / 1e6; }

// ------------------------------------------------------------------------------------------------ TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::fprintf(stderr, "tensorrt: %s\n", msg);
    }
};

void* dev_alloc(std::size_t floats) {
    void* p = nullptr;
    if (cudaMalloc(&p, floats * sizeof(float)) != cudaSuccess) { std::fprintf(stderr, "error: cudaMalloc\n"); std::exit(4); }
    return p;
}
void upload(void* dev, const float* host, std::size_t floats) { cudaMemcpy(dev, host, floats * sizeof(float), cudaMemcpyHostToDevice); }
void download(float* host, const void* dev, std::size_t floats) { cudaMemcpy(host, dev, floats * sizeof(float), cudaMemcpyDeviceToHost); }

// One engine with fixed shapes; the tensors are bound by name to device buffers that several engines may share.
struct Net {
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx;
    std::vector<std::string> names;
    std::vector<void*> bind;

    bool load(nvinfer1::IRuntime& rt, const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "error: cannot read %s\n", path.c_str()); return false; }
        const std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        engine.reset(rt.deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine) { std::fprintf(stderr, "error: %s is not an engine of this TensorRT version\n", path.c_str()); return false; }
        ctx.reset(engine->createExecutionContext());
        for (int i = 0; i < engine->getNbIOTensors(); i++) names.push_back(engine->getIOTensorName(i));
        bind.assign(names.size(), nullptr);
        return ctx != nullptr;
    }
    std::size_t size(const char* name) const {
        const nvinfer1::Dims d = engine->getTensorShape(name);
        std::size_t n = 1;
        for (int i = 0; i < d.nbDims; i++) n *= std::size_t(d.d[i]);
        return n;
    }
    // Binds the tensor to the buffer (expected number of floats checked).
    void set(const char* name, void* dev, std::size_t floats) {
        for (std::size_t i = 0; i < names.size(); i++)
            if (names[i] == name) {
                if (size(name) != floats) { std::fprintf(stderr, "error: tensor %s has %zu values, expected %zu\n", name, size(name), floats); std::exit(4); }
                bind[i] = dev;
                return;
            }
        std::fprintf(stderr, "error: engine has no tensor %s\n", name);
        std::exit(4);
    }
    void run() {
        for (std::size_t i = 0; i < bind.size(); i++)
            if (!bind[i]) { std::fprintf(stderr, "error: tensor %s not bound\n", names[i].c_str()); std::exit(4); }
        if (!ctx->executeV2(bind.data())) { std::fprintf(stderr, "error: TensorRT execution failed\n"); std::exit(4); }
    }
};

// ------------------------------------------------------------------------------------------------ proposal
struct Hints {
    int count = 0, hypotheses = 0;
    double T[NHINT][16];
    float score[NHINT];
    double prep_ms = 0, net_ms = 0, post_ms = 0, net1_ms = 0;   // net1: the position stage, part of net
};

class Proposer {
public:
    bool init(const std::string& dir, const std::string& tag, const FioInfo& info) {
        info_ = info;
        rt_.reset(nvinfer1::createInferRuntime(logger_));
        const char* graphs[4] = {"stage1e", "stage2e", "refine", "score"};
        Net* nets[4] = {&stage1_, &stage2_, &refine_, &score_};
        for (int i = 0; i < 4; i++)
            if (!nets[i]->load(*rt_, dir + "/" + graphs[i] + "." + tag + ".engine")) return false;
        const char* graphs1[3] = {"stage2e1", "refine1", "score1"};
        Net* nets1[3] = {&stage2one_, &refine1_, &score1_};
        one_ = true;
        for (const char* g : graphs1) one_ = one_ && std::ifstream(dir + "/" + g + "." + tag + ".engine").good();
        for (int i = 0; i < 3 && one_; i++)
            if (!nets1[i]->load(*rt_, dir + "/" + graphs1[i] + "." + tag + ".engine")) return false;
        // device buffers; the pose stage, the refinement and the verifier share points, cameras and poses
        d1_pts_ = dev_alloc(K1 * NP * 3); d1_cam_ = dev_alloc(K1 * NP); d1_q_ = dev_alloc(K1 * 4);
        d1_qo_ = dev_alloc(K1 * 4); d1_po_ = dev_alloc(K1 * 3); d1_seg_ = dev_alloc(K1 * NP);
        d2_pts_ = dev_alloc(K2 * NP * 3); d2_cam_ = dev_alloc(K2 * NP); d2_q0_ = dev_alloc(K2 * 4);
        d2_qa_ = dev_alloc(K2 * 4); d2_pa_ = dev_alloc(K2 * 3); d2_qb_ = dev_alloc(K2 * 4); d2_pb_ = dev_alloc(K2 * 3);
        d2_logit_ = dev_alloc(K2);
        stage1_.set("pts_in", d1_pts_, K1 * NP * 3); stage1_.set("cam_in", d1_cam_, K1 * NP); stage1_.set("q_in", d1_q_, K1 * 4);
        stage1_.set("q", d1_qo_, K1 * 4); stage1_.set("p", d1_po_, K1 * 3); stage1_.set("seg", d1_seg_, K1 * NP);
        stage2_.set("pts_in", d2_pts_, K2 * NP * 3); stage2_.set("cam_in", d2_cam_, K2 * NP); stage2_.set("q_in", d2_q0_, K2 * 4);
        stage2_.set("q", d2_qa_, K2 * 4); stage2_.set("p", d2_pa_, K2 * 3);
        refine_.set("pts_in", d2_pts_, K2 * NP * 3); refine_.set("q_in", d2_qa_, K2 * 4); refine_.set("p_in", d2_pa_, K2 * 3);
        refine_.set("q", d2_qb_, K2 * 4); refine_.set("p", d2_pb_, K2 * 3);
        score_.set("pts_in", d2_pts_, K2 * NP * 3); score_.set("cam_in", d2_cam_, K2 * NP); score_.set("q_in", d2_qb_, K2 * 4);
        score_.set("p_in", d2_pb_, K2 * 3); score_.set("logit", d2_logit_, K2);
        if (one_) {   // the first 32 rows of the same buffers
            const int K = ORIENT;
            stage2one_.set("pts_in", d2_pts_, K * NP * 3); stage2one_.set("cam_in", d2_cam_, K * NP); stage2one_.set("q_in", d2_q0_, K * 4);
            stage2one_.set("q", d2_qa_, K * 4); stage2one_.set("p", d2_pa_, K * 3);
            refine1_.set("pts_in", d2_pts_, K * NP * 3); refine1_.set("q_in", d2_qa_, K * 4); refine1_.set("p_in", d2_pa_, K * 3);
            refine1_.set("q", d2_qb_, K * 4); refine1_.set("p", d2_pb_, K * 3);
            score1_.set("pts_in", d2_pts_, K * NP * 3); score1_.set("cam_in", d2_cam_, K * NP); score1_.set("q_in", d2_qb_, K * 4);
            score1_.set("p_in", d2_pb_, K * 3); score1_.set("logit", d2_logit_, K);
        }
        std::vector<float> q0(K2 * 4);
        for (int k = 0; k < K2; k++) std::memcpy(&q0[std::size_t(k) * 4], kOrient[k % ORIENT], 4 * sizeof(float));
        upload(d2_q0_, q0.data(), q0.size());
        // host memory
        for (int i = 0; i < 3; i++) {
            lo_[i] = float(info.wv_min[i]);
            dim_[i] = int(std::ceil((info.wv_max[i] - info.wv_min[i]) / CELL)) + 1;
        }
        const std::size_t cells = std::size_t(dim_[0]) * dim_[1] * dim_[2];
        start_.assign(cells + 1, 0); fill_.assign(cells, 0); order_.assign(MAX_CLOUD, 0); cell_.assign(MAX_CLOUD, 0);
        cen_.reserve(cells); far_.reserve(cells); cand_.reserve(MAX_CLOUD);
        h1_pts_.assign(K1 * NP * 3, 0); h1_cam_.assign(K1 * NP, 0); h1_q_.assign(K1 * 4, 0);
        h1_qo_.assign(K1 * 4, 0); h1_po_.assign(K1 * 3, 0); h1_seg_.assign(K1 * NP, 0);
        h2_pts_.assign(K2 * NP * 3, 0); h2_cam_.assign(K2 * NP, 0);
        h2_q_.assign(K2 * 4, 0); h2_p_.assign(K2 * 3, 0); h2_logit_.assign(K2, 0);
        return true;
    }

    // Cloud (n points, rig frame, camera 0 / 1) -> hints.
    void propose(const float* P, const uint8_t* C, int n, uint64_t seed, Hints& out) {
        const double t0 = now_ms();
        out = Hints();
        rng_.seed(seed);
        build_grid(P, n);
        const int ns = seeds(P);
        if (ns == 0) return;
        int nreal[K1];
        for (int i = 0; i < K1; i++) {
            const int src = i < ns ? i : 0;   // fixed shape: the rows after the last seed are copies of the first
            if (i < ns) nreal[i] = point_set(P, C, seed_[i].data(), &h1_pts_[std::size_t(i) * NP * 3], &h1_cam_[std::size_t(i) * NP]);
            else {
                std::memcpy(&h1_pts_[std::size_t(i) * NP * 3], &h1_pts_[std::size_t(src) * NP * 3], NP * 3 * sizeof(float));
                std::memcpy(&h1_cam_[std::size_t(i) * NP], &h1_cam_[std::size_t(src) * NP], NP * sizeof(float));
            }
            float q[4], norm = 0;
            for (float& v : q) { v = gauss_(rng_); norm += v * v; }
            norm = std::sqrt(norm);
            for (int j = 0; j < 4; j++) h1_q_[std::size_t(i) * 4 + j] = q[j] / norm;
        }
        const double t1 = now_ms();
        upload(d1_pts_, h1_pts_.data(), h1_pts_.size()); upload(d1_cam_, h1_cam_.data(), h1_cam_.size()); upload(d1_q_, h1_q_.data(), h1_q_.size());
        stage1_.run();
        download(h1_po_.data(), d1_po_, h1_po_.size()); download(h1_seg_.data(), d1_seg_, h1_seg_.size());
        const double t2 = now_ms();
        // end positions, share of object points, groups of end positions
        double of[K1], e[K1][3];
        for (int i = 0; i < ns; i++) {
            int obj = 0;
            for (int j = 0; j < nreal[i]; j++) obj += h1_seg_[std::size_t(i) * NP + j] > 0;
            of[i] = double(obj) / double(std::max(nreal[i], 1));
            for (int a = 0; a < 3; a++) e[i][a] = double(h1_po_[std::size_t(i) * 3 + a]) + double(seed_[i][a]);
        }
        int idx[K1];
        for (int i = 0; i < ns; i++) idx[i] = i;
        std::stable_sort(idx, idx + ns, [&](int a, int b) { return of[a] > of[b]; });
        int first[K1], member[K1], ncl = 0;   // first: the best member of each group; member: group of each seed
        double weight[K1];
        for (int k = 0; k < ns; k++) {
            const int i = idx[k];
            int g = -1;
            for (int c = 0; c < ncl && g < 0; c++) {
                const double* f = e[first[c]];
                if (std::sqrt((e[i][0] - f[0]) * (e[i][0] - f[0]) + (e[i][1] - f[1]) * (e[i][1] - f[1]) + (e[i][2] - f[2]) * (e[i][2] - f[2])) < CENTER_M) g = c;
            }
            if (g < 0) { g = ncl++; first[g] = i; weight[g] = 0; }
            member[i] = g;
            weight[g] += of[i];
        }
        int cl[K1];
        for (int c = 0; c < ncl; c++) cl[c] = c;
        std::stable_sort(cl, cl + ncl, [&](int a, int b) { return weight[a] > weight[b]; });
        const int nc = std::min(ncl, CENTERS);
        double cen[CENTERS][3];
        for (int c = 0; c < nc; c++) {
            double w = 0, s[3] = {0, 0, 0};
            for (int i = 0; i < ns; i++)
                if (member[i] == cl[c]) {
                    const double wi = of[i] + 1e-3;
                    w += wi;
                    for (int a = 0; a < 3; a++) s[a] += wi * e[i][a];
                }
            for (int a = 0; a < 3; a++) cen[c][a] = s[a] / w;
        }
        // pose stage: the point set of each position, 32 times
        const bool single = one_ && nc == 1;
        const std::size_t rows = single ? ORIENT : K2;
        for (int c = 0; c < (single ? 1 : CENTERS); c++) {
            float* X = &h2_pts_[std::size_t(c) * ORIENT * NP * 3];
            float* cam = &h2_cam_[std::size_t(c) * ORIENT * NP];
            if (c < nc) {
                const float s[3] = {float(cen[c][0]), float(cen[c][1]), float(cen[c][2])};
                point_set(P, C, s, X, cam);
            } else {
                std::memcpy(X, h2_pts_.data(), NP * 3 * sizeof(float));
                std::memcpy(cam, h2_cam_.data(), NP * sizeof(float));
            }
            for (int o = 1; o < ORIENT; o++) {
                std::memcpy(X + std::size_t(o) * NP * 3, X, NP * 3 * sizeof(float));
                std::memcpy(cam + std::size_t(o) * NP, cam, NP * sizeof(float));
            }
        }
        const double t3 = now_ms();
        upload(d2_pts_, h2_pts_.data(), rows * NP * 3); upload(d2_cam_, h2_cam_.data(), rows * NP);
        (single ? stage2one_ : stage2_).run();
        (single ? refine1_ : refine_).run();
        (single ? score1_ : score_).run();
        download(h2_q_.data(), d2_qb_, rows * 4); download(h2_p_.data(), d2_pb_, rows * 3); download(h2_logit_.data(), d2_logit_, rows);
        const double t4 = now_ms();
        // hypotheses above the minimum score, best first; one per group of agreeing hypotheses
        const int K = nc * ORIENT;
        int good[K2], ng = 0;
        double sc[K2], p[K2][3];
        for (int k = 0; k < K; k++) {
            sc[k] = 1.0 / (1.0 + std::exp(-double(h2_logit_[std::size_t(k)])));
            for (int a = 0; a < 3; a++) p[k][a] = double(h2_p_[std::size_t(k) * 3 + a]) + cen[k / ORIENT][a];
            if (sc[k] >= MIN_SCORE) good[ng++] = k;
        }
        std::stable_sort(good, good + ng, [&](int a, int b) { return sc[a] > sc[b]; });
        const double cosmin = std::cos(GROUP_DEG * M_PI / 180.0 / 2.0);
        int lead[K2], nl = 0;
        for (int g = 0; g < ng; g++) {
            const int i = good[g];
            const float* qi = &h2_q_[std::size_t(i) * 4];
            bool joined = false;
            for (int l = 0; l < nl && !joined; l++) {
                const int j = lead[l];
                const float* qj = &h2_q_[std::size_t(j) * 4];
                const double d = std::sqrt((p[i][0] - p[j][0]) * (p[i][0] - p[j][0]) + (p[i][1] - p[j][1]) * (p[i][1] - p[j][1]) + (p[i][2] - p[j][2]) * (p[i][2] - p[j][2]));
                const double dot = double(qi[0]) * qj[0] + double(qi[1]) * qj[1] + double(qi[2]) * qj[2] + double(qi[3]) * qj[3];
                joined = d < GROUP_M && std::fabs(dot) > cosmin;
            }
            if (!joined) lead[nl++] = i;
        }
        out.hypotheses = K;
        out.count = std::min(nl, NHINT);
        for (int h = 0; h < out.count; h++) {
            const int i = lead[h];
            const double w = h2_q_[std::size_t(i) * 4], x = h2_q_[std::size_t(i) * 4 + 1], y = h2_q_[std::size_t(i) * 4 + 2], z = h2_q_[std::size_t(i) * 4 + 3];
            const double R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                                 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                                 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
            double* T = out.T[h];
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) T[r * 4 + c] = R[r * 3 + c];
                // the hypothesis is the pose of the model centroid: object origin = p - R centroid
                T[r * 4 + 3] = p[i][r] - (R[r * 3] * info_.centroid[0] + R[r * 3 + 1] * info_.centroid[1] + R[r * 3 + 2] * info_.centroid[2]);
            }
            T[12] = T[13] = T[14] = 0;
            T[15] = 1;
            out.score[h] = float(sc[i]);
        }
        const double t5 = now_ms();
        out.prep_ms = (t1 - t0) + (t3 - t2);
        out.net_ms = (t2 - t1) + (t4 - t3);
        out.net1_ms = t2 - t1;
        out.post_ms = t5 - t4;
    }

    bool one_position() const { return one_; }

private:
    int cell_index(const float* p, int c[3]) const {
        for (int a = 0; a < 3; a++) c[a] = std::min(dim_[a] - 1, std::max(0, int((p[a] - lo_[a]) / CELL)));
        return (c[0] * dim_[1] + c[1]) * dim_[2] + c[2];
    }
    // Points sorted by 3 cm cell (counting sort): order_[start_[cell] .. start_[cell + 1]) are the points of a cell.
    void build_grid(const float* P, int n) {
        std::fill(start_.begin(), start_.end(), 0);
        int c[3];
        for (int i = 0; i < n; i++) {
            cell_[std::size_t(i)] = cell_index(P + 3 * i, c);
            start_[std::size_t(cell_[std::size_t(i)]) + 1]++;
        }
        for (std::size_t k = 1; k < start_.size(); k++) start_[k] += start_[k - 1];
        std::copy(start_.begin(), start_.end() - 1, fill_.begin());
        for (int i = 0; i < n; i++) order_[std::size_t(fill_[std::size_t(cell_[std::size_t(i)])]++)] = i;
    }
    // Up to K1 seeds: farthest-point selection among the mean points of the cells with enough points.
    int seeds(const float* P) {
        cen_.clear();
        for (std::size_t k = 0; k + 1 < start_.size(); k++) {
            const int m = start_[k + 1] - start_[k];
            if (m < MIN_CELL_POINTS) continue;
            double s[3] = {0, 0, 0};
            for (int j = start_[k]; j < start_[k + 1]; j++)
                for (int a = 0; a < 3; a++) s[a] += double(P[3 * order_[std::size_t(j)] + a]);
            cen_.push_back({float(s[0] / m), float(s[1] / m), float(s[2] / m)});
        }
        if (cen_.empty()) return 0;
        far_.assign(cen_.size(), 1e30f);
        int ns = 0;
        std::size_t next = std::uniform_int_distribution<std::size_t>(0, cen_.size() - 1)(rng_);
        while (ns < K1 && ns < int(cen_.size())) {
            seed_[ns] = cen_[next];
            const std::array<float, 3>& s = seed_[ns++];
            float best = -1;
            for (std::size_t k = 0; k < cen_.size(); k++) {
                const float dx = cen_[k][0] - s[0], dy = cen_[k][1] - s[1], dz = cen_[k][2] - s[2];
                far_[k] = std::min(far_[k], dx * dx + dy * dy + dz * dz);
                if (far_[k] > best) { best = far_[k]; next = k; }
            }
        }
        return ns;
    }
    // NP random points within SET_RADIUS of s, relative to s, and their camera as -1 / +1. Fewer points: the first
    // chosen one fills the rest; none: zeros. Returns the number of real points.
    int point_set(const float* P, const uint8_t* C, const float* s, float* X, float* cam) {
        cand_.clear();
        int c0[3], c1[3];
        const float a[3] = {s[0] - SET_RADIUS, s[1] - SET_RADIUS, s[2] - SET_RADIUS}, b[3] = {s[0] + SET_RADIUS, s[1] + SET_RADIUS, s[2] + SET_RADIUS};
        cell_index(a, c0);
        cell_index(b, c1);
        for (int x = c0[0]; x <= c1[0]; x++)
            for (int y = c0[1]; y <= c1[1]; y++) {
                const std::size_t row = std::size_t((x * dim_[1] + y) * dim_[2]);
                for (int j = start_[row + std::size_t(c0[2])]; j < start_[row + std::size_t(c1[2]) + 1]; j++) {
                    const int i = order_[std::size_t(j)];
                    const float dx = P[3 * i] - s[0], dy = P[3 * i + 1] - s[1], dz = P[3 * i + 2] - s[2];
                    if (dx * dx + dy * dy + dz * dz < SET_RADIUS * SET_RADIUS) cand_.push_back(i);
                }
            }
        const int m = std::min<int>(int(cand_.size()), NP);
        for (int k = 0; k < m; k++) {   // the first m of a random permutation
            const std::size_t j = std::uniform_int_distribution<std::size_t>(std::size_t(k), cand_.size() - 1)(rng_);
            std::swap(cand_[std::size_t(k)], cand_[j]);
        }
        for (int k = 0; k < NP; k++) {
            if (m == 0) { X[3 * k] = X[3 * k + 1] = X[3 * k + 2] = 0; cam[k] = 0; continue; }
            const int i = cand_[std::size_t(k < m ? k : 0)];
            for (int d = 0; d < 3; d++) X[3 * k + d] = P[3 * i + d] - s[d];
            cam[k] = C[i] ? 1.0f : -1.0f;
        }
        return m;
    }

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> rt_;
    Net stage1_, stage2_, refine_, score_, stage2one_, refine1_, score1_;
    bool one_ = false;   // the one-position engines are there
    FioInfo info_{};
    void *d1_pts_ = nullptr, *d1_cam_ = nullptr, *d1_q_ = nullptr, *d1_qo_ = nullptr, *d1_po_ = nullptr, *d1_seg_ = nullptr;
    void *d2_pts_ = nullptr, *d2_cam_ = nullptr, *d2_q0_ = nullptr, *d2_qa_ = nullptr, *d2_pa_ = nullptr, *d2_qb_ = nullptr, *d2_pb_ = nullptr, *d2_logit_ = nullptr;
    std::vector<float> h1_pts_, h1_cam_, h1_q_, h1_qo_, h1_po_, h1_seg_, h2_pts_, h2_cam_, h2_q_, h2_p_, h2_logit_;
    float lo_[3] = {0, 0, 0};
    int dim_[3] = {1, 1, 1};
    std::vector<int> start_, fill_, order_, cell_, cand_;
    std::vector<std::array<float, 3>> cen_;
    std::vector<float> far_;
    std::array<float, 3> seed_[K1];
    std::mt19937_64 rng_;
    std::normal_distribution<float> gauss_{0.0f, 1.0f};
};

long read_long(const char* path) {
    long v = -1;
    if (FILE* f = std::fopen(path, "r")) {
        if (std::fscanf(f, "%ld", &v) != 1) v = -1;
        std::fclose(f);
    }
    return v;
}
// Diagnostics of the scenes mode (--probe): the GPU clock (Hz), the core this thread is on and that core's clock (kHz).
struct Probe {
    long gpu_hz, cpu_khz;
    int core;
    static Probe now() {
        Probe p;
        p.core = sched_getcpu();
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", p.core);
        p.cpu_khz = read_long(path);
        p.gpu_hz = read_long("/sys/class/devfreq/17000000.gpu/cur_freq");
        return p;
    }
};

double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, std::size_t(q * double(v.size())))];
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string config = "/home/gchang/tracker/config/tracker.json", engines = ".", tag = "fp16", scenes, out = "hints.bin";
    bool always = false, probe = false;
    long step = 1, gap_ms = 0;
    double status_s = 10;
    long limit = 0;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--always") always = true;
        else if (a == "--probe") probe = true;
        else if (i + 1 >= argc) { std::fprintf(stderr, "option %s needs a value\n", a.c_str()); return 2; }
        else if (a == "--config") config = argv[++i];
        else if (a == "--engines") engines = argv[++i];
        else if (a == "--tag") tag = argv[++i];
        else if (a == "--status") status_s = std::atof(argv[++i]);
        else if (a == "--scenes") scenes = argv[++i];
        else if (a == "--out") out = argv[++i];
        else if (a == "--limit") limit = std::atol(argv[++i]);
        else if (a == "--step") step = std::max(1L, std::atol(argv[++i]));
        else if (a == "--gap-ms") gap_ms = std::atol(argv[++i]);
        else { std::fprintf(stderr, "usage: %s --config FILE --engines DIR [--tag fp16] [--always] [--status S] [--scenes FILE --out FILE [--limit N]]\n", argv[0]); return 2; }
    }
    char err[512] = "";
    void* fio = fio_open(config.c_str(), err, sizeof err);
    if (!fio) { std::fprintf(stderr, "error: %s\n", err); return 3; }
    FioInfo info;
    fio_info(fio, &info);
    std::unique_ptr<Proposer> prop(new Proposer());
    if (!prop->init(engines, tag, info)) return 4;
    std::printf("flow: engines %s (%s)%s\n", engines.c_str(), tag.c_str(), prop->one_position() ? ", with the one-position engines" : "");
    std::vector<float> P(std::size_t(MAX_CLOUD) * 3);
    std::vector<uint8_t> C(MAX_CLOUD);
    std::unique_ptr<Hints> hints(new Hints());

    if (!scenes.empty()) {   // ---- a scenes file -> hints file
        if (fio_default_cameras(fio, err, sizeof err) != 0) { std::fprintf(stderr, "error: %s\n", err); return 3; }
        FILE* fs = std::fopen(scenes.c_str(), "rb");
        char magic[8];
        uint32_t h[4];
        if (!fs || std::fread(magic, 1, 8, fs) != 8 || std::memcmp(magic, "TRKSCENE", 8) != 0 || std::fread(h, 4, 4, fs) != 4 || h[0] != 1) {
            std::fprintf(stderr, "error: %s is not a scenes file\n", scenes.c_str());
            return 3;
        }
        const uint32_t count = limit > 0 ? std::min<uint32_t>(h[1], uint32_t(limit)) : h[1];
        const std::size_t n = std::size_t(h[2]) * h[3];
        std::vector<uint16_t> a(n), b(n);
        FILE* fo = std::fopen(out.c_str(), "wb");
        FILE* ft = std::fopen((out + ".times.csv").c_str(), "w");
        if (!fo || !ft) { std::fprintf(stderr, "error: cannot write %s\n", out.c_str()); return 3; }
        const uint32_t head[3] = {1, uint32_t((count + step - 1) / step), NHINT};
        std::fwrite("TRKHINTS", 1, 8, fo);
        std::fwrite(head, 4, 3, fo);
        std::fprintf(ft, "scene,cloud_ms,prep_ms,net_ms,post_ms,points,hints%s\n", probe ? ",net1_ms,gpu_hz_before,gpu_hz_after,core_before,core_after,cpu_khz_before,cpu_khz_after" : "");
        std::vector<double> total;
        for (uint32_t s = 0; s < count; s++) {
            double truth[16];
            if (s % step != 0) {
                std::fseek(fs, long(128 + 4 * n), SEEK_CUR);
                continue;
            }
            if (std::fread(truth, 8, 16, fs) != 16 || std::fread(a.data(), 2, n, fs) != n || std::fread(b.data(), 2, n, fs) != n) {
                std::fprintf(stderr, "error: %s: short file at scene %u\n", scenes.c_str(), s);
                return 3;
            }
            const Probe pb = probe ? Probe::now() : Probe();
            const double t0 = now_ms();
            const int np = fio_cloud(fio, a.data(), b.data(), STRIDE, P.data(), C.data(), MAX_CLOUD);
            const double t_cloud = now_ms() - t0;
            prop->propose(P.data(), C.data(), np, s, *hints);
            const uint32_t nh = uint32_t(hints->count);
            double zero[16] = {0};
            std::fwrite(&nh, 4, 1, fo);
            for (int k = 0; k < NHINT; k++) std::fwrite(k < hints->count ? hints->T[k] : zero, 8, 16, fo);
            std::fprintf(ft, "%u,%.3f,%.3f,%.3f,%.3f,%d,%d", s, t_cloud, hints->prep_ms, hints->net_ms, hints->post_ms, np, hints->count);
            if (probe) {
                const Probe pa = Probe::now();
                std::fprintf(ft, ",%.3f,%ld,%ld,%d,%d,%ld,%ld", hints->net1_ms, pb.gpu_hz, pa.gpu_hz, pb.core, pa.core, pb.cpu_khz, pa.cpu_khz);
            }
            std::fprintf(ft, "\n");
            if (gap_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
            total.push_back(t_cloud + hints->prep_ms + hints->net_ms + hints->post_ms);
        }
        std::fclose(fo);
        std::fclose(ft);
        std::printf("%zu scenes -> %s; ms per scene: median %.1f, p90 %.1f, max %.1f\n", total.size(), out.c_str(), quantile(total, 0.5),
                    quantile(total, 0.9), quantile(total, 1.0));
        fio_close(fio);
        return 0;
    }

    // ---- live
    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);
    if (fio_hints_open(fio, err, sizeof err) != 0) { std::fprintf(stderr, "error: hint buffer %s: %s\n", info.hint_buffer, err); return 3; }
    if (err[0]) std::printf("note: %s\n", err);
    std::printf("flow: hint buffer %s, pair buffer %s, %s\n", info.hint_buffer, info.pair_buffer,
                always ? "always" : "while the tracker is in search state");
    bool attached = false;
    uint64_t n_prop = 0, n_hint = 0, n_empty = 0;
    std::vector<double> v_prep, v_net, v_post, v_lat;
    for (auto* v : {&v_prep, &v_net, &v_post, &v_lat}) v->reserve(4096);
    const double t_start = now_ms();
    double t_status = t_start;
    FioPair pair;
    while (!g_stop) {
        const double now = now_ms();
        if (now - t_status >= status_s * 1e3) {
            t_status = now;
            if (!v_net.empty())
                std::printf("[flow +%.0fs] proposals %" PRIu64 " (with hints %" PRIu64 ", empty cloud %" PRIu64 ") | ms p50: cloud+prep %.1f, "
                            "networks %.0f, grouping %.2f, capture->publish %.0f (p90 %.0f)\n", (now - t_start) / 1e3, n_prop, n_hint, n_empty,
                            quantile(v_prep, 0.5), quantile(v_net, 0.5), quantile(v_post, 0.5), quantile(v_lat, 0.5), quantile(v_lat, 0.9));
            else
                std::printf("[flow +%.0fs] proposals %" PRIu64 " (idle: tracker state %d, pair buffer %s)\n", (now - t_start) / 1e3, n_prop,
                            fio_tracker_state(fio), attached ? "attached" : "none");
            for (auto* v : {&v_prep, &v_net, &v_post, &v_lat}) v->clear();
        }
        if (!attached) {
            if (fio_attach_pairs(fio, err, sizeof err) != 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            attached = true;
            std::printf("[flow] pair buffer attached: %s\n", err);
        }
        if (!always && fio_tracker_state(fio) != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        const int rc = fio_read_cloud(fio, 200, STRIDE, P.data(), C.data(), MAX_CLOUD, &pair);
        if (rc == 2) {
            attached = false;
            std::printf("[flow] pair buffer writer gone\n");
            continue;
        }
        if (rc != 0) continue;
        const double t_cloud = double(fio_mono_ns() - pair.read_done_mono_ns) / 1e6;
        if (pair.points < 50) { n_empty++; continue; }
        prop->propose(P.data(), C.data(), pair.points, pair.pair_seq, *hints);
        fio_hints_publish(fio, &pair, hints->count, &hints->T[0][0], hints->score, float(hints->prep_ms + t_cloud), float(hints->net_ms),
                          float(hints->post_ms), hints->hypotheses);
        n_prop++;
        n_hint += hints->count > 0;
        if (v_net.size() < v_net.capacity()) {
            v_prep.push_back(hints->prep_ms + t_cloud);
            v_net.push_back(hints->net_ms);
            v_post.push_back(hints->post_ms);
            v_lat.push_back(double(fio_mono_ns() - pair.meas_time_mono_ns) / 1e6);
        }
    }
    fio_close(fio);
    std::printf("flow: stopped after %" PRIu64 " proposals\n", n_prop);
    return 0;
}
