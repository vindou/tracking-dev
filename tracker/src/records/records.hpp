// Records of the pose buffer and the cloud buffer, and the network messages (bridge -> viewer).
// Little-endian (the Jetson and the Mac are little-endian), fixed size, version first. Offsets: README.md.
#pragma once

#include <cstddef>
#include <cstdint>

namespace trk {

// ----------------------------------------------------------------------------------------------- pose record
constexpr uint32_t kPoseRecordVersion = 1;
constexpr const char* kPoseRecordType = "pose";
constexpr uint32_t kPoseSlots = 256;

enum TrackState : uint32_t { kStateSearch = 0, kStateTrack = 1, kStateLost = 2 };

// Reject reasons of the fit check (bit mask).
enum RejectReason : uint32_t {
    kRejectNone = 0,
    kRejectFewPoints = 1u << 0,
    kRejectRmse = 1u << 1,
    kRejectInliers = 1u << 2,
    kRejectEigen = 1u << 3,
    kRejectJump = 1u << 4,
    kRejectNoSolve = 1u << 5,     // no ICP run in this cycle (search in progress, no points, no pair)
    kRejectVisibility = 1u << 6,  // free-space violations: the cameras see through where the model surface would be
};

// Stage times (microseconds) in the pose record.
enum Stage : uint32_t {
    kStageRead = 0,      // pair reader: wait end -> rows copied and checked
    kStageBuild = 1,     // cloud builder
    kStageFilter = 2,    // scene filter
    kStageSolve = 3,     // pose solver (ICP; in search: all candidates of this cycle)
    kStageCheck = 4,     // fit check
    kStageTrack = 5,     // tracking filter
    kStagePublish = 6,   // publish of the previous cycle (pose + cloud copy)
    kStageTotal = 7,     // read start -> pose ready (this cycle, without publish)
    kStageCount = 8
};

struct PoseRecord {                      // 640 bytes
    uint32_t version;                    // 0   kPoseRecordVersion
    uint32_t track_state;                // 4   TrackState
    uint64_t pose_seq;                   // 8   0, 1, 2, ... per tracker instance
    uint64_t pair_seq;                   // 16  pair_seq of the pair used
    uint64_t writer_instance_id;         // 24  instance id of the tracker (pose buffer writer)
    uint64_t capture_instance_id;        // 32  instance id of the pair buffer writer (capture or replay)
    int64_t meas_time_mono_ns;           // 40  mean capture time of the two frames, CLOCK_MONOTONIC
    int64_t publish_time_mono_ns;        // 48  CLOCK_MONOTONIC just before the pose is published
    double pair_dt_ms;                   // 56  capture-time difference B - A of the pair
    uint32_t valid;                      // 64  1: the ICP pose of this cycle passed the fit check
    uint32_t reject_reasons;             // 68  RejectReason bits (0 when valid)
    double position[3];                  // 72  filtered position of the object frame in the rig frame (m)
    double orientation[4];               // 96  filtered orientation, quaternion w, x, y, z
    double velocity[3];                  // 128 m/s, rig frame
    double angular_velocity[3];          // 152 rad/s, rig frame
    double icp_position[3];              // 176 unfiltered ICP pose (last ICP result of this cycle, valid or not)
    double icp_orientation[4];           // 200 w, x, y, z
    double covariance[36];               // 232 6x6 row-major, order (px, py, pz, rx, ry, rz), rig frame, of the ICP pose
    double fit_rmse;                     // 520 m, point-to-point RMSE of the inlier correspondences (Open3D inlier_rmse)
    double inlier_ratio;                 // 528 inlier correspondences / points (Open3D fitness)
    double min_eigenvalue;               // 536 smallest eigenvalue of the normalized 6x6 matrix
    uint32_t point_count;                // 544 points given to the solver
    uint32_t icp_iterations;             // 548 iterations of the last ICP run (sum over the distance sequence)
    uint32_t search_done;                // 552 search: candidates done in the current round
    uint32_t search_total;               // 556 search: candidates per round
    float stage_us[kStageCount];         // 560 stage times, microseconds (Stage)
    int64_t capture_time_mono_ns[2];     // 592 capture time of frame A and frame B, CLOCK_MONOTONIC
    uint64_t pairs_skipped;              // 608 total pairs not used (newer pair available, torn read)
    uint32_t rejected_in_row;            // 616 consecutive rejected cycles
    float free_space_violation;          // 620 fraction of checked model points that the cameras see through
    uint8_t reserved[16];                // 624
};
static_assert(sizeof(PoseRecord) == 640, "layout");
static_assert(offsetof(PoseRecord, covariance) == 232, "layout");
static_assert(offsetof(PoseRecord, fit_rmse) == 520, "layout");
static_assert(offsetof(PoseRecord, stage_us) == 560, "layout");
static_assert(offsetof(PoseRecord, capture_time_mono_ns) == 592, "layout");

// ----------------------------------------------------------------------------------------------- cloud record
constexpr uint32_t kCloudRecordVersion = 1;
constexpr const char* kCloudRecordType = "cloud";
constexpr uint32_t kCloudSlots = 4;
constexpr uint32_t kCloudMaxPoints = 8192;

struct CloudPoint {                      // 16 bytes
    float x, y, z;                       // rig frame, m
    uint8_t inlier;                      // 1 / 0: inlier / outlier of the last ICP iteration; 2: not given to the solver
    uint8_t camera;                      // 0 = A, 1 = B
    uint16_t reserved;
};
struct CloudRecord {                     // 64 + 8192 * 16 = 131136 bytes
    uint32_t version;                    // 0   kCloudRecordVersion
    uint32_t point_count;                // 4   valid entries in points[]
    uint64_t pose_seq;                   // 8   pose record of the same cycle
    uint64_t pair_seq;                   // 16
    int64_t meas_time_mono_ns;           // 24
    uint32_t track_state;                // 32
    uint8_t reserved[28];                // 36
    CloudPoint points[kCloudMaxPoints];  // 64
};
static_assert(sizeof(CloudPoint) == 16, "layout");
static_assert(sizeof(CloudRecord) == 64 + 16 * kCloudMaxPoints, "layout");
static_assert(offsetof(CloudRecord, points) == 64, "layout");

// ----------------------------------------------------------------------------------------------- network
constexpr uint32_t kNetMagic = 0x4E4B5254;   // "TRKN"
constexpr uint16_t kNetVersion = 1;
enum NetType : uint16_t { kNetPose = 1, kNetCloud = 2, kNetStatus = 3 };

struct NetHeader {                       // 16 bytes
    uint32_t magic;                      // 0  kNetMagic
    uint16_t type;                       // 4  NetType
    uint16_t version;                    // 6  kNetVersion
    uint32_t payload_len;                // 8  bytes after the header
    uint32_t reserved;                   // 12
};
static_assert(sizeof(NetHeader) == 16, "layout");

struct NetCloudPoint {                   // 8 bytes
    int16_t x, y, z;                     // 0.1 mm relative to the message center
    uint8_t inlier;
    uint8_t camera;
};
struct NetCloud {                        // payload of kNetCloud: 48 + 8192 * 8 = 65584 bytes (fixed size)
    uint32_t version;                    // 0  1
    uint32_t point_count;                // 4
    uint64_t pose_seq;                   // 8
    uint64_t pair_seq;                   // 16
    double center[3];                    // 24 rig frame, m
    NetCloudPoint points[kCloudMaxPoints];  // 48
};
static_assert(sizeof(NetCloudPoint) == 8, "layout");
static_assert(sizeof(NetCloud) == 48 + 8 * kCloudMaxPoints, "layout");

struct NetStatus {                       // payload of kNetStatus: 64 bytes, once per second
    uint32_t version;                    // 0  1
    uint32_t tracker_attached;           // 4  1: the bridge reads a live pose buffer
    uint64_t poses_read;                 // 8
    uint64_t clouds_read;                // 16
    uint64_t poses_dropped;              // 24 discarded from the send queue (queue full)
    uint64_t clouds_dropped;             // 32
    uint64_t tracker_instance_id;        // 40
    int64_t bridge_time_mono_ns;         // 48
    uint64_t reserved;                   // 56
};
static_assert(sizeof(NetStatus) == 64, "layout");

}  // namespace trk
