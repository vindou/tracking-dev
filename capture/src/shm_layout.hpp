// Shared-memory layout, version 2. Used by the writer (ShmSink) and by readers. All fields are little-endian, naturally
// aligned, fixed size. Offsets are checked with static_assert below; README.md lists them for readers in other languages.
//
//   offset 0                 ShmHeader (512 bytes)
//   offset slots_offset      slot 0 .. slot_count-1, each slot_stride bytes
//   slot + 0                 uint64 seq        (odd: being written; 2n+2: holds pair n complete)
//   slot + 64                PairRecord (128 bytes)
//   slot + 512               depth image A (image_bytes, uint16 little-endian, row-major, depth units -> * depth_scale = m)
//   slot + 512 + image_bytes depth image B
//
// Version 2 (gaps 1 and 3 of the review): writer instance id, heartbeat, "replaced" state, publish time in the record.
// The fields at offsets 0-111 have the same meaning and position as in version 1, so that a reader of either version can
// read magic and layout_version and reject the other version with a clear message.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "common.hpp"

namespace cap {

constexpr uint64_t kShmMagic = 0x5249415035333444ULL;  // bytes "D435PAIR" in memory order (little-endian)
constexpr uint32_t kShmLayoutVersion = 2;
constexpr uint32_t kWriterRunning = 1;
constexpr uint32_t kWriterStopped = 2;    // stopped cleanly
constexpr uint32_t kWriterReplaced = 3;   // a new writer took over the name; replaced_by_instance tells which
constexpr std::size_t kShmHeaderSize = 512;
constexpr std::size_t kShmRecordOffset = 64;   // within a slot
constexpr std::size_t kShmImageOffset = 512;   // within a slot
constexpr uint32_t kHeartbeatIntervalMs = 50;  // writer updates the heartbeat at least this often
constexpr uint32_t kLivenessTimeoutMs = 1000;  // heartbeat older than this: writer not alive

struct CamRecord {             // 48 bytes
    double global_ts_ms;       // SDK frame timestamp (Global Time domain), ms since 1970 (CLOCK_REALTIME base)
    int64_t hw_ts_us;          // metadata FRAME_TIMESTAMP, camera clock, us (-1: missing)
    int64_t sensor_ts_us;      // metadata SENSOR_TIMESTAMP, camera clock, us (-1: missing)
    int64_t arrival_mono_ns;   // CLOCK_MONOTONIC at callback entry, ns
    int64_t frame_counter;     // metadata FRAME_COUNTER (-1: missing)
    int32_t exposure_us;       // metadata ACTUAL_EXPOSURE (-1: missing)
    int32_t domain;            // rs2_timestamp_domain (2 = Global Time)
};

struct PairRecord {            // 128 bytes
    uint64_t pair_seq;         // 0   0, 1, 2, ... (no gaps on the writer side; restarts at 0 for each writer instance)
    double dt_ms;              // 8   pairing time B minus pairing time A, ms
    CamRecord cam[2];          // 16  [0] = A, [1] = B
    int64_t publish_mono_ns;   // 112 CLOCK_MONOTONIC when the matcher hands the pair to the sink (just before the copy)
    int64_t publish_real_ns;   // 120 CLOCK_REALTIME at the same moment (same base as global_ts_ms)
};

struct CameraInfo {            // 96 bytes
    char serial[32];           // NUL-terminated
    float depth_scale;         // metres per depth unit
    uint32_t width;
    uint32_t height;
    float ppx, ppy, fx, fy;
    int32_t distortion_model;  // rs2_distortion
    float coeffs[5];
    uint32_t reserved[3];
};

struct ShmHeader {
    uint64_t magic;                          // 0   kShmMagic
    uint32_t layout_version;                 // 8   2
    uint32_t header_size;                    // 12  512
    uint32_t slot_count;                     // 16  4
    uint32_t image_width;                    // 20  848
    uint32_t image_height;                   // 24  480
    uint32_t bytes_per_pixel;                // 28  2
    uint64_t image_bytes;                    // 32  814080
    uint64_t slot_stride;                    // 40
    uint64_t slots_offset;                   // 48  512
    uint64_t total_size;                     // 56
    int32_t writer_pid;                      // 64  information only (not used for the liveness check)
    std::atomic<uint32_t> writer_state;      // 68  1 running, 2 stopped cleanly, 3 replaced
    std::atomic<uint64_t> pairs_published;   // 72  number of pairs written (pair n is complete when > n)
    std::atomic<uint32_t> notify_word;       // 80  futex word, +1 per pair and per state change
    uint32_t record_offset;                  // 84  64 (PairRecord offset in a slot)
    uint32_t image_offset;                   // 88  512 (image A offset in a slot)
    uint32_t record_size;                    // 92  sizeof(PairRecord) = 128
    double frame_period_ms;                  // 96  measured; valid when pairs_published > 0
    double tolerance_ms;                     // 104 pairing tolerance (half period)
    CameraInfo cam[2];                       // 112 A, 208 B
    uint64_t writer_instance_id;             // 304 random, non-zero, different for each start of the writer
    std::atomic<uint64_t> heartbeat_mono_ns; // 312 CLOCK_MONOTONIC of the last heartbeat
    std::atomic<uint64_t> heartbeat_count;   // 320 +1 per heartbeat
    uint32_t heartbeat_interval_ms;          // 328 50
    uint32_t liveness_timeout_ms;            // 332 1000: heartbeat older than this -> writer not alive
    int64_t writer_start_real_ns;            // 336 CLOCK_REALTIME at creation (information)
    std::atomic<uint64_t> replaced_by_instance;  // 344 instance id of the writer that replaced this one (state 3)
    uint8_t reserved[512 - 352];             // 352 .. 511
};

struct ShmSlotHead {
    std::atomic<uint64_t> seq;               // 0
    uint8_t pad[56];
};

static_assert(sizeof(CamRecord) == 48, "layout");
static_assert(sizeof(PairRecord) == 128, "layout");
static_assert(offsetof(PairRecord, publish_mono_ns) == 112, "layout");
static_assert(sizeof(CameraInfo) == 96, "layout");
static_assert(offsetof(ShmHeader, layout_version) == 8, "layout");
static_assert(offsetof(ShmHeader, writer_state) == 68, "layout");
static_assert(offsetof(ShmHeader, pairs_published) == 72, "layout");
static_assert(offsetof(ShmHeader, notify_word) == 80, "layout");
static_assert(offsetof(ShmHeader, frame_period_ms) == 96, "layout");
static_assert(offsetof(ShmHeader, cam) == 112, "layout");
static_assert(offsetof(ShmHeader, writer_instance_id) == 304, "layout");
static_assert(offsetof(ShmHeader, heartbeat_mono_ns) == 312, "layout");
static_assert(offsetof(ShmHeader, heartbeat_count) == 320, "layout");
static_assert(offsetof(ShmHeader, heartbeat_interval_ms) == 328, "layout");
static_assert(offsetof(ShmHeader, writer_start_real_ns) == 336, "layout");
static_assert(offsetof(ShmHeader, replaced_by_instance) == 344, "layout");
static_assert(sizeof(ShmHeader) == kShmHeaderSize, "layout");
static_assert(sizeof(ShmSlotHead) == 64, "layout");
static_assert(kShmRecordOffset + sizeof(PairRecord) <= kShmImageOffset, "layout");
static_assert(std::atomic<uint64_t>::is_always_lock_free && std::atomic<uint32_t>::is_always_lock_free,
              "shared atomics must be lock-free");

constexpr std::size_t shm_slot_stride(std::size_t image_bytes) { return kShmImageOffset + 2 * image_bytes; }
constexpr std::size_t shm_total_size(std::size_t image_bytes, std::size_t slots) {
    return kShmHeaderSize + slots * shm_slot_stride(image_bytes);
}

// Copy of the data that the slot sequence counter protects (record and images), on the writer and the reader side.
// Reader and writer may run this at the same time on the same bytes: that is the seqlock's intended, detected race
// (a read that overlaps a write is discarded). It is the only code excluded from ThreadSanitizer (gap 5). With
// ThreadSanitizer the copy is a volatile word loop (a plain memcpy call would be intercepted and checked by TSan).
void seqlock_copy(void* dst, const void* src, std::size_t n);

}  // namespace cap
