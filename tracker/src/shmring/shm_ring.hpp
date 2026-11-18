// Generic single-writer, multi-reader shared-memory ring of fixed-size records. Same protocol as the capture pair
// buffer (layout version 2): per-slot sequence counter (seqlock), futex notification, heartbeat, writer instance id,
// reader conditions stopped / not alive / replaced, attach again by name. The writer never waits.
// Layout: DESIGN.md section 3 and README.md "Shared-memory rings".
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace trk {

constexpr uint64_t kRingMagic = 0x00474E49524B5254ULL;  // "TRKRING\0" in memory order
constexpr uint32_t kRingLayoutVersion = 2;
constexpr uint32_t kRingHeaderSize = 512;
constexpr uint32_t kRingRecordOffset = 64;               // record offset within a slot
constexpr uint32_t kWriterRunning = 1, kWriterStopped = 2, kWriterReplaced = 3;
constexpr uint32_t kRingHeartbeatMs = 50;
constexpr uint32_t kRingLivenessMs = 1000;

struct RingHeader {
    uint64_t magic;                                // 0
    uint32_t layout_version;                       // 8
    uint32_t header_size;                          // 12
    uint32_t slot_count;                           // 16
    uint32_t record_size;                          // 20
    uint64_t slot_stride;                          // 24
    uint64_t slots_offset;                         // 32
    uint64_t total_size;                           // 40
    char record_type[16];                          // 48
    uint32_t record_version;                       // 64
    int32_t writer_pid;                            // 68 information only
    std::atomic<uint32_t> writer_state;            // 72
    std::atomic<uint32_t> notify_word;             // 76
    std::atomic<uint64_t> records_published;       // 80
    uint64_t writer_instance_id;                   // 88
    std::atomic<uint64_t> heartbeat_mono_ns;       // 96
    std::atomic<uint64_t> heartbeat_count;         // 104
    uint32_t heartbeat_interval_ms;                // 112
    uint32_t liveness_timeout_ms;                  // 116
    int64_t writer_start_real_ns;                  // 120
    std::atomic<uint64_t> replaced_by_instance;    // 128
    uint8_t reserved[512 - 136];                   // 136
};
static_assert(sizeof(RingHeader) == kRingHeaderSize, "layout");
static_assert(offsetof(RingHeader, record_version) == 64, "layout");
static_assert(offsetof(RingHeader, records_published) == 80, "layout");
static_assert(offsetof(RingHeader, heartbeat_mono_ns) == 96, "layout");
static_assert(offsetof(RingHeader, replaced_by_instance) == 128, "layout");
static_assert(std::atomic<uint64_t>::is_always_lock_free && std::atomic<uint32_t>::is_always_lock_free, "lock-free");

inline uint64_t ring_slot_stride(uint32_t record_size) {
    return (uint64_t(kRingRecordOffset) + record_size + 63) / 64 * 64;
}

// Copy of the seqlock-protected data (both sides). Excluded from ThreadSanitizer (the only exclusion): a read that
// overlaps a write is the intended, detected race of the seqlock.
void ring_seqlock_copy(void* dst, const void* src, std::size_t n);

class RingWriter {
public:
    // Creates the object. An existing object of the same name is marked replaced (its readers are woken) and its name
    // removed first; note describes it. Starts the heartbeat thread.
    static std::unique_ptr<RingWriter> create(const std::string& name, const char* record_type, uint32_t record_version,
                                              uint32_t record_size, uint32_t slot_count, std::string& err,
                                              std::string* note = nullptr);
    ~RingWriter();
    RingWriter(const RingWriter&) = delete;
    RingWriter& operator=(const RingWriter&) = delete;

    // Copies one record into the next slot. Never waits. Returns the record's sequence number.
    uint64_t publish(const void* record);
    // Zero-copy: begin() marks the next slot as being written and returns the record area; commit() completes it.
    void* begin();
    uint64_t commit();

    void close();   // heartbeat stop, state stopped (unless replaced), wake readers, unmap, unlink (unless replaced)
    const std::string& name() const { return name_; }
    uint64_t instance_id() const { return instance_; }
    uint64_t published() const { return next_; }
    const uint8_t* mapping() const { return base_; }
    std::size_t mapping_size() const { return size_; }

private:
    RingWriter() = default;
    void heartbeat_loop();
    std::string name_;
    uint8_t* base_ = nullptr;
    std::size_t size_ = 0;
    RingHeader* hdr_ = nullptr;
    uint64_t instance_ = 0, next_ = 0;
    uint8_t* open_slot_ = nullptr;
    bool closed_ = false;
    std::thread hb_th_;
    std::atomic<uint32_t> hb_word_{0};
    std::atomic<bool> hb_stop_{false};
};

class RingReader {
public:
    enum class Result { Ok, Timeout, WriterStopped, WriterNotAlive, WriterReplaced };

    // Attach by name. record_type / record_version / record_size must match (else err, nullptr). Reading starts with
    // the first record published after the attach.
    static std::unique_ptr<RingReader> attach(const std::string& name, const char* record_type, uint32_t record_version,
                                              uint32_t record_size, std::string& err);
    static std::unique_ptr<RingReader> attach_in_process(const RingWriter& w, const char* record_type,
                                                         uint32_t record_version, uint32_t record_size,
                                                         std::string& err);
    ~RingReader();

    // Attach again by the same name to a new writer instance; false (err) while it is still the same instance.
    bool reattach(std::string& err);

    // Next record after the last one returned (into rec, record_size bytes). timeout_ns < 0: no limit. Waits on the
    // futex, at most 200 ms per wait to check the writer state and heartbeat. copy_delay_ns: test hook.
    Result next(void* rec, int64_t timeout_ns, int64_t copy_delay_ns = 0);
    // Newest complete record, without waiting (Timeout if none newer than the last one returned). Skipped records
    // are not counted as missing.
    Result latest(void* rec);

    bool writer_alive() const;
    int64_t heartbeat_age_ns() const;
    const RingHeader& header() const { return *hdr_; }
    uint64_t instance_id() const { return instance_; }
    uint64_t attaches() const { return attaches_; }
    uint64_t received() const { return received_; }
    uint64_t missing() const { return missing_; }
    uint64_t discarded_reads() const { return discarded_; }
    uint64_t next_seq() const { return next_; }
    static const char* result_name(Result r);

private:
    RingReader() = default;
    static std::unique_ptr<RingReader> from_mapping(const uint8_t* base, std::size_t size, bool owns, const char* type,
                                                    uint32_t version, uint32_t rsize, std::string& err);
    bool read_slot(uint64_t n, void* rec, int64_t copy_delay_ns);
    void release();
    std::string name_, type_;
    uint32_t version_ = 0, rsize_ = 0;
    const uint8_t* base_ = nullptr;
    std::size_t size_ = 0;
    bool owns_ = false;
    const RingHeader* hdr_ = nullptr;
    uint64_t instance_ = 0, attaches_ = 0;
    uint64_t next_ = 0, received_ = 0, missing_ = 0, discarded_ = 0;
};

}  // namespace trk
