// POSIX shared-memory pair sink (writer) and reader. Layout: shm_layout.hpp. Protocol: DESIGN.md section 5, README.md.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "event_signal.hpp"
#include "pair_sink.hpp"
#include "shm_layout.hpp"

namespace cap {

class ShmSink : public PairSink {
public:
    // Creates the shared-memory object. If an object of the same name exists (a previous writer that crashed, or one that
    // still runs), it is marked "replaced" (readers are woken) before its name is removed. Starts the heartbeat thread.
    // Returns nullptr and sets err on failure; note describes a replaced object.
    static std::unique_ptr<ShmSink> create(const std::string& name, const CameraInfo cams[2], std::size_t image_bytes,
                                           std::string& err, std::string* note = nullptr);
    ~ShmSink() override;

    // Never waits (no lock, no blocking call): seqlock write of the slot, then a futex wake.
    void on_pair(const PairRecord& rec, const uint16_t* depth_a, const uint16_t* depth_b) override;
    void on_period(double period_ms, double tolerance_ms) override;
    const char* name() const override { return "shm"; }

    // Stops the heartbeat, marks the writer stopped, wakes readers, unmaps and unlinks (unless another writer has
    // replaced this object: then the name belongs to the new object and is left alone). Idempotent.
    void close();
    const std::string& shm_name() const { return name_; }
    uint64_t instance_id() const { return instance_; }
    uint64_t heartbeats() const;

    // In-process access to the mapping (unit tests: a reader on the same mapping lets ThreadSanitizer see both sides).
    const uint8_t* mapping() const { return base_; }
    std::size_t mapping_size() const { return size_; }

private:
    ShmSink() = default;
    void heartbeat_loop();

    std::string name_;
    uint8_t* base_ = nullptr;
    std::size_t size_ = 0;
    std::size_t image_bytes_ = 0;
    ShmHeader* hdr_ = nullptr;
    uint64_t instance_ = 0;
    bool closed_ = false;

    std::thread hb_th_;
    EventSignal hb_sig_;                 // wakes the heartbeat thread for the stop (futex; no mutex)
    std::atomic<bool> hb_stop_{false};
};

class ShmReader {
public:
    enum class Result {
        Ok,              // rec (and the images) hold the next pair
        Timeout,         // nothing new within the timeout; the writer is alive
        WriterStopped,   // the writer stopped cleanly (after all its pairs were returned)
        WriterNotAlive,  // heartbeat older than the liveness timeout: the writer died or hangs
        WriterReplaced,  // a new writer took over the name: call reattach() (pairs of the old writer that were not
                         // read yet are skipped and counted as missing)
    };

    // Attaches read-only by name. Returns nullptr and sets err on failure (missing object, header not ready, other layout
    // version, sizes). Reading starts with the first pair published after attach.
    static std::unique_ptr<ShmReader> attach(const std::string& name, std::string& err);
    // Reader on the mapping of a writer in the same process (unit tests).
    static std::unique_ptr<ShmReader> attach_in_process(const ShmSink& sink, std::string& err);
    ~ShmReader();

    // Attaches again by the same name to a *new* writer (other instance id). Returns false and sets err if the name is
    // missing, the object is not ready, or it is still the same writer instance. On success the reader continues with
    // the first pair published after the new attach; the counters below continue (totals over all writers).
    bool reattach(std::string& err);

    // Next pair after the last one returned. depth_a / depth_b: buffers of image_bytes() bytes, or nullptr to skip the
    // image copy. timeout_ns < 0: no timeout. While no pair comes, the reader sleeps on the futex and wakes at least every
    // 200 ms to check the writer state and heartbeat (liveness check; no polling while pairs arrive).
    // copy_delay_ns > 0 sleeps between the record copy and the image copy of the first read attempt of this call
    // (test hook: provokes a slot change during the read).
    Result next(PairRecord& rec, uint16_t* depth_a, uint16_t* depth_b, int64_t timeout_ns, int64_t copy_delay_ns = 0);

    // Writer state as seen now: running and heartbeat younger than the liveness timeout.
    bool writer_alive() const;
    int64_t heartbeat_age_ns() const;

    const ShmHeader& header() const { return *hdr_; }
    std::size_t image_bytes() const { return std::size_t(hdr_->image_bytes); }
    uint64_t instance_id() const { return instance_; }
    uint64_t attaches() const { return attaches_; }
    uint64_t received() const { return received_; }
    uint64_t missing() const { return missing_; }            // pair numbers never returned (overwritten or torn)
    uint64_t discarded_reads() const { return discarded_; }  // reads discarded because the slot changed during the read
    uint64_t start_seq() const { return start_seq_; }
    uint64_t next_seq() const { return next_; }

    static const char* result_name(Result r);

private:
    ShmReader() = default;
    static std::unique_ptr<ShmReader> from_mapping(const uint8_t* base, std::size_t size, bool owns, std::string& err);
    void release();

    std::string name_;
    const uint8_t* base_ = nullptr;
    std::size_t size_ = 0;
    bool owns_ = false;
    const ShmHeader* hdr_ = nullptr;
    uint64_t instance_ = 0, attaches_ = 0;
    uint64_t next_ = 0, start_seq_ = 0, received_ = 0, missing_ = 0, discarded_ = 0;
};

}  // namespace cap
