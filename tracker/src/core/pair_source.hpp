// Stage 1: pair reader on the capture pair buffer (capture layout version 2). Attach, liveness check and reattach use
// the capture library (cap::ShmReader); the read itself is done here so that only the image rows (and columns) of the
// crop regions are copied, followed by the slot sequence check (seqlock), as the capture README describes.
#pragma once
#include <cstdint>
#include <memory>
#include <string>

#include "shm_layout.hpp"
#include "shm_sink.hpp"
#include "tracker_core.hpp"

namespace trk {

class PairSource {
public:
    enum class Status { Ok, Timeout, WriterGone };
    // Attach once (false: not available now).
    bool attach(const std::string& name, std::string& err);
    // After WriterGone: attach again to a new writer instance (false while there is none).
    bool reattach(std::string& err);
    bool attached() const { return r_ != nullptr; }

    // Waits for a pair newer than the last one used, takes the newest, copies the record and the crop rows of both
    // images into img[c] (full-size buffers, row stride = image width) and checks the slot counter. Pairs not used
    // because a newer one was available, or because the slot changed during the copy, are counted.
    // copy_ns: time from the wake-up to the end of the checked copy.
    Status read_newest(const CropRect crops[2], cap::PairRecord& rec, uint16_t* const img[2], int64_t timeout_ns,
                       int64_t& copy_ns);

    const cap::ShmHeader& header() const { return r_->header(); }
    uint64_t instance_id() const { return r_ ? r_->instance_id() : 0; }
    uint64_t skipped() const { return skipped_; }     // newer pair available
    uint64_t torn() const { return torn_; }           // slot changed during the copy
    uint64_t used() const { return used_; }
    uint64_t attaches() const { return attaches_; }
    const char* gone_reason() const { return gone_; }

private:
    std::unique_ptr<cap::ShmReader> r_;
    std::string name_;
    uint64_t next_ = 0, skipped_ = 0, torn_ = 0, used_ = 0, attaches_ = 0;
    const char* gone_ = "";
};

}  // namespace trk
