#include "pair_source.hpp"

#include <atomic>

#include "../common/clock.hpp"
#include "event_signal.hpp"   // capture: shared_word_wait

namespace trk {

bool PairSource::attach(const std::string& name, std::string& err) {
    name_ = name;
    r_ = cap::ShmReader::attach(name, err);
    if (!r_) return false;
    next_ = r_->header().pairs_published.load(std::memory_order_acquire);
    attaches_++;
    return true;
}

bool PairSource::reattach(std::string& err) {
    if (!r_) return attach(name_, err);
    if (!r_->reattach(err)) return false;
    next_ = r_->header().pairs_published.load(std::memory_order_acquire);
    attaches_++;
    return true;
}

PairSource::Status PairSource::read_newest(const CropRect crops[2], cap::PairRecord& rec, uint16_t* const img[2],
                                           int64_t timeout_ns, int64_t& copy_ns) {
    const cap::ShmHeader& h = r_->header();
    const uint8_t* base = reinterpret_cast<const uint8_t*>(&h);
    auto* notify = const_cast<std::atomic<uint32_t>*>(&h.notify_word);
    const int64_t deadline = mono_ns() + timeout_ns;
    constexpr int64_t kCheckNs = 200000000;   // liveness check at least every 200 ms while waiting
    for (;;) {
        const uint32_t w = notify->load(std::memory_order_acquire);
        const uint64_t pub = h.pairs_published.load(std::memory_order_acquire);
        const uint32_t st = h.writer_state.load(std::memory_order_acquire);
        if (st == cap::kWriterReplaced) { gone_ = "replaced"; return Status::WriterGone; }
        if (pub > next_) {
            const int64_t t0 = mono_ns();
            const uint64_t n = pub - 1;   // the newest pair
            skipped_ += n - next_;
            next_ = n + 1;
            const uint8_t* slot = base + h.slots_offset + (n % h.slot_count) * h.slot_stride;
            auto* seq = reinterpret_cast<const std::atomic<uint64_t>*>(slot);
            const uint64_t s1 = seq->load(std::memory_order_acquire);
            if (s1 != 2 * n + 2) { torn_++; continue; }
            cap::seqlock_copy(&rec, slot + h.record_offset, sizeof rec);
            const std::size_t W = h.image_width;
            for (int c = 0; c < 2; c++) {
                const CropRect& r = crops[c];
                if (r.empty()) continue;
                const uint8_t* src = slot + h.image_offset + std::size_t(c) * h.image_bytes;
                const std::size_t cols = std::size_t(r.u1 - r.u0 + 1);
                for (int v = r.v0; v <= r.v1; v++) {
                    const std::size_t off = std::size_t(v) * W + std::size_t(r.u0);
                    cap::seqlock_copy(img[c] + off, src + off * 2, cols * 2);
                }
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq->load(std::memory_order_relaxed) != s1) { torn_++; continue; }
            used_++;
            copy_ns = mono_ns() - t0;
            return Status::Ok;
        }
        if (st == cap::kWriterStopped) { gone_ = "stopped"; return Status::WriterGone; }
        if (!r_->writer_alive()) { gone_ = "not_alive"; return Status::WriterGone; }
        const int64_t now = mono_ns();
        if (now >= deadline) return Status::Timeout;
        cap::shared_word_wait(notify, w, std::min<int64_t>(deadline - now, kCheckNs));
    }
}

}  // namespace trk
