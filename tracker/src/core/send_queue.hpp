// Send queue of the telemetry bridge (trk_bridge). Header-only so that the unit tests can use it.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "records/records.hpp"

namespace trk {

// Send queue without allocation after start: one preallocated slot pool per message type (pose, cloud, status), each
// large enough for the whole byte limit, and a fixed-size ring of queue entries in arrival order. All storage is
// written once at start so that it is resident from the beginning. Limit and discard rule as before: when adding a
// message would exceed max_bytes, the oldest cloud message is discarded first, then the oldest other message.
class SendQueue {
public:
    static constexpr int kTypes = 3;   // index 0 pose, 1 cloud, 2 status
    static int tindex(uint16_t type) { return type == kNetPose ? 0 : type == kNetCloud ? 1 : 2; }
    static uint32_t msg_size(int t) {
        return uint32_t(sizeof(NetHeader)) + (t == 0 ? uint32_t(sizeof(PoseRecord)) : t == 1 ? uint32_t(sizeof(NetCloud))
                                                                                         : uint32_t(sizeof(NetStatus)));
    }
    static constexpr uint32_t kMaxMsg = uint32_t(sizeof(NetHeader) + sizeof(NetCloud));

    static constexpr std::size_t kNone = ~std::size_t(0);

    explicit SendQueue(long max_bytes) : max_(std::max<long>(max_bytes, long(kMaxMsg))) {
        std::size_t total = 0;
        for (int t = 0; t < kTypes; t++) {
            const std::size_t n = std::size_t(max_ / long(msg_size(t))) + 1;   // byte limit reached before the pool is empty
            pool_[t].assign(n * msg_size(t), 0);                               // zero-filled: resident from the start
            free_[t].assign(n, 0);
            for (std::size_t i = 0; i < n; i++) free_[t][i] = uint32_t(n - 1 - i);
            nfree_[t] = n;
            total += n;
        }
        ring_.assign(total + 1, Entry{});
    }
    // type: kNetPose / kNetCloud / kNetStatus; payload of the fixed size of that type
    void push(uint16_t type, const void* payload) {
        const int t = tindex(type);
        const uint32_t sz = msg_size(t);
        std::lock_guard<std::mutex> lk(mu_);
        if (!client_) return;   // no client: nothing is queued
        while (bytes_ + long(sz) > max_ && live_ > 0) {
            // full: discard the oldest cloud message first, then the oldest message
            std::size_t victim = kNone, first = kNone;
            for (std::size_t k = head_; k != tail_; k = next(k)) {
                if (ring_[k].type < 0) continue;
                if (first == kNone) first = k;
                if (ring_[k].type == 1) { victim = k; break; }
            }
            if (victim == kNone) victim = first;
            if (ring_[victim].type == 1) dropped_cloud_++;
            else if (ring_[victim].type == 0) dropped_pose_++;
            release(victim);
        }
        if (nfree_[t] == 0) {   // cannot happen (pool sized for the byte limit); counted as a drop
            if (t == 0) dropped_pose_++; else if (t == 1) dropped_cloud_++;
            return;
        }
        if (next(tail_) == head_) compact();
        const uint32_t slot = free_[t][--nfree_[t]];
        uint8_t* dst = pool_[t].data() + std::size_t(slot) * sz;
        const NetHeader h{kNetMagic, type, kNetVersion, sz - uint32_t(sizeof(NetHeader)), 0};
        std::memcpy(dst, &h, sizeof h);
        std::memcpy(dst + sizeof h, payload, sz - sizeof h);
        ring_[tail_] = Entry{int8_t(t), slot};
        tail_ = next(tail_);
        bytes_ += long(sz);
        live_++;
    }
    // copies the oldest message into out (kMaxMsg bytes); returns its size and type, or 0 when the queue is empty
    uint32_t pop(uint8_t* out, uint16_t& type) {
        std::lock_guard<std::mutex> lk(mu_);
        while (head_ != tail_ && ring_[head_].type < 0) head_ = next(head_);
        if (head_ == tail_) return 0;
        const Entry e = ring_[head_];
        const uint32_t sz = msg_size(e.type);
        std::memcpy(out, pool_[e.type].data() + std::size_t(e.slot) * sz, sz);
        type = e.type == 0 ? kNetPose : e.type == 1 ? kNetCloud : kNetStatus;
        release(head_);
        head_ = next(head_);
        return sz;
    }
    void set_client(bool c) {
        std::lock_guard<std::mutex> lk(mu_);
        client_ = c;
        if (!c)
            while (head_ != tail_) { if (ring_[head_].type >= 0) release(head_); head_ = next(head_); }
    }
    uint64_t dropped_pose() const { return dropped_pose_.load(); }
    uint64_t dropped_cloud() const { return dropped_cloud_.load(); }

private:
    struct Entry { int8_t type = -1; uint32_t slot = 0; };   // type -1: discarded (skipped by pop)
    std::size_t next(std::size_t k) const { return k + 1 == ring_.size() ? 0 : k + 1; }
    void release(std::size_t k) {   // returns the slot of a live entry to its pool and marks the entry discarded
        Entry& e = ring_[k];
        free_[e.type][nfree_[e.type]++] = e.slot;
        bytes_ -= long(msg_size(e.type));
        live_--;
        e.type = -1;
    }
    void compact() {   // removes discarded entries (ring full of entries, some of them discarded); no allocation
        std::size_t w = head_;
        for (std::size_t k = head_; k != tail_; k = next(k))
            if (ring_[k].type >= 0) { ring_[w] = ring_[k]; w = next(w); }
        tail_ = w;
    }
    std::mutex mu_;
    std::vector<uint8_t> pool_[kTypes];
    std::vector<uint32_t> free_[kTypes];
    std::size_t nfree_[kTypes] = {0, 0, 0};
    std::vector<Entry> ring_;
    std::size_t head_ = 0, tail_ = 0, live_ = 0;
    long bytes_ = 0, max_;
    bool client_ = false;
    std::atomic<uint64_t> dropped_pose_{0}, dropped_cloud_{0};
};

}  // namespace trk
