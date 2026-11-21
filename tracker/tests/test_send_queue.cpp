// Send queue of the bridge (core/send_queue.hpp): message order and content, byte limit and discard rule (oldest cloud
// first, then the oldest message), a stalled client (no pop) over many messages, clearing on disconnect, and no heap
// allocation after construction.
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "alloc_counter.hpp"
#include "core/send_queue.hpp"
#include "test_framework.hpp"

using namespace trk;

namespace {
// payload buffer large enough for every type; the first 8 bytes carry an id
struct Payload {
    std::vector<uint8_t> b = std::vector<uint8_t>(sizeof(NetCloud), 0);
    const void* with(uint64_t id) { std::memcpy(b.data(), &id, 8); return b.data(); }
};
struct Popped { uint16_t type; uint64_t id; uint32_t size; bool header_ok; };
bool pop1(SendQueue& q, std::vector<uint8_t>& buf, Popped& p) {
    uint16_t type = 0;
    const uint32_t n = q.pop(buf.data(), type);
    if (n == 0) return false;
    NetHeader h;
    std::memcpy(&h, buf.data(), sizeof h);
    p.type = type;
    p.size = n;
    std::memcpy(&p.id, buf.data() + sizeof h, 8);
    p.header_ok = h.magic == kNetMagic && h.type == type && h.version == kNetVersion && h.payload_len + sizeof h == n;
    return true;
}
const uint32_t kPose = sizeof(NetHeader) + sizeof(PoseRecord);
const uint32_t kCloud = sizeof(NetHeader) + sizeof(NetCloud);
}  // namespace

TEST(send_queue_order_content_and_no_client) {
    SendQueue q(1 << 20);
    Payload pl;
    std::vector<uint8_t> buf(SendQueue::kMaxMsg);
    q.push(kNetPose, pl.with(1));   // no client yet: not queued
    Popped p;
    CHECK(!pop1(q, buf, p));
    q.set_client(true);
    q.push(kNetPose, pl.with(10));
    q.push(kNetCloud, pl.with(11));
    q.push(kNetStatus, pl.with(12));
    q.push(kNetPose, pl.with(13));
    const uint16_t types[4] = {kNetPose, kNetCloud, kNetStatus, kNetPose};
    const uint32_t sizes[4] = {kPose, kCloud, uint32_t(sizeof(NetHeader) + sizeof(NetStatus)), kPose};
    for (int i = 0; i < 4; i++) {
        CHECK(pop1(q, buf, p));
        CHECK_EQ(p.type, types[i]);
        CHECK_EQ(p.id, uint64_t(10 + i));
        CHECK_EQ(p.size, sizes[i]);
        CHECK(p.header_ok);
    }
    CHECK(!pop1(q, buf, p));
    q.push(kNetPose, pl.with(20));
    q.set_client(false);   // disconnect clears the queue
    q.set_client(true);
    CHECK(!pop1(q, buf, p));
}

TEST(send_queue_discards_oldest_cloud_first_then_oldest) {
    // room for 3 clouds and 3 poses (198768 bytes) but not for a fourth cloud
    SendQueue q(200000);
    q.set_client(true);
    Payload pl;
    std::vector<uint8_t> buf(SendQueue::kMaxMsg);
    for (int i = 0; i < 3; i++) { q.push(kNetPose, pl.with(100 + i)); q.push(kNetCloud, pl.with(200 + i)); }
    q.push(kNetCloud, pl.with(203));   // discards cloud 200
    CHECK_EQ(q.dropped_cloud(), 1u);
    CHECK_EQ(q.dropped_pose(), 0u);
    const uint64_t expect[6] = {100, 101, 201, 102, 202, 203};
    Popped p;
    for (int i = 0; i < 6; i++) { CHECK(pop1(q, buf, p)); CHECK_EQ(p.id, expect[i]); }
    CHECK(!pop1(q, buf, p));
    // poses only: the oldest poses are discarded, the newest max_bytes / size remain in order
    SendQueue q2(100000);
    q2.set_client(true);
    const int n = 2000, keep = 100000 / int(kPose);
    for (int i = 0; i < n; i++) q2.push(kNetPose, pl.with(uint64_t(i)));
    CHECK_EQ(q2.dropped_pose(), uint64_t(n - keep));
    int got = 0;
    uint64_t prev = 0;
    bool in_order = true;
    while (pop1(q2, buf, p)) {
        if (got == 0) CHECK_EQ(p.id, uint64_t(n - keep));
        else in_order = in_order && p.id == prev + 1;
        prev = p.id;
        got++;
    }
    CHECK_EQ(got, keep);
    CHECK(in_order);
}

TEST(send_queue_stalled_client_and_no_allocation) {
    // a client that never reads: 200000 messages (pose + cloud every 3rd + status every 90th), then everything that is
    // left is read; bytes stay within the limit, order is kept, nothing is allocated after construction
    const long max_bytes = 1048576;
    std::unique_ptr<SendQueue> q(new SendQueue(max_bytes));
    q->set_client(true);
    Payload pl;
    std::vector<uint8_t> buf(SendQueue::kMaxMsg);
    alloc_count_start();
    uint64_t id = 0;
    for (int i = 0; i < 200000; i++) {
        q->push(kNetPose, pl.with(id++));
        if (i % 3 == 0) q->push(kNetCloud, pl.with(id++));
        if (i % 90 == 0) q->push(kNetStatus, pl.with(id++));
    }
    long bytes = 0;
    int msgs = 0;
    uint64_t prev = 0;
    bool in_order = true;
    Popped p;
    while (pop1(*q, buf, p)) {
        if (msgs > 0) in_order = in_order && p.id > prev;
        prev = p.id;
        bytes += p.size;
        msgs++;
    }
    // after reading: the queue works normally again
    q->push(kNetPose, pl.with(id));
    const bool again = pop1(*q, buf, p) && p.id == id;
    const uint64_t allocs = alloc_count_stop();
    std::printf("  left in queue %d messages, %ld bytes (limit %ld); dropped poses %llu clouds %llu; allocations %llu%s\n",
                msgs, bytes, max_bytes, (unsigned long long)q->dropped_pose(), (unsigned long long)q->dropped_cloud(),
                (unsigned long long)allocs, alloc_counting_available() ? "" : " (not counted in sanitizer builds)");
    CHECK_LE(bytes, max_bytes);
    CHECK_GE(bytes, max_bytes - long(kCloud));
    CHECK(in_order);
    CHECK(again);
    CHECK_GE(q->dropped_pose(), 1u);
    CHECK_GE(q->dropped_cloud(), 1u);
    if (alloc_counting_available()) CHECK_EQ(allocs, 0u);
}
