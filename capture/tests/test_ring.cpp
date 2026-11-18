// T1: ring buffer tests. Run with ThreadSanitizer: build with -DCAPTURE_TSAN=ON.
#include <atomic>
#include <cstdint>
#include <thread>

#include "../src/event_signal.hpp"
#include "../src/spsc_ring.hpp"
#include "test_framework.hpp"

using namespace cap;

namespace {
struct Item {
    uint64_t seq;
    uint64_t a;   // derived from seq: detects changed items
    uint64_t b;
};
uint64_t mix(uint64_t x) { return x * 0x9E3779B97F4A7C15ULL ^ (x >> 7); }
}  // namespace

TEST(ring_full_and_fifo) {
    SpscRing<Item> r(8);
    for (uint64_t i = 0; i < 8; i++) {
        Item* s = r.claim();
        CHECK(s != nullptr);
        if (s) *s = Item{i, mix(i), ~i};
        r.publish();
    }
    CHECK(r.claim() == nullptr);            // full: writer does not wait, counts an overflow
    CHECK_EQ(r.overflows(), 1u);
    CHECK_EQ(r.max_fill(), 8u);
    CHECK_EQ(r.available(), 8u);
    CHECK_EQ(r.peek(3)->seq, 3u);          // look ahead without consuming
    CHECK_EQ(r.peek(0)->seq, 0u);
    r.pop();
    CHECK_EQ(r.peek(0)->seq, 1u);
    Item* s = r.claim();
    CHECK(s != nullptr);
    if (s) *s = Item{8, mix(8), ~8ULL};
    r.publish();
    for (uint64_t i = 1; i <= 8; i++) {
        CHECK_EQ(r.peek(0)->seq, i);
        r.pop();
    }
    CHECK_EQ(r.available(), 0u);
}

// 10 million items, one writer thread, one reader thread. The reader is slowed down at times so that the ring fills up
// and overflows happen. Every item the reader gets must be unchanged and in order; every missing item must be counted.
TEST(ring_10M_two_threads) {
    constexpr uint64_t N = 10000000;
    SpscRing<Item> r(8);
    EventSignal sig;
    std::atomic<bool> done{false};
    uint64_t writer_overflows = 0;

    std::thread writer([&] {
        for (uint64_t i = 0; i < N; i++) {
            Item* s = r.claim();
            if (!s) {
                // The ring never waits: item i is lost and counted. The test writer then yields so that the reader can
                // catch up and most items still go through the ring.
                writer_overflows++;
                std::this_thread::yield();
                continue;
            }
            s->seq = i;
            s->a = mix(i);
            s->b = ~i;
            r.publish();
            if ((i & 63) == 0) sig.notify();
        }
        done.store(true, std::memory_order_release);
        sig.notify();
    });

    uint64_t received = 0, bad = 0, out_of_order = 0, last = 0;
    bool first = true;
    uint64_t spin = 0;
    for (;;) {
        const uint32_t seen = sig.value();
        const bool fin = done.load(std::memory_order_acquire);
        bool got = false;
        while (r.available()) {
            const Item* s = r.peek(0);
            if (s->a != mix(s->seq) || s->b != ~s->seq) bad++;
            if (!first && s->seq <= last) out_of_order++;
            last = s->seq;
            first = false;
            r.pop();
            received++;
            got = true;
            if ((received % 100000) == 0) {
                for (volatile int k = 0; k < 20000; k = k + 1) {  // slow reader now and then -> overflows
                }
            }
        }
        if (fin && !r.available()) break;
        if (!got) {
            if (++spin % 4096 == 0) sig.wait(seen, 100000);
            else std::this_thread::yield();
        }
    }
    writer.join();
    std::printf("  received=%llu overflows=%llu max_fill=%llu\n", (unsigned long long)received,
                (unsigned long long)r.overflows(), (unsigned long long)r.max_fill());
    CHECK_EQ(bad, 0u);
    CHECK_EQ(out_of_order, 0u);
    CHECK_EQ(received + writer_overflows, N);
    CHECK_EQ(r.overflows(), writer_overflows);
    CHECK_LE(r.max_fill(), 8u);
    CHECK_GE(received, N / 2);           // the data path is really exercised
    CHECK_GE(writer_overflows, 1u);      // and so is the overflow path (slow reader phases)
}
