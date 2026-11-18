// T1: shared-memory sink. A reader slower than the writer and a reader that stops must not slow the writer down; every
// read that the reader accepts must be complete and unchanged.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "../src/shm_sink.hpp"
#include "test_framework.hpp"

using namespace cap;

namespace {

std::string shm_name(const char* tag) { return "/cap_t_" + std::to_string(getpid()) + "_" + tag; }

void fill_images(std::vector<uint16_t>& a, std::vector<uint16_t>& b, uint64_t seq) {
    std::fill(a.begin(), a.end(), uint16_t(seq * 3 + 1));
    std::fill(b.begin(), b.end(), uint16_t(seq * 5 + 2));
}

PairRecord make_rec(uint64_t seq) {
    PairRecord r{};
    r.pair_seq = seq;
    r.dt_ms = double(seq) * 0.001;
    r.cam[0].frame_counter = int64_t(seq) + 10;
    r.cam[1].frame_counter = int64_t(seq) + 20;
    return r;
}

// Accepted reads must be consistent: record and both images belong to the same pair.
bool consistent(const PairRecord& r, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    const uint16_t va = uint16_t(r.pair_seq * 3 + 1), vb = uint16_t(r.pair_seq * 5 + 2);
    if (r.cam[0].frame_counter != int64_t(r.pair_seq) + 10 || r.cam[1].frame_counter != int64_t(r.pair_seq) + 20) return false;
    for (std::size_t i = 0; i < a.size(); i += 997)
        if (a[i] != va || b[i] != vb) return false;
    return a.back() == va && b.back() == vb && a.front() == va && b.front() == vb;
}

CameraInfo cams[2];

struct WriteStats {
    double p50_us = 0, p99_us = 0, max_us = 0;
    double period_us = 0;   // actual time per pair of the write loop (> the nominal period in slow sanitizer builds)
};

// Writes n pairs, one every period_us, and measures the time of each write.
WriteStats write_pairs(ShmSink& s, uint64_t first, uint64_t n, int period_us) {
    std::vector<uint16_t> a(kWidth * kHeight), b(kWidth * kHeight);
    std::vector<double> t;
    t.reserve(n);
    auto next = std::chrono::steady_clock::now();
    const auto t_start = next;
    for (uint64_t i = first; i < first + n; i++) {
        fill_images(a, b, i);
        const PairRecord r = make_rec(i);
        const auto t0 = std::chrono::steady_clock::now();
        s.on_pair(r, a.data(), b.data());
        t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        next += std::chrono::microseconds(period_us);
        std::this_thread::sleep_until(next);
    }
    std::sort(t.begin(), t.end());
    WriteStats w;
    w.p50_us = t[t.size() / 2];
    w.p99_us = t[std::size_t(double(t.size()) * 0.99)];
    w.max_us = t.back();
    w.period_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count() / double(n);
    return w;
}

}  // namespace

TEST(shm_roundtrip_and_unlink) {
    const std::string name = shm_name("rt");
    std::string err;
    auto s = ShmSink::create(name, cams, kImageBytes, err);
    CHECK(s != nullptr);
    if (!s) {
        std::printf("  %s\n", err.c_str());
        return;
    }
    s->on_period(11.128, 5.564);
    auto r = ShmReader::attach(name, err);
    CHECK(r != nullptr);
    if (!r) return;
    CHECK_EQ(r->header().slot_count, 4u);
    CHECK_EQ(r->image_bytes(), kImageBytes);
    std::vector<uint16_t> a(kWidth * kHeight), b(kWidth * kHeight), ra(kWidth * kHeight), rb(kWidth * kHeight);
    for (uint64_t i = 0; i < 3; i++) {
        fill_images(a, b, i);
        s->on_pair(make_rec(i), a.data(), b.data());
    }
    for (uint64_t i = 0; i < 3; i++) {
        PairRecord rec;
        CHECK(r->next(rec, ra.data(), rb.data(), 100000000) == ShmReader::Result::Ok);
        CHECK_EQ(rec.pair_seq, i);
        CHECK(consistent(rec, ra, rb));
    }
    PairRecord rec;
    CHECK(r->next(rec, nullptr, nullptr, 10000000) == ShmReader::Result::Timeout);
    CHECK_EQ(r->header().frame_period_ms, 11.128);
    s->close();
    CHECK(r->next(rec, nullptr, nullptr, 10000000) == ShmReader::Result::WriterStopped);
    CHECK_EQ(r->missing(), 0u);
    // the object is unlinked
    const int fd = shm_open(name.c_str(), O_RDONLY, 0);
    CHECK(fd < 0);
    if (fd >= 0) close(fd);
}

// Writer at 500 pairs/s (faster than the camera's 90). Phase 1: no reader (reference). Phase 2: a reader thread that
// is slower than the writer and sleeps in the middle of each read so that the writer overwrites the slot under it.
// Phase 3: a reader process that attached and then stopped (SIGSTOP).
TEST(shm_slow_and_stopped_reader_do_not_slow_writer) {
    const std::string name = shm_name("sr");
    std::string err;
    auto s = ShmSink::create(name, cams, kImageBytes, err);
    CHECK(s != nullptr);
    if (!s) return;
    constexpr uint64_t N = 1500;
    constexpr int kPeriodUs = 2000;
    const WriteStats w0 = write_pairs(*s, 0, N, kPeriodUs);

    // phase 2: slow reader thread. Its pause and copy delay are scaled to the actual writer period (2 ms in the normal
    // build: 3 ms pause, 9 ms copy delay; the write loop is slower under ThreadSanitizer because the test images are
    // filled by instrumented code), so that the reader is always slower than the writer.
    const double per_us = std::max(double(kPeriodUs), w0.period_us);
    const int64_t pause_us = int64_t(1.5 * per_us), copy_delay_ns = int64_t(4.5 * per_us * 1000.0);
    std::printf("  writer period %.0f us -> reader pause %lld us, copy delay %lld us\n", per_us, (long long)pause_us,
                (long long)(copy_delay_ns / 1000));
    std::atomic<bool> stop{false};
    uint64_t accepted = 0, bad = 0, missing = 0, discarded = 0;
    std::thread reader([&] {
        std::string e;
        auto r = ShmReader::attach(name, e);
        if (!r) return;
        std::vector<uint16_t> ra(kWidth * kHeight), rb(kWidth * kHeight);
        uint64_t last = 0, iter = 0;
        bool first = true;
        while (!stop.load()) {
            PairRecord rec;
            // Every second read has a copy delay of 4.5 writer periods (> 4 slots): the slot is overwritten during
            // that read. After each read the reader pauses 1.5 periods, so it is slower than the writer.
            const int64_t delay = (iter++ % 2) ? copy_delay_ns : 0;
            const auto res = r->next(rec, ra.data(), rb.data(), 50000000 + 4 * copy_delay_ns, delay);
            std::this_thread::sleep_for(std::chrono::microseconds(pause_us));
            if (res != ShmReader::Result::Ok) {
                if (res == ShmReader::Result::WriterStopped) break;
                continue;
            }
            if (!consistent(rec, ra, rb)) bad++;
            if (!first && rec.pair_seq <= last) bad++;
            last = rec.pair_seq;
            first = false;
            accepted++;
        }
        missing = r->missing();
        discarded = r->discarded_reads();
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const WriteStats w1 = write_pairs(*s, N, N, kPeriodUs);
    stop.store(true);
    reader.join();

    // phase 3: stopped reader process
    const pid_t child = fork();
    if (child == 0) {
        std::string e;
        auto r = ShmReader::attach(name, e);
        raise(SIGSTOP);
        PairRecord rec;
        if (r) r->next(rec, nullptr, nullptr, 1000000);
        _exit(0);
    }
    int st = 0;
    waitpid(child, &st, WUNTRACED);   // returns when the child has stopped itself
    CHECK(WIFSTOPPED(st));
    const WriteStats w2 = write_pairs(*s, 2 * N, N, kPeriodUs);
    kill(child, SIGKILL);
    waitpid(child, &st, 0);
    s->close();

    std::printf("  writer p50/p99/max us: no reader %.0f/%.0f/%.0f | slow reader %.0f/%.0f/%.0f | stopped reader "
                "%.0f/%.0f/%.0f\n",
                w0.p50_us, w0.p99_us, w0.max_us, w1.p50_us, w1.p99_us, w1.max_us, w2.p50_us, w2.p99_us, w2.max_us);
    std::printf("  slow reader: accepted=%llu missing=%llu discarded_reads=%llu bad=%llu\n", (unsigned long long)accepted,
                (unsigned long long)missing, (unsigned long long)discarded, (unsigned long long)bad);
    CHECK_EQ(bad, 0u);                     // no accepted read is torn or out of order
    CHECK_GE(accepted, 10u);
    CHECK_GE(missing, N / 2);              // the slow reader misses pairs (expected) ...
    CHECK_GE(discarded, 1u);               // ... and detects slots changed during its read
    // the writer does not slow down: same order of time as without a reader
    CHECK_LE(w1.p50_us, 2.0 * w0.p50_us + 200.0);
    CHECK_LE(w1.p99_us, 2.0 * w0.p99_us + 500.0);
    CHECK_LE(w2.p50_us, 2.0 * w0.p50_us + 200.0);
    CHECK_LE(w2.p99_us, 2.0 * w0.p99_us + 500.0);
    CHECK_LE(w1.max_us, 20000.0);
    CHECK_LE(w2.max_us, 20000.0);
}

TEST(shm_reader_skips_overwritten_pairs) {
    const std::string name = shm_name("sk");
    std::string err;
    auto s = ShmSink::create(name, cams, kImageBytes, err);
    CHECK(s != nullptr);
    if (!s) return;
    auto r = ShmReader::attach(name, err);
    CHECK(r != nullptr);
    if (!r) return;
    std::vector<uint16_t> a(kWidth * kHeight), b(kWidth * kHeight);
    for (uint64_t i = 0; i < 10; i++) {
        fill_images(a, b, i);
        s->on_pair(make_rec(i), a.data(), b.data());
    }
    PairRecord rec;
    std::vector<uint16_t> ra(kWidth * kHeight), rb(kWidth * kHeight);
    CHECK(r->next(rec, ra.data(), rb.data(), 0) == ShmReader::Result::Ok);
    CHECK_EQ(rec.pair_seq, 6u);           // 0..5 were overwritten (4 slots)
    CHECK_EQ(r->missing(), 6u);
    CHECK(consistent(rec, ra, rb));
    s->close();
}
