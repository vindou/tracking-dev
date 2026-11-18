// Shared-memory layout version 2 (gaps 1 and 5):
//  - liveness: a SIGKILLed writer is detected within 2 s; the same reader object attaches to a new writer and gets all
//    its pairs; a writer that replaces a running writer is followed; a clean stop leaves no object.
//  - a reader of version 2 rejects a version 1 object with a clear message.
//  - stress: 1 writer at full speed, 3 readers at different speeds, on the writer's own mapping (so that
//    ThreadSanitizer sees both sides); every element of every accepted read is checked (mixed data is detected).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../src/shm_sink.hpp"
#include "test_framework.hpp"

extern char** environ;

using namespace cap;

namespace {

std::string shm_name(const char* tag) { return "/cap_v2_" + std::to_string(getpid()) + "_" + tag; }

uint64_t mix64(uint64_t x) {   // splitmix64 finaliser
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Data pattern: every 64-bit word of image c of pair seq is mix64(key(seq, c)) + word index. Two different pairs give
// different values in every word, so a read that mixes data of two pairs fails the check.
void fill_pattern(uint16_t* img, std::size_t bytes, uint64_t seq, int cam) {
    const uint64_t k = mix64(seq * 2 + uint64_t(cam));
    auto* w = reinterpret_cast<uint64_t*>(img);
    for (std::size_t i = 0; i < bytes / 8; i++) w[i] = k + i;
}
std::size_t pattern_errors(const uint16_t* img, std::size_t bytes, uint64_t seq, int cam) {
    const uint64_t k = mix64(seq * 2 + uint64_t(cam));
    auto* w = reinterpret_cast<const uint64_t*>(img);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < bytes / 8; i++) bad += w[i] != k + i;
    return bad;
}
PairRecord pattern_rec(uint64_t seq) {
    PairRecord r{};
    r.pair_seq = seq;
    r.dt_ms = double(seq % 1000) * 0.001;
    for (int c = 0; c < 2; c++) {
        r.cam[c].frame_counter = int64_t(mix64(seq + 7 * uint64_t(c)) >> 2);
        r.cam[c].hw_ts_us = int64_t(seq) * 11111 + c;
        r.cam[c].global_ts_ms = double(seq) * 11.1;
    }
    return r;
}
bool rec_ok(const PairRecord& r) {
    const PairRecord e = pattern_rec(r.pair_seq);
    return r.dt_ms == e.dt_ms && r.cam[0].frame_counter == e.cam[0].frame_counter &&
           r.cam[1].frame_counter == e.cam[1].frame_counter && r.cam[0].hw_ts_us == e.cam[0].hw_ts_us &&
           r.cam[1].hw_ts_us == e.cam[1].hw_ts_us && r.cam[0].global_ts_ms == e.cam[0].global_ts_ms;
}

CameraInfo g_cams[2];

pid_t spawn_writer(const std::string& name, int period_us, int first_pair_delay_ms) {
    const std::string p = std::to_string(period_us), d = std::to_string(first_pair_delay_ms);
    const char* argv[] = {tf::self_path().c_str(), "--helper-shm-writer", name.c_str(), p.c_str(), d.c_str(), nullptr};
    pid_t pid = -1;
    if (posix_spawn(&pid, tf::self_path().c_str(), nullptr, nullptr, const_cast<char* const*>(argv), environ) != 0)
        return -1;
    return pid;
}

bool object_exists(const std::string& name) {
    const int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) return false;
    close(fd);
    return true;
}

// Attach by name, retrying every 20 ms (the helper may not have created the object yet), up to timeout_s.
std::unique_ptr<ShmReader> attach_retry(const std::string& name, double timeout_s) {
    std::string err;
    const int64_t end = mono_ns() + int64_t(timeout_s * 1e9);
    while (mono_ns() < end) {
        auto r = ShmReader::attach(name, err);
        if (r) return r;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::printf("  attach %s failed: %s\n", name.c_str(), err.c_str());
    return nullptr;
}

bool reattach_retry(ShmReader& r, double timeout_s) {
    std::string err;
    const int64_t end = mono_ns() + int64_t(timeout_s * 1e9);
    while (mono_ns() < end) {
        if (r.reattach(err)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::printf("  reattach failed: %s\n", err.c_str());
    return false;
}

struct ReadRun {
    uint64_t ok = 0, bad = 0, first_seq = UINT64_MAX, last_seq = 0, gaps = 0;
    int64_t first_recv_real_ns = 0, first_publish_real_ns = 0;
    ShmReader::Result end = ShmReader::Result::Ok;
    int64_t end_mono_ns = 0;
};

// Reads up to n pairs (n = 0: until the result is not Ok), with images, and checks each one.
ReadRun read_pairs(ShmReader& r, uint64_t n, int64_t timeout_ns) {
    ReadRun rr;
    std::vector<uint16_t> a(r.image_bytes() / 2), b(r.image_bytes() / 2);
    while (n == 0 || rr.ok < n) {
        PairRecord rec;
        const auto res = r.next(rec, a.data(), b.data(), timeout_ns);
        if (res != ShmReader::Result::Ok) {
            rr.end = res;
            rr.end_mono_ns = mono_ns();
            return rr;
        }
        if (rr.ok == 0) {
            rr.first_seq = rec.pair_seq;
            rr.first_recv_real_ns = real_ns();
            rr.first_publish_real_ns = rec.publish_real_ns;
        } else if (rec.pair_seq != rr.last_seq + 1) {
            rr.gaps += rec.pair_seq - rr.last_seq - 1;
        }
        rr.last_seq = rec.pair_seq;
        if (!rec_ok(rec) || pattern_errors(a.data(), a.size() * 2, rec.pair_seq, 0) ||
            pattern_errors(b.data(), b.size() * 2, rec.pair_seq, 1))
            rr.bad++;
        rr.ok++;
    }
    rr.end_mono_ns = mono_ns();
    return rr;
}

std::atomic<bool> g_helper_stop{false};
void helper_on_term(int) { g_helper_stop.store(true); }

}  // namespace

// Helper writer process: capture_tests --helper-shm-writer NAME PERIOD_US FIRST_PAIR_DELAY_MS
// Creates the object, waits FIRST_PAIR_DELAY_MS (like the settle time of the real writer), then writes one pair every
// PERIOD_US until SIGTERM (clean stop: close) or SIGKILL.
int helper_shm_writer(int argc, char** argv) {
    if (argc < 5) return 2;
    const std::string name = argv[2];
    const int period_us = std::atoi(argv[3]);
    const int delay_ms = std::atoi(argv[4]);
    std::signal(SIGTERM, helper_on_term);
    std::string err;
    auto s = ShmSink::create(name, g_cams, kImageBytes, err);
    if (!s) {
        std::fprintf(stderr, "helper: %s\n", err.c_str());
        return 3;
    }
    std::vector<uint16_t> a(kImageBytes / 2), b(kImageBytes / 2);
    const auto t_end_delay = std::chrono::steady_clock::now() + std::chrono::milliseconds(delay_ms);
    while (!g_helper_stop.load() && std::chrono::steady_clock::now() < t_end_delay)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    s->on_period(double(period_us) / 1000.0, double(period_us) / 2000.0);
    auto next = std::chrono::steady_clock::now();
    for (uint64_t i = 0; !g_helper_stop.load(); i++) {
        fill_pattern(a.data(), kImageBytes, i, 0);
        fill_pattern(b.data(), kImageBytes, i, 1);
        PairRecord r = pattern_rec(i);
        r.publish_mono_ns = mono_ns();
        r.publish_real_ns = real_ns();
        s->on_pair(r, a.data(), b.data());
        next += std::chrono::microseconds(period_us);
        std::this_thread::sleep_until(next);
    }
    s->close();
    return 0;
}

// Gap 1: writer killed -> detected <= 2 s; new writer -> same reader object gets its pairs within 2 s of its first
// pair, with no missing number; clean stop -> WriterStopped and no object left.
TEST(shm_v2_killed_writer_detected_and_new_writer_followed) {
    const std::string name = shm_name("kill");
    const pid_t w1 = spawn_writer(name, 11111, 300);
    CHECK(w1 > 0);
    auto r = attach_retry(name, 5.0);
    CHECK(r != nullptr);
    if (!r) {
        kill(w1, SIGKILL);
        waitpid(w1, nullptr, 0);
        shm_unlink(name.c_str());
        return;
    }
    const uint64_t inst1 = r->instance_id();
    ReadRun rr1 = read_pairs(*r, 40, 2000000000LL);
    CHECK_EQ(rr1.ok, 40u);
    CHECK_EQ(rr1.bad, 0u);
    CHECK_EQ(rr1.gaps, 0u);
    CHECK_EQ(rr1.first_seq, 0u);
    CHECK(r->writer_alive());

    const int64_t t_kill = mono_ns();
    kill(w1, SIGKILL);
    waitpid(w1, nullptr, 0);
    ReadRun drain = read_pairs(*r, 0, 10000000000LL);   // remaining pairs, then the condition
    const double detect_s = double(drain.end_mono_ns - t_kill) / 1e9;
    std::printf("  writer 1 (instance %016llx) SIGKILLed: result %s after %.3f s (heartbeat age %.3f s)\n",
                (unsigned long long)inst1, ShmReader::result_name(drain.end), detect_s,
                double(r->heartbeat_age_ns()) / 1e9);
    CHECK(drain.end == ShmReader::Result::WriterNotAlive);
    CHECK_LE(detect_s, 2.0);
    CHECK_EQ(drain.bad, 0u);
    CHECK(!r->writer_alive());
    std::string err;
    CHECK(!r->reattach(err));   // the name still points to the dead writer's object: same instance, refused
    CHECK(object_exists(name));  // a killed writer cannot remove its object

    // New writer: it marks the old object replaced and creates a new one. The same reader object attaches again.
    const pid_t w2 = spawn_writer(name, 11111, 300);
    CHECK(w2 > 0);
    CHECK(reattach_retry(*r, 5.0));
    CHECK(r->instance_id() != inst1);
    CHECK_EQ(r->attaches(), 2u);
    ReadRun rr2 = read_pairs(*r, 60, 2000000000LL);
    const double first_delay_s = double(rr2.first_recv_real_ns - rr2.first_publish_real_ns) / 1e9;
    std::printf("  writer 2 (instance %016llx): first pair %llu received %.4f s after its publish; %llu pairs, gaps %llu, "
                "bad %llu\n",
                (unsigned long long)r->instance_id(), (unsigned long long)rr2.first_seq, first_delay_s,
                (unsigned long long)rr2.ok, (unsigned long long)rr2.gaps, (unsigned long long)rr2.bad);
    CHECK_EQ(rr2.ok, 60u);
    CHECK_EQ(rr2.first_seq, 0u);   // attached before the new writer's first pair: nothing missed
    CHECK_LE(first_delay_s, 2.0);
    CHECK_EQ(rr2.gaps, 0u);
    CHECK_EQ(rr2.bad, 0u);

    // Clean stop of writer 2: WriterStopped, and the object is removed.
    kill(w2, SIGTERM);
    int st = 0;
    waitpid(w2, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0);
    ReadRun rr3 = read_pairs(*r, 0, 3000000000LL);
    CHECK(rr3.end == ShmReader::Result::WriterStopped);
    CHECK_EQ(rr3.bad, 0u);
    CHECK_EQ(rr3.gaps, 0u);
    CHECK(!object_exists(name));
    CHECK_EQ(r->missing(), 0u);
}

// Gap 1: a second writer starts while the first still runs. The reader gets WriterReplaced, attaches to the new one;
// the first writer's clean stop does not remove the new writer's name.
TEST(shm_v2_replaced_writer_followed) {
    const std::string name = shm_name("repl");
    const pid_t w1 = spawn_writer(name, 11111, 0);
    auto r = attach_retry(name, 5.0);
    CHECK(r != nullptr);
    if (!r) {
        kill(w1, SIGKILL);
        waitpid(w1, nullptr, 0);
        shm_unlink(name.c_str());
        return;
    }
    const uint64_t inst1 = r->instance_id();
    CHECK_EQ(read_pairs(*r, 10, 2000000000LL).bad, 0u);
    const pid_t w2 = spawn_writer(name, 11111, 300);
    ReadRun d = read_pairs(*r, 0, 5000000000LL);
    std::printf("  running writer replaced: result %s\n", ShmReader::result_name(d.end));
    CHECK(d.end == ShmReader::Result::WriterReplaced);
    CHECK_EQ(d.bad, 0u);
    CHECK_EQ(d.gaps, 0u);
    CHECK(reattach_retry(*r, 5.0));
    CHECK(r->instance_id() != inst1);
    // The old writer stops cleanly while the reader reads the new one. (The test waits for the old writer's exit only
    // after reading: under AddressSanitizer the exit of a helper takes seconds (leak check), and a reader that does not
    // read for seconds misses pairs.)
    kill(w1, SIGTERM);
    ReadRun rr = read_pairs(*r, 30, 2000000000LL);
    CHECK_EQ(rr.ok, 30u);
    CHECK_EQ(rr.first_seq, 0u);
    CHECK_EQ(rr.gaps, 0u);
    CHECK_EQ(rr.bad, 0u);
    int st1 = 0;
    waitpid(w1, &st1, 0);
    CHECK(WIFEXITED(st1) && WEXITSTATUS(st1) == 0);
    CHECK(object_exists(name));   // the old writer's clean stop left the new writer's name alone
    CHECK(r->writer_alive());
    kill(w2, SIGTERM);
    waitpid(w2, nullptr, 0);
    CHECK(read_pairs(*r, 0, 3000000000LL).end == ShmReader::Result::WriterStopped);
    CHECK(!object_exists(name));
}

// Layout versions: a version 2 reader rejects a version 1 object (and the message says so).
TEST(shm_v2_rejects_v1_object) {
    const std::string name = shm_name("v1");
    const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    CHECK(fd >= 0);
    if (fd < 0) return;
    const std::size_t size = 4096 * 4;
    CHECK(ftruncate(fd, off_t(size)) == 0);
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    CHECK(p != MAP_FAILED);
    if (p == MAP_FAILED) {
        shm_unlink(name.c_str());
        return;
    }
    auto* bytes = static_cast<uint8_t*>(p);
    const uint64_t magic = kShmMagic;
    const uint32_t v1 = 1, hs = 512;
    std::memcpy(bytes + 0, &magic, 8);
    std::memcpy(bytes + 8, &v1, 4);
    std::memcpy(bytes + 12, &hs, 4);
    std::string err;
    auto r = ShmReader::attach(name, err);
    std::printf("  attach to a version 1 object: %s\n", r ? "accepted" : err.c_str());
    CHECK(r == nullptr);
    CHECK(err.find("layout version 1") != std::string::npos);
    CHECK(err.find("supports only version 2") != std::string::npos);
    munmap(p, size);
    shm_unlink(name.c_str());
}

// Gap 5: 1 writer at full speed (no sleep between pairs) and 3 readers at different speeds for CAP_STRESS_S seconds
// (default 30). The readers use the writer's mapping (attach_in_process) so that ThreadSanitizer sees the writer and
// the readers. Every word of every accepted image and the record are checked: 0 accepted reads with mixed data.
TEST(shm_v2_stress_1_writer_3_readers) {
    double dur = 30;
    if (const char* e = std::getenv("CAP_STRESS_S")) dur = std::atof(e);
    {   // negative control: the check detects an image whose second half belongs to the next pair
        std::vector<uint16_t> x(kImageBytes / 2), y(kImageBytes / 2);
        fill_pattern(x.data(), kImageBytes, 41, 0);
        fill_pattern(y.data(), kImageBytes, 42, 0);
        std::copy(y.begin() + long(y.size() / 2), y.end(), x.begin() + long(x.size() / 2));
        CHECK_EQ(pattern_errors(x.data(), kImageBytes, 41, 0), kImageBytes / 16);
        CHECK_EQ(pattern_errors(y.data(), kImageBytes, 42, 0), 0u);
    }
    const std::string name = shm_name("stress");
    std::string err;
    auto s = ShmSink::create(name, g_cams, kImageBytes, err);
    CHECK(s != nullptr);
    if (!s) return;
    std::atomic<bool> stop{false};
    struct RStat {
        uint64_t ok = 0, mixed = 0, order = 0, missing = 0, discarded = 0, other = 0;
    };
    RStat rs[3];
    // reader 0: as fast as possible; reader 1: 1 ms pause after each read; reader 2: 5 ms pause and every 4th read
    // with a 2 ms delay between record and image copy (the writer overwrites the slot during the read).
    const int pause_us[3] = {0, 1000, 5000};
    std::vector<std::thread> th;
    for (int k = 0; k < 3; k++) {
        th.emplace_back([&, k] {
            std::string e;
            auto r = ShmReader::attach_in_process(*s, e);
            if (!r) {
                rs[k].other++;
                return;
            }
            std::vector<uint16_t> a(kImageBytes / 2), b(kImageBytes / 2);
            uint64_t last = 0, iter = 0;
            bool first = true;
            while (!stop.load(std::memory_order_relaxed)) {
                PairRecord rec;
                const int64_t delay = (k == 2 && (iter++ % 4) == 0) ? 2000000 : 0;
                const auto res = r->next(rec, a.data(), b.data(), 100000000, delay);
                if (res == ShmReader::Result::WriterStopped) break;
                if (res != ShmReader::Result::Ok) {
                    if (res != ShmReader::Result::Timeout) rs[k].other++;
                    continue;
                }
                if (!rec_ok(rec) || pattern_errors(a.data(), kImageBytes, rec.pair_seq, 0) ||
                    pattern_errors(b.data(), kImageBytes, rec.pair_seq, 1))
                    rs[k].mixed++;
                if (!first && rec.pair_seq <= last) rs[k].order++;
                first = false;
                last = rec.pair_seq;
                rs[k].ok++;
                if (pause_us[k]) std::this_thread::sleep_for(std::chrono::microseconds(pause_us[k]));
            }
            rs[k].missing = r->missing();
            rs[k].discarded = r->discarded_reads();
        });
    }
    std::vector<uint16_t> a(kImageBytes / 2), b(kImageBytes / 2);
    uint64_t n = 0;
    const int64_t end = mono_ns() + int64_t(dur * 1e9);
    while (mono_ns() < end) {
        fill_pattern(a.data(), kImageBytes, n, 0);
        fill_pattern(b.data(), kImageBytes, n, 1);
        PairRecord rec = pattern_rec(n);
        rec.publish_mono_ns = mono_ns();
        rec.publish_real_ns = real_ns();
        s->on_pair(rec, a.data(), b.data());
        n++;
    }
    stop.store(true);
    for (auto& t : th) t.join();
    s->close();
    std::printf("  %.0f s, writer pairs %llu (%.0f/s)\n", dur, (unsigned long long)n, double(n) / dur);
    uint64_t mixed = 0, order = 0, discarded = 0, other = 0;
    for (int k = 0; k < 3; k++) {
        std::printf("  reader %d (pause %d us): accepted %llu, mixed %llu, order errors %llu, missing %llu, discarded "
                    "reads %llu, other results %llu\n",
                    k, pause_us[k], (unsigned long long)rs[k].ok, (unsigned long long)rs[k].mixed,
                    (unsigned long long)rs[k].order, (unsigned long long)rs[k].missing,
                    (unsigned long long)rs[k].discarded, (unsigned long long)rs[k].other);
        CHECK_GE(rs[k].ok, 10u);
        mixed += rs[k].mixed;
        order += rs[k].order;
        discarded += rs[k].discarded;
        other += rs[k].other;
    }
    std::printf("  accepted reads with mixed data: %llu\n", (unsigned long long)mixed);
    CHECK_EQ(mixed, 0u);
    CHECK_EQ(order, 0u);
    CHECK_EQ(other, 0u);
    CHECK_GE(discarded, 1u);   // the test did provoke reads that overlapped a write (and they were rejected)
    CHECK(!object_exists(name));
}
