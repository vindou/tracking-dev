// Phase 2: generic shared-memory ring. Same test types as the capture pair buffer: roundtrip and unlink, slow and
// stopped reader do not slow the writer, a fast reader misses nothing, writer killed with SIGKILL (detected, new
// writer followed), writer replaced, 30 s stress test with a detectable data pattern (ThreadSanitizer sees both
// sides through attach_in_process), latest().
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../src/common/clock.hpp"
#include "../src/records/records.hpp"
#include "../src/shmring/shm_ring.hpp"
#include "test_framework.hpp"

extern char** environ;
using namespace trk;

namespace {

std::string ring_name(const char* tag) { return "/trk_t_" + std::to_string(getpid()) + "_" + tag; }

uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
// Pattern: word 0 = sequence number, word i = mix64(seq) + i. Two records differ in every word.
void fill(uint8_t* rec, std::size_t size, uint64_t seq) {
    auto* w = reinterpret_cast<uint64_t*>(rec);
    w[0] = seq;
    const uint64_t k = mix64(seq);
    for (std::size_t i = 1; i < size / 8; i++) w[i] = k + i;
}
bool check(const uint8_t* rec, std::size_t size, uint64_t* seq_out = nullptr) {
    auto* w = reinterpret_cast<const uint64_t*>(rec);
    const uint64_t seq = w[0], k = mix64(seq);
    if (seq_out) *seq_out = seq;
    for (std::size_t i = 1; i < size / 8; i++)
        if (w[i] != k + i) return false;
    return true;
}

constexpr uint32_t kSize = sizeof(PoseRecord);   // 640: tests with the pose record size, stress also with the cloud

bool exists(const std::string& name) {
    const int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) return false;
    close(fd);
    return true;
}

pid_t spawn_writer(const std::string& name, int period_us, int first_delay_ms) {
    const std::string p = std::to_string(period_us), d = std::to_string(first_delay_ms);
    const char* argv[] = {tf::self_path().c_str(), "--helper-ring-writer", name.c_str(), p.c_str(), d.c_str(), nullptr};
    pid_t pid = -1;
    if (posix_spawn(&pid, tf::self_path().c_str(), nullptr, nullptr, const_cast<char* const*>(argv), environ) != 0)
        return -1;
    return pid;
}

std::unique_ptr<RingReader> attach_retry(const std::string& name, uint32_t size, double timeout_s) {
    std::string err;
    const int64_t end = mono_ns() + int64_t(timeout_s * 1e9);
    while (mono_ns() < end) {
        auto r = RingReader::attach(name, "test", 1, size, err);
        if (r) return r;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::printf("  attach failed: %s\n", err.c_str());
    return nullptr;
}

struct Run {
    uint64_t ok = 0, bad = 0, gaps = 0, first = UINT64_MAX, last = 0;
    RingReader::Result end = RingReader::Result::Ok;
    int64_t end_ns = 0;
};
Run read_n(RingReader& r, uint64_t n, int64_t timeout_ns) {
    Run run;
    std::vector<uint8_t> buf(r.header().record_size);
    while (n == 0 || run.ok < n) {
        const auto res = r.next(buf.data(), timeout_ns);
        if (res != RingReader::Result::Ok) {
            run.end = res;
            run.end_ns = mono_ns();
            return run;
        }
        uint64_t seq = 0;
        if (!check(buf.data(), buf.size(), &seq)) run.bad++;
        if (run.ok == 0) run.first = seq;
        else if (seq != run.last + 1) run.gaps += seq - run.last - 1;
        run.last = seq;
        run.ok++;
    }
    run.end_ns = mono_ns();
    return run;
}

std::atomic<bool> g_stop{false};
void on_term(int) { g_stop.store(true); }

}  // namespace

// capture_tests-style helper: trk_tests --helper-ring-writer NAME PERIOD_US FIRST_DELAY_MS
int helper_ring_writer(int argc, char** argv) {
    if (argc < 5) return 2;
    std::signal(SIGTERM, on_term);
    std::string err;
    auto w = RingWriter::create(argv[2], "test", 1, kSize, kPoseSlots, err);
    if (!w) {
        std::fprintf(stderr, "helper: %s\n", err.c_str());
        return 3;
    }
    const int period = std::atoi(argv[3]);
    std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(argv[4])));
    std::vector<uint8_t> rec(kSize);
    auto next = std::chrono::steady_clock::now();
    for (uint64_t i = 0; !g_stop.load(); i++) {
        fill(rec.data(), kSize, i);
        w->publish(rec.data());
        next += std::chrono::microseconds(period);
        std::this_thread::sleep_until(next);
    }
    w->close();
    return 0;
}

TEST(ring_roundtrip_latest_unlink) {
    const std::string name = ring_name("rt");
    std::string err;
    auto w = RingWriter::create(name, "test", 1, kSize, 8, err);
    CHECK(w != nullptr);
    if (!w) return;
    auto r = RingReader::attach(name, "test", 1, kSize, err);
    CHECK(r != nullptr);
    if (!r) return;
    std::string err2;
    CHECK(RingReader::attach(name, "test", 2, kSize, err2) == nullptr);   // other record version: rejected
    CHECK(RingReader::attach(name, "pose", 1, kSize, err2) == nullptr);   // other record type
    std::vector<uint8_t> rec(kSize), out(kSize);
    for (uint64_t i = 0; i < 5; i++) {
        fill(rec.data(), kSize, i);
        const uint64_t seq = w->publish(rec.data());   // CHECK_EQ evaluates its arguments twice
        CHECK_EQ(seq, i);
    }
    Run run = read_n(*r, 5, 100000000);
    CHECK_EQ(run.ok, 5u);
    CHECK_EQ(run.first, 0u);
    CHECK_EQ(run.bad, 0u);
    CHECK(r->next(out.data(), 10000000) == RingReader::Result::Timeout);
    for (uint64_t i = 5; i < 20; i++) {
        fill(rec.data(), kSize, i);
        w->publish(rec.data());
    }
    uint64_t seq = 0;
    CHECK(r->latest(out.data()) == RingReader::Result::Ok);
    CHECK(check(out.data(), kSize, &seq));
    CHECK_EQ(seq, 19u);
    CHECK(r->latest(out.data()) == RingReader::Result::Timeout);   // nothing newer
    w->close();
    CHECK(r->next(out.data(), 10000000) == RingReader::Result::WriterStopped);
    CHECK(!exists(name));
}

// A reader faster than the writer gets every record (writer 2 kHz, 256 slots, 3 s).
TEST(ring_fast_reader_misses_nothing) {
    const std::string name = ring_name("fast");
    std::string err;
    auto w = RingWriter::create(name, "test", 1, kSize, kPoseSlots, err);
    CHECK(w != nullptr);
    if (!w) return;
    auto r = RingReader::attach_in_process(*w, "test", 1, kSize, err);
    CHECK(r != nullptr);
    if (!r) return;
    constexpr uint64_t N = 6000;
    Run run;
    std::thread th([&] { run = read_n(*r, N, 1000000000LL); });
    std::vector<uint8_t> rec(kSize);
    auto next = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < N; i++) {
        fill(rec.data(), kSize, i);
        w->publish(rec.data());
        next += std::chrono::microseconds(500);
        std::this_thread::sleep_until(next);
    }
    th.join();
    std::printf("  fast reader: received %llu, missing %llu, gaps %llu, bad %llu\n", (unsigned long long)run.ok,
                (unsigned long long)r->missing(), (unsigned long long)run.gaps, (unsigned long long)run.bad);
    CHECK_EQ(run.ok, N);
    CHECK_EQ(r->missing(), 0u);
    CHECK_EQ(run.gaps, 0u);
    CHECK_EQ(run.bad, 0u);
    w->close();
}

// Slow reader (in-process) and stopped reader (process, SIGSTOP): the writer's time per record stays in the range of
// the run without a reader (same criteria as the capture test).
TEST(ring_slow_and_stopped_reader_do_not_slow_writer) {
    const std::string name = ring_name("slow");
    std::string err;
    constexpr uint32_t kCloud = sizeof(CloudRecord);
    auto w = RingWriter::create(name, "test", 1, kCloud, kCloudSlots, err);
    CHECK(w != nullptr);
    if (!w) return;
    auto write = [&](uint64_t first, uint64_t n, std::vector<double>& t) {
        std::vector<uint8_t> rec(kCloud);
        auto next = std::chrono::steady_clock::now();
        for (uint64_t i = first; i < first + n; i++) {
            fill(rec.data(), kCloud, i);
            const int64_t t0 = mono_ns();
            w->publish(rec.data());
            t.push_back(double(mono_ns() - t0) / 1000.0);
            next += std::chrono::microseconds(2000);
            std::this_thread::sleep_until(next);
        }
        std::sort(t.begin(), t.end());
    };
    constexpr uint64_t N = 1500;
    std::vector<double> t0, t1, t2;
    write(0, N, t0);
    std::atomic<bool> stop{false};
    uint64_t accepted = 0, bad = 0;
    std::thread reader([&] {
        std::string e;
        auto r = RingReader::attach(name, "test", 1, kCloud, e);
        if (!r) return;
        std::vector<uint8_t> buf(kCloud);
        uint64_t iter = 0;
        while (!stop.load()) {
            const int64_t delay = (iter++ % 2) ? 9000000 : 0;   // > 4 slots x 2 ms: overwritten during the read
            const auto res = r->next(buf.data(), 50000000, delay);
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            if (res != RingReader::Result::Ok) continue;
            if (!check(buf.data(), kCloud)) bad++;
            accepted++;
        }
        std::printf("  slow reader: accepted %llu, missing %llu, discarded reads %llu, bad %llu\n",
                    (unsigned long long)accepted, (unsigned long long)r->missing(),
                    (unsigned long long)r->discarded_reads(), (unsigned long long)bad);
        CHECK_GE(r->discarded_reads(), 1u);
        CHECK_GE(r->missing(), N / 2);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    write(N, N, t1);
    stop.store(true);
    reader.join();
    const pid_t child = fork();
    if (child == 0) {
        std::string e;
        auto r = RingReader::attach(name, "test", 1, kCloud, e);
        raise(SIGSTOP);
        _exit(r ? 0 : 1);
    }
    int st = 0;
    waitpid(child, &st, WUNTRACED);
    CHECK(WIFSTOPPED(st));
    write(2 * N, N, t2);
    kill(child, SIGKILL);
    waitpid(child, &st, 0);
    w->close();
    auto q = [](const std::vector<double>& v, double p) { return v[std::size_t(double(v.size() - 1) * p)]; };
    std::printf("  writer p50/p99/max us: no reader %.0f/%.0f/%.0f | slow reader %.0f/%.0f/%.0f | stopped reader "
                "%.0f/%.0f/%.0f\n", q(t0, .5), q(t0, .99), t0.back(), q(t1, .5), q(t1, .99), t1.back(), q(t2, .5),
                q(t2, .99), t2.back());
    CHECK_EQ(bad, 0u);
    CHECK_GE(accepted, 10u);
    CHECK_LE(q(t1, .5), 2.0 * q(t0, .5) + 200.0);
    CHECK_LE(q(t1, .99), 2.0 * q(t0, .99) + 500.0);
    CHECK_LE(q(t2, .5), 2.0 * q(t0, .5) + 200.0);
    CHECK_LE(q(t2, .99), 2.0 * q(t0, .99) + 500.0);
    CHECK_LE(t1.back(), 20000.0);
    CHECK_LE(t2.back(), 20000.0);
}

// Writer killed with SIGKILL: detected within 2 s; a new writer: the same reader gets all its records.
TEST(ring_killed_writer_detected_and_new_writer_followed) {
    const std::string name = ring_name("kill");
    const pid_t w1 = spawn_writer(name, 2000, 300);
    auto r = attach_retry(name, kSize, 5.0);
    CHECK(r != nullptr);
    if (!r) {
        kill(w1, SIGKILL);
        waitpid(w1, nullptr, 0);
        shm_unlink(name.c_str());
        return;
    }
    const uint64_t inst1 = r->instance_id();
    Run a = read_n(*r, 200, 2000000000LL);
    CHECK_EQ(a.ok, 200u);
    CHECK_EQ(a.first, 0u);
    CHECK_EQ(a.gaps, 0u);
    const int64_t t_kill = mono_ns();
    kill(w1, SIGKILL);
    waitpid(w1, nullptr, 0);
    Run d = read_n(*r, 0, 10000000000LL);
    const double detect_s = double(d.end_ns - t_kill) / 1e9;
    std::printf("  writer SIGKILLed: %s after %.3f s\n", RingReader::result_name(d.end), detect_s);
    CHECK(d.end == RingReader::Result::WriterNotAlive);
    CHECK_LE(detect_s, 2.0);
    std::string err;
    CHECK(!r->reattach(err));
    const pid_t w2 = spawn_writer(name, 2000, 300);
    bool ok = false;
    for (int i = 0; i < 250 && !ok; i++) {
        ok = r->reattach(err);
        if (!ok) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    CHECK(ok);
    CHECK(r->instance_id() != inst1);
    CHECK_EQ(r->attaches(), 2u);
    Run b = read_n(*r, 500, 2000000000LL);
    std::printf("  new writer: first %llu, received %llu, gaps %llu, bad %llu\n", (unsigned long long)b.first,
                (unsigned long long)b.ok, (unsigned long long)b.gaps, (unsigned long long)b.bad);
    CHECK_EQ(b.first, 0u);
    CHECK_EQ(b.ok, 500u);
    CHECK_EQ(b.gaps, 0u);
    CHECK_EQ(b.bad, 0u);
    kill(w2, SIGTERM);
    int st = 0;
    waitpid(w2, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0);
    CHECK(read_n(*r, 0, 3000000000LL).end == RingReader::Result::WriterStopped);
    CHECK(!exists(name));
}

// Writer replaced while it runs: WriterReplaced, reattach, all records of the new writer.
TEST(ring_replaced_writer_followed) {
    const std::string name = ring_name("repl");
    const pid_t w1 = spawn_writer(name, 2000, 0);
    auto r = attach_retry(name, kSize, 5.0);
    CHECK(r != nullptr);
    if (!r) {
        kill(w1, SIGKILL);
        waitpid(w1, nullptr, 0);
        shm_unlink(name.c_str());
        return;
    }
    CHECK_EQ(read_n(*r, 50, 2000000000LL).bad, 0u);
    const uint64_t inst1 = r->instance_id();
    const pid_t w2 = spawn_writer(name, 2000, 300);
    Run d = read_n(*r, 0, 5000000000LL);
    CHECK(d.end == RingReader::Result::WriterReplaced);
    std::string err;
    bool ok = false;
    for (int i = 0; i < 250 && !ok; i++) {
        ok = r->reattach(err);
        if (!ok) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    CHECK(ok);
    CHECK(r->instance_id() != inst1);
    kill(w1, SIGTERM);
    Run b = read_n(*r, 300, 2000000000LL);
    CHECK_EQ(b.first, 0u);
    CHECK_EQ(b.gaps, 0u);
    CHECK_EQ(b.bad, 0u);
    int st = 0;
    waitpid(w1, &st, 0);
    CHECK(exists(name));   // the old writer's clean stop leaves the new name alone
    kill(w2, SIGTERM);
    waitpid(w2, nullptr, 0);
    CHECK(!exists(name));
}

// 1 writer at full speed, 3 readers at different speeds, TRK_STRESS_S seconds (default 30), cloud-size records.
TEST(ring_stress_1_writer_3_readers) {
    double dur = 30;
    if (const char* e = std::getenv("TRK_STRESS_S")) dur = std::atof(e);
    constexpr uint32_t kCloud = sizeof(CloudRecord);
    {   // negative control: a record whose second half belongs to the next record is detected
        std::vector<uint8_t> x(kCloud), y(kCloud);
        fill(x.data(), kCloud, 41);
        fill(y.data(), kCloud, 42);
        std::copy(y.begin() + kCloud / 2, y.end(), x.begin() + kCloud / 2);
        CHECK(!check(x.data(), kCloud));
        CHECK(check(y.data(), kCloud));
    }
    const std::string name = ring_name("stress");
    std::string err;
    auto w = RingWriter::create(name, "test", 1, kCloud, kCloudSlots, err);
    CHECK(w != nullptr);
    if (!w) return;
    struct S { uint64_t ok = 0, mixed = 0, order = 0, missing = 0, discarded = 0, other = 0; } rs[3];
    const int pause_us[3] = {0, 1000, 5000};
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    for (int k = 0; k < 3; k++)
        th.emplace_back([&, k] {
            std::string e;
            auto r = RingReader::attach_in_process(*w, "test", 1, kCloud, e);
            if (!r) { rs[k].other++; return; }
            std::vector<uint8_t> buf(kCloud);
            uint64_t last = 0, iter = 0;
            bool first = true;
            while (!stop.load(std::memory_order_relaxed)) {
                const int64_t delay = (k == 2 && (iter++ % 4) == 0) ? 2000000 : 0;
                const auto res = r->next(buf.data(), 100000000, delay);
                if (res == RingReader::Result::WriterStopped) break;
                if (res != RingReader::Result::Ok) { if (res != RingReader::Result::Timeout) rs[k].other++; continue; }
                uint64_t seq = 0;
                if (!check(buf.data(), kCloud, &seq)) rs[k].mixed++;
                if (!first && seq <= last) rs[k].order++;
                first = false;
                last = seq;
                rs[k].ok++;
                if (pause_us[k]) std::this_thread::sleep_for(std::chrono::microseconds(pause_us[k]));
            }
            rs[k].missing = r->missing();
            rs[k].discarded = r->discarded_reads();
        });
    std::vector<uint8_t> rec(kCloud);
    uint64_t n = 0;
    const int64_t end = mono_ns() + int64_t(dur * 1e9);
    while (mono_ns() < end) {
        fill(rec.data(), kCloud, n++);
        w->publish(rec.data());
    }
    stop.store(true);
    for (auto& t : th) t.join();
    w->close();
    std::printf("  %.0f s, writer records %llu (%.0f/s)\n", dur, (unsigned long long)n, double(n) / dur);
    uint64_t mixed = 0, discarded = 0;
    for (int k = 0; k < 3; k++) {
        std::printf("  reader %d (pause %d us): accepted %llu, mixed %llu, order errors %llu, missing %llu, discarded "
                    "reads %llu, other %llu\n", k, pause_us[k], (unsigned long long)rs[k].ok,
                    (unsigned long long)rs[k].mixed, (unsigned long long)rs[k].order,
                    (unsigned long long)rs[k].missing, (unsigned long long)rs[k].discarded,
                    (unsigned long long)rs[k].other);
        CHECK_GE(rs[k].ok, 10u);
        CHECK_EQ(rs[k].order, 0u);
        CHECK_EQ(rs[k].other, 0u);
        mixed += rs[k].mixed;
        discarded += rs[k].discarded;
    }
    std::printf("  accepted reads with mixed data: %llu\n", (unsigned long long)mixed);
    CHECK_EQ(mixed, 0u);
    CHECK_GE(discarded, 1u);
    CHECK(!exists(name));
}
