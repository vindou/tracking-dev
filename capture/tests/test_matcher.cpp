// T1: pairing engine with synthetic timestamp sequences (cases a-f of the task), plus a threaded pipeline test
// (producers -> rings -> matcher thread -> sink) that also runs under ThreadSanitizer.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <thread>
#include <vector>

#include "../src/matcher.hpp"
#include "../src/pairing_engine.hpp"
#include "../src/producer.hpp"
#include "test_framework.hpp"

using namespace cap;

namespace {

constexpr double P = 11.128;                 // ms, true frame period of the synthetic cameras
constexpr double kBase = 1.7e12;             // ms, global time base (like CLOCK_REALTIME in ms)
constexpr int64_t kMono0 = 1000000000000LL;  // ns, monotonic time of global time kBase
constexpr double kLatency = 9.0;             // ms, delivery latency

struct Ev {
    int64_t arrival;
    FrameInfo f;
};

// Scenario description. phase_b(t_s) = time offset of camera B frames relative to A (ms) at stream time t_s.
struct Scenario {
    double duration_s = 60;
    std::function<double(double)> phase_b = [](double) { return 2.0; };
    std::set<int64_t> drop_a;                  // frame indices of A that never arrive
    int hw_domain_frames = 0;                  // first frames of each stream in Hardware Clock domain
    // arrival override for A: frames [from, to] arrive together at (to_arrival)
    int64_t burst_from = -1, burst_to = -1;
    double burst_arrival_k = 0;                // arrival time of the burst, in A frame periods
    // restart of B: frames with t in [restart_at_s, restart_at_s + restart_gap_s) do not exist; later frames epoch 1
    double restart_at_s = -1, restart_gap_s = 0.5;
    double phase_after_restart = -3.0;
    double start_s[2] = {0, 0};               // stream start of A / B (s): frames before it do not exist
};

std::vector<Ev> make_events(const Scenario& sc) {
    std::vector<Ev> ev;
    const int64_t n = int64_t(sc.duration_s * 1000.0 / P);
    uint64_t id = 0;
    for (int cam = 0; cam < 2; cam++) {
        int64_t kb_epoch1 = 0;
        for (int64_t k = 0; k < n; k++) {
            const double t_rel = k * P;                       // ms since stream start (camera A clock)
            const double t_s = t_rel / 1000.0;
            double off = 0;
            uint64_t epoch = 0;
            int64_t counter = 1000 + k;
            if (cam == 1) {
                if (sc.restart_at_s >= 0 && t_s >= sc.restart_at_s) {
                    if (t_s < sc.restart_at_s + sc.restart_gap_s) continue;
                    epoch = 1;
                    counter = kb_epoch1++;
                    off = sc.phase_after_restart;
                } else {
                    off = sc.phase_b(t_s);
                    counter = 50000 + k;
                }
            } else if (sc.drop_a.count(k)) {
                continue;
            }
            if (t_s < sc.start_s[cam]) continue;   // this stream started later
            FrameInfo f;
            f.cam = cam;
            f.epoch = epoch;
            f.id = 0;
            const bool hw = (cam == 1 && epoch == 1) ? (counter < sc.hw_domain_frames) : (k < sc.hw_domain_frames);
            f.time_usable = !hw;
            f.t_ms = hw ? t_rel + off : kBase + t_rel + off;
            f.hw_ts_us = int64_t(std::llround((t_rel + off) * 1000.0)) + (cam ? 777000000 : 123000000);
            f.counter = counter;
            int64_t arr = kMono0 + int64_t(std::llround((t_rel + off + kLatency) * 1e6));
            if (cam == 0 && sc.burst_from >= 0 && k >= sc.burst_from && k <= sc.burst_to)
                arr = kMono0 + int64_t(std::llround((sc.burst_arrival_k * P + kLatency) * 1e6)) + (k - sc.burst_from) * 1000;
            f.arrival_ns = arr;
            ev.push_back({arr, f});
        }
    }
    std::stable_sort(ev.begin(), ev.end(), [](const Ev& x, const Ev& y) { return x.arrival < y.arrival; });
    for (auto& e : ev) e.f.id = ++id;
    return ev;
}

struct Recorder : EngineOutput {
    std::vector<std::pair<FrameInfo, FrameInfo>> pairs;
    std::vector<double> dts;
    std::vector<std::pair<FrameInfo, Discard>> discards;
    std::vector<uint64_t> emitted[2];   // ids in emission order per camera
    void on_pair(const FrameInfo& a, const FrameInfo& b, double dt) override {
        pairs.push_back({a, b});
        dts.push_back(dt);
        emitted[0].push_back(a.id);
        emitted[1].push_back(b.id);
    }
    void on_discard(const FrameInfo& f, Discard r) override {
        discards.push_back({f, r});
        emitted[f.cam].push_back(f.id);
    }
    uint64_t count(int cam, Discard r) const {
        uint64_t n = 0;
        for (auto& d : discards)
            if (d.first.cam == cam && d.second == r) n++;
        return n;
    }
};

struct Result {
    Recorder rec;
    EngineCounters ctr;
    double tol = 0;
    std::vector<uint64_t> pushed[2];
};

// Feeds the events in arrival order; timeouts are processed at their deadlines like the matcher thread does.
Result run(const std::vector<Ev>& ev) {
    Result r;
    PairingEngine eng;
    for (const Ev& e : ev) {
        for (int guard = 0; guard < 1000; guard++) {
            const int64_t dl = eng.next_deadline_ns();
            if (dl > e.arrival) break;
            eng.process(dl, r.rec);
        }
        eng.push(e.f, r.rec);
        r.pushed[e.f.cam].push_back(e.f.id);
        eng.process(e.arrival, r.rec);
    }
    const int64_t end = ev.back().arrival + 10000000000LL;
    for (int guard = 0; guard < 1000 && eng.next_deadline_ns() <= end; guard++) eng.process(eng.next_deadline_ns(), r.rec);
    eng.process(end, r.rec);
    r.ctr = eng.counters();
    r.tol = eng.tolerance_ms();
    CHECK_EQ(eng.queued(0) + eng.queued(1), 0u);   // nothing left behind (no deadlock)
    return r;
}

// Invariants for every scenario: each frame leaves exactly once, FIFO per camera, no frame in two pairs, pairs in time
// order, all differences within tolerance, counters agree with what was emitted.
void check_invariants(const Result& r) {
    for (int c = 0; c < 2; c++) {
        CHECK(r.rec.emitted[c] == r.pushed[c]);      // exactly once, same order as pushed
        CHECK_EQ(r.ctr.pushed[c], uint64_t(r.pushed[c].size()));
        CHECK_EQ(r.ctr.unpaired(c), r.rec.count(c, Discard::Unpaired));
        CHECK_EQ(r.ctr.paired[c], uint64_t(r.rec.pairs.size()));
    }
    CHECK_EQ(r.ctr.pairs, uint64_t(r.rec.pairs.size()));
    std::set<uint64_t> used;
    for (std::size_t i = 0; i < r.rec.pairs.size(); i++) {
        const auto& p = r.rec.pairs[i];
        CHECK(used.insert(p.first.id).second);
        CHECK(used.insert(p.second.id).second);
        CHECK_EQ(p.first.cam, 0);
        CHECK_EQ(p.second.cam, 1);
        CHECK(p.first.time_usable && p.second.time_usable);
        CHECK_LE(std::fabs(r.rec.dts[i]), r.tol + 1e-9);
        if (i) {
            CHECK(p.first.t_ms > r.rec.pairs[i - 1].first.t_ms);
            CHECK(p.second.t_ms > r.rec.pairs[i - 1].second.t_ms);
        }
    }
}

}  // namespace

TEST(matcher_a_constant_offsets) {
    for (double off : {0.0, 2.0, 5.0, -5.0, 5.3, -2.0}) {
        Scenario sc;
        sc.phase_b = [off](double) { return off; };
        Result r = run(make_events(sc));
        check_invariants(r);
        std::printf("  offset %+.1f ms: pairs=%llu unpaired A/B=%llu/%llu changes=%llu tol=%.4f\n", off,
                    (unsigned long long)r.ctr.pairs, (unsigned long long)r.ctr.unpaired(0),
                    (unsigned long long)r.ctr.unpaired(1), (unsigned long long)r.ctr.partner_changes, r.tol);
        CHECK_LE(std::fabs(r.tol - P / 2), 1e-6);         // period measured from hardware timestamps
        CHECK_EQ(r.ctr.partner_changes, 0u);
        for (int c = 0; c < 2; c++) {
            // gap 4: pairing begins cleanly; only near half a period a boundary frame may stay without partner
            if (std::fabs(std::fabs(off) - P / 2) > 0.5) CHECK_EQ(r.ctr.unpaired(c), 0u);
            else CHECK_LE(r.ctr.unpaired(c), 1u);
            CHECK_GE(r.ctr.paired[c] + 2, r.ctr.eligible(c));
        }
        // 60 s, settle 5 s: about 55 s of pairs
        CHECK_GE(r.ctr.pairs, uint64_t(54.5 * 1000 / P));
        // timestamps near 1.7e12 ms have a double resolution of about 2.4e-4 ms
        for (std::size_t i = 0; i < r.rec.dts.size(); i++) CHECK_LE(std::fabs(r.rec.dts[i] - off), 1e-3);
    }
}

TEST(matcher_b_offset_through_half_period_with_noise) {
    Scenario sc;
    sc.duration_s = 180;
    // drifts from 4 ms to 7 ms (through P/2 = 5.564 ms) with +-2 ms slow noise (20 s period)
    sc.phase_b = [](double t) { return 4.0 + 3.0 * t / 180.0 + 2.0 * std::sin(2 * M_PI * t / 20.0); };
    Result r = run(make_events(sc));
    check_invariants(r);
    std::printf("  pairs=%llu unpaired A/B=%llu/%llu changes=%llu timeouts A/B=%llu/%llu\n",
                (unsigned long long)r.ctr.pairs, (unsigned long long)r.ctr.unpaired(0),
                (unsigned long long)r.ctr.unpaired(1), (unsigned long long)r.ctr.partner_changes,
                (unsigned long long)r.ctr.timeout_decisions[0], (unsigned long long)r.ctr.timeout_decisions[1]);
    CHECK_GE(r.ctr.partner_changes, 2u);                  // the partner changes, several times with noise
    for (int c = 0; c < 2; c++) {
        // each partner change leaves at most one frame without partner on one camera
        CHECK_LE(r.ctr.unpaired(c), r.ctr.partner_changes + 2);
    }
    // count partner changes independently: key = counterB - counterA
    uint64_t changes = 0;
    for (std::size_t i = 1; i < r.rec.pairs.size(); i++) {
        const auto& p = r.rec.pairs[i];
        const auto& q = r.rec.pairs[i - 1];
        if (p.second.counter - p.first.counter != q.second.counter - q.first.counter) changes++;
    }
    CHECK_EQ(changes, r.ctr.partner_changes);
}

TEST(matcher_c_single_missing_frames) {
    Scenario base;
    Result r0 = run(make_events(base));
    Scenario sc;
    for (int64_t k : {700, 1000, 1001 + 5, 1500, 2000, 2600, 3100, 3500, 4000, 4700}) sc.drop_a.insert(k);
    Result r = run(make_events(sc));
    check_invariants(r);
    CHECK_EQ(r.ctr.pushed[0] + 10, r0.ctr.pushed[0]);
    CHECK_EQ(r.ctr.pairs + 10, r0.ctr.pairs);             // each missing A frame costs exactly one pair
    CHECK_EQ(r.ctr.unpaired(1), r0.ctr.unpaired(1) + 10); // its B partner is counted unpaired
    CHECK_EQ(r.ctr.unpaired(0), r0.ctr.unpaired(0));
    CHECK_EQ(r.ctr.partner_changes, 0u);                  // frame counters keep their numbering: no change
}

TEST(matcher_d_gap_then_burst) {
    Scenario base;
    Result r0 = run(make_events(base));
    // gap of 3 periods: A frames 2001..2003 all arrive at the time of frame 2003
    {
        Scenario sc;
        sc.burst_from = 2001;
        sc.burst_to = 2003;
        sc.burst_arrival_k = 2003;
        Result r = run(make_events(sc));
        check_invariants(r);
        CHECK_EQ(r.ctr.pairs, r0.ctr.pairs);              // nothing waited longer than 3 periods: all paired
        CHECK_EQ(r.ctr.unpaired(0), r0.ctr.unpaired(0));
        CHECK_EQ(r.ctr.unpaired(1), r0.ctr.unpaired(1));
    }
    // gap of 5.5 periods: A frames 2001..2005 arrive together at 2005.5 periods; B frames 2001 and 2002 wait more than
    // 3 periods and are discarded; then A 2001 and 2002 have no partner left.
    {
        Scenario sc;
        sc.burst_from = 2001;
        sc.burst_to = 2005;
        sc.burst_arrival_k = 2005.5;
        Result r = run(make_events(sc));
        check_invariants(r);
        CHECK_EQ(r.ctr.pairs + 2, r0.ctr.pairs);
        CHECK_EQ(r.ctr.unpaired(1), r0.ctr.unpaired(1) + 2);
        CHECK_EQ(r.ctr.unpaired(0), r0.ctr.unpaired(0) + 2);
        CHECK_EQ(r.ctr.timeout_decisions[1], r0.ctr.timeout_decisions[1] + 2);
    }
}

TEST(matcher_e_startup_hardware_clock_domain) {
    Scenario sc;
    sc.hw_domain_frames = 7;
    Result r = run(make_events(sc));
    check_invariants(r);
    for (auto& p : r.rec.pairs) {
        CHECK(p.first.arrival_ns - kMono0 >= 5000000000LL);   // nothing before the settle period
        CHECK(p.second.arrival_ns - kMono0 >= 5000000000LL);
    }
    for (int c = 0; c < 2; c++) {
        CHECK_GE(r.ctr.discards[c][int(Discard::Settle)], 7u);
        CHECK_LE(r.ctr.unpaired(c), 1u);
    }
    CHECK_GE(r.ctr.pairs, uint64_t(54.5 * 1000 / P));
}

TEST(matcher_f_restart_of_one_stream) {
    Scenario sc;
    sc.duration_s = 60;
    sc.restart_at_s = 30;
    sc.restart_gap_s = 0.5;
    sc.phase_after_restart = -3.0;
    sc.hw_domain_frames = 5;   // the restarted stream also starts in Hardware Clock domain
    Result r = run(make_events(sc));
    check_invariants(r);
    CHECK_EQ(r.ctr.epoch_changes[1], 1u);
    CHECK_EQ(r.ctr.epoch_changes[0], 0u);
    // first epoch-1 frame of B arrives at 30.5 s + latency; no pair before its 5 s settle period has passed
    int64_t first_b1 = INT64_MAX;
    uint64_t pairs_after = 0, pairs_before = 0;
    for (auto& p : r.rec.pairs) {
        if (p.second.epoch == 1) {
            pairs_after++;
            first_b1 = std::min(first_b1, p.second.arrival_ns);
        } else {
            pairs_before++;
            CHECK(p.second.arrival_ns - kMono0 < 30000000000LL + 20000000LL);
        }
    }
    CHECK(first_b1 - kMono0 >= int64_t((30.5 + 5.0) * 1e9));
    CHECK_GE(pairs_before, uint64_t(24.5 * 1000 / P));
    CHECK_GE(pairs_after, uint64_t(24.0 * 1000 / P));
    CHECK_EQ(r.ctr.partner_changes, 0u);                  // a new epoch resets the partner reference
    CHECK_GE(r.ctr.discards[0][int(Discard::PartnerSettling)], uint64_t(5.0 * 1000 / P));
    // While B is stopped (0.5 s), A frames wait for a partner and are discarded as unpaired after 3 periods: the A
    // frames that arrived after the last old B frame and timed out before the first new B frame arrived.
    const auto ev = make_events(sc);
    int64_t last_b0 = 0, first_b1_arr = INT64_MAX;
    for (auto& e : ev)
        if (e.f.cam == 1) {
            if (e.f.epoch == 0) last_b0 = std::max(last_b0, e.arrival);
            else first_b1_arr = std::min(first_b1_arr, e.arrival);
        }
    uint64_t expect_a = 0;
    for (auto& e : ev)
        if (e.f.cam == 0 && e.arrival > last_b0 && e.arrival + int64_t(3 * P * 1e6) <= first_b1_arr) expect_a++;
    std::printf("  unpaired A=%llu (expected about %llu) B=%llu, pairs before/after restart=%llu/%llu\n",
                (unsigned long long)r.ctr.unpaired(0), (unsigned long long)expect_a,
                (unsigned long long)r.ctr.unpaired(1), (unsigned long long)pairs_before, (unsigned long long)pairs_after);
    CHECK_LE(r.ctr.unpaired(0), expect_a + 3);
    CHECK_GE(r.ctr.unpaired(0) + 3, expect_a);
    CHECK_LE(r.ctr.unpaired(1), 3u);
    for (auto& p : r.rec.pairs) CHECK(std::fabs((p.second.t_ms - p.first.t_ms) - (p.second.epoch ? -3.0 : 2.0)) < 1e-3);
}

// Gap 4: the camera that settles second must find its partner for its first settled frame. The cameras start at
// different times (as in the start/stop runs: B about 60 ms after A, so A settles first) with B later or earlier than A.
// Away from half a period no frame may be counted unpaired.
TEST(matcher_g_first_pair_after_settle) {
    for (double start_b : {0.06, 0.5, -0.06, -0.5}) {
        for (double off : {3.0, 4.5, 1.0, 0.2, -0.2, -1.0, -3.0, -4.5}) {
            Scenario sc;
            sc.duration_s = 20;
            sc.phase_b = [off](double) { return off; };
            if (start_b > 0) sc.start_s[1] = start_b;
            else sc.start_s[0] = -start_b;
            Result r = run(make_events(sc));
            check_invariants(r);
            const bool ok = r.ctr.unpaired(0) == 0 && r.ctr.unpaired(1) == 0;
            std::printf("  %s starts %.2f s later, offset %+.1f ms: pairs=%llu unpaired A/B=%llu/%llu partner_settling A/B=%llu/%llu%s\n",
                        start_b > 0 ? "B" : "A", std::fabs(start_b), off, (unsigned long long)r.ctr.pairs,
                        (unsigned long long)r.ctr.unpaired(0), (unsigned long long)r.ctr.unpaired(1),
                        (unsigned long long)r.ctr.discards[0][int(Discard::PartnerSettling)],
                        (unsigned long long)r.ctr.discards[1][int(Discard::PartnerSettling)], ok ? "" : "  <-- unpaired");
            CHECK_EQ(r.ctr.unpaired(0), 0u);
            CHECK_EQ(r.ctr.unpaired(1), 0u);
            CHECK_GE(r.ctr.pairs, uint64_t(14.0 * 1000 / P));
        }
    }
    // The same after a restart of B (new epoch, B settles again while A streams): B's first settled frame pairs.
    for (double off : {3.0, -3.0}) {
        Scenario sc;
        sc.duration_s = 30;
        sc.restart_at_s = 10;
        sc.phase_after_restart = off;
        Result r = run(make_events(sc));
        check_invariants(r);
        std::printf("  restart of B, new offset %+.1f ms: unpaired A/B=%llu/%llu (A: frames while B was absent)\n", off,
                    (unsigned long long)r.ctr.unpaired(0), (unsigned long long)r.ctr.unpaired(1));
        CHECK_EQ(r.ctr.unpaired(1), 0u);
    }
}

// Threads: two producer threads deliver synthetic frames into their rings; the matcher thread pairs them into a
// StatsSink. Checks the whole chain (and runs under ThreadSanitizer).
TEST(pipeline_threads) {
    EventSignal sig;
    FrameProducer pa(kCamA, sig), pb(kCamB, sig);
    SdkGlobalTimestampSource ts;
    StatsSink sink;
    EngineConfig cfg;
    cfg.settle_ns = 200000000;               // 0.2 s for the test
    cfg.period_measure_after_ns = 50000000;
    cfg.min_period_frames = 10;
    Matcher m(pa, pb, ts, sink, sig, cfg);
    m.start();
    constexpr int N = 1500;
    // 5 ms: the 3-period timeout (15 ms) is longer than ordinary OS scheduling delays of the producer threads.
    // (With 2 ms periods a 6 ms scheduling hiccup made a frame time out correctly, but the test then failed.)
    const double period = 5.0;               // ms: fast synthetic stream
    // 64 KB per frame (deliver copies what it is given): keeps the producers on schedule under ThreadSanitizer.
    std::vector<uint16_t> img(32768, 7);
    // One common start time: arrival follows the timestamps, as with real cameras.
    const int64_t t0 = mono_ns() + 20000000;
    auto prod = [&](FrameProducer& p, double off) {
        for (int k = 0; k < N; k++) {
            const int64_t due = t0 + int64_t((k * period + off) * 1e6);
            while (mono_ns() < due) std::this_thread::yield();
            FrameMeta fm;
            fm.ts_ms = kBase + k * period + off;
            fm.domain = kDomainGlobalTime;
            fm.hw_ts_us = int64_t((k * period + off) * 1000);
            fm.frame_counter = k;
            fm.exposure_us = 1000;
            p.deliver(fm, img.data(), img.size() * 2, mono_ns(), real_ns());
        }
    };
    std::thread ta(prod, std::ref(pa), 0.0), tb(prod, std::ref(pb), 0.4);
    ta.join();
    tb.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    m.stop();
    const MatcherSnapshot s = m.snapshot();
    std::printf("  pairs=%llu unpaired=%llu/%llu overflows=%llu/%llu order_errors=%llu\n", (unsigned long long)sink.pairs(),
                (unsigned long long)s.engine.unpaired(0), (unsigned long long)s.engine.unpaired(1),
                (unsigned long long)pa.ring().overflows(), (unsigned long long)pb.ring().overflows(),
                (unsigned long long)s.order_errors);
    CHECK_EQ(s.order_errors, 0u);
    CHECK_EQ(sink.seq_errors(), 0u);
    CHECK_EQ(sink.pairs(), s.engine.pairs);
    // with the period measured as 2 ms, everything after settle should pair (allow overflow losses under TSan)
    const uint64_t lost = pa.ring().overflows() + pb.ring().overflows();
    CHECK_GE(s.engine.pairs + lost + 2, s.engine.eligible(0) < s.engine.eligible(1) ? s.engine.eligible(0) : s.engine.eligible(1));
    CHECK_GE(s.engine.pairs, uint64_t(N * 3 / 4));
}
