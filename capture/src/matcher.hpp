// Matcher thread (R4): sleeps on the event signal, drains both rings into the pairing engine, gives pairs to the sink,
// and pops ring slots in FIFO order. It never polls: it waits for a callback signal or for the next timeout deadline.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

#include "event_signal.hpp"
#include "histogram.hpp"
#include "pair_sink.hpp"
#include "pairing_engine.hpp"
#include "producer.hpp"
#include "spsc_ring.hpp"
#include "timestamp_source.hpp"

namespace cap {

// Element of the optional CSV queue: the pair record plus the restart generation (epoch) of each frame.
struct PairCsvItem {
    PairRecord rec;
    uint64_t epoch[2];
};

// Output delay (gap 3): publish time (CLOCK_REALTIME) minus SDK global timestamp of the older and of the newer frame
// of the pair, 0.1 ms bins 0-60 ms. Separate histograms for usual pairs and for pairs whose difference is within
// kNearHalfMarginMs of half a frame period (there the matcher may wait for the next frame to decide).
constexpr double kNearHalfMarginMs = 0.5;
enum DelayCat { kDelayUsual = 0, kDelayNearHalf = 1 };
enum DelayWhich { kDelayOlder = 0, kDelayNewer = 1 };

struct MatcherSnapshot {
    EngineCounters engine;
    double period_ms = 0;
    double camera_period_ms[2] = {0, 0};
    bool period_known = false;
    bool settled[2] = {false, false};
    uint64_t csv_overflows = 0;
    uint64_t order_errors = 0;   // internal consistency check: emitted frame was not the ring front (must stay 0)
};

class Matcher {
public:
    Matcher(FrameProducer& a, FrameProducer& b, TimestampSource& ts, PairSink& sink, EventSignal& sig,
            const EngineConfig& cfg = EngineConfig());
    ~Matcher();

    // Optional queue for CSV output (filled by the matcher, drained by a non-critical thread).
    void set_csv_queue(SpscRing<PairCsvItem>* q, EventSignal* q_sig) { csv_q_ = q; csv_sig_ = q_sig; }
    void set_affinity(int cpu) { cpu_ = cpu; }

    void start();
    void stop();     // sets the stop flag, wakes the thread and joins it

    MatcherSnapshot snapshot();                      // any thread
    const AtomicHistogram& dt_hist() const { return dt_hist_; }
    const AtomicHistogram& out_delay_hist(int cat, int which) const { return out_delay_[cat][which]; }
    // Last pair publish time (CLOCK_MONOTONIC) and the epochs of its frames (recovery measurement), any thread.
    int64_t last_pair_mono_ns() const { return last_pair_mono_ns_.load(std::memory_order_relaxed); }
    uint64_t last_pair_epoch(int cam) const { return last_pair_epoch_[cam].load(std::memory_order_relaxed); }
    int64_t dt_min_ns() const { return dt_min_ns_.load(std::memory_order_relaxed); }
    int64_t dt_max_ns() const { return dt_max_ns_.load(std::memory_order_relaxed); }
    uint64_t pairs() const { return pairs_.load(std::memory_order_relaxed); }

private:
    class Out;
    void run();
    void publish_snapshot(bool force);

    FrameProducer* prod_[2];
    TimestampSource& ts_;
    PairSink& sink_;
    EventSignal& sig_;
    PairingEngine engine_;
    SpscRing<PairCsvItem>* csv_q_ = nullptr;
    EventSignal* csv_sig_ = nullptr;
    int cpu_ = -1;

    std::thread th_;
    std::atomic<bool> stop_{false};

    // matcher-thread state
    std::size_t mirrored_[2] = {0, 0};
    uint64_t next_seq_ = 0;
    bool period_sent_ = false;
    uint64_t csv_overflows_ = 0;
    uint64_t order_errors_ = 0;

    // shared statistics
    AtomicHistogram dt_hist_;                        // dt = B - A, 0.01 ms bins, -6 .. +6 ms
    std::atomic<int64_t> dt_min_ns_{INT64_MAX}, dt_max_ns_{INT64_MIN};
    std::atomic<uint64_t> pairs_{0};
    AtomicHistogram out_delay_[2][2];               // [DelayCat][DelayWhich], 0.1 ms bins, 0-60 ms
    std::atomic<int64_t> last_pair_mono_ns_{0};
    std::atomic<uint64_t> last_pair_epoch_[2] = {{0}, {0}};
    std::mutex snap_mu_;                             // matcher uses try_lock only (never waits)
    MatcherSnapshot snap_;
};

}  // namespace cap
