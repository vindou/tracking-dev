#include "matcher.hpp"

#include <algorithm>
#include <cmath>

#include "thread_util.hpp"

namespace cap {

class Matcher::Out : public EngineOutput {
public:
    explicit Out(Matcher& m) : m_(m) {}

    void on_pair(const FrameInfo& a, const FrameInfo& b, double dt_ms) override {
        auto* sa = static_cast<const DepthSlot*>(a.ref);
        auto* sb = static_cast<const DepthSlot*>(b.ref);
        check_front(kCamA, sa);
        check_front(kCamB, sb);
        if (!m_.period_sent_) {
            m_.sink_.on_period(m_.engine_.period_ms(), m_.engine_.tolerance_ms());
            m_.period_sent_ = true;
        }
        PairRecord rec{};
        rec.pair_seq = m_.next_seq_++;
        rec.dt_ms = dt_ms;
        fill(rec.cam[0], sa->meta);
        fill(rec.cam[1], sb->meta);
        // Publish time (gap 3): taken just before the sink copies the pair (the copy into shared memory takes about
        // 0.3 ms more before the pair becomes visible to readers).
        rec.publish_mono_ns = mono_ns();
        rec.publish_real_ns = real_ns();
        m_.sink_.on_pair(rec, sa->depth, sb->depth);
        {
            const double pub_ms = double(rec.publish_real_ns) / 1e6;
            const double ga = rec.cam[0].global_ts_ms, gb = rec.cam[1].global_ts_ms;
            const double half = m_.engine_.period_ms() / 2.0;
            const int cat = std::fabs(std::fabs(dt_ms) - half) <= kNearHalfMarginMs ? kDelayNearHalf : kDelayUsual;
            m_.out_delay_[cat][kDelayOlder].add(pub_ms - std::min(ga, gb));
            m_.out_delay_[cat][kDelayNewer].add(pub_ms - std::max(ga, gb));
            m_.last_pair_epoch_[0].store(sa->meta.epoch, std::memory_order_relaxed);
            m_.last_pair_epoch_[1].store(sb->meta.epoch, std::memory_order_relaxed);
            m_.last_pair_mono_ns_.store(rec.publish_mono_ns, std::memory_order_relaxed);
        }
        if (m_.csv_q_) {
            if (PairCsvItem* q = m_.csv_q_->claim()) {
                q->rec = rec;
                q->epoch[0] = sa->meta.epoch;
                q->epoch[1] = sb->meta.epoch;
                m_.csv_q_->publish();
                m_.csv_sig_->notify();
            } else {
                m_.csv_overflows_++;
            }
        }
        m_.dt_hist_.add(dt_ms);
        const int64_t dns = int64_t(std::llround(dt_ms * 1e6));
        atomic_min(m_.dt_min_ns_, dns);
        atomic_max(m_.dt_max_ns_, dns);
        m_.pairs_.fetch_add(1, std::memory_order_relaxed);
        pop(kCamA);
        pop(kCamB);
    }

    void on_discard(const FrameInfo& f, Discard) override {
        check_front(f.cam, static_cast<const DepthSlot*>(f.ref));
        pop(f.cam);
    }

private:
    static void fill(CamRecord& c, const FrameMeta& m) {
        c.global_ts_ms = m.ts_ms;
        c.hw_ts_us = m.hw_ts_us;
        c.sensor_ts_us = m.sensor_ts_us;
        c.arrival_mono_ns = m.arrival_mono_ns;
        c.frame_counter = m.frame_counter;
        c.exposure_us = m.exposure_us;
        c.domain = m.domain;
    }
    void check_front(int cam, const DepthSlot* s) {
        if (m_.prod_[cam]->ring().available() == 0 || m_.prod_[cam]->ring().peek(0) != s) m_.order_errors_++;
    }
    void pop(int cam) {
        m_.prod_[cam]->ring().pop();
        m_.mirrored_[cam]--;
    }
    Matcher& m_;
};

Matcher::Matcher(FrameProducer& a, FrameProducer& b, TimestampSource& ts, PairSink& sink, EventSignal& sig,
                 const EngineConfig& cfg)
    : prod_{&a, &b}, ts_(ts), sink_(sink), sig_(sig), engine_(cfg), dt_hist_(-6.0, 0.01, 1200),
      out_delay_{{AtomicHistogram(0.0, 0.1, 600), AtomicHistogram(0.0, 0.1, 600)},
                 {AtomicHistogram(0.0, 0.1, 600), AtomicHistogram(0.0, 0.1, 600)}} {}

Matcher::~Matcher() { stop(); }

void Matcher::start() {
    stop_.store(false);
    th_ = std::thread([this] { run(); });
}

void Matcher::stop() {
    if (!th_.joinable()) return;
    stop_.store(true, std::memory_order_release);
    sig_.notify();
    th_.join();
}

MatcherSnapshot Matcher::snapshot() {
    std::lock_guard<std::mutex> lk(snap_mu_);
    return snap_;
}

void Matcher::publish_snapshot(bool force) {
    std::unique_lock<std::mutex> lk(snap_mu_, std::defer_lock);
    if (force) lk.lock();
    else if (!lk.try_lock()) return;   // the status thread holds it: skip, the next iteration publishes
    snap_.engine = engine_.counters();
    snap_.period_ms = engine_.period_ms();
    snap_.camera_period_ms[0] = engine_.camera_period_ms(0);
    snap_.camera_period_ms[1] = engine_.camera_period_ms(1);
    snap_.period_known = engine_.period_known();
    snap_.settled[0] = engine_.settled(0);
    snap_.settled[1] = engine_.settled(1);
    snap_.csv_overflows = csv_overflows_;
    snap_.order_errors = order_errors_;
}

void Matcher::run() {
    set_thread_name("matcher");
    if (cpu_ >= 0) set_thread_affinity(cpu_);
    Out out(*this);
    while (!stop_.load(std::memory_order_acquire)) {
        const uint32_t seen = sig_.value();
        for (int c = 0; c < 2; c++) {
            SpscRing<DepthSlot>& r = prod_[c]->ring();
            while (mirrored_[c] < r.available()) {
                const DepthSlot* s = r.peek(mirrored_[c]);
                mirrored_[c]++;
                FrameInfo f;
                f.cam = c;
                f.epoch = s->meta.epoch;
                f.time_usable = ts_.pairing_time_ms(c, s->meta, f.t_ms);
                f.hw_ts_us = s->meta.hw_ts_us;
                f.counter = s->meta.frame_counter;
                f.arrival_ns = s->meta.arrival_mono_ns;
                f.ref = s;
                engine_.push(f, out);
            }
        }
        engine_.process(mono_ns(), out);
        publish_snapshot(false);
        if (stop_.load(std::memory_order_acquire)) break;
        const int64_t dl = engine_.next_deadline_ns();
        int64_t timeout = -1;
        if (dl != INT64_MAX) {
            timeout = dl - mono_ns();
            if (timeout <= 0) continue;      // a deadline has passed: decide now
        }
        sig_.wait(seen, timeout);
    }
    publish_snapshot(true);
}

}  // namespace cap
