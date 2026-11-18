#include "pairing_engine.hpp"


namespace cap {

const char* discard_name(Discard d) {
    switch (d) {
        case Discard::Settle: return "settle";
        case Discard::Domain: return "domain";
        case Discard::PartnerSettling: return "partner_settling";
        case Discard::Restart: return "restart";
        case Discard::Unpaired: return "unpaired";
        case Discard::QueueFull: return "queue_full";
        default: return "?";
    }
}

PairingEngine::PairingEngine(const EngineConfig& cfg) : cfg_(cfg) {}

double PairingEngine::period_ms() const {
    const Cam& a = cams_[0];
    const Cam& b = cams_[1];
    if (a.period_frozen && b.period_frozen) return 0.5 * (a.period_ms + b.period_ms);
    if (a.period_frozen) return a.period_ms;
    if (b.period_frozen) return b.period_ms;
    return cfg_.fallback_period_ms;
}

int64_t PairingEngine::timeout_ns() const { return int64_t(cfg_.timeout_periods * period_ms() * 1e6); }

void PairingEngine::push(const FrameInfo& in, EngineOutput& out) {
    FrameInfo f = in;
    Cam& cs = cams_[f.cam];
    const int other = 1 - f.cam;
    ctr_.pushed[f.cam]++;

    if (!cs.has_epoch || f.epoch != cs.epoch) {
        if (cs.has_epoch) {
            // Restart of this camera: its queued frames belong to the old stream; the other camera's queued frames can
            // no longer find a partner. Both leave now, before the new frame, so FIFO order is kept.
            ctr_.epoch_changes[f.cam]++;
            flush(f.cam, Discard::Restart, out);
            flush(other, Discard::PartnerSettling, out);
            key_valid_ = false;
        }
        cs.has_epoch = true;
        cs.epoch = f.epoch;
        cs.settle_start_ns = f.arrival_ns;
        cs.settled = false;
        cs.first_valid = false;
        cs.p_first = false;  // the measured period of a camera is kept across a restart once frozen
    }

    const int64_t since_start = f.arrival_ns - cs.settle_start_ns;
    if (!cs.period_frozen && f.hw_ts_us >= 0 && f.counter >= 0 && since_start >= cfg_.period_measure_after_ns) {
        if (!cs.p_first) {
            cs.p_first = true;
            cs.hw0 = cs.hw1 = f.hw_ts_us;
            cs.c0 = cs.c1 = f.counter;
        } else if (f.counter > cs.c1) {
            cs.hw1 = f.hw_ts_us;
            cs.c1 = f.counter;
        }
    }
    if (!cs.settled && f.time_usable && since_start >= cfg_.settle_ns) {
        if (!cs.period_frozen) {
            if (cs.p_first && cs.c1 - cs.c0 >= cfg_.min_period_frames && cs.hw1 > cs.hw0) {
                cs.period_ms = double(cs.hw1 - cs.hw0) / 1000.0 / double(cs.c1 - cs.c0);
            } else {
                cs.period_ms = cfg_.fallback_period_ms;
                ctr_.period_fallback[f.cam]++;
            }
            cs.period_frozen = true;
        }
        cs.settled = true;
    }
    f.settle_ok = cs.settled && f.time_usable;
    if (f.settle_ok && !cs.first_valid) {
        cs.first_valid = true;
        cs.first_t = f.t_ms;
    }

    if (cs.size == kQueueCap) pop_discard(f.cam, Discard::QueueFull, out);  // not expected: queue >= ring capacity
    cs.q[(cs.head + cs.size) % kQueueCap] = f;
    cs.size++;
}

bool PairingEngine::eligible(const FrameInfo& f) const {
    const Cam& cs = cams_[f.cam];
    return f.epoch == cs.epoch && f.time_usable && f.settle_ok && cams_[1 - f.cam].settled;
}

Discard PairingEngine::ineligible_reason(const FrameInfo& f) const {
    const Cam& cs = cams_[f.cam];
    if (f.epoch != cs.epoch) return Discard::Restart;
    if (!f.time_usable) return cs.settled ? Discard::Domain : Discard::Settle;
    if (!f.settle_ok) return Discard::Settle;
    return Discard::PartnerSettling;
}

void PairingEngine::pop_discard(int cam, Discard r, EngineOutput& out) {
    Cam& cs = cams_[cam];
    const FrameInfo f = cs.at(0);
    // A frame older than the other camera's first eligible frame of its epoch had its candidate partners discarded
    // during that camera's settle period: it is not really unpaired (gap 4).
    const Cam& os = cams_[1 - cam];
    if (r == Discard::Unpaired && os.first_valid && f.t_ms < os.first_t) r = Discard::PartnerSettling;
    cs.head = (cs.head + 1) % kQueueCap;
    cs.size--;
    ctr_.discards[cam][int(r)]++;
    out.on_discard(f, r);
}

void PairingEngine::flush(int cam, Discard r, EngineOutput& out) {
    while (cams_[cam].size) pop_discard(cam, r, out);
}

void PairingEngine::pop_pair(int64_t, bool by_timeout, int x_cam, EngineOutput& out) {
    Cam& ca = cams_[kCamA];
    Cam& cb = cams_[kCamB];
    const FrameInfo a = ca.at(0);
    const FrameInfo b = cb.at(0);
    ca.head = (ca.head + 1) % kQueueCap;
    ca.size--;
    cb.head = (cb.head + 1) % kQueueCap;
    cb.size--;
    ctr_.pairs++;
    ctr_.paired[0]++;
    ctr_.paired[1]++;
    if (by_timeout) ctr_.timeout_decisions[x_cam]++;
    if (a.counter >= 0 && b.counter >= 0) {
        const int64_t key = b.counter - a.counter;
        if (key_valid_ && key != last_key_) ctr_.partner_changes++;
        last_key_ = key;
        key_valid_ = true;
    }
    out.on_pair(a, b, b.t_ms - a.t_ms);
}

// One decision. Returns false when nothing more can be decided at now_ns.
bool PairingEngine::step(int64_t now_ns, EngineOutput& out) {
    const int64_t tmo = timeout_ns();
    // 1. Frames that cannot take part in pairing leave first (FIFO per camera). Exception: frames of a settled camera
    //    whose partner camera is still settling are held (newest hold_max_frames, at most the timeout), so that the
    //    first settled frame of the partner camera finds its partner (gap 4).
    bool held = false;
    for (int c = 0; c < 2; c++) {
        if (cams_[c].size && !eligible(cams_[c].at(0))) {
            const FrameInfo& f = cams_[c].at(0);
            const Discard r = ineligible_reason(f);
            if (r == Discard::PartnerSettling && cams_[c].size <= cfg_.hold_max_frames && now_ns - f.arrival_ns < tmo) {
                held = true;
                continue;
            }
            pop_discard(c, r, out);
            return true;
        }
    }
    if (held) return false;   // wait for the partner camera to settle, a newer frame, or the timeout
    const bool ha = cams_[0].size > 0;
    const bool hb = cams_[1].size > 0;

    // 2. Only one camera has a frame: wait for the other one, at most the timeout.
    if (ha != hb) {
        const int c = ha ? 0 : 1;
        const FrameInfo& f = cams_[c].at(0);
        if (now_ns - f.arrival_ns >= tmo) {
            ctr_.timeout_decisions[c]++;
            pop_discard(c, Discard::Unpaired, out);
            return true;
        }
        return false;
    }
    if (!ha) return false;

    // 3. Both fronts eligible: mutual-nearest decision.
    const FrameInfo& a = cams_[0].at(0);
    const FrameInfo& b = cams_[1].at(0);
    const int xc = (a.t_ms <= b.t_ms) ? 0 : 1;          // stream with the earlier front
    const FrameInfo& x = cams_[xc].at(0);
    const FrameInfo& y = cams_[1 - xc].at(0);
    const double d = y.t_ms - x.t_ms;                    // >= 0
    const double half = period_ms() / 2.0;               // tolerance

    if (d > half) {                                      // x is farther than the tolerance from every remaining frame
        pop_discard(xc, Discard::Unpaired, out);
        return true;
    }
    if (d <= half - cfg_.ambiguity_ms) {                 // no later X frame can be nearer to y
        pop_pair(now_ns, false, xc, out);
        return true;
    }
    // Near half a period: the next X frame decides whom y prefers.
    if (cams_[xc].size >= 2 && eligible(cams_[xc].at(1))) {
        const FrameInfo& x1 = cams_[xc].at(1);
        if (x1.t_ms - y.t_ms < d) {
            pop_discard(xc, Discard::Unpaired, out);     // y's nearest is x1
        } else {
            pop_pair(now_ns, false, xc, out);
        }
        return true;
    }
    if (cams_[xc].size >= 2 || now_ns - x.arrival_ns >= tmo) {
        // The next X frame is not eligible (it will be discarded), or it did not come in time: x is y's nearest.
        pop_pair(now_ns, cams_[xc].size < 2, xc, out);
        return true;
    }
    return false;                                        // wait for X's next frame or the timeout
}

void PairingEngine::process(int64_t now_ns, EngineOutput& out) {
    while (step(now_ns, out)) {
    }
}

int64_t PairingEngine::next_deadline_ns() const {
    int64_t d = INT64_MAX;
    const int64_t tmo = timeout_ns();
    for (int c = 0; c < 2; c++)
        if (cams_[c].size) {
            const int64_t t = cams_[c].at(0).arrival_ns + tmo;
            if (t < d) d = t;
        }
    return d;
}

}  // namespace cap
