// Fixed-bin histogram with atomic bins: add() is lock-free and allocation-free (usable in the callback).
// Readers take snapshots; the difference of two snapshots gives the distribution of an interval.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cap {

class AtomicHistogram {
public:
    // Bins [lo + i*width, lo + (i+1)*width) for i < nbins, plus one underflow and one overflow bin.
    AtomicHistogram(double lo, double width, std::size_t nbins)
        : lo_(lo), width_(width), n_(nbins), bins_(new std::atomic<uint64_t>[nbins + 2]) {
        for (std::size_t i = 0; i < n_ + 2; i++) bins_[i].store(0, std::memory_order_relaxed);
    }

    void add(double v) {
        std::size_t i;
        if (v < lo_) i = 0;
        else {
            const double k = (v - lo_) / width_;
            i = k >= double(n_) ? n_ + 1 : std::size_t(k) + 1;
        }
        bins_[i].fetch_add(1, std::memory_order_relaxed);
    }

    // counts[0] = underflow, counts[1..n] = bins, counts[n+1] = overflow
    void snapshot(std::vector<uint64_t>& counts) const {
        counts.resize(n_ + 2);
        for (std::size_t i = 0; i < n_ + 2; i++) counts[i] = bins_[i].load(std::memory_order_relaxed);
    }

    double lo() const { return lo_; }
    double width() const { return width_; }
    std::size_t nbins() const { return n_; }

    // Value at quantile q (0..1) of a snapshot (or snapshot difference): upper edge of the bin that reaches q.
    // Returns NaN when empty, +inf when the quantile is in the overflow bin, lo when in the underflow bin.
    double quantile(const std::vector<uint64_t>& c, double q) const;
    // Lower edge of the first / upper edge of the last non-empty bin.
    double min_value(const std::vector<uint64_t>& c) const;
    double max_value(const std::vector<uint64_t>& c) const;
    static uint64_t total(const std::vector<uint64_t>& c);
    static std::vector<uint64_t> diff(const std::vector<uint64_t>& now, const std::vector<uint64_t>& before);

private:
    double lo_, width_;
    std::size_t n_;
    std::unique_ptr<std::atomic<uint64_t>[]> bins_;
};

// Lock-free running maximum (CAS loop).
inline void atomic_max(std::atomic<int64_t>& m, int64_t v) {
    int64_t cur = m.load(std::memory_order_relaxed);
    while (v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}
inline void atomic_min(std::atomic<int64_t>& m, int64_t v) {
    int64_t cur = m.load(std::memory_order_relaxed);
    while (v < cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}

}  // namespace cap
