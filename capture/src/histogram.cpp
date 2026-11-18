#include "histogram.hpp"

#include <cmath>
#include <limits>

namespace cap {

uint64_t AtomicHistogram::total(const std::vector<uint64_t>& c) {
    uint64_t t = 0;
    for (uint64_t v : c) t += v;
    return t;
}

std::vector<uint64_t> AtomicHistogram::diff(const std::vector<uint64_t>& now, const std::vector<uint64_t>& before) {
    std::vector<uint64_t> d(now.size(), 0);
    for (std::size_t i = 0; i < now.size(); i++) d[i] = now[i] - (i < before.size() ? before[i] : 0);
    return d;
}

double AtomicHistogram::quantile(const std::vector<uint64_t>& c, double q) const {
    const uint64_t t = total(c);
    if (t == 0) return std::numeric_limits<double>::quiet_NaN();
    const double target = q * double(t);
    uint64_t acc = 0;
    for (std::size_t i = 0; i < c.size(); i++) {
        acc += c[i];
        if (double(acc) >= target && c[i]) {
            if (i == 0) return lo_;
            if (i == n_ + 1) return std::numeric_limits<double>::infinity();
            return lo_ + double(i) * width_;  // upper edge of bin i-1
        }
    }
    return std::numeric_limits<double>::infinity();
}

double AtomicHistogram::min_value(const std::vector<uint64_t>& c) const {
    for (std::size_t i = 0; i < c.size(); i++)
        if (c[i]) return i == 0 ? -std::numeric_limits<double>::infinity() : lo_ + double(i - 1) * width_;
    return std::numeric_limits<double>::quiet_NaN();
}

double AtomicHistogram::max_value(const std::vector<uint64_t>& c) const {
    for (std::size_t i = c.size(); i-- > 0;)
        if (c[i]) return i == n_ + 1 ? std::numeric_limits<double>::infinity() : lo_ + double(i) * width_;
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace cap
