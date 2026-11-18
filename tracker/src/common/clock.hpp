// Clocks (ns). CLOCK_MONOTONIC is the tracker's time base.
#pragma once
#include <cstdint>
#include <ctime>

namespace trk {
inline int64_t clock_ns(clockid_t c) {
    timespec ts;
    clock_gettime(c, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}
inline int64_t mono_ns() { return clock_ns(CLOCK_MONOTONIC); }
inline int64_t real_ns() { return clock_ns(CLOCK_REALTIME); }
}  // namespace trk
