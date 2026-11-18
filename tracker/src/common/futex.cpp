#include "futex.hpp"

#include <thread>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <ctime>
#endif

namespace trk {

#if defined(__linux__)
void futex_wake_all(std::atomic<uint32_t>* word) {
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAKE, INT32_MAX, nullptr, nullptr, 0);
}
void futex_wait(std::atomic<uint32_t>* word, uint32_t seen, int64_t timeout_ns) {
    if (word->load(std::memory_order_acquire) != seen) return;
    timespec ts, *pts = nullptr;
    if (timeout_ns >= 0) {
        ts.tv_sec = time_t(timeout_ns / 1000000000LL);
        ts.tv_nsec = long(timeout_ns % 1000000000LL);
        pts = &ts;
    }
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAIT, seen, pts, nullptr, 0);
}
#else
void futex_wake_all(std::atomic<uint32_t>*) {}
void futex_wait(std::atomic<uint32_t>* word, uint32_t seen, int64_t timeout_ns) {
    if (word->load(std::memory_order_acquire) != seen) return;
    int64_t t = timeout_ns < 0 || timeout_ns > 1000000 ? 1000000 : timeout_ns;
    std::this_thread::sleep_for(std::chrono::nanoseconds(t));
}
#endif

}  // namespace trk
