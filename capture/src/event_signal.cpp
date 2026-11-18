#include "event_signal.hpp"

#include <climits>
#include <ctime>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <thread>
#include <chrono>
#endif

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t), "futex word must be 32 bits");
static_assert(std::atomic<uint32_t>::is_always_lock_free, "futex word must be lock-free");

namespace cap {

#if defined(__linux__)
namespace {
long futex(void* addr, int op, uint32_t val, const timespec* ts) {
    return syscall(SYS_futex, addr, op, val, ts, nullptr, 0);
}
timespec to_ts(int64_t ns) {
    timespec ts;
    ts.tv_sec = ns / 1000000000LL;
    ts.tv_nsec = ns % 1000000000LL;
    return ts;
}
}  // namespace

EventSignal::EventSignal() = default;
EventSignal::~EventSignal() = default;

void EventSignal::notify() {
    seq_.fetch_add(1, std::memory_order_release);
    futex(&seq_, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr);
}

void EventSignal::wait(uint32_t seen, int64_t timeout_ns) {
    if (timeout_ns == 0) return;
    if (timeout_ns < 0) {
        futex(&seq_, FUTEX_WAIT_PRIVATE, seen, nullptr);
    } else {
        timespec ts = to_ts(timeout_ns);  // FUTEX_WAIT: relative timeout on CLOCK_MONOTONIC
        futex(&seq_, FUTEX_WAIT_PRIVATE, seen, &ts);
    }
}

void shared_word_wake_all(std::atomic<uint32_t>* word) { futex(word, FUTEX_WAKE, INT_MAX, nullptr); }

void shared_word_wait(std::atomic<uint32_t>* word, uint32_t seen, int64_t timeout_ns) {
    if (timeout_ns < 0) {
        futex(word, FUTEX_WAIT, seen, nullptr);
    } else {
        timespec ts = to_ts(timeout_ns);
        futex(word, FUTEX_WAIT, seen, &ts);
    }
}

#elif defined(__APPLE__)

EventSignal::EventSignal() : sem_(dispatch_semaphore_create(0)) {}
EventSignal::~EventSignal() { dispatch_release(sem_); }

void EventSignal::notify() {
    seq_.fetch_add(1, std::memory_order_release);
    dispatch_semaphore_signal(sem_);
}

void EventSignal::wait(uint32_t seen, int64_t timeout_ns) {
    if (timeout_ns == 0) return;
    // The semaphore may hold extra counts from earlier notifies; the loop re-checks the counter.
    const dispatch_time_t deadline =
        timeout_ns < 0 ? DISPATCH_TIME_FOREVER : dispatch_time(DISPATCH_TIME_NOW, timeout_ns);
    while (seq_.load(std::memory_order_acquire) == seen) {
        if (dispatch_semaphore_wait(sem_, deadline) != 0) return;  // timed out
    }
}

void shared_word_wake_all(std::atomic<uint32_t>*) {}

void shared_word_wait(std::atomic<uint32_t>* word, uint32_t seen, int64_t timeout_ns) {
    // macOS has no public process-shared futex. Unit tests only: sleep in 1 ms steps.
    int64_t waited = 0;
    while (word->load(std::memory_order_acquire) == seen && (timeout_ns < 0 || waited < timeout_ns)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        waited += 1000000;
    }
}
#else
#error "unsupported platform"
#endif

}  // namespace cap
