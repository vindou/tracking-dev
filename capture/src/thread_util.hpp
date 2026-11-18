// Thread naming and optional CPU affinity (R8: off by default, no real-time priorities).
#pragma once

#include <pthread.h>

#if defined(__linux__)
#include <sched.h>
#endif

namespace cap {

inline void set_thread_name(const char* name) {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#elif defined(__APPLE__)
    pthread_setname_np(name);
#endif
}

// Returns false if not supported or the call failed.
inline bool set_thread_affinity(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

}  // namespace cap
