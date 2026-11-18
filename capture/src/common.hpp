// Constants and small helpers shared by all components. No librealsense dependency.
#pragma once

#include <cstddef>
#include <cstdint>
#include <ctime>

namespace cap {

constexpr int kNumCameras = 2;
constexpr int kCamA = 0;
constexpr int kCamB = 1;

constexpr int kWidth = 848;
constexpr int kHeight = 480;
constexpr int kFps = 90;  // requested stream rate (the camera delivers about 89.87 fps)
constexpr std::size_t kImageBytes = std::size_t(kWidth) * kHeight * 2;  // Z16: 814,080 bytes

constexpr std::size_t kRingSlots = 8;     // per camera (R3)
constexpr std::size_t kShmSlots = 4;      // pair slots in shared memory (R5)

// Expected frame period from the investigation (report exp 3). The program measures the real period from hardware
// timestamps; this value is only used before the measurement exists or when hardware timestamps are missing.
constexpr double kExpectedPeriodMs = 11.128;
constexpr double kExpectedFps = 89.87;    // R6 frame-rate check: +- kFpsWarnTolerance
constexpr double kFpsWarnTolerance = 0.5;

// Timestamp domains, same values as rs2_timestamp_domain.
constexpr int kDomainHardwareClock = 0;
constexpr int kDomainSystemTime = 1;
constexpr int kDomainGlobalTime = 2;

inline int64_t now_ns(clockid_t c) {
    timespec ts;
    clock_gettime(c, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}
inline int64_t mono_ns() { return now_ns(CLOCK_MONOTONIC); }
inline int64_t real_ns() { return now_ns(CLOCK_REALTIME); }

inline const char* cam_name(int cam) { return cam == kCamA ? "A" : "B"; }

}  // namespace cap
