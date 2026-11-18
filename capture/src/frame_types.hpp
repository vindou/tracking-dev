// Frame metadata in a ring slot, and the ring slot itself.
#pragma once

#include <cstdint>

#include "common.hpp"

namespace cap {

struct FrameMeta {
    uint64_t epoch = 0;            // camera restart generation
    double ts_ms = 0;              // SDK frame timestamp (rs2::frame::get_timestamp)
    int32_t domain = -1;           // rs2_timestamp_domain
    int32_t exposure_us = -1;      // metadata ACTUAL_EXPOSURE, -1 if missing
    int64_t hw_ts_us = -1;         // metadata FRAME_TIMESTAMP (camera clock, us), -1 if missing
    int64_t sensor_ts_us = -1;     // metadata SENSOR_TIMESTAMP (camera clock, us), -1 if missing
    int64_t frame_counter = -1;    // metadata FRAME_COUNTER, -1 if missing
    int64_t arrival_mono_ns = 0;   // CLOCK_MONOTONIC at callback entry
    int64_t arrival_real_ns = 0;   // CLOCK_REALTIME at callback entry (latency vs global timestamp)
    uint32_t data_bytes = 0;       // bytes copied into the slot
};

struct alignas(64) DepthSlot {
    FrameMeta meta;
    alignas(64) uint16_t depth[kWidth * kHeight];
};

}  // namespace cap
