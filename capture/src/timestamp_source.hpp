// Source of the time used for pairing (R4). The matcher asks it for every frame.
// First implementation: the SDK global timestamp. A later implementation can use an own clock model (for example a
// regression of the hardware timestamp against host time) without changes to the matcher.
#pragma once

#include "common.hpp"
#include "frame_types.hpp"

namespace cap {

class TimestampSource {
public:
    virtual ~TimestampSource() = default;
    // Writes the pairing time in ms and returns true if the frame can be paired with this source.
    virtual bool pairing_time_ms(int cam, const FrameMeta& m, double& t_ms) = 0;
    virtual const char* name() const = 0;
};

class SdkGlobalTimestampSource : public TimestampSource {
public:
    bool pairing_time_ms(int, const FrameMeta& m, double& t_ms) override {
        t_ms = m.ts_ms;
        return m.domain == kDomainGlobalTime;
    }
    const char* name() const override { return "sdk_global"; }
};

}  // namespace cap
