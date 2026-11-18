// One D435i: startup checks and options (R1), the SDK callback (R2), and restart (R6). librealsense-specific.
#pragma once

#include <atomic>
#include <string>

#include <librealsense2/rs.hpp>

#include "producer.hpp"
#include "shm_layout.hpp"

namespace cap {

class Camera {
public:
    Camera(int cam, std::string serial, FrameProducer& prod, int callback_cpu);

    // Finds the devices of both serial numbers in one query, before any of them is opened. (With the RSUSB backend,
    // creating a device object for a camera that is already open fails with "USB busy"; such entries are skipped.)
    static void find_devices(rs2::context& ctx, const std::string serials[2], rs2::device out[2]);

    // Checks USB 3 and firmware, sets and reads back the options, selects the depth profile (848x480 Z16 90 fps), reads
    // depth scale and intrinsics, and opens the sensor. Returns false with err set.
    bool open(const rs2::device& dev, std::string& err);
    bool start(std::string& err);
    void stop_close();                            // never throws
    // stop + close + open + start of this camera only; starts a new epoch first. Returns false with err set.
    // If the old device handle does not open (device re-enumerated) or after hardware_reset(), the camera is found
    // again by serial number and all startup checks of open() run again.
    bool restart(rs2::context& ctx, std::string& err);
    // Test (gap 2): real hardware reset of the device (rs2::device::hardware_reset) while it streams. The device
    // disconnects and re-enumerates; the next restart() finds it again by serial. Never throws.
    bool hardware_reset(std::string& err);

    CameraInfo info() const { return info_; }
    const std::string& serial() const { return serial_; }
    const std::string& firmware() const { return fw_; }
    const std::string& usb() const { return usb_; }
    std::string option_report() const { return opt_report_; }
    uint64_t full_opens() const { return full_opens_; }   // restarts that found the device again by serial
    uint64_t callback_errors() const { return cb_errors_.load(std::memory_order_relaxed); }

    static bool firmware_at_least(const std::string& fw, const std::string& min);

private:
    void on_frame(const rs2::frame& f);
    bool set_and_check(rs2_option opt, float value, std::string& err);

    const int cam_;
    const std::string serial_;
    FrameProducer& prod_;
    const int cb_cpu_;
    rs2::device dev_;
    rs2::sensor sensor_;
    rs2::stream_profile profile_;
    bool opened_ = false, started_ = false;
    bool need_find_ = false;   // the device handle is stale (hardware reset): restart() finds the device again
    uint64_t full_opens_ = 0;
    CameraInfo info_{};
    std::string fw_, usb_, opt_report_;
    std::atomic<uint64_t> cb_errors_{0};
};

}  // namespace cap
