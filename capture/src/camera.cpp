#include "camera.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

#include "thread_util.hpp"

namespace cap {

namespace {
const char* kMinFirmware = "5.17.3.10";

std::vector<int> version_parts(const std::string& v) {
    std::vector<int> p;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, '.')) p.push_back(std::atoi(item.c_str()));
    return p;
}
}  // namespace

bool Camera::firmware_at_least(const std::string& fw, const std::string& min) {
    const auto a = version_parts(fw), b = version_parts(min);
    for (std::size_t i = 0; i < std::max(a.size(), b.size()); i++) {
        const int x = i < a.size() ? a[i] : 0, y = i < b.size() ? b[i] : 0;
        if (x != y) return x > y;
    }
    return true;
}

Camera::Camera(int cam, std::string serial, FrameProducer& prod, int callback_cpu)
    : cam_(cam), serial_(std::move(serial)), prod_(prod), cb_cpu_(callback_cpu) {}

bool Camera::set_and_check(rs2_option opt, float value, std::string& err) {
    if (!sensor_.supports(opt)) {
        err = std::string("option not supported: ") + rs2_option_to_string(opt);
        return false;
    }
    sensor_.set_option(opt, value);
    const float rb = sensor_.get_option(opt);
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s=%g (readback %g) ", rs2_option_to_string(opt), double(value), double(rb));
    opt_report_ += buf;
    if (std::fabs(rb - value) > 1e-3f) {
        err = std::string("option ") + rs2_option_to_string(opt) + " readback " + std::to_string(rb) + " != " +
              std::to_string(value);
        return false;
    }
    return true;
}

void Camera::find_devices(rs2::context& ctx, const std::string serials[2], rs2::device out[2]) {
    out[0] = rs2::device();
    out[1] = rs2::device();
    rs2::device_list list;
    try {
        list = ctx.query_devices();
    } catch (...) {
        return;   // device list not available now (for example during re-enumeration): nothing found
    }
    for (uint32_t i = 0; i < list.size(); i++) {
        try {
            rs2::device d = list[i];
            if (!d.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)) continue;
            const std::string s = d.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            for (int c = 0; c < 2; c++)
                if (!serials[c].empty() && s == serials[c]) out[c] = d;
        } catch (...) {
            // device in use (for example the other camera, already open) or not accessible: skip it
        }
    }
}

bool Camera::open(const rs2::device& dev, std::string& err) {
    const std::string who = std::string("camera ") + cam_name(cam_) + " (" + serial_ + "): ";
    try {
        dev_ = dev;
        if (!dev_) {
            err = who + "not found (check the USB connection and the serial number in the config file)";
            return false;
        }
        usb_ = dev_.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) ? dev_.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) : "";
        fw_ = dev_.supports(RS2_CAMERA_INFO_FIRMWARE_VERSION) ? dev_.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) : "";
        if (usb_.rfind("3.", 0) != 0) {
            err = who + "USB type descriptor is \"" + usb_ + "\", not USB 3 (\"3.x\"). Use a USB 3 cable and port. Refusing to start.";
            return false;
        }
        if (fw_.empty() || !firmware_at_least(fw_, kMinFirmware)) {
            err = who + "firmware " + (fw_.empty() ? std::string("unknown") : fw_) + " is older than " + kMinFirmware +
                  ". Refusing to start.";
            return false;
        }
        auto ds = dev_.first<rs2::depth_sensor>();
        sensor_ = ds;
        opt_report_.clear();
        if (!set_and_check(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.f, err) ||
            !set_and_check(RS2_OPTION_INTER_CAM_SYNC_MODE, 0.f, err) ||
            !set_and_check(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f, err) ||
            !set_and_check(RS2_OPTION_FRAMES_QUEUE_SIZE, 16.f, err)) {
            err = who + err;
            return false;
        }
        profile_ = rs2::stream_profile();
        for (auto&& p : sensor_.get_stream_profiles()) {
            auto vp = p.as<rs2::video_stream_profile>();
            if (vp && vp.stream_type() == RS2_STREAM_DEPTH && vp.format() == RS2_FORMAT_Z16 && vp.width() == kWidth &&
                vp.height() == kHeight && vp.fps() == kFps) {
                profile_ = p;
                break;
            }
        }
        if (!profile_) {
            err = who + "depth profile 848x480 Z16 90 fps not available";
            return false;
        }
        const rs2_intrinsics in = profile_.as<rs2::video_stream_profile>().get_intrinsics();
        std::memset(&info_, 0, sizeof info_);
        std::snprintf(info_.serial, sizeof info_.serial, "%s", serial_.c_str());
        info_.depth_scale = ds.get_depth_scale();
        info_.width = uint32_t(in.width);
        info_.height = uint32_t(in.height);
        info_.ppx = in.ppx;
        info_.ppy = in.ppy;
        info_.fx = in.fx;
        info_.fy = in.fy;
        info_.distortion_model = int32_t(in.model);
        for (int i = 0; i < 5; i++) info_.coeffs[i] = in.coeffs[i];
        sensor_.open(profile_);
        opened_ = true;
        return true;
    } catch (const rs2::error& e) {
        err = who + "librealsense error: " + e.what();
        return false;
    } catch (const std::exception& e) {
        err = who + "error: " + e.what();
        return false;
    } catch (...) {
        err = who + "unknown exception";
        return false;
    }
}

bool Camera::start(std::string& err) {
    try {
        prod_.set_stream_start(mono_ns());
        sensor_.start([this](rs2::frame f) { on_frame(f); });
        started_ = true;
        return true;
    } catch (const std::exception& e) {
        err = std::string("camera ") + cam_name(cam_) + ": start failed: " + e.what();
        return false;
    } catch (...) {
        err = std::string("camera ") + cam_name(cam_) + ": start failed: unknown exception";
        return false;
    }
}

void Camera::stop_close() {
    try {
        if (started_) sensor_.stop();
    } catch (...) {
    }
    started_ = false;
    try {
        if (opened_) sensor_.close();
    } catch (...) {
    }
    opened_ = false;
}

bool Camera::restart(rs2::context& ctx, std::string& err) {
    stop_close();                 // after this returns, the old callback does not run any more
    prod_.begin_epoch(mono_ns());
    bool fast = false;
    if (!need_find_) {
        try {
            sensor_.open(profile_);
            opened_ = true;
            fast = true;
        } catch (...) {
            // The device may have re-enumerated (cable, hardware reset): find it again below.
        }
    }
    if (!fast) {
        // Release the old handles first, then find the device by serial and redo all startup checks.
        try {
            sensor_ = rs2::sensor();
            dev_ = rs2::device();
        } catch (...) {
        }
        const std::string serials[2] = {cam_ == kCamA ? serial_ : std::string(), cam_ == kCamB ? serial_ : std::string()};
        rs2::device found[2];
        find_devices(ctx, serials, found);
        if (!found[cam_]) {
            err = std::string("camera ") + cam_name(cam_) + " (" + serial_ + "): not found (absent)";
            return false;
        }
        if (!open(found[cam_], err)) return false;
        need_find_ = false;
        full_opens_++;
    }
    return start(err);
}

bool Camera::hardware_reset(std::string& err) {
    try {
        if (!dev_) {
            err = "no device handle";
            return false;
        }
        need_find_ = true;
        dev_.hardware_reset();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
    } catch (...) {
        err = "unknown exception";
    }
    return false;
}

// SDK callback thread. No allocation, no lock, no I/O (R2). rs2 accessors used here do not allocate in our code path.
void Camera::on_frame(const rs2::frame& f) {
    const int64_t t0 = mono_ns();
    const int64_t tr = real_ns();
    if (cb_cpu_ >= 0) {
        static thread_local bool pinned = false;   // trivially-initialised thread_local: no allocation
        if (!pinned) {
            set_thread_affinity(cb_cpu_);
            pinned = true;
        }
    }
    try {
        FrameMeta m;
        m.ts_ms = f.get_timestamp();
        m.domain = int32_t(f.get_frame_timestamp_domain());
        if (f.supports_frame_metadata(RS2_FRAME_METADATA_FRAME_TIMESTAMP))
            m.hw_ts_us = f.get_frame_metadata(RS2_FRAME_METADATA_FRAME_TIMESTAMP);
        if (f.supports_frame_metadata(RS2_FRAME_METADATA_SENSOR_TIMESTAMP))
            m.sensor_ts_us = f.get_frame_metadata(RS2_FRAME_METADATA_SENSOR_TIMESTAMP);
        if (f.supports_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER))
            m.frame_counter = f.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
        if (f.supports_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE))
            m.exposure_us = int32_t(f.get_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE));
        prod_.deliver(m, f.get_data(), std::size_t(f.get_data_size()), t0, tr);
    } catch (...) {
        cb_errors_.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace cap
