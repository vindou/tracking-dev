// Background map (T6): occupancy grid of the empty scene, 20 mm cells over the work volume. A point in an occupied
// cell is removed by the scene filter.
// File: "TRKBGMAP", u32 version 1, u32 reserved, f64 origin[3], f64 cell, u32 nx, ny, nz, u32 reserved, then nx*ny*nz
// bytes (0/1), index (ix * ny + iy) * nz + iz.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "calib.hpp"

namespace trk {

class BackgroundMap {
public:
    void init(const Box& volume, double cell);
    bool empty() const { return occ_.empty(); }
    void mark(const Vec3& p);                       // marks the cell of p (if inside)
    bool occupied(const Vec3& p) const {
        if (occ_.empty()) return false;
        const long ix = long((p.x() - origin_.x()) / cell_), iy = long((p.y() - origin_.y()) / cell_),
                   iz = long((p.z() - origin_.z()) / cell_);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ || iz >= nz_) return false;
        return occ_[std::size_t((ix * ny_ + iy) * nz_ + iz)] != 0;
    }
    std::size_t occupied_cells() const;
    bool save(const std::string& path, std::string& err) const;
    bool load(const std::string& path, std::string& err);
    double cell() const { return cell_; }
private:
    Vec3 origin_ = Vec3::Zero();
    double cell_ = 0.02;
    long nx_ = 0, ny_ = 0, nz_ = 0;
    std::vector<uint8_t> occ_;
};

}  // namespace trk
