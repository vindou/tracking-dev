#include "bgmap.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace trk {

void BackgroundMap::init(const Box& v, double cell) {
    origin_ = v.min;
    cell_ = cell;
    nx_ = long(std::ceil((v.max.x() - v.min.x()) / cell)) + 1;
    ny_ = long(std::ceil((v.max.y() - v.min.y()) / cell)) + 1;
    nz_ = long(std::ceil((v.max.z() - v.min.z()) / cell)) + 1;
    occ_.assign(std::size_t(nx_ * ny_ * nz_), 0);
}

void BackgroundMap::mark(const Vec3& p) {
    const long ix = long(std::floor((p.x() - origin_.x()) / cell_)), iy = long(std::floor((p.y() - origin_.y()) / cell_)),
               iz = long(std::floor((p.z() - origin_.z()) / cell_));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ || iz >= nz_) return;
    occ_[std::size_t((ix * ny_ + iy) * nz_ + iz)] = 1;
}

std::size_t BackgroundMap::occupied_cells() const {
    std::size_t n = 0;
    for (uint8_t o : occ_) n += o;
    return n;
}

bool BackgroundMap::save(const std::string& path, std::string& err) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot write " + path; return false; }
    const uint32_t ver = 1, res = 0, n[3] = {uint32_t(nx_), uint32_t(ny_), uint32_t(nz_)};
    const double o[3] = {origin_.x(), origin_.y(), origin_.z()};
    bool ok = std::fwrite("TRKBGMAP", 1, 8, f) == 8 && std::fwrite(&ver, 4, 1, f) == 1 && std::fwrite(&res, 4, 1, f) == 1 &&
              std::fwrite(o, 8, 3, f) == 3 && std::fwrite(&cell_, 8, 1, f) == 1 && std::fwrite(n, 4, 3, f) == 3 &&
              std::fwrite(&res, 4, 1, f) == 1 && std::fwrite(occ_.data(), 1, occ_.size(), f) == occ_.size();
    ok = std::fclose(f) == 0 && ok;
    if (!ok) err = "write error " + path;
    return ok;
}

bool BackgroundMap::load(const std::string& path, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot read " + path; return false; }
    char magic[8];
    uint32_t ver = 0, res = 0, n[3] = {0, 0, 0};
    double o[3];
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "TRKBGMAP", 8) == 0 && std::fread(&ver, 4, 1, f) == 1 &&
              ver == 1 && std::fread(&res, 4, 1, f) == 1 && std::fread(o, 8, 3, f) == 3 && std::fread(&cell_, 8, 1, f) == 1 &&
              std::fread(n, 4, 3, f) == 3 && std::fread(&res, 4, 1, f) == 1 && n[0] * uint64_t(n[1]) * n[2] < 100000000ULL;
    if (ok) {
        origin_ = Vec3(o[0], o[1], o[2]);
        nx_ = n[0]; ny_ = n[1]; nz_ = n[2];
        occ_.assign(std::size_t(nx_ * ny_ * nz_), 0);
        ok = std::fread(occ_.data(), 1, occ_.size(), f) == occ_.size();
    }
    std::fclose(f);
    if (!ok) { err = path + ": not a background map (version 1)"; occ_.clear(); }
    return ok;
}

}  // namespace trk
