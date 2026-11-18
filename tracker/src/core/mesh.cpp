#include "mesh.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace trk {

bool load_stl(const std::string& path, Mesh& m, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot read " + path; return false; }
    char hdr[80];
    uint32_t n = 0;
    bool ok = std::fread(hdr, 1, 80, f) == 80 && std::fread(&n, 4, 1, f) == 1 && n > 0 && n < 10000000;
    m = Mesh();
    for (uint32_t i = 0; ok && i < n; i++) {
        float d[12];
        uint16_t attr;
        ok = std::fread(d, 4, 12, f) == 12 && std::fread(&attr, 2, 1, f) == 1;
        m.v0.emplace_back(d[3], d[4], d[5]);
        m.v1.emplace_back(d[6], d[7], d[8]);
        m.v2.emplace_back(d[9], d[10], d[11]);
    }
    std::fclose(f);
    if (!ok) { err = path + ": not a binary STL"; return false; }
    m.bmin = m.bmax = m.v0[0];
    for (std::size_t i = 0; i < m.size(); i++)
        for (const auto* p : {&m.v0[i], &m.v1[i], &m.v2[i]}) {
            m.bmin = m.bmin.cwiseMin(*p);
            m.bmax = m.bmax.cwiseMax(*p);
        }
    return true;
}

bool load_model(const std::string& path, Model& m, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot read " + path; return false; }
    char magic[8];
    uint32_t ver = 0, n = 0;
    double b[6];
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "TRKMODEL", 8) == 0 && std::fread(&ver, 4, 1, f) == 1 &&
              ver == 1 && std::fread(&n, 4, 1, f) == 1 && n > 0 && n < 1000000 && std::fread(b, 8, 6, f) == 6;
    m = Model();
    for (uint32_t i = 0; ok && i < n; i++) {
        double d[6];
        ok = std::fread(d, 8, 6, f) == 6;
        m.points.emplace_back(d[0], d[1], d[2]);
        m.normals.emplace_back(d[3], d[4], d[5]);
    }
    std::fclose(f);
    if (!ok) { err = path + ": not a model file (version 1)"; return false; }
    m.bmin = Eigen::Vector3d(b[0], b[1], b[2]);
    m.bmax = Eigen::Vector3d(b[3], b[4], b[5]);
    for (const auto& p : m.points) m.centroid += p;
    m.centroid /= double(m.points.size());
    for (const auto& p : m.points) m.radius = std::max(m.radius, (p - m.centroid).norm());
    return true;
}

}  // namespace trk
