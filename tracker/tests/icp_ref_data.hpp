// Reader of tests/data/icp_ref.bin (written by tools/icp_reference.py with the Open3D library).
#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct IcpRefCase {
    uint32_t kernel = 0, noise = 0, max_iter = 30;
    double k = 0, max_dist = 0, rel_fit = 0, rel_rmse = 0;
    Eigen::Matrix4d init, truth, T;
    std::vector<Eigen::Vector3d> src;
    double fitness = 0, rmse = 0;
    uint32_t iters = 0, ncorr = 0;
};
struct IcpRefData {
    std::vector<Eigen::Vector3d> tp, tn;
    std::vector<IcpRefCase> cases;
};

inline bool load_icp_ref(const std::string& path, IcpRefData& d) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto rd = [&](void* p, std::size_t n) { return std::fread(p, 1, n, f) == n; };
    auto m4 = [&](Eigen::Matrix4d& m) {
        double v[16];
        if (!rd(v, sizeof v)) return false;
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) m(r, c) = v[r * 4 + c];
        return true;
    };
    char magic[8];
    uint32_t ver = 0, nc = 0, nt = 0;
    bool ok = rd(magic, 8) && std::string(magic, 8) == "ICPREF01" && rd(&ver, 4) && ver == 1 && rd(&nc, 4) && rd(&nt, 4);
    for (uint32_t i = 0; ok && i < nt; i++) {
        double v[6];
        ok = rd(v, sizeof v);
        d.tp.emplace_back(v[0], v[1], v[2]);
        d.tn.emplace_back(v[3], v[4], v[5]);
    }
    for (uint32_t c = 0; ok && c < nc; c++) {
        IcpRefCase k;
        uint32_t res = 0, n = 0;
        ok = rd(&k.kernel, 4) && rd(&k.noise, 4) && rd(&k.k, 8) && rd(&k.max_dist, 8) && rd(&k.max_iter, 4) &&
             rd(&res, 4) && rd(&k.rel_fit, 8) && rd(&k.rel_rmse, 8) && m4(k.init) && m4(k.truth) && rd(&n, 4) && rd(&res, 4);
        for (uint32_t i = 0; ok && i < n; i++) {
            double v[3];
            ok = rd(v, sizeof v);
            k.src.emplace_back(v[0], v[1], v[2]);
        }
        ok = ok && m4(k.T) && rd(&k.fitness, 8) && rd(&k.rmse, 8) && rd(&k.iters, 4) && rd(&k.ncorr, 4);
        d.cases.push_back(std::move(k));
    }
    std::fclose(f);
    return ok;
}
