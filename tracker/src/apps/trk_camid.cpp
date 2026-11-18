// T4, camera identification: the operator covers the camera on the -Y side of the rig (D435i_1 in the nominal
// transforms); the camera whose valid depth drops is that camera. The serial numbers are written to the configuration.
//
//   trk_camid [--config config/tracker.json] [--source shm:/d435_pairs | FILE] [--write] [--timed BASE_S]
//
// Steps: 1. 2 s baseline (fraction of pixels with depth, per camera). 2. "Cover the camera on the -Y side now" (and
// wait for Enter, or with --timed, without a prompt). 3. Watch until one camera keeps less than 20 % of its baseline
// for 1 s (the other at least 70 %), at most 30 s. 4. Report; with --write, set cameras.D435i_1.serial and
// cameras.D435i_2.serial in the configuration file (the rest of the file is kept).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/json.hpp"
#include "core/pairinput.hpp"

using namespace trk;

namespace {
double valid_fraction(const uint16_t* img, std::size_t n) {
    std::size_t v = 0;
    for (std::size_t i = 0; i < n; i += 4) v += img[i] != 0;
    return double(v) / double((n + 3) / 4);
}

// Replaces the value of "serial" inside the object "name" of the configuration text (keeps the rest unchanged).
bool set_serial(std::string& text, const std::string& name, const std::string& serial) {
    const auto p = text.find("\"" + name + "\"");
    if (p == std::string::npos) return false;
    const auto s = text.find("\"serial\"", p);
    if (s == std::string::npos) return false;
    const auto q1 = text.find('"', text.find(':', s) + 1);
    const auto q2 = text.find('"', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) return false;
    text.replace(q1 + 1, q2 - q1 - 1, serial);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    std::string cfgp = "config/tracker.json", source = "shm:/d435_pairs";
    bool write = false;
    double timed = -1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) cfgp = argv[++i];
        else if (a == "--source" && i + 1 < argc) source = argv[++i];
        else if (a == "--write") write = true;
        else if (a == "--timed" && i + 1 < argc) timed = std::atof(argv[++i]);
        else { std::fprintf(stderr, "usage: trk_camid [--config F] [--source shm:NAME|FILE] [--write] [--timed S]\n"); return 2; }
    }
    PairInput in;
    std::string err;
    if (!in.open(source, 20, err)) { std::fprintf(stderr, "cannot open %s: %s\n", source.c_str(), err.c_str()); return 3; }
    const std::size_t n = in.pixels();
    std::vector<uint16_t> a(n), b(n);
    cap::PairRecord rec;
    const std::string serial[2] = {in.cameras()[0].serial, in.cameras()[1].serial};
    // 1. baseline
    double base[2] = {0, 0};
    int nb = 0;
    const int base_pairs = timed > 0 ? int(timed * 90) : 180;
    while (nb < base_pairs && in.next(rec, a.data(), b.data())) {
        base[0] += valid_fraction(a.data(), n);
        base[1] += valid_fraction(b.data(), n);
        nb++;
    }
    if (nb == 0) { std::fprintf(stderr, "no pairs\n"); return 3; }
    base[0] /= nb;
    base[1] /= nb;
    std::printf("baseline over %d pairs: camera A %s %.1f %% valid, camera B %s %.1f %% valid\n", nb, serial[0].c_str(),
                base[0] * 100, serial[1].c_str(), base[1] * 100);
    // 2. instruction
    std::printf(">>> Cover the camera on the -Y side of the rig (D435i_1 side) completely now%s\n",
                timed > 0 ? "" : ", then press Enter");
    std::fflush(stdout);
    if (timed <= 0) { std::string line; std::getline(std::cin, line); }
    // 3. watch for 30 s
    int run = 0, which = -1, seen = 0;
    double last[2] = {0, 0};
    while (seen < 30 * 90 && in.next(rec, a.data(), b.data())) {
        seen++;
        last[0] = valid_fraction(a.data(), n);
        last[1] = valid_fraction(b.data(), n);
        int cand = -1;
        for (int c = 0; c < 2; c++)
            if (last[c] < 0.2 * base[c] && last[1 - c] >= 0.7 * base[1 - c]) cand = c;
        if (cand >= 0 && (run == 0 || cand == which)) { which = cand; run++; }
        else { run = 0; which = -1; }
        if (run >= 90) break;
    }
    if (run < 90) {
        std::printf("RESULT: no camera lost its depth data for 1 s (last valid fractions A %.1f %%, B %.1f %%): assignment unchanged\n",
                    last[0] * 100, last[1] * 100);
        return 1;
    }
    const std::string s1 = serial[which], s2 = serial[1 - which];
    std::printf("RESULT: camera %s (serial %s) lost its depth data (%.1f %% -> %.1f %%): it is on the -Y side.\n"
                "        D435i_1 = %s (camera %s), D435i_2 = %s (camera %s)\n",
                which ? "B" : "A", s1.c_str(), base[which] * 100, last[which] * 100, s1.c_str(), which ? "B" : "A",
                s2.c_str(), which ? "A" : "B");
    if (write) {
        std::ifstream f(cfgp);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string text = ss.str();
        if (!set_serial(text, "D435i_1", s1) || !set_serial(text, "D435i_2", s2)) {
            std::fprintf(stderr, "cannot find cameras.D435i_1/2.serial in %s\n", cfgp.c_str());
            return 3;
        }
        std::string perr;
        Json::parse(text, perr);
        if (!perr.empty()) { std::fprintf(stderr, "internal error: %s\n", perr.c_str()); return 3; }
        std::ofstream o(cfgp);
        o << text;
        std::printf("written to %s\n", cfgp.c_str());
    }
    return 0;
}
