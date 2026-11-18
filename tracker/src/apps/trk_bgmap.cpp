// T6, background map: records pairs of the empty scene (object removed) and writes an occupancy grid with 20 mm cells
// over the work volume. A cell is occupied if points fell into it in at least --min-hits pairs; occupied cells are
// grown by --dilate cells. Set "background_map" in the configuration to use it.
//   trk_bgmap --out FILE [--config config/tracker.json] [--source shm:/d435_pairs | FILE] [--pairs 100] [--cell 0.02]
//             [--min-hits 3] [--dilate 0]
#include <cstdio>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/pairinput.hpp"
#include "core/tracker_core.hpp"

using namespace trk;

int main(int argc, char** argv) {
    std::string out, cfgp = "config/tracker.json", source = "shm:/d435_pairs";
    int pairs = 100, min_hits = 3, dilate = 0;
    double cell = 0.02;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--out") out = argv[i + 1];
        else if (a == "--config") cfgp = argv[i + 1];
        else if (a == "--source") source = argv[i + 1];
        else if (a == "--pairs") pairs = std::atoi(argv[i + 1]);
        else if (a == "--cell") cell = std::atof(argv[i + 1]);
        else if (a == "--min-hits") min_hits = std::atoi(argv[i + 1]);
        else if (a == "--dilate") dilate = std::atoi(argv[i + 1]);
    }
    if (out.empty()) { std::fprintf(stderr, "usage: trk_bgmap --out FILE [--config F] [--source shm:NAME|FILE] [--pairs N]\n"); return 2; }
    TrackerConfig cfg;
    std::string err;
    if (!load_tracker_config(cfgp, cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    cfg.background_map.clear();
    TrackerCore core;
    if (!core.init(cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    PairInput in;
    if (!in.open(source, 20, err)) { std::fprintf(stderr, "cannot open %s: %s\n", source.c_str(), err.c_str()); return 3; }
    if (!core.set_cameras(in.cameras(), err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    const Box& v = cfg.work_volume;
    const long nx = long((v.max.x() - v.min.x()) / cell) + 1, ny = long((v.max.y() - v.min.y()) / cell) + 1,
               nz = long((v.max.z() - v.min.z()) / cell) + 1;
    std::vector<uint16_t> hits(std::size_t(nx * ny * nz), 0), seen_pair(hits.size(), 0);
    std::vector<uint16_t> a(in.pixels()), b(in.pixels());
    cap::PairRecord rec;
    int k = 0;
    long points = 0;
    while (k < pairs && in.next(rec, a.data(), b.data())) {
        k++;
        for (int c = 0; c < 2; c++) {
            const Camera& cam = core.camera(c);
            const uint16_t* img = c ? b.data() : a.data();
            for (int y = 0; y < cam.height; y += 2)
                for (int x = 0; x < cam.width; x += 2) {
                    const uint16_t d = img[std::size_t(y) * cam.width + x];
                    if (!d) continue;
                    const double z = d * cam.depth_scale;
                    if (z < cfg.min_depth || z > cfg.max_depth) continue;
                    const Vec3 p = xform(cam.T_rig_cam, cam.backproject(x, y, z));
                    if (!v.contains(p)) continue;
                    const long ix = long((p.x() - v.min.x()) / cell), iy = long((p.y() - v.min.y()) / cell),
                               iz = long((p.z() - v.min.z()) / cell);
                    const std::size_t idx = std::size_t((ix * ny + iy) * nz + iz);
                    if (seen_pair[idx] != k) { seen_pair[idx] = uint16_t(k); hits[idx]++; }
                    points++;
                }
        }
    }
    BackgroundMap map;
    map.init(v, cell);
    long occupied = 0;
    for (long ix = 0; ix < nx; ix++)
        for (long iy = 0; iy < ny; iy++)
            for (long iz = 0; iz < nz; iz++) {
                if (hits[std::size_t((ix * ny + iy) * nz + iz)] < min_hits) continue;
                for (long dx = -dilate; dx <= dilate; dx++)
                    for (long dy = -dilate; dy <= dilate; dy++)
                        for (long dz = -dilate; dz <= dilate; dz++)
                            map.mark(v.min + Vec3((ix + dx + 0.5) * cell, (iy + dy + 0.5) * cell, (iz + dz + 0.5) * cell));
                occupied++;
            }
    if (!map.save(out, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    std::printf("%s: %d pairs, %ld points in the work volume, %ld cells with >= %d hits, %zu occupied cells after dilation "
                "(%ld x %ld x %ld cells of %.0f mm)\n", out.c_str(), k, points, occupied, min_hits, map.occupied_cells(), nx, ny,
                nz, cell * 1000);
    return k == pairs ? 0 : 1;
}
