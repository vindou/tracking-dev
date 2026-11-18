// Writes the point cloud of one pair of a pair file (rig frame, all valid pixels with a stride, camera label) as CSV.
//   trk_clouddump --in FILE --out CSV [--index K] [--stride N] [--config config/tracker.json] [--calib FILE]
#include <cstdio>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/tracker_core.hpp"
#include "records/pairfile.hpp"

using namespace trk;

int main(int argc, char** argv) {
    std::string in, out, cfgp = "config/tracker.json", calib;
    int index = 0, stride = 2;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--in") in = argv[i + 1];
        else if (a == "--out") out = argv[i + 1];
        else if (a == "--index") index = std::atoi(argv[i + 1]);
        else if (a == "--stride") stride = std::atoi(argv[i + 1]);
        else if (a == "--config") cfgp = argv[i + 1];
        else if (a == "--calib") calib = argv[i + 1];
    }
    TrackerConfig cfg;
    std::string err;
    if (!load_tracker_config(cfgp, cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    if (!calib.empty()) {
        Camera tmp[2];
        tmp[0].name = cfg.cam_name[0];
        tmp[1].name = cfg.cam_name[1];
        if (!apply_calibration_file(calib, tmp, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        cfg.cam_T[0] = tmp[0].T_rig_cam;
        cfg.cam_T[1] = tmp[1].T_rig_cam;
    }
    TrackerCore core;
    if (!core.init(cfg, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    PairFileReader f;
    if (!f.open(in, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    cap::CameraInfo ci[2] = {f.header().cam[0], f.header().cam[1]};
    if (!core.set_cameras(ci, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    const std::size_t n = std::size_t(f.header().width) * f.header().height;
    std::vector<uint16_t> a(n), b(n);
    PairChunkHeader ch;
    for (int k = 0; k <= index; k++)
        if (!f.next(ch, a.data(), b.data(), err)) { std::fprintf(stderr, "pair %d not found\n", index); return 3; }
    FILE* o = std::fopen(out.c_str(), "w");
    if (!o) return 3;
    std::fprintf(o, "x,y,z,camera,u,v,depth_m\n");
    long cnt = 0;
    for (int c = 0; c < 2; c++) {
        const Camera& cam = core.camera(c);
        const uint16_t* img = c ? b.data() : a.data();
        for (int v = 0; v < cam.height; v += stride)
            for (int u = 0; u < cam.width; u += stride) {
                const uint16_t d = img[std::size_t(v) * cam.width + u];
                if (!d) continue;
                const double z = d * cam.depth_scale;
                const Vec3 p = xform(cam.T_rig_cam, cam.backproject(u, v, z));
                std::fprintf(o, "%.5f,%.5f,%.5f,%d,%d,%d,%.4f\n", p.x(), p.y(), p.z(), c, u, v, z);
                cnt++;
            }
    }
    std::fclose(o);
    std::printf("%s: %ld points of pair %d\n", out.c_str(), cnt, index);
    return 0;
}
