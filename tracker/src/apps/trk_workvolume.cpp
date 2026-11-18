// Work volume: the region that both cameras see, from the camera transforms and intrinsics, with a minimum camera
// distance (default 0.2 m) and a maximum distance; printed as JSON for the "work_volume" entry of the configuration.
//   trk_workvolume [--min-depth 0.2] [--max-depth 0.8] [--z-min 0.07] [--calib FILE]
#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/calib.hpp"
#include "core/synth.hpp"

using namespace trk;

int main(int argc, char** argv) {
    double min_d = 0.2, max_d = 0.8, z_min = 0.07;
    std::string calib;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--min-depth") min_d = std::atof(argv[i + 1]);
        else if (a == "--max-depth") max_d = std::atof(argv[i + 1]);
        else if (a == "--z-min") z_min = std::atof(argv[i + 1]);
        else if (a == "--calib") calib = argv[i + 1];
    }
    Camera cams[2];
    default_cameras(cams);
    std::string err;
    if (!calib.empty() && !apply_calibration_file(calib, cams, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    double frac = 0;
    const Box b = compute_work_volume(cams, min_d, max_d, z_min, &frac);
    std::printf("{\"work_volume\": {\"min\": [%.3f, %.3f, %.3f], \"max\": [%.3f, %.3f, %.3f], \"min_camera_distance\": %.3f, "
                "\"max_camera_distance\": %.3f, \"z_min\": %.3f}}\n", b.min.x(), b.min.y(), b.min.z(), b.max.x(), b.max.y(),
                b.max.z(), min_d, max_d, z_min);
    std::fprintf(stderr, "both cameras see %.0f %% of this box\n", frac * 100);
    return 0;
}
