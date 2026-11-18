// T2 core: depth image that a camera sees for a given object pose (ray casting against the mesh), with an optional
// background plane (rig frame z = plane_z), Gaussian depth noise and quantization to depth units.
#pragma once
#include <cstdint>
#include <random>

#include "camera.hpp"
#include "mesh.hpp"

namespace trk {

struct RenderOptions {
    double noise_sigma_m = 0.0;      // Gaussian noise on z before quantization
    bool plane = false;              // background plane z = plane_z (rig frame)
    double plane_z = 0.0;
    double max_depth_m = 4.0;        // farther: 0 (no data)
};

// Renders into out (width * height uint16 depth units; 0 = no data). T_rig_obj == nullptr: no object.
void render_depth(const Mesh& mesh, const Camera& cam, const Mat4* T_rig_obj, const RenderOptions& opt,
                  std::mt19937_64& rng, uint16_t* out);

}  // namespace trk
