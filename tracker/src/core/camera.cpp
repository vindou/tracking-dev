#include "camera.hpp"

namespace trk {

// Nominal transforms from the corrected rig assembly (assets/test_rig_main_assem.step of 2026-11-17 22:38): the task's
// values with the cameras 52.7 mm farther out in y (see DEVIATIONS.md).
Mat4 nominal_T_rig_cam(int index) {
    Mat4 T = Mat4::Identity();
    if (index == 1) {
        T << -1, 0, 0, 17.5, 0, -0.906307787, 0.422618262, -240.664, 0, 0.422618262, 0.906307787, 56.1495, 0, 0, 0, 1;
    } else {
        T << 1, 0, 0, -17.5, 0, 0.906307787, -0.422618262, 240.664, 0, 0.422618262, 0.906307787, 56.1495, 0, 0, 0, 1;
    }
    T.block<3, 1>(0, 3) /= 1000.0;   // mm -> m
    return T;
}

}  // namespace trk
