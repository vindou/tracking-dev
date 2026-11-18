// No memory allocation in the tracker loop: heap allocations counted inside TrackerCore::process (stages 2-6) in
// track state and in search state (synthetic pairs). Not counted in sanitizer builds (they replace operator new).
#include <cstdio>

#include "alloc_counter.hpp"
#include "synth_harness.hpp"
#include "test_framework.hpp"

using namespace trkt;

TEST(core_process_does_not_allocate) {
    Camera cams[2];
    default_cameras(cams);
    std::mt19937_64 rng(77);
    SynthScenario s;
    s.T0 = random_pose_in_view(cams, object_mesh(), rng);
    s.render.noise_sigma_m = 0.001;
    TrackerConfig cfg = test_config();
    TrackerCore core;
    std::string err;
    CHECK(core.init(cfg, err));
    cap::CameraInfo ci[2];
    camera_infos(cams, ci);
    CHECK(core.set_cameras(ci, err));
    SynthGenerator gen(object_mesh(), cams, s);
    std::vector<uint16_t> a(std::size_t(cams[0].width) * cams[0].height), b(a.size());
    std::unique_ptr<CloudRecord> cloud(new CloudRecord());
    PoseRecord pose;
    uint64_t allocs_search = 0, allocs_track = 0;
    int n_search = 0, n_track = 0;
    for (int k = 0; k < 200; k++) {
        cap::PairRecord rec;
        Mat4 truth;
        bool present;
        gen.make(k, a.data(), b.data(), rec, truth, present);
        CropRect cr[2];
        core.crops(cr);
        const uint16_t* img[2] = {a.data(), b.data()};
        const TrackState st = core.state();
        alloc_count_start();
        core.process(rec, img, pose, (k % 6) == 0 ? cloud.get() : nullptr);
        const uint64_t n = alloc_count_stop();
        if (st == kStateSearch) { allocs_search += n; n_search++; } else { allocs_track += n; n_track++; }
    }
    std::printf("  %d cycles in search, %d in track; heap allocations in process(): search %llu, track %llu%s\n", n_search,
                n_track, (unsigned long long)allocs_search, (unsigned long long)allocs_track,
                alloc_counting_available() ? "" : " (not counted in sanitizer builds)");
    CHECK_GE(n_search, 1);
    CHECK_GE(n_track, 50);
    if (alloc_counting_available()) {
        CHECK_EQ(allocs_search, 0u);
        CHECK_EQ(allocs_track, 0u);
    }
}
