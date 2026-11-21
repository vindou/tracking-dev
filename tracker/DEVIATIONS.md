# Deviations and interpretations

Each item: what the task says, what was done, and why. **Deviation** marks a difference from the task; the other items
are interpretations of open points.

## Third-party code

1. **Open3D port.** Source: `isl-org/Open3D`, tag **v0.20.0**, commit **b6c5e196384ad71e75b6e6f9c5da22d046221f1d**
   (2026-09-16), MIT license; the copyright notice is kept in every ported file (`src/icp/`). The tag was chosen
   instead of the current main branch because the reference test runs the pip package open3d 0.20.0: port and
   reference are the same code. Ported, with the Open3D names, namespaces and structure:
   - `pipelines/registration/Registration.cpp`: `RegistrationICP`, `GetRegistrationResultAndCorrespondences`,
     `EvaluateRegistration`; `Registration.h`: `ICPConvergenceCriteria`, `RegistrationResult`;
   - `TransformationEstimation.cpp/.h`: `TransformationEstimation`, `TransformationEstimationPointToPlane`
     (`ComputeTransformation`, `ComputeRMSE`);
   - `RobustKernel.cpp/.h`: `L2Loss`, `L1Loss`, `HuberLoss`, `CauchyLoss`, `GMLoss`, `TukeyLoss` (unchanged);
   - `utility/Eigen.cpp/.h`: `ComputeJTJandJTr`, `SolveJacobianSystemAndObtainExtrinsicMatrix`, `SolveLinearSystemPSD`,
     `TransformVector6dToMatrix4d`;
   - `geometry/KDTreeFlann.cpp/.h`: `SetGeometry`, `SetRawData`, `SearchHybrid` (nanoflann `KDTreeEigenMatrixAdaptor`,
     column-major 3 x N, metric L2, leaf size 15, as in Open3D);
   - `geometry/Geometry3D.cpp` `TransformPoints` / `TransformNormals` in a minimal `geometry::PointCloud`.
   Changes in the port: TBB `parallel_reduce` -> sequential loops (same sums in index order); logging removed;
   `LogError` on a non-positive distance returns the initial transform; `SolveLinearSystemPSD` keeps the dense path
   only (the sparse path is not used by ICP); `RegistrationResult::num_iteration_` added (Open3D does not report the
   iteration count).
2. **nanoflann v1.5.0** (commit f1e47f0dbae72c11b008fd0d5b1191d9ba1f29bb), the version Open3D v0.20.0 builds with, and
   **Eigen 3.4.0** (commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c; Open3D pins a development snapshot of Eigen). Both
   are header-only and copied to `third_party/` with their license files. Eigen is compiled with `EIGEN_MPL2_ONLY`.
3. Open3D's Python API does not return the iteration count. `tools/icp_reference.py` finds it as the smallest
   `max_iteration` whose result equals the 30-iteration result (transform difference < 1e-12).
4. Reference test cases: "20 random poses, with and without noise, L2 and Tukey" = 20 poses x {no noise, 1 mm noise}
   x {L2, Tukey (k = 5 mm)} = 80 cases; source 1000 points sampled on the mesh, start up to 10 degrees / 15 mm from the
   truth, correspondence distance 25 mm, Open3D default criteria (1e-6, 1e-6, 30). In all cases the clouds overlap fully
   (fitness 1); the outlier handling is covered by the tracker tests, not by the reference test.
5. **Optimizations** (`src/icp/RegistrationPrealloc.*`, a second function next to the unchanged port): tree built once
   for the model; preallocated work arrays and a template JTJ loop instead of `std::function` (no allocation, checked
   with a counting `operator new`); a fixed-dimension (3) nanoflann index next to the original dynamic one (nanoflann
   allocates a vector per query when the dimension is dynamic; it takes the dimension from the matrix type, so the
   fixed index uses a 3 x N fixed-row copy of the data); fixed-size 6x6 LDLT instead of the dynamic matrices of
   `SolveLinearSystemPSD`; the distance sequence; the last iteration's JTJ / JTr returned. With one distance it gives
   the reference result (max matrix difference 5.8e-16, same iteration counts; TEST_REPORT phase 3).
6. **Deviation - Tukey kernel scaled per distance stage.** A fixed k = 5 mm gives weight 0 to every residual above
   5 mm, so ICP cannot move from a start a few centimetres away (search found no correct pose; diagnostic in
   TEST_REPORT phase 5). The tracker uses k = max(5 mm, 0.5 x stage distance): 25 / 10 / 5 mm for 50 / 20 / 10 mm.
   (`icp.tukey_k_m`, `icp.tukey_k_ratio`.) The ported kernels are unchanged; the reference test uses fixed k = 5 mm.
7. The tracker's convergence criteria are 1e-5 (10 um), not Open3D's default 1e-6, to save iterations in the 5 ms
   cycle. The reference test uses the default (its data file holds the criteria).

## Model, time, records

8. **T1 sampling.** trimesh's even sampler needs scipy, which is not a permitted package. `make_model.py` takes 30000
   area-weighted random samples and selects 3000 by farthest-point selection (numpy). Normals are the face normals.
   Checks: 74 triangles, closed, 102.5 x 92.5 x 67.84 mm, outward normals (`models/test_object.model.json`).
9. **Deviation - time conversion.** "Converted with the arrival times in the pair record": the arrival times are
   `CLOCK_MONOTONIC` at the callback, which comes about 9.6 ms after the capture, so they alone would need a latency
   model. The pair record also carries `publish_mono_ns` and `publish_real_ns`, read at the same instant by the capture
   program; their difference converts the SDK global timestamp (realtime base) exactly:
   `t_mono = global_ts * 1e6 - (publish_real - publish_mono)`. The arrival times are kept in the record and give the
   delivery latency as a check.
10. The pose record is 640 bytes (DESIGN version 1 said 512; the 6x6 covariance did not fit). It has a version field
    and a free-space violation field added for item 14.
11. The covariance of the ICP pose is `rmse^2 * JTJ^-1` (a common approximation; correlated depth noise makes it
    optimistic), transformed to the output pose (DESIGN section 5). The Kalman filter uses it plus a floor
    (0.3 mm sigma).
12. No third-party JSON library is permitted: `src/common/json.*` is a small reader/writer.

## Tracker

13. **Work volume.** The task gives the minimum camera distance (0.2 m) but no maximum; 0.8 m was chosen (the shared
    region at larger distances has little object detail at 431 px focal length). `trk_workvolume` samples a 5 mm grid,
    keeps the points both cameras see at 0.2-0.8 m and above the rig plate (z >= 0.07 m), and gives their bounding box
    (x +-0.765, y +-0.24, z 0.20-0.85 m). Both cameras see 45 % of that box; the filter uses the box.
14. **Deviation - additional fit-check criterion (free-space violations).** With the five listed criteria, the search
    accepted poses turned by about 180 degrees: from a partial view the flipped object explains the visible points
    equally well. The added check projects model points that face a camera at the candidate pose and counts those
    whose pixel shows a surface farther away (the camera sees through where the object would be). Reject if more than
    15 % (`fit_check.max_free_space_violation`) or fewer than 50 points could be checked. Reject bit 64.
15. **Fit-check defaults, selected from the synthetic tests** (TEST_REPORT phase 5): smallest normalized eigenvalue
    0.01 (correct poses: min 0.0168; one flat face only: max 0.0059); RMSE 4 mm (synthetic correct poses: max
    1.66 mm with 1 mm noise; real D435 data at 0.4-0.7 m showed 2.3-2.6 mm, so the limit has margin); inlier ratio 0.6;
    at least 150 points; jump 30 mm / 20 degrees (0.2 m/s moves 2.2 mm per cycle); free-space violations 15 %.
16. **Search.** Start positions: object-sized clusters of the cloud (connected 4 cm cells), up to 4, instead of only the
    cloud centre (the real scene contains other structures that pull the centre away). 72 orientations spread evenly
    (farthest-point selection on random rotations), coarse ICP (50 / 20 mm, 10 iterations) on at most 250 points near
    the start position, then the 4 best (fitness minus violations) refined with the full sequence and the fit check;
    the best one that passes is taken at the end of the round. An "early accept" of a clearly good coarse candidate is
    implemented but off: it accepted 180-degree flips (TEST_REPORT phase 5).
17. "Time limit of 8 ms for each cycle": the search stops before a step that would end after 8 ms (step times
    estimated from earlier steps) but runs at least one step per cycle. The whole cycle in search (cloud of up to
    90000 samples + search) can exceed the 11.1 ms pair period; then newer pairs are taken and older ones counted as
    skipped.
18. Lost state: the crop box margin grows by (lost_box_factor - 1) x half the largest object extent; the velocity and
    angular rate are set to zero (prediction holds the last pose). After 3 rejected cycles track -> lost, after 45
    (0.5 s) lost -> search.
19. Orientation filter: alpha-beta on SO(3) (alpha 0.6, beta 0.1); position: constant-velocity Kalman filter with
    white-noise acceleration 4 m^2/s^3. Values chosen with the moving-object test.
20. The tracker reads the pair buffer itself (rows and columns of the crop region, then the slot counter), using the
    capture library only for attach, liveness and reattach. No file in `capture/` was changed.
21. Status line every 10 s from a separate thread that reads atomics (it sleeps 50 ms between checks; it is not the
    loop thread).

## Tools

22. T3: `trk_record` keeps the pairs in memory and writes the file at the end (the eMMC cannot take 145 MB/s; 300 real
    pairs = 389 MB coded). The file codes zero runs; real depth images compress little.
23. T3 replay writes with the capture program's own writer (`cap::ShmSink`) and shifts every time field by the same
    offsets (monotonic and realtime), so the clock relations of the original pairs are kept.
24. T4: "the camera on the -Y side" = the camera that the nominal transforms call D435i_1 (y = -188 mm). The manual
    part was not run (no one can cover a camera); the test replays synthetic files in which camera A or B has no depth
    from frame 250 on (`trk_synth --blank`), and the configuration starts with the serials swapped.
25. T5 reports the single-camera poses, their difference, the correction of camera B, and - above 20 mm / 5 degrees -
    the difference under other assumptions (camera frame rotated 180 degrees about x, y or z; names swapped;
    translations in cm or inches). A name swap rotates the whole scene about the rig Z axis, so T5 cannot detect it
    (T4 does).
26. T6: a cell is occupied if points fell into it in at least 3 of the 100 pairs; no dilation by default. Tested with
    a synthetic background plane at z = 0.55 m (behind the object as the upward-looking cameras see it).

## Tests

27. Degenerate view ("only one flat face visible"): the measured scene is rendered from the largest flat face of the
    object alone (4 triangles, 59.7 cm2), turned toward the cameras.
28. Static accuracy: the tracker starts in track state 5 mm / 3 degrees from the truth and runs 5 cycles; the
    unfiltered ICP pose of the 5th cycle is compared with the truth.
29. Acquisition time is counted in pairs (one cycle per pair, 11.13 ms each), with the real computation (the 8 ms
    slice limit applies); the right pose = within 5 mm and 5 degrees.
30. Replay test: both runs start in track state at the same pose (`--initial-pose`, `--force-track`), because the
    time-sliced search acquires at a pair that depends on the computing time; the comparison from search is reported
    too.
31. "Viewer stopped for 10 s": the viewer process is stopped with SIGSTOP while connected (the TCP connection stays
    open, the bridge queue fills, drops are counted), then continued.
32. Pose messages received by the viewer are counted against the poses the tracker published while the viewer was
    connected (first to last pose number received).

## Changes after the first real-scene runs (2026-11-17, with the user)

33. **Deviation - camera transforms.** The nominal transforms of the task came from a rig assembly that the user found
    to be assembled wrongly. The corrected assembly (`assets/test_rig_main_assem.step`) moves both camera bodies 52.7 mm
    outward in y and does not rotate them (matching the old and new camera bodies: residual 0.14 mm). The tracker uses
    y = -240.664 / +240.664 mm (was -187.964 / +187.964); `assets/d435i_depth_transforms.json` was updated to the same
    values. Evidence on the real scene (object held in view of both cameras): median distance of camera B's points to
    camera A's 34.0 mm with the old transforms, 3.3 mm with the new ones (TEST_REPORT phase 7). A reflection of the
    x offsets of the depth origins was also checked (the user saw the frustum apex on the other side); it made the
    agreement worse (17.4 mm) and was not applied.
34. The work volume was recomputed for the new camera positions: x +-0.765, y +-0.29, z 0.235-0.825 m.
35. The first real-scene runs (search never accepting, T5 failing) had no object in view: the user was away. Those
    results are superseded; the later real-scene runs have the object held by hand from above (the user cannot leave it
    on a support), so the "static object" values include the motion of a hand.
36. **Deviation - cloud buffer content and rate (user request for the demo).** The cloud record holds the whole filtered
    cloud (crop box in track and lost, work volume in search), thinned to at most 8192 points, with a flag per point:
    1 inlier / 0 outlier of the last ICP iteration, 2 not given to the solver. The task asks for "the points that the
    solver used"; those are the points with flag 0 or 1. `"cloud_content": "solver"` in the configuration restores
    the task's content. The cloud rate is 30 Hz instead of 15 Hz (the viewer's display rate).
37. Search start positions (found necessary with the object held by hand, where object and arm form one cluster):
    candidates are object-sized clusters and density peaks (4 cm cells with the most points in their 12 cm
    neighbourhood); they are ranked by min(points of camera A, points of camera B) within reach, because the object is in
    view of both cameras and clutter mostly of one; the search decides after each start position (refines its best
    candidates and accepts a passing one) instead of after all of them.
38. Viewer (user requests for the demo video): rig mesh gray; a show / hide switch for the camera frames and frustums,
    each frustum drawn in two sections (red 0-0.3 m "dead zone", green 0.3-0.8 m); a fixed-size phase panel (ACQUIRING
    with progress, TRACKING updating / predicting with the reason, LOST); position and orientation text; object mesh
    solid when the fit is accepted, transparent when predicted. The 0.3 m dead zone is drawn only; the tracker keeps
    the task's 0.2 m minimum camera distance.
39. The input files of the task were moved to `assets/` (`test_object.step`, `test_rig_main_assem.step`,
    `d435i_depth_transforms.json`) at the user's request.
40. T5 on the real scene was skipped at the user's request (2026-11-17). The tracker uses the uncalibrated transforms
    of the corrected CAD assembly. The T5 tool is tested with synthetic data (camera B correction recovered within
    1.29 mm / 0.18 degrees on the Jetson, `logs/jetson/t5_synth_correction`; a 180-degree axis convention found as the cause); the frame-convention check on the real
    scene was done with the cameras' agreement on the held object instead (item 33).

## Changes during the final test runs (2026-11-18)

41. **Bridge send queue without allocation after start.** The first 30-minute run failed the memory criterion: the
    bridge's resident memory rose once by 628 kB (+26.7 %) when its queue filled for the first time, because each queued
    message was a separate heap allocation and the C library kept the freed memory. The queue
    (`src/core/send_queue.hpp`, moved out of `trk_bridge.cpp` for the unit tests) now holds one preallocated, zero-filled
    slot pool per message type (each large enough for the whole byte limit) and a fixed ring of entries; the discard
    rule is unchanged (oldest cloud message first, then the oldest message; both counted). New unit tests:
    `send_queue_*` (order, content, discard rule, 200000 messages to a stalled client with 0 allocations).
42. **Tracker work arrays written at start.** In the same run the tracker's resident memory rose once by 3.2 MB
    (+25.4 %) at the first change from search to track: `std::vector<Eigen::Vector3d>::resize` does not write the
    elements (Eigen does not initialize them), so the pages of the point arrays became resident only when track mode
    first used more of them, i.e. page faults inside the loop. The arrays (and the ICP workspace) are now filled with
    zeros at start ("all memory allocated at start" now also means resident at start). `core_process_does_not_allocate`
    and the ICP allocation test still report 0 allocations.
43. `trk_tracker --final-json` writes `null` instead of `nan` for timing quantiles of a run without cycles (`nan` is not
    valid JSON; it stopped the evaluation of one link run).
44. **Test harness faults (no effect on the system under test).** The first two attempts of the link and viewer tests
    did not run the system: (a) `tegrastats`, started in the background by the monitor helper, kept the SSH session
    open and the script waited for it; (b) `~` in the Jetson log path was expanded on the Mac; (c) `cd DIR && nohup
    CMD &` put the whole list in the background, so each SSH call waited until its program ended and capture, tracker
    and bridge ran one after the other; the `$!` in the pid files was escaped once too often; (d) the next capture
    program started while the cameras were still re-enumerating after the previous one ("camera B not found"); the
    harness now waits until no capture program runs, then 5 s. These attempts are not counted as test results; their
    logs were overwritten by the repeated runs.
45. `tools/sync_to_jetson.sh` deleted the real-scene background map `config/bgmap_real.bin` (made on the Jetson by the
    T6 real-scene run) because `rsync --delete` mirrors `tracker/` from the Mac. The script now keeps `config/*.bin`
    and `config/calib*.json` on the Jetson. The map was not used by any live test (`background_map` is empty in the
    configuration).
46. **Finding: acquisition of static scene structure without a background map.** In the later live runs without the
    object, the search accepted a pose on static structure about 0.24 m wide at (-0.17, 0.22, 0.42) m in the rig frame
    (present in the scene recordings of 2026-11-17 and 2026-11-18; whether it is new or was moved closer could not be
    determined without the user). The fits are marginal (median RMSE 3.5 mm against the 4 mm limit, smallest eigenvalue
    0.015 against 0.01) and jitter (position SD 2.7 mm, orientation 2 degrees), but they pass every fit-check
    criterion. The intended defence is the background map (README demo step 2); without it, these fit-check limits do
    not separate this structure from the object. The limits were not changed: tighter limits would first have to be
    checked against fits on the real object, and the user was not available to hold it.

47. **Transparent huge pages turned off for the tracker and the bridge (long run, attempt 2).** Attempt 2 of the
    30-minute run (with D41 and D42) failed the memory criterion again: tracker 15,252 -> 16,020 kB (+5.04 %), bridge
    6,488 -> 8,388 kB (+29.3 %). Both rose once, between 150 and 180 s, and then stayed flat for 27 minutes (no
    leak pattern). The Jetson runs transparent huge pages in "always" mode with `khugepaged/max_ptes_none` = 511, so
    khugepaged replaces any 2 MB range of anonymous memory that has one resident page with a full huge page; it passes
    over all processes about every 9.5 min (179 full scans in 28 h of uptime), so each process steps by up to 2 MB at
    a random time in the run. A 5.5-minute probe without a viewer and a 12-minute bridge-only probe showed no step and
    `AnonHugePages` 0 (`logs/jetson/thp_probe/`; no evidence either way; the collapse counter of the system rose by 1 during the second probe).
    The programs now call `prctl(PR_SET_THP_DISABLE)` at start (`src/common/no_thp.hpp`, Linux only, no privileges;
    `/proc/<pid>/status` shows `THP_enabled: 0`); this also removes the stall while khugepaged copies a range. The
    system setting was not changed (no `sudo`). The long test now also writes `mem_detail.csv` (resident memory by kind
    and `AnonHugePages`, every 30 s) as a diagnostic; the criterion and its evaluation are unchanged. Attempt 3 (the last
    allowed) passed: tracker and bridge +0.0 % over 30 min, `AnonHugePages` 0 (`logs/long_30min/`).

## Flow module (2026-11-20, user request)

48. **Addition - pose hypotheses from outside.** Not in the task: `TrackerCore::set_hints` / `try_hints` (DESIGN
    section 13), the tool `trk_hint_eval`, and the unit test `synth_pose_hints` (a hypothesis 20 mm / 12 degrees off is
    accepted in the first cycle in 20 of 20 poses; a hypothesis 0.29 m away is never accepted and the search that
    follows finds the pose in 20 of 20). The tracker still links only Eigen, nanoflann and the capture library; the
    network that makes the hypotheses is in `../flow` (Python, PyTorch) and is not part of the tracker programs.
    Without hints the search cycle is the same code path as before (the 20 earlier tests pass unchanged).

