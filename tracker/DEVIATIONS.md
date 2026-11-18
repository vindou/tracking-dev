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
