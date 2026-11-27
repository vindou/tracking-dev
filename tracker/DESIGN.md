# tracker: design

Pose tracking of one known rigid object (`assets/test_object.step`) from the depth pairs of `capture/`, and a live viewer.
Version 1 of this document was written before the main code (after phase 1); section 12 lists what changed during
the implementation.

```
Jetson                                                               MacBook
d435_capture -> pair buffer (/d435_pairs, capture layout v2)
                    |
                 tracker  -> pose buffer  (/trk_pose,  256 slots) --\
                    |                                                 trk_bridge == TCP ==> viewer.py (viser)
                    +------> cloud buffer (/trk_cloud,   4 slots) --/
```

Conventions (fixed by the task): meters, seconds, radians; quaternions w, x, y, z; rig frame = assembly frame of the
camera transforms; object frame = frame of `test_object.step`; output pose = `T_rig_from_object`; time base
`CLOCK_MONOTONIC` of the Jetson; all binary records and network messages little-endian, fixed size, with a version.

## 1. Components and directories

| Path | Content |
|---|---|
| `tracker/third_party/` | Eigen 3.4.0 and nanoflann 1.5.0 headers (the only libraries besides C++17 and POSIX) |
| `tracker/src/common/` | clocks, a small JSON reader (no third-party JSON library is permitted), configuration, geometry helpers |
| `tracker/src/shmring/` | generic single-writer, multi-reader shared-memory ring (protocol of the pair buffer, layout v2) |
| `tracker/src/records/` | pose record, cloud record, network messages, record/replay file format |
| `tracker/src/icp/` | port of the Open3D legacy point-to-plane ICP (namespaces and names kept) |
| `tracker/src/core/` | model, cameras, pair reader (partial reads), cloud builder, scene filter, solver, fit check, filter, track states |
| `tracker/src/apps/` | `trk_tracker`, `trk_bridge`, tools T2 (`trk_synth`), T3 (`trk_record`, `trk_replay`), T4 (`trk_camid`), T5 (`trk_calib`), T6 (`trk_bgmap`) |
| `tracker/tests/` | unit, synthetic and acceptance tests |
| `tracker/tools/` | `make_model.py` (T1, Mac), `icp_reference.py` (Open3D reference, Mac) |
| `viewer/` | `viewer.py`, `requirements.txt`, `demo.sh` |
| `models/` | STL and model files from T1 |

The tracker links `capture_core` (built from `capture/` without changes) for the pair-buffer attach, reattach and
liveness check, and reads the slots itself so that it can copy only the rows of the crop region.

## 2. Threads

| Process / thread | Does | Must not |
|---|---|---|
| tracker: loop | waits for the pair notification (futex), takes the newest pair, runs the 6 stages, publishes the pose (every cycle) and the cloud (15 Hz) | allocate, write files, use the network, wait for a reader |
| tracker: status | every 10 s: one status line from atomic counters and stage-time histograms | touch loop data other than atomics / try-lock snapshot |
| tracker: 2 heartbeat threads | heartbeat of the pose and cloud buffers (part of `shmring`) | anything else |
| tracker: main | configuration, setup, `sigtimedwait(SIGINT, SIGTERM)`, ordered stop | - |
| bridge: reader | waits on the pose-buffer notification (and the cloud buffer's), copies new records into the send queue | wait for the network |
| bridge: network | accepts one client, sends from the queue with non-blocking `send`, `TCP_NODELAY` | block the reader |
| viewer: receiver | reads the socket, keeps the newest pose and cloud, writes each pose to CSV | - |
| viewer: display | updates the viser scene at about 30 Hz | - |

All loop memory is allocated at start: image row buffers, point arrays (maximum point count), ICP work arrays, search
state. The ICP port uses preallocated correspondence and transformed-point arrays (section 6).

## 3. Shared-memory ring (`shmring`)

Same protocol as the pair buffer (capture README "Shared memory"): a header, then `slot_count` slots; each slot
starts with a 64-bit sequence counter (odd while written, `2n+2` when it holds record n), the record at slot offset 64.
Writer: `seq = 2n+1`, release fence, copy, `seq = 2n+2` (release), `published = n+1`, `notify_word += 1`, futex wake;
never waits. Heartbeat thread every 50 ms; liveness timeout 1 s; instance id per writer start; a new writer marks an
old object of the same name `replaced` and wakes its readers before it removes the name. Reader: results `Ok`,
`Timeout`, `WriterStopped`, `WriterNotAlive`, `WriterReplaced`; `reattach()` by name; futex wait with a 200 ms
timeout only for the liveness check. Additional reader call `latest()`: copy the newest complete record.

Header (512 bytes):

| Offset | Type | Field |
|---|---|---|
| 0 | u64 | magic (`TRKRING\0` = 0x00474E49524B5254) |
| 8 | u32 | layout_version = 2 |
| 12 | u32 | header_size = 512 |
| 16 | u32 | slot_count |
| 20 | u32 | record_size |
| 24 | u64 | slot_stride (64 + record_size rounded up to 64) |
| 32 | u64 | slots_offset = 512 |
| 40 | u64 | total_size |
| 48 | char[16] | record_type ("pose", "cloud"), NUL-terminated |
| 64 | u32 | record_version |
| 68 | i32 | writer_pid (information) |
| 72 | u32 atomic | writer_state: 1 running, 2 stopped, 3 replaced |
| 76 | u32 atomic | notify_word |
| 80 | u64 atomic | records_published |
| 88 | u64 | writer_instance_id |
| 96 | u64 atomic | heartbeat_mono_ns |
| 104 | u64 atomic | heartbeat_count |
| 112 | u32 | heartbeat_interval_ms = 50 |
| 116 | u32 | liveness_timeout_ms = 1000 |
| 120 | i64 | writer_start_real_ns |
| 128 | u64 atomic | replaced_by_instance |
| 136 | | reserved to 512 |

Pose buffer `/trk_pose`: 256 slots, pose record (section 9). Cloud buffer `/trk_cloud`: 4 slots, cloud record.

## 4. Time base

Measurement time of a pose = mean capture time of the two frames, in `CLOCK_MONOTONIC`. The SDK global timestamp of a
frame is in the `CLOCK_REALTIME` base. The pair record carries `publish_mono_ns` and `publish_real_ns`, read by the
capture program at the same instant, so `offset = publish_real_ns - publish_mono_ns` converts exactly:
`t_capture_mono = global_ts_ms * 1e6 - offset`. The arrival times (`arrival_mono_ns`) of the record are used as a check:
`arrival_mono - t_capture_mono` is the delivery latency (about 9.6 ms in the capture tests) and must be positive.
(The task names the arrival times for the conversion; with arrival times alone a latency model would be needed. See
DEVIATIONS.)

## 5. Stages of one cycle

1. **Pair reader.** Wait on the pair notification; on wake take the newest published pair (older unread pairs are
   skipped and counted). Copy the record; copy only the image rows of the crop region of each camera; then check the
   slot sequence counter (seqlock). A changed counter discards the pair (counted). Capture restarts: `WriterNotAlive`
   or `WriterReplaced` -> reattach by name (retry every 100 ms while no writer exists), intrinsics re-read.
2. **Cloud builder.** Crop region per camera = bounding rectangle of the 8 projected corners of the predicted object
   box (+ margin), or the whole work volume in search. Every n-th valid pixel in rows and columns (n chosen so that the
   expected point count stays below the configured maximum) -> 3D point in the camera frame (pinhole, intrinsics from
   the pair header, depth unit from the header) -> rig frame with `T_rig_from_camera` (calibration file, or nominal).
   Camera label kept per point.
3. **Scene filter.** Keep points inside the work volume (axis-aligned box in the rig frame) and, in track state, inside
   the predicted object box; remove points in occupied cells of the background map (20 mm grid) if one is loaded.
4. **Pose solver.** Open3D point-to-plane ICP port with a Tukey kernel, source = measured points, target = model points
   with normals, correspondence distances 50 / 20 / 10 mm in sequence. Start = filter prediction (track) or search
   candidates. Result `T_object_from_rig`; output pose = inverse.
5. **Fit check.** Reject if: points < min, RMSE > max, inlier ratio < min, smallest eigenvalue of the normalized 6x6
   matrix < min, jump from the prediction > max (translation and rotation). Limits in the configuration file, defaults
   from the synthetic tests.
6. **Tracking filter.** Position: constant-velocity Kalman filter. Orientation: alpha-beta filter on SO(3). Without a
   valid pose: prediction only.

Every stage is timed (`CLOCK_MONOTONIC`) in every cycle; the times go into the pose record and into histograms.

### Normalized 6x6 matrix and covariance
ICP returns `JTJ` (6x6, order rotation then translation, increment parameters of `T_object_from_rig`, last iteration)
and the correspondences. The rotation part is moved to the centroid `c` of the matched source points and scaled by
the model radius `L`: `A = [[I/L, -[c]x/L], [0, I]]`, `H_n = A JTJ A^T / sum(w)`. A full view of the object gives
`lambda_min(H_n)` of order 0.1; a single plane gives about 0 (rotation about the normal, translation in the plane).
Covariance of the increment: `C_x = rmse^2 * JTJ^-1`. Output pose `T_rig_from_object = T^-1`: a left increment
`(omega, t)` on `T` changes the output position by `-R^T t` and the output rotation (rig frame) by `-R^T omega`, so
`C_out = J C_x J^T` with `J = blockdiag(-R^T, -R^T)` reordered to (position, rotation).

## 6. ICP port (Open3D, MIT license)

Source `isl-org/Open3D` tag v0.20.0 (commit in DEVIATIONS). Ported with the Open3D names and structure, in namespace
`open3d::pipelines::registration` / `open3d::utility` / `open3d::geometry`, copyright notice kept:
`RegistrationICP`, `GetRegistrationResultAndCorrespondences`, `ICPConvergenceCriteria`, `RegistrationResult`,
`TransformationEstimationPointToPlane::{ComputeTransformation, ComputeRMSE}`, the robust kernels (`L2Loss`, `L1Loss`,
`HuberLoss`, `CauchyLoss`, `GMLoss`, `TukeyLoss`), `utility::ComputeJTJandJTr`,
`utility::SolveJacobianSystemAndObtainExtrinsicMatrix`, `utility::SolveLinearSystemPSD` (dense path),
`utility::TransformVector6dToMatrix4d`, and `geometry::KDTreeFlann::SearchHybrid` on nanoflann 1.5.0 with the same
adaptor and leaf size (15). TBB parallel reductions become sequential loops (same sums, other order: differences at
the 1e-15 level). A minimal `geometry::PointCloud` (points, normals, `Transform`) replaces the Open3D class.

Reference test before any optimization: `tools/icp_reference.py` runs Open3D 0.20.0 (pip) on test clouds and writes
inputs and results; the port must give the same transform (max element difference < 1e-6), fitness, RMSE and
iteration count. Open3D does not return the iteration count; the script finds it as the smallest `max_iteration` that
gives the same result as the full run.

Optimizations afterwards, each re-checked with the reference test: tree for the model built once; no allocation in
the ICP call (preallocated arrays); distance sequence 50/20/10 mm; the last iteration's 6x6 matrix returned.

## 7. Track states

- **Search.** Crop = full work volume. Start translation = centroid of the cloud minus the rotated model centroid;
  start orientations = a fixed set of K orientations spread evenly over SO(3). Each candidate: ICP with the distance
  sequence. Work is spread over cycles with an 8 ms limit per cycle; the best candidate that passes the fit check
  (lowest RMSE among those with the highest inlier ratio band) is taken when all K are done -> Track.
- **Track.** Crop and start pose from the prediction.
- **Lost.** After N rejected cycles in track: box enlarged by a factor; after M rejected cycles: Search.

## 8. Filter equations

Position, state `x = [p, v]` (6), time step `dt` from measurement times:
`F = [[I, dt I], [0, I]]`, `Q = q_a * [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]` (white-noise acceleration, `q_a` in
m^2/s^3), measurement `z = p_icp`, `H = [I 0]`, `R = max(C_pos, r_min I)` from the ICP covariance (clamped below).
Standard predict/update.

Orientation (alpha-beta on SO(3)): prediction `q_pred = exp(omega dt) * q`; innovation `e = log(q_meas * q_pred^-1)`
(rotation vector, rig frame); update `q = exp(alpha e) * q_pred`, `omega = omega + (beta / dt) e`. Without a
measurement: prediction only.

## 9. Records

All little-endian, fixed size, `version` first. Full offsets: README.md.
- Pose record (version 1, 640 bytes): sequence numbers, instance ids, measurement / publish times, pair dt, state,
  valid flag, reject reasons, filtered position / quaternion / velocity / angular velocity, ICP pose, 6x6 covariance,
  RMSE, inlier ratio, smallest eigenvalue, point count, iteration count, search progress, stage times, capture times
  of both frames, pairs skipped, free-space violation fraction.
- Cloud record (version 1): pose sequence number, point count, center, then 8192 points
  `{f32 x, y, z; u8 inlier; u8 camera; u16 reserved}` (16 bytes) in the rig frame.

## 10. Network protocol (bridge -> viewer)

TCP, `TCP_NODELAY`, one client. Each message: header `{u32 magic 'TRKN', u16 type, u16 version, u32 payload_len,
u32 reserved}` (16 bytes), then the payload. Type 1 = pose (payload = pose record). Type 2 = cloud (fixed size:
`{u32 version, u32 count, u64 pose_seq, f64 center[3]}` then 8192 points `{i16 x, y, z (0.1 mm relative to center),
u8 inlier, u8 camera}`). Type 3 = bridge status (counters). Send queue with a drop policy: oldest cloud messages first,
then oldest pose messages; all drops counted. The queue (`src/core/send_queue.hpp`) allocates nothing after start: one
zero-filled slot pool per message type, each large enough for the byte limit, and a fixed ring of entries in arrival
order (a discarded entry is skipped; the ring is compacted in place when it is full of entries).

## 11. Record/replay file (T2, T3)

`TRKPAIRS` file version 1: header (magic, version, intrinsics and serials as in the pair header, pair count), then per
pair the 128-byte pair record and both images, each image run-length coded (zero runs + literal runs of u16), with
its coded size. Replay writes the pairs into a pair buffer with the capture layout (version 2) at the original
intervals, with all time fields shifted to the current clocks, so the tracker sees the same as from the capture.

## 12. Changes during the implementation

Version 1 above was written before the main code. Changes found necessary while implementing and testing (details and
reasons: DEVIATIONS.md; measurements: TEST_REPORT.md):

1. **Kernel per distance stage.** A fixed Tukey k = 5 mm stops ICP from moving when the start is centimetres away. The
   kernel is Tukey with k = max(5 mm, 0.5 x the stage distance). `RegistrationICPPrealloc` takes one kernel per stage.
2. **Free-space check in the fit check (reject bit 64).** From a partial view a pose turned by about 180 degrees can fit
   the visible points as well as the true pose. Model points that face a camera at the candidate pose are projected
   into that camera's image; if the pixel shows a surface farther than the model surface by more than 10 mm, the camera
   sees through the object there (violation). Points with no depth or with a nearer surface (occlusion) are not
   counted. Reject if violations > 15 % of the checked points or fewer than 50 checked. The same fraction ranks the
   search candidates (score = fitness - violations).
3. **Search start positions.** Instead of the centre of the whole cloud: connected components of occupied 4 cm cells
   of the filtered cloud whose extent fits the object (up to 4, largest first); each candidate uses only the points
   within (model radius + 20 mm) of its start position. Candidates = start positions x 72 orientations, coarse ICP with
   the first two distances (10 iterations), the 4 best refined with the full sequence and checked; the best passing
   one is accepted at the end of the round. Search samples the image denser (about every 3rd pixel) than track,
   because the full image is the crop.
4. **Time slicing.** The search estimates the duration of its next step from earlier steps (running average) and stops
   before the 8 ms limit, but always runs one step per cycle (a first slow step had stopped the search for good).
5. **Track budget.** 350 solver points in track state (was 800) and convergence criteria 1e-5: solver p50 1.2 ms on the
   Jetson with unchanged accuracy.
6. **Pair reader.** `PairSource` (stage 1) uses the capture library's `ShmReader` for attach, liveness and reattach, and
   reads the newest slot itself (record, crop rows and columns, sequence check).
7. **Records.** Pose record 640 bytes (above). The network cloud message is fixed size (8192 points of 8 bytes).
8. **Live programs on Linux only** (signals with `sigtimedwait`, thread names); the Mac builds the tests and tools.

### Threads, final

| Process | Threads |
|---|---|
| trk_tracker | main (signals), loop, status, 2 ring heartbeats; with `hints.threads` = n > 1 also n - 1 pool threads `trk-hint` (section 14), idle except during a cycle that tries hypotheses |
| trk_bridge | main (signals, status line), reader (pose buffer futex wait, cloud `latest()`), network (accept, non-blocking send) |
| viewer.py | main (display loop, ~30 Hz), receiver (socket, CSV), viser server threads |

## 13. Pose hypotheses from outside (flow module, added 2026-11-20)

`TrackerCore::set_hints(T_rig_from_object[], n)` stores up to 32 pose hypotheses (fixed array, no allocation). In the
next cycle in search state they are tried before the orientation grid: for each, the points within (model radius +
20 mm) of its position, the full ICP sequence from the hypothesis, and the fit check of the search (no jump test). The
best passing one (fitness minus free-space violations, then RMSE) is accepted and the state becomes track. If none
passes, the normal search runs in the same cycle; without hints the cycle is unchanged. The hypotheses come from
`../flow` (a learned flow on SE(3)); they only choose where ICP starts, every accepted pose still passes the same
fit check. The hint ICP runs are not time-sliced in this version (12 hypotheses: about 12 refine steps in one cycle).
`trk_hint_eval` runs the core on independent static scenes with and without hypotheses (formats in its header).

### Live path (added 2026-11-26)

```
d435_capture -> /d435_pairs --+--> trk_tracker --> /trk_pose, /trk_cloud
                              |        ^  reads the newest hint record while in search state
                              +--> trk_flow (../flow/live) --> /trk_hints (4 slots, hint record)
```

- **Hint record** (`records.hpp`, version 1, 2176 bytes): count, hint sequence number, pair sequence number and
  writer instance of the pair used, its measurement time (as in the pose record), publish time, timings, then up to
  16 scores and poses `T_rig_from_object`, best first. Ring `/trk_hints` (`hints.buffer`), same protocol as the other
  rings.
- **Inference process** `trk_flow` (`../flow/live/trk_flow.cpp`; TensorRT, not part of the tracker programs). While
  the newest pose record says search, it takes the newest pair, computes the hypotheses and publishes them. It gets
  everything from the tracker side through the C interface `src/flowio` (library `trk_flowio`): configuration and
  calibration, the pair reader (`PairSource`), the cloud for the network (every 2nd pixel, work volume), the state of
  the tracker and the hint writer. The cloud of the cloud buffer is not used: thinned to 8192 points it has about
  300 points near the object in cluttered scenes, and the network takes 512.
- **Tracker loop** (`trk_tracker`). In search state, before the pair is read: `latest()` on the hint ring (attach
  attempts at most twice per second, in search state only). A record is given to `set_hints` if its measurement time
  is at most `hints.max_age_ms` (400) old; older records are counted as stale (typically a proposal that was finished
  after the tracker had found the object by itself). No writer: the cycle is unchanged. `--no-hints` switches the
  reading off.
- **Pose record**: `hints_tried` (offset 624) and `by_hint` (625), from the reserved bytes; record version unchanged.
- **Cost in the tracker**: the cycle that tries the hypotheses is not time-sliced: median 6.8 ms, maximum 49 ms in the
  live test (16 hypotheses at most).
- **Test without cameras**: `trk_scenes` shows scenes of a scenes file in the pair buffer (empty images for 0.7 s,
  then the scene for 2 s); `../flow/live_test.sh` runs it with the tracker and the inference process on the Jetson,
  with and without hints (results: `../flow/README.md`).

## 14. Switchable CPU paths (added 2026-11-26)

A profile of the core on the evaluation scenes (`cmake -DTRK_PROFILE=ON`, `src/common/prof.hpp`, printed by
`trk_hint_eval`) shows where a search step and a hint trial spend their time: the nearest-neighbour queries of the
ICP take 95 % and 90 %; the normal equations 3 to 4 %, the free-space check 1 to 2 %, the 6x6 solve and the point
transforms together under 1 %; cloud building and scene filter 0.25 ms per cycle. So the paths below concern the
correspondence search and the hint trials. Every one is an option; the defaults are the code as it was.

| Option | Values | Effect on the result |
|---|---|---|
| `icp.correspondence_search` | `kdtree` (default) | the Open3D port (`SearchHybrid1`) |
| | `bounded` | the same correspondences: the tree search starts with the bound min(radius, distance to the correspondence of the previous iteration) (`KDTreeFlann::SearchBounded1`) |
| | `table_exact` | the same correspondences from a grid (`icp/ClosestTable`, cells of `icp.table_cell_m` = 2 mm): per cell the list of all model points that can be the closest for a query in the cell (all within d + 2 h of the centre: d its distance to the model, h the half diagonal); cells with more than 24 use the bounded tree search. 25 MB, built at init |
| | `table` | per cell the model point closest to the centre; a query takes the best of it and its 8 nearest model points. 3.6 MB. Not always the closest point (98.7 % in the unit test, at most 2.2 mm farther), so poses differ slightly |
| `hints.threads` | 1 (default) .. 8 | hint trials (`try_hints`) on a fixed pool (`core/work_pool.hpp`): n - 1 threads created at init plus the loop thread; per worker one ICP workspace and one point buffer, allocated at init. The trials are independent; afterwards the hints are gone through in their order with the rule of the sequential code (best score, then RMSE, then the lower index), so the pose record does not depend on thread timing |
| build: `TRK_TUNE` (default on) | | `-mcpu=cortex-a78ae` on the Jetson (`-mcpu=native` elsewhere); before, the compiler targeted a generic ARMv8-A processor |
| build: `TRK_LTO` (default off) | | link-time optimization |

Notes:
- The search cycle stays near the time slice (`states.search_time_limit_ms`, 8 ms) whatever the ICP costs: a faster
  ICP means more candidates per cycle and fewer cycles until the object is found. One round is up to 4 positions x
  (72 + 4) ICP runs and is cut off by the 180 cycles of `trk_hint_eval` in many scenes, which is why the classical
  result of that tool varies by a few scenes with the speed of the machine. The benchmark therefore times a whole
  round without slicing (`config/bench_noslice.json`).
- 4 hint threads are meant for the Jetson (6 cores): the loop thread and 3 pool threads work during the trial, 2 cores
  stay for the capture and the inference process. The cycle that tries the hypotheses is still not time-sliced.
- Tools: `tools/cpu_bench_run.sh` (one benchmark run), `tools/cpu_bench_table.py` (table and scene-by-scene comparison
  with a reference run), `tools/bench_cpu_jetson.sh` (all variants on the Jetson in one command, with the thread
  sanitizer run of the hint tests). Unit tests: `tests/test_cpu_paths.cpp`.
- Not done, because the profile gives them at most a few percent: SIMD for the cloud builder, the work-volume test,
  the point transforms and the normal equations; single precision.
