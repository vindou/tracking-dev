# tracker

Pose tracking of one known rigid object (`assets/test_object.step`) from the depth pairs of `capture/` (two D435i on a
Jetson Orin Nano), with a live viewer on a Mac.

```
Jetson                                                                MacBook
d435_capture -> /d435_pairs -> trk_tracker -> /trk_pose  (256 slots) --\
                                    |                                    trk_bridge == TCP :9870 ==> viewer.py (viser, :8080)
                                    +-------> /trk_cloud (4 slots)  --/
```

Design: [DESIGN.md](DESIGN.md). Tests: [TEST_REPORT.md](TEST_REPORT.md). Deviations and interpretations:
[DEVIATIONS.md](DEVIATIONS.md). Conventions: meters, seconds, radians; quaternions w, x, y, z; output pose =
`T_rig_from_object`; time base `CLOCK_MONOTONIC` of the Jetson.

## Build

Libraries: C++17, POSIX, Eigen 3.4.0 and nanoflann 1.5.0 (headers in `third_party/`, copied from the Mac), and the
capture library (`capture_core`, built from `../capture` without changes).

Jetson (`~/tracker`, `~/capture`, `~/models` side by side):

```
cd ~/tracker
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && make -C build -j6
cmake -S . -B build-tsan -DTRK_TSAN=ON && make -C build-tsan -j6 trk_tests      # ThreadSanitizer
cmake -S . -B build-asan -DTRK_ASAN=ON && make -C build-asan -j6 trk_tests      # AddressSanitizer + UBSan
```

From the Mac: `tracker/tools/sync_to_jetson.sh [normal] [tsan] [asan]` copies `tracker/` and `models/` and builds.
The Mac builds the tests and the offline tools (`cmake -S tracker -B tracker/build-mac -G Ninja && ninja -C tracker/build-mac`);
the live programs (signals, Linux thread names) are built on Linux only.

Python (Mac): a virtual environment in the repository (`.venv/`, ignored by git) with the five permitted packages:

```
python3 -m venv .venv && .venv/bin/pip install viser numpy trimesh cascadio open3d
```

`viewer/requirements.txt` lists what the viewer needs (viser, numpy, trimesh).

## Programs

| Program | Where | Purpose |
|---|---|---|
| `tools/make_model.py` (T1) | Mac | STEP -> `models/test_object.stl` + `.model` (3000 points with face normals); `--rig` -> `models/rig.stl` |
| `tools/icp_reference.py` | Mac | Open3D reference results for the ICP port test (`tests/data/icp_ref.bin`) |
| `trk_tracker` | Jetson | the tracker (pair buffer -> pose and cloud buffers) |
| `trk_bridge` | Jetson | telemetry bridge (pose and cloud buffers -> TCP) |
| `viewer/viewer.py` | Mac | live viewer; `viewer/demo.sh` runs the whole demo |
| `trk_synth` (T2) | both | synthetic depth pairs for a pose and motion -> pair file |
| `trk_record`, `trk_replay` (T3) | Jetson | pair buffer -> pair file; pair file -> pair buffer at the original timing |
| `trk_camid` (T4) | Jetson | camera identification (cover the -Y camera) -> serials in the configuration |
| `trk_calib` (T5) | Jetson | single-camera solves, camera-frame check, calibration file |
| `trk_bgmap` (T6) | Jetson | background map of the empty scene (20 mm grid) |
| `trk_scenes` | Jetson | test source: scenes of a scenes file in the pair buffer, one after another (live test of the hint path, `../flow/live_test.sh`) |
| `libtrk_flowio` | both | shared library, C interface of the tracker side for the inference process `../flow/live/trk_flow` (`src/flowio/flowio.h`) |
| `trk_hint_eval` | both | acquisition on independent static scenes, classical search with and without pose hypotheses of the flow module (`../flow`) |
| `trk_offline` | both | tracker core on a pair file -> pose CSV (offline reference, analysis) |
| `trk_posedump` | Jetson | pose buffer -> CSV |
| `trk_clouddump`, `trk_workvolume` | both | point cloud of one pair -> CSV; work volume from the camera geometry |

Run (Jetson, from `~/tracker`):

```
~/capture/build/d435_capture --config ~/capture/capture.conf &      # pair buffer /d435_pairs
build/trk_tracker --config config/tracker.json                       # Ctrl-C / SIGTERM: clean stop
build/trk_bridge --config config/tracker.json                        # listens on 192.168.55.1:9870
```

Mac: `.venv/bin/python viewer/viewer.py --host 192.168.55.1` and open http://localhost:8080.

## Demo procedure

Manual steps are marked **(manual)**.

1. Build on the Jetson (above). Make the models on the Mac once: `.venv/bin/python tracker/tools/make_model.py` and
   `... make_model.py --rig assets/test_rig_main_assem.step`, then `tracker/tools/sync_to_jetson.sh`.
2. Start the capture program on the Jetson: `~/capture/build/d435_capture --config ~/capture/capture.conf &`.
3. **Camera identification (manual, once per rig set-up).** `build/trk_camid --config config/tracker.json --write`.
   When the program says so, **cover the camera on the -Y side of the rig** (the one that the nominal transforms call
   D435i_1, at y = -188 mm) completely, press Enter, and keep it covered until the result is printed. The serial
   numbers are written to `cameras.D435i_1/2.serial` in the configuration.
4. **Background map (manual, optional).** **Remove the object** from the scene. `build/trk_bgmap --out
   config/bgmap.bin --config config/tracker.json` records 100 pairs and writes the map. Set
   `"background_map": "bgmap.bin"` in `config/tracker.json`. **Put the object back** on its support.
5. **Calibration (manual: the object must stand still).** `build/trk_calib --out config/calib.json --config
   config/tracker.json` (about 4 s of pairs). It prints the object pose from each camera alone and their difference.
   If the difference is within 20 mm and 5 degrees, set `"calibration_file": "calib.json"` in `cameras`. If not, read
   the camera-frame check in its output (axis convention, names, units) before using the file.
6. Stop the capture program (Ctrl-C or `kill -TERM`). On the Mac: `viewer/demo.sh`. It starts capture, tracker and
   bridge on the Jetson, starts the viewer and prints the URL (http://localhost:8080). The object appears when the
   tracker leaves search state (normally within 1 s).
7. Stop with Ctrl-C: the viewer, the bridge, the tracker and the capture program stop, and the script prints whether
   any process or shared-memory object is left. Logs: `viewer/demo_logs/<time>/` (`poses.csv` = every pose received).

Viewer controls: "Clear path trace", "Show cloud", "Pose shown: filtered / unfiltered (ICP)". Text: state, position,
RMSE, inlier ratio, pose rate, calculation time, delay from capture to pose publish (measured on the Jetson).

## Configuration (`config/tracker.json`)

One JSON file, version 1; paths relative to the file. Main entries: buffer names; `model_file`, `mesh_file`;
`cameras.D435i_1/2` (`serial`, `T_rig_from_camera_mm` nominal), `cameras.calibration_file`; `work_volume` (min, max in
the rig frame, from `trk_workvolume`: region both cameras see, 0.2-0.8 m from the cameras, above z = 0.07 m);
`background_map`; `cloud` (point budgets, crop margins); `icp` (distance sequence 50/20/10 mm, Tukey kernel with
k = max(5 mm, 0.5 x distance), iterations, convergence); `fit_check` (limits); `filter` (Kalman and orientation filter
gains); `states` (lost and search thresholds, search time limit 8 ms per cycle, orientations); `bridge` (address
192.168.55.1, port 9870, queue bytes); `cloud_rate_hz` (30; the task's value is 15, see DEVIATIONS 36); `status_interval_s` (10).

## Shared-memory rings (pose and cloud buffers)

Same protocol as the capture pair buffer (`capture/README.md`), with a generic header. All values little-endian,
naturally aligned. Names: `/trk_pose` (256 slots), `/trk_cloud` (4 slots).

### Header (offset 0, 512 bytes)

| Offset | Type | Field | Value / meaning |
|---|---|---|---|
| 0 | u64 | magic | `0x00474E49524B5254` (bytes `TRKRING\0`); written last (release) when the header is ready |
| 8 | u32 | layout_version | 2 |
| 12 | u32 | header_size | 512 |
| 16 | u32 | slot_count | 256 (pose), 4 (cloud) |
| 20 | u32 | record_size | 640 (pose), 131136 (cloud) |
| 24 | u64 | slot_stride | 64 + record_size rounded up to 64: 704 (pose), 131200 (cloud) |
| 32 | u64 | slots_offset | 512 |
| 40 | u64 | total_size | the object may be larger (pages) |
| 48 | char[16] | record_type | "pose" / "cloud", NUL-terminated |
| 64 | u32 | record_version | 1 |
| 68 | i32 | writer_pid | information only (not a liveness signal) |
| 72 | u32 atomic | writer_state | 1 running, 2 stopped cleanly, 3 replaced by a new writer |
| 76 | u32 atomic | notify_word | +1 per record and per state change; futex word (`FUTEX_WAIT`, not private) |
| 80 | u64 atomic | records_published | record n is available when `records_published > n` |
| 88 | u64 | writer_instance_id | random, non-zero, new for each tracker start |
| 96 | u64 atomic | heartbeat_mono_ns | `CLOCK_MONOTONIC` of the last heartbeat (every 50 ms) |
| 104 | u64 atomic | heartbeat_count | |
| 112 | u32 | heartbeat_interval_ms | 50 |
| 116 | u32 | liveness_timeout_ms | 1000 |
| 120 | i64 | writer_start_real_ns | `CLOCK_REALTIME` at creation |
| 128 | u64 atomic | replaced_by_instance | instance that replaced this writer (state 3) |
| 136 | | reserved | to 512 |

Slot k at `512 + k * slot_stride`: `u64 seq` at 0 (odd: being written; `2n + 2`: holds record n), the record at 64.
Record n is in slot `n % slot_count`.

### Attach sequence, read loop, liveness, reattach

1. `shm_open(name, O_RDONLY)`; missing: no tracker yet, retry (100 ms). `fstat`; size < 512: retry. `mmap` read-only.
2. Load `magic` (acquire); not `TRKRING\0`: header not ready, retry. Check `layout_version == 2`, `record_type`,
   `record_version == 1` and `record_size` (otherwise reject: other program version), header and slot sizes.
3. Remember `writer_instance_id`; start at `next = records_published` (or `max(0, published - slot_count)`).
4. Loop: `w = notify_word` (acquire), `pub = records_published` (acquire). If `writer_state == 3`: go to reattach.
   If `next < pub`: skip to `pub - slot_count` if behind; read slot `next % slot_count`: `s1 = seq` (acquire); if
   `s1 != 2*next + 2` it was overwritten (missing); else copy the record, acquire fence, `s2 = seq`; if `s2 != s1` the
   copy is torn: discard (missing); else use it. `next++`. If nothing new: `writer_state == 2` -> stopped;
   liveness check; else `futex(&notify_word, FUTEX_WAIT, w, <= 200 ms)`.
5. Liveness: alive if `writer_state == 1` and `CLOCK_MONOTONIC - heartbeat_mono_ns <= liveness_timeout_ms` (same host).
6. Reattach (writer not alive, replaced, or stopped): every 100 ms run steps 1-3 again; accept only a different
   `writer_instance_id`.

`tracker/src/shmring/shm_ring.*` (C++) and `trk_posedump` implement this.

### Pose record (version 1, 640 bytes)

| Offset | Type | Field |
|---|---|---|
| 0 | u32 | version (1) |
| 4 | u32 | track_state: 0 search, 1 track, 2 lost |
| 8 | u64 | pose_seq (0, 1, 2, ... per tracker instance) |
| 16 | u64 | pair_seq of the pair used |
| 24 | u64 | writer_instance_id (tracker) |
| 32 | u64 | capture_instance_id (pair-buffer writer) |
| 40 | i64 | meas_time_mono_ns: mean capture time of the two frames (`CLOCK_MONOTONIC`) |
| 48 | i64 | publish_time_mono_ns |
| 56 | f64 | pair_dt_ms: capture-time difference B - A |
| 64 | u32 | valid: 1 if the ICP pose of this cycle passed the fit check |
| 68 | u32 | reject_reasons bits: 1 points, 2 RMSE, 4 inlier ratio, 8 eigenvalue, 16 jump, 32 no solve (search), 64 free-space violations |
| 72 | f64[3] | position (filtered), m, rig frame |
| 96 | f64[4] | orientation (filtered) w, x, y, z |
| 128 | f64[3] | linear velocity, m/s |
| 152 | f64[3] | angular velocity, rad/s, rig frame |
| 176 | f64[3] | ICP position (unfiltered) |
| 200 | f64[4] | ICP orientation (unfiltered) |
| 232 | f64[36] | 6x6 covariance of the ICP pose, row-major, order (px, py, pz, rx, ry, rz), rig frame |
| 520 | f64 | fit RMSE (m, inlier point-to-point) |
| 528 | f64 | inlier ratio |
| 536 | f64 | smallest eigenvalue of the normalized 6x6 matrix |
| 544 | u32 | point count (given to the solver) |
| 548 | u32 | ICP iteration count (all distance stages) |
| 552 | u32 | search candidates done in the round |
| 556 | u32 | search candidates per round |
| 560 | f32[8] | stage times, us: read, build, filter, solve, check, track filter, publish (previous cycle), total |
| 592 | i64[2] | capture time of frame A and B (`CLOCK_MONOTONIC`) |
| 608 | u64 | pairs skipped (total) |
| 616 | u32 | rejected cycles in a row |
| 620 | f32 | free-space violation fraction |
| 624 | u8 | external pose hypotheses tried in this cycle (DESIGN.md 13) |
| 625 | u8 | 1: the pose accepted in this cycle started from an external hypothesis |
| 626 | | reserved to 640 |

### Hint record (version 1, 2176 bytes; ring `/trk_hints`, 4 slots, written by `../flow/live/trk_flow`)

`u32 version` 0, `u32 count` 4, `u64 hint_seq` 8, `u64 pair_seq` 16, `u64 capture_instance_id` 24,
`i64 meas_time_mono_ns` 32, `i64 publish_time_mono_ns` 40, `f32 prep_ms` 48, `f32 net_ms` 52, `f32 post_ms` 56,
`u32 hypotheses` 60, `f32 score[16]` 64, `f64 T_rig_from_object[16][16]` 128 (row-major 4x4, best first).

### Cloud record (version 1, 131136 bytes; 15 Hz)

`u32 version` 0, `u32 point_count` 4, `u64 pose_seq` 8, `u64 pair_seq` 16, `i64 meas_time_mono_ns` 24, `u32 track_state`
32, reserved to 64; then 8192 points of 16 bytes: `f32 x, y, z` (rig frame, m), `u8 inlier`, `u8 camera` (0 = A,
1 = B), `u16` reserved. The points are those given to the solver in that cycle.

## Network protocol (trk_bridge -> viewer)

TCP, server on the Jetson (`bridge.listen_address`, default 192.168.55.1 = USB network interface `l4tbr0`, port 9870),
`TCP_NODELAY`, one client at a time (a second client is closed at once). The client only reads. Every message:

| Offset | Type | Field |
|---|---|---|
| 0 | u32 | magic `0x4E4B5254` ("TRKN") |
| 4 | u16 | type: 1 pose, 2 cloud, 3 bridge status |
| 6 | u16 | version 1 |
| 8 | u32 | payload length |
| 12 | u32 | reserved |
| 16 | | payload |

- Type 1 (640 bytes): the pose record as above, for every pose.
- Type 2 (65584 bytes, fixed): `u32 version` 0, `u32 point_count` 4, `u64 pose_seq` 8, `u64 pair_seq` 16, `f64 center[3]`
  24 (rig frame, m), then 8192 points of 8 bytes: `i16 x, y, z` in units of 0.1 mm relative to the center, `u8 inlier`,
  `u8 camera`. Point = center + 1e-4 * (x, y, z).
- Type 3 (64 bytes, once per second): `u32 version`, `u32 tracker_attached`, `u64 poses_read`, `u64 clouds_read`,
  `u64 poses_dropped`, `u64 clouds_dropped`, `u64 tracker_instance_id`, `i64 bridge_time_mono_ns`, `u64 reserved`.

The bridge never waits for the network: a queue (1 MB) between its reader and network threads; when it is full, the
oldest cloud messages are discarded first, then the oldest pose messages (counted, sent in type 3). `viewer/protocol.py`
decodes all three types.

## Pair file (record / replay, T2 / T3)

Header 512 bytes: `"TRKPAIRS"`, `u32 version` (1), `u32 header_size` (512), `u32 width`, `u32 height`,
`u32 bytes_per_pixel` (2), `u32 source` (0 capture, 1 synthetic), `u64 pair_count`, `i64 created_real_ns`,
`f64 frame_period_ms`, `f64 tolerance_ms`, reserved, `CameraInfo[2]` at 128 (capture layout, 96 bytes each). Per pair:
272-byte chunk header (`u32 "PAIR"`, `u32` coded size A, `u32` coded size B, `u32 flags` (1: truth present),
`PairRecord` (capture layout v2, 128 bytes), `f64 truth[16]` (`T_rig_from_object` row-major, synthetic files)), then
both images, run-length coded: repeated `u16` zero-run length, `u16` literal count, literal values.

## Tests

`build/trk_tests [filter]` (unit and synthetic tests; `TRK_STRESS_S`, `TRK_N_POSES` shorten them);
`tests/acceptance/run_jetson.sh` (unit tests in three builds, replay, 3 x 60 s live, capture restart),
`tests/acceptance/link_tests.sh` (Mac: link, viewer pause, viewer first, tracker SIGKILL, 30 min long run),
`tests/acceptance/phase7_tools.sh` (T4, T5, T6). Results: [TEST_REPORT.md](TEST_REPORT.md).
