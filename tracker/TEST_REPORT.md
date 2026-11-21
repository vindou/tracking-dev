# Test report: object tracker and live viewer

Date of the final runs: 2026-11-18 (Mac). The Jetson clock is about 8 h behind and set to 2023; Jetson log time stamps
use that clock. Every value below comes from the log file named in its row (paths relative to `tracker/logs/`).
Deviations and interpretations are numbered in `DEVIATIONS.md` and referred to as "D<n>".

Result counts: **PASS** the criterion was met; **FAIL** the criterion was not met; **BLOCKED** the test could not be
run as specified (reason given); **REPORT** the task asks only for values.

## Summary

| Area | Result |
|---|---|
| Unit and synthetic tests (Mac and Jetson; normal, TSan, ASan+UBSan builds) | 20 of 20 pass in each build, 0 sanitizer reports |
| Replay | Track variant PASS (0.0000 mm); search variant FAIL (0.65 mm, acquisition on a different pair) |
| Live, 3 x 60 s | Pair use PASS, capture counters PASS; track-state timing BLOCKED (no object in view during the runs) |
| Static object | BLOCKED (no object in view) |
| Capture restart, tracker SIGKILL | PASS |
| Long run 30 min | Attempts 1 and 2 FAIL (memory); attempt 3 PASS after D41, D42, D47 |
| Link and viewer | 3 x 60 s PASS (100 %), viewer stopped PASS, viewer first PASS |
| Phase 7 tools | T4 PASS, T5 synthetic PASS, T5 real scene skipped (user), T6 PASS |

## Phase 1: model (T1)

| Test | Result | Values | Log |
|---|---|---|---|
| Object model from `test_object.step` | PASS | 74 triangles, closed, 102.5 x 92.5 x 67.84 mm, 3000 model points, normals point out | `t1_object.log` |
| Rig mesh from `test_rig_main_assem.step` | PASS | 6 parts | `t1_rig.log` |

## Unit and synthetic tests

Run on the Mac (`unit/unit_mac.log`, 20 tests, 0 failed) and on the Jetson in three builds:
`jetson/unit/unit_normal.log`, `jetson/unit/unit_tsan.log`, `jetson/unit/unit_asan.log` (ASan + UBSan), each 20 tests,
0 failed, 0 sanitizer reports (summary: `jetson/run_unit2.out`). Values below are from the Jetson normal build unless noted.

| Test (task table) | Criterion | Result | Values |
|---|---|---|---|
| Phase 2 buffer tests | no mixed data, no missing records for a fast reader, 0 sanitizer reports | PASS | fast reader 6000 received, 0 missing; stress 30 s, 977,775 records, 3 readers: 0 mixed, 0 order errors; writer p99 25-27 us with no / slow / stopped reader; killed writer detected after 1.001 s. Also Mac: `phase2_ring_mac.log`, `phase2/ring_{normal,tsan,asan}.log` |
| ICP reference (phase 3) | \|dT\| < 1e-6 with the same iteration count | PASS | 80 cases against Open3D 0.20.0: max \|dT\| 5.6e-16, iteration mismatches 0; preallocated variant max \|dT\| 5.8e-16, 0 heap allocations. Reference: `phase3_icp_reference_py.log`; Mac: `phase3_icp_mac.log`, `phase3/` |
| Static accuracy, 100 poses, 1 mm noise | position RMS <= 1 mm, orientation RMS <= 1 deg | PASS | 100 passed the fit check, 0 rejected; position RMS 0.255 mm (max 0.569), orientation RMS 0.307 deg (max 0.650); RMSE p50 1.56 mm |
| Degenerate view (one flat face) | fit check rejects | PASS | 100 cycles, 0 valid; min eigenvalue max 0.0057 < 0.01 |
| Acquisition, 100 poses | >= 90 % in track within 1 s | PASS | 100 % (100 of 100); wrong poses accepted 0; time to track p50/p90/max 0.60/0.68/0.70 s |
| Moving object, 0.2 m/s, 30 deg/s, offset 0-5.6 ms | stays valid, position RMS <= 3 mm | PASS | all offsets: 0 cycles out of track, 0 invalid of 135; filtered position RMS 0.20-0.43 mm |
| Object removed and put back | track -> lost -> search -> track, no invalid pose marked valid | PASS | 5 runs, states T-L-S-T, invalid marked valid 0 |
| All unit tests with ASan + UBSan | 0 reports | PASS | `jetson/unit/unit_asan.log` |
| (additional) No heap allocation in the cycle | 0 | PASS | `core_process_does_not_allocate`: search 0, track 0 |
| (additional) Bridge send queue | order, discard rule, no allocation | PASS | 200,000 messages to a stalled client: 1514 left, 983,392 bytes (limit 1,048,576), 0 allocations (D41) |

## Replay and live tests

Live tests were run with the cameras on the Jetson. **No test object was in view** during live_1..3, capture restart
and the link tests (the user was away; D35). The tracker either searched the whole time (live_1..3) or locked onto
static scene structure (link and long runs; D46). tegrastats and the new kernel lines are in each test folder
(`tegrastats.log`, `kernel_new.log`). The kernel lines are the UVC/USB enumeration of the two D435i at capture start
(84 lines per start), 0 USB error lines in all runs.

| Test | Criterion | Result | Values | Log |
|---|---|---|---|---|
| Replay of a synthetic file, track start | same poses as offline within 0.1 mm | PASS | 600 pairs compared, max difference 0.0000 mm, 0.0000 deg | `jetson/replay_synth_track/compare.out` |
| Replay, search start (additional variant) | same | FAIL | 527 pairs compared, max 0.6489 mm, 0.6459 deg. Offline and live acquire on different pairs (60 and 73), so the filter histories differ; D30 | `jetson/replay_synth_search/compare.out` |
| Calculation time in track state, 3 x 60 s live | 99 % <= 5 ms; report against 2 ms | BLOCKED | No object in view: live_1..3 were 100 % search. Supplementary (not a substitute): **synthetic replay on the Jetson**, track state, 600 cycles: p50 2.25 ms, p99 2.66 ms, max 4.07 ms; <= 5 ms 100 %, **<= 2 ms 20 %** (goal not met). **Live track state on the false lock** (link runs, 490-726 cycles each): p50 7.15-7.24 ms, p99 13.8-15.1 ms, <= 5 ms 10.6-14.5 %: the ICP runs 25 iterations at p50 (p90 43-44) on a structure that does not match the model, against 4 (p90 4, max 6) in the synthetic replay | `jetson/replay_synth_track/tracker.json`, `live.csv`; `link_{1,2,3}/viewer_poses.csv` (`t_total_us`, `track_state` = 1) |
| Calculation time, all states (additional) | none | REPORT | live_1..3 (search): p50 7.72-7.76 ms, p99 11.4-11.9 ms, max 21.3-26.0 ms; search slice limit 8 ms | `jetson/live_{1,2,3}/eval.out` |
| Pair use, 3 x 60 s live | >= 99 % | PASS | 99.964 %, 99.947 %, 100.0 % | `jetson/live_{1,2,3}/eval.out` |
| Effect on capture, 3 x 60 s live | 0 dropped, 0 overflows | PASS | A and B: dropped 0, overflows 0, restarts 0, callback errors 0 in all three runs | `jetson/live_{1,2,3}/eval.out`, `capture.json` |
| Static object, 3 x 60 s live | report only | BLOCKED | No object in view. The false lock on static structure (D46) gave position SD 2.7 mm and orientation SD 2 deg; this is not the object and not a measurement of the requirement | `long_30min/viewer_poses.csv` |
| Capture restart during tracking | tracker continues without restart | PASS | tracker detached at 23.5 s, attached to the new capture instance at 30.0 s, same tracker process, 2511 poses | `jetson/caprestart_1/tracker.log`, `eval.out` |
| Tracker SIGKILL, then started | bridge and viewer continue | PASS | viewer connects 1 (no reconnect needed), 2 tracker instances seen: 1580 and 1796 poses, 0 gaps; bridge attaches 2 | `tracker_kill/result.txt` |
| Long run, 30 min, attempt 1 | no crash, memory growth <= 5 %, counters | FAIL | no crash; tracker +25.4 %, bridge +26.7 %. Causes found and fixed: D41, D42 | logs overwritten by attempt 2; values in D41/D42 |
| Long run, 30 min, attempt 2 | same | FAIL | no crash; tracker 15,252 -> 16,020 kB (+5.0 %, above the limit by rounding: +5.04 %), bridge 6,488 -> 8,388 kB (+29.3 %), viewer -36 %. Both stepped once between 150 and 180 s and stayed flat for the remaining 27 min. Cause: transparent huge pages (D47). Counters: 162,857 cycles, pairs used 162,857 of 162,950, skipped 93, cycle p99 13.01 ms; bridge poses read/sent/dropped 162,857/161,687/0; viewer 161,687 poses, 1 connect; capture 0 dropped, 0 overflows | log of attempt 2 replaced by attempt 3; values copied from `long_30min/result.txt` of attempt 2 into D47 |
| Long run, 30 min, attempt 3 (with D47) | same | **PASS** | no crash, both logs ended normally; tracker 15,252 -> 15,252 kB (+0.0 %), bridge 6,476 -> 6,476 kB (+0.0 %), viewer 49,920 -> 27,952 kB (-44 %); `AnonHugePages` 0 throughout. Counters: 162,818 cycles, pairs used 162,818 of 162,916, skipped 98, cycle p99 13.15 ms; bridge poses read/sent/dropped 162,818/161,676/0; viewer 161,676 poses, 46,002 clouds, 1 connect; capture A and B 0 dropped, 0 overflows | `long_30min/result.txt`, `rss.csv`, `mem_detail.csv` |

## Link and viewer tests

| Test | Criterion | Result | Values | Log |
|---|---|---|---|---|
| Pose messages received, run 1 | >= 99 % published, in sequence | PASS | 5404 of 5404 (100.00 %), gaps 0, out of order 0; bridge drops 0 | `link_1/result.txt` |
| run 2 | same | PASS | 5415 of 5415 (100.00 %), gaps 0, out of order 0 | `link_2/result.txt` |
| run 3 | same | PASS | 5412 of 5412 (100.00 %), gaps 0, out of order 0 | `link_3/result.txt` |
| Viewer stopped for 10 s | tracker cycle time unchanged; bridge counts discards | PASS | cycle p50/p99 before 7.57/12.80 ms, stopped 7.50/13.17 ms, after 7.51/13.64 ms; bridge discarded 230 clouds, 0 poses (D31) | `viewer_pause/result.txt` |
| Viewer started with no bridge | waits, connects later | PASS | connected once the bridge started; 1527 poses | `viewer_first/result.txt` |

## Phase 7: calibration and set-up tools

| Test | Result | Values | Log |
|---|---|---|---|
| T4 camera identification, cover A | PASS | camera 234322070408 lost its depth (1.7 % -> 0.0 %), written as D435i_1 | `jetson/t4_cover_A/` |
| T4, cover B | PASS | 234322070144 written as D435i_1 | `jetson/t4_cover_B/` |
| T5 calibration, synthetic | PASS | corrected B vs true B 1.29 mm, 0.180 deg (limits 2 mm, 0.5 deg) | `jetson/t5_synth_correction/` |
| T5 axis convention, synthetic | PASS | rotated-frame input detected, closest assumption reported | `jetson/t5_synth_axis/result.txt` |
| T5 real scene | not run | skipped at the user's request (D40). The camera frame assumption was checked instead by overlaying the rig CAD and the camera frames in the viewer with the user (D33) | |
| T6 background map, synthetic | PASS | plane points in occupied cells 100 %, object points 0 %; without map 0 valid, with map 143 of 200 valid, 0 wrong | `jetson/t6_synth/` |
| T6 real scene | ran | 100 pairs, 435 occupied cells; the map file was later deleted by the sync script (D45) | `jetson/t6_real/bgmap.log` |

The earlier real-scene analysis (`phase7_real_scene*/analysis.txt`) used the first rig assembly, which the user later
corrected (D33); it is kept for the record only.

## Findings for the demo

1. **False lock on static structure (D46).** Without a background map the search accepts a pose on a structure
   about 0.24 m wide at (-0.17, 0.22, 0.42) m in the rig frame. Build the background map (`DEMO.md`, set-up step 2)
   with the scene empty before the demo.
2. **Track-state timing on the object live is not yet measured.** With the object in view, run
   `tests/acceptance/run_jetson.sh live` on the Jetson (3 x 60 s) to complete the track-state timing and static-object rows.
