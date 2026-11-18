# Test report

Hardware: Jetson Orin Nano, two D435i (A = 234322070408, B = 234322070144), firmware 5.17.3.10, USB 3 (5000M), librealsense
2.59.1 (RSUSB backend), depth 848x480 Z16 @90. Compiler on the Jetson: g++ 11.4.0.

Log paths: `~/capture_logs/...` on the Jetson. Copies are in `test_logs/` in this directory, with the same names. The
`pairs.csv` files are left out of the copies because they are 42 MB. Each T2 run directory contains:

- `stdout.log`: status lines and final counters.
- `final.json`: final counters.
- `latency_hist.csv`.
- `tegrastats.log`.
- `kernel_new.log`: new kernel lines.
- `meta.txt`: USB tree, leftover shm and processes.
- `reader.json`: in the reader tests only.

`t2/evaluate.out` is the evaluator output for all runs and `t2/verdict.json` its machine-readable form. Both come from
`tests/acceptance/evaluate.py`.

## Summary

| Test | Result | Key values | Log |
|---|---|---|---|
| Build, `-Wall -Wextra -Werror` | PASS | 0 warnings (Jetson, all targets) | `build_cmake.log`, `build_make.log` |
| T1 unit tests, Mac | PASS | 15/15 | `t1_mac.log` |
| T1 unit tests, Jetson | PASS | 15/15 | `t1_jetson.log` |
| T1 under ThreadSanitizer | PASS (on the Jetson) | ring 10M, matcher a-f, pipeline, fps: 0 TSan reports. The Mac TSan is broken (DEVIATIONS 1). | `t1_jetson_tsan.log`, `t1_mac_tsan_unavailable.log` |
| T2 baseline 3 x 60 s | PASS | 0 drops, 0 overflows, 0 restarts; paired 99.98-100 %; all dt within ±5.564 ms | `t2/baseline_{1,2,3}` |
| T2 long 45 min | PASS | 242,188 pairs; 0 drops, 0 overflows, 0 restarts; paired 99.996 %; 19 partner changes | `t2/long_1` |
| T2 reader 3 x 60 s | PASS | reader got 4941/4940/4940 pairs = all published; 0 missing; 0 discarded reads | `t2/reader_{1,2,3}` |
| T2 reader SIGSTOP 60 s | PASS | counters as baseline: 0 drops, 0 overflows, 0 restarts, 100/99.98 % paired | `t2/reader_stopped_1` |
| T2 memory load 3 x 60 s | PASS | no crash or deadlock; 0 drops, 0 overflows. Callback p99 reached 2.19 ms, which does not meet R2 (see below). | `t2/memload_{1,2,3}` |
| T2 start/stop 20 cycles | PASS | 20/20: pairs supplied (443-444 each), no stall, no restart, shm removed | `t2/startstop_01..20` |
| T2 watchdog | PASS | A restarted once (detected 158 ms after its last frame, restart took 465 ms); pairs continued | `t2/watchdog_1` |
| Cleanup (rule 8) | PASS | in all 32 runs no process of mine and no shm object of mine remained | `t2/*/meta.txt` |

*The gap closure of the second task, with its own summary table and a new regression series, is in the section "Gap
closure (second task)" below.*

No test failed and none is blocked. One requirement is not met in every condition: **R2, callback p99 < 0.5 ms, is not met under
memory load** (see T2 memory load).

## T1 unit tests

`capture_tests` runs 15 tests in a test program of my own (`tests/test_framework.hpp`).

| Test | What it checks | Result (Jetson) |
|---|---|---|
| `ring_full_and_fifo` | 8 slots, a full ring discards the new item and counts an overflow, FIFO order | PASS |
| `ring_10M_two_threads` | 10,000,000 items, producer and consumer threads, sequence and content checked | PASS: 9,971,378 received, 28,622 overflows (all counted), max fill 8 |
| `matcher_a_constant_offsets` | offsets 0, 2, 5, -5, 5.3, -2 ms | PASS |
| `matcher_b_offset_through_half_period_with_noise` | drift through half a period, ±2 ms slow noise | PASS |
| `matcher_c_single_missing_frames` | single missing frames on A and B | PASS |
| `matcher_d_gap_then_burst` | 3-period gap, then a burst | PASS |
| `matcher_e_startup_hardware_clock_domain` | startup in the Hardware Clock domain, then Global Time | PASS |
| `matcher_f_restart_of_one_stream` | restart of one stream (new epoch) | PASS |
| `pipeline_threads` | callbacks, rings, matcher thread and sink on real threads | PASS: 1458 pairs, 0 order errors |
| `shm_roundtrip_and_unlink` | header, records, images; object removed on close | PASS |
| `shm_slow_and_stopped_reader_do_not_slow_writer` | slow reader and SIGSTOP'd reader (forked) | PASS (values below) |
| `shm_reader_skips_overwritten_pairs` | a late reader sees overwritten pairs as missing | PASS |
| `fps_monitor_*` (3 tests) | no false warning over 1 h at 89.87 fps; warns at 60/89.0/91.0 fps; gaps | PASS |

Each matcher case a-f checks the following:

- No frame is used twice.
- Pairs are in time order.
- Every pair is within tolerance.
- The pairs are exactly the expected ones.
- The counters are exact (paired, unpaired and the discard reasons).

In `shm_slow_and_stopped_reader_do_not_slow_writer` the writer publishes a pair every 2 ms. The writer's time per pair
(p50/p99/max) was:

| Reader | p50 | p99 | max |
|---|---|---|---|
| None | 319 us | 417 us | 479 us |
| Slow | 357 us | 706 us | 2037 us |
| Stopped | 320 us | 460 us | 558 us |

The writer has no wait path. Its higher p99 with a slow reader comes from memory-bandwidth contention: the reader copies
1.6 MB per read at the same time. The pass criterion and how I interpreted "does not slow down" are in DEVIATIONS 34.

The slow reader accepted 342 reads, saw 1155 pairs as missing and discarded 232 torn reads. None of the accepted reads was
torn.

On the Mac the same 15 tests pass (`test_logs/t1_mac.log`).

## T2 on the Jetson (final series, started 21:16 Jetson time)

Every run recorded `tegrastats` at 1 s intervals and the new kernel log lines. After the normal uvcvideo re-probe lines of
the RSUSB backend are filtered out, **no run had a relevant kernel line**: no USB error, xhci message, reset or disconnect
(`kern=0` in `evaluate.out`).

In the tables:

- **cb99** is the callback time p99 over the whole run.
- **lat** is the delivery latency p50/p99 (arrival `CLOCK_REALTIME` minus SDK global timestamp).
- **dt** is the range of t(B) − t(A) over all pairs.
- **CPU** is the mean CPU over the run from tegrastats.
- **Tj** is the maximum junction temperature.

For every run the evaluator also checked `pairs.csv` independently:

- No frame counter of either camera appears twice.
- Pairs are in strictly increasing time order.
- Every |dt| is at most the tolerance.

Every run passed these checks, and `order_errors` was 0 in every run.

### Baseline (3 x 60 s)

Criteria: 0 drops, 0 overflows, 0 restarts, ≥ 99 % of frames paired after settle, all dt within tolerance (5.564 ms).
I also checked R2: callback p99 < 0.5 ms.

| Run | Pairs | Drops A/B | Overflows A/B | Restarts | Paired A/B | Unpaired A/B | dt [ms] | cb99 A/B [ms] | lat A [ms] | CPU | Tj |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_1 | 4941 | 0/0 | 0/0 | 0 | 100/100 % | 0/0 | -1.19 .. -0.23 | 0.40/0.41 | 10.0/11.2 | 11.9 % | 43.4 °C |
| baseline_2 | 4943 | 0/0 | 0/0 | 0 | 100/100 % | 0/0 | -3.43 .. -2.06 | 0.30/0.29 | 9.5/10.7 | 9.7 % | 43.1 °C |
| baseline_3 | 4939 | 0/0 | 0/0 | 0 | 100/99.98 % | 0/1 | 1.18 .. 2.48 | 0.46/0.29 | 9.7/11.0 | 10.1 % | 43.0 °C |

Measured frame period: 11.12833 ms; tolerance 5.5642 ms. Callback maximum (single frames): 0.42-1.46 ms.

The offset between the cameras is different in every run, because the cameras are free-running and their phase depends on
when each stream started.

### Long run (45 min)

`t2/long_1`: 2700 s.

- **Pairs and pairing:** 242,188 pairs; 0 drops, 0 overflows, 0 restarts. 99.996 % of eligible frames were paired:
  10 unpaired frames on A and 10 on B, out of 242,198 eligible per camera.
- **Timing:** dt ranged from −5.563 to +5.564 ms. Callback p99 was 0.37/0.37 ms and maximum 2.08/1.91 ms. Latency
  p50/p99 was 9.6/10.8 ms.
- **System:** CPU 8.9 %, Tj max 43.9 °C, 0 frame-rate warnings, 0 relevant kernel lines.

**Partner-change behaviour.** There were 19 partner changes, all between 661 s and 1009 s of the run. I analysed them from
`pairs.csv`; the analysis is in the session, not a log file.

- At every change dt was at ±5.56 ms, which is exactly half the frame period. The phase between the two free-running
  cameras drifted to the half-period boundary and stayed there for about 6 minutes.
- In that time the counter difference `counter(B) − counter(A)` alternated between −95 and −96 at each change. dt jumped
  between −5.56 and +5.56 ms: the matcher changed between the two nearly equidistant partners.
- Outside that window the phase was elsewhere and there were no changes. At the end of the run dt was about +3.7 ms.

At these changes the run had:

- 0 duplicate frames: the independent check found no frame counter used twice.
- 0 order errors.
- No dt beyond the tolerance.
- No stall: 0 timeouts while both cameras were up.

Each change costs at most one unpaired frame per camera: 10 per camera over 19 changes.

The superseded first series had a similar episode: 31 changes between 1687 s and 1997 s (`~/capture_logs/t2_series1/long_1`).

### Reader attached (3 x 60 s)

`d435_pair_reader` was attached for the whole run. Criteria: the reader receives every pair with no missing sequence
number, and the capture counters meet the baseline criteria.

| Run | Pairs published | Reader received | Missing | Seq gaps | Discarded reads | Drops | Overflows | Restarts | Paired A/B | cb99 A/B |
|---|---|---|---|---|---|---|---|---|---|---|
| reader_1 | 4941 | 4941 (seq 0..4940) | 0 | 0 | 0 | 0/0 | 0/0 | 0 | 99.96/99.96 % | 0.33/0.32 |
| reader_2 | 4940 | 4940 (seq 0..4939) | 0 | 0 | 0 | 0/0 | 0/0 | 0 | 100/100 % | 0.33/0.34 |
| reader_3 | 4940 | 4940 (seq 0..4939) | 0 | 0 | 0 | 0/0 | 0/0 | 0 | 100/99.98 % | 0.33/0.32 |

reader_1 had 3 partner changes, with dt reaching ±5.56 ms, and all of its pairs were still received.

### Reader stopped with SIGSTOP (60 s)

`t2/reader_stopped_1`: the reader was attached, then stopped with SIGSTOP 10 s after start. It stayed stopped until the
capture ended.

The capture counters were unaffected:

- 4941 pairs; 0 drops, 0 overflows, 0 restarts.
- Paired 100/99.98 %; dt within tolerance.
- Callback p99 0.36/0.31 ms; latency 9.7/11.1 ms. These values are the same as in the baseline.

### Memory load (3 x 60 s, `cpuload` with 4 threads)

`rs_investigation/src/build/cpuload 4` ran during each run. Criteria: no crash, no deadlock, counters reported.

| Run | rc | Pairs | Drops | Overflows | Restarts | Paired A/B | cb99 A/B [ms] | cbmax A/B [ms] | max fill | lat99 A/B | CPU | Tj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| memload_1 | 0 | 4942 | 0/0 | 0/0 | 0 | 100/100 % | **2.19**/**0.74** | 20.43/1.48 | 2/2 | 12.3/10.0 | 72.5 % | 50.5 °C |
| memload_2 | 0 | 4940 | 0/0 | 0/0 | 0 | 100/99.98 % | 0.35/**0.74** | 2.64/3.29 | 2/2 | 9.6/11.4 | 72.3 % | 52.9 °C |
| memload_3 | 0 | 4941 | 0/0 | 0/0 | 0 | 100/100 % | **0.96**/0.37 | 9.35/1.50 | 1/1 | 9.8/10.2 | 72.3 % | 53.9 °C |

The test passes: every run exited cleanly, finished on time and reported its counters.

**R2 (callback p99 < 0.5 ms) is not met under this load.** In 4 of the 6 camera-runs p99 was 0.74-2.19 ms, and the callback
maximum was up to 20.4 ms. The callback is the 814,080-byte memcpy plus metadata reads, and it slows down when four threads
saturate memory bandwidth.

The ring absorbed this: maximum fill was 2 of 8 slots, with 0 overflows and 0 drops. The superseded first series had one
frame drop on camera B in `memload_2` under the same load (`~/capture_logs/t2_series1/memload_2`).

Without load, callback p99 was below 0.5 ms in all 29 runs (maximum 0.48 ms, in startstop_20).

### Start/stop (20 cycles of 10 s)

`t2/startstop_01` to `startstop_20`. In all 20 cycles:

- rc = 0, and the run ended within its 60 s timeout.
- Both cameras delivered frames.
- Pairs were supplied: 443-444 per cycle.
- 0 drops, 0 overflows, 0 restarts.
- `/d435_pairs` was removed, and no process remained.

In 12 of the 20 cycles B had 1 unpaired frame, which still meets the criterion (paired ≥ 99.775 %). The likely cause is the
following: the first frame after both cameras have settled finds its nearest partner already discarded during the settle
period. I did not verify this.

Two cycles had partner changes (startstop_14: 2, startstop_18: 1), because their phase was at the half-period boundary.

The `fps` column in `evaluate.out` is frames divided by run time, measured from program start. In 10 s runs this reads low
for B (89.3), because B's stream starts later than A's. It is not the frame-rate monitor: that measures over 2 s from frame
arrival times, and gave 0 warnings in every run.

### Watchdog

`t2/watchdog_1`: `--test-ignore A:20:1` makes camera A's callback ignore frames from 20 s to 21 s after start.

- **Detection:** at +20.2 s the watchdog reported "camera A: no frame for 158 ms (> 10 frame periods)". That is 10 periods
  = 111 ms, plus up to 50 ms from the supervisor's poll interval.
- **Restart:** stop, close, open and start of camera A took 465 ms, ending at epoch 1. 42 frames were ignored before the
  stream was stopped.
- **Camera B:** B continued without interruption (0 drops, 0 overflows). Its frames had no partner while A was down: 88
  were counted unpaired (87 by timeout). During A's 5 s re-settle 453 more were counted `partner_settling`.
- **Pairs:** after the re-settle, pairs continued with a new phase: dt 3.3-4.7 ms against 0.5-1.2 ms before the restart.
  In total there were 4401 pairs, and pairs were in order (order_errors = 0).
- **Counters:** restarts A/B = 1/0, 0 frame-rate warnings, 0 relevant kernel lines.

Paired was 100 % for A and 98.04 % for B. This test has no paired-percentage criterion: B's unpaired frames are the frames
captured while A was down.

## Superseded first T2 series

The first full T2 series (`~/capture_logs/t2_series1`) also passed every test. During it the frame-rate check gave false
warnings although no frame was dropped:

- `long_1`: 12 warnings.
- `reader_2`: 1 warning.
- `startstop_11`: 1 warning.

The check counted frames in 1 s windows, which has a resolution of ±1 fps. It was replaced by an arrival-time monitor over
2 s with a unit test (DEVIATIONS 23), and the whole T2 suite was run again. This report uses that second series, in
`~/capture_logs/t2` and `test_logs/t2`.

## Gap closure (second task)

All logs of this section are on the Jetson in `~/capture_logs/{gap,t2v2,t2v2_reader_rerun,unit}` and copied to
`test_logs/gap/`, `test_logs/t2v2/`, `test_logs/t2v2_reader_rerun/` (without `pairs*.csv`). Gap tests:
`tests/acceptance/gap_tests.sh`, evaluated by `tests/acceptance/evaluate_gaps.py` (`gap/evaluate_gaps.out`,
`gap/gap_verdict.json`). Regression: `tests/acceptance/t2_all.sh` and `evaluate.py` with the unchanged criteria
(`t2v2/evaluate.out`, `t2v2/verdict.json`).

| Gap | Result | Key values | Log |
|---|---|---|---|
| 1 Writer liveness and reattach | **Closed** | SIGKILL detected after 0.995-1.194 s (limit 2 s); new writer's first pair received 2.0-2.7 ms after its publish (limit 2 s); 0 missing in all 6 writer instances; no object left after the clean stop | `gap/kill_1`, `gap/unit/` |
| 2 Real hardware reset | **Closed** (cable test written, not run) | 20/20 resets seen in the kernel log; pairs again 7.6-9.1 s after re-enumeration (limit 15 s); other camera 0 drops, 0 overflows; RSS end -15.5 % vs. after first recovery; 0 order errors, 0 frames in two pairs | `gap/hwreset_1` |
| 3 Output delay | **Closed** (reported) | Usual: older frame p50/p99 12.1/15.0 ms, newer 9.8/11.1 ms. Near half a period: older 20.8/21.8 ms, newer 15.5/16.5 ms (+8.7 / +5.7 ms at p50) | `t2v2/long_1` |
| 4 Unpaired frame at startup | **Closed for the startup cause; partly closed by the criterion's wording** | Startup: 0 of 20 cycles (before: 12 of 20). 3 cycles had 1 unpaired frame each, all at a half-period crossing in mid-run. Pass if "near half" means any pair of the cycle; 1 failing cycle (`startstop_20`) if it means the cycle's median dt | `t2v2/startstop_*`, `gap/gap_verdict.json` |
| 5 Sanitizers | **Closed** | TSan, ASan + UBSan + LSan: 20/20 tests each, 0 reports; stress test 1 writer + 3 readers x 30 s: 0 accepted reads with mixed data | `gap/unit/`, `gap5_*` |
| Layout v2 / v1 rejection | Done | v2 reader rejects a v1 object with a version message; the saved v1 reader rejects a v2 object ("unsupported layout version or size") | `gap/version_check/` |
| Regression | **1 failure** | All tests pass except the reader test: in `reader_1` camera A's callback p99 was 0.520 ms (limit 0.5 ms). A diagnostic rerun of the 3 reader runs passed (0.32-0.43 ms). | `t2v2/`, `t2v2_reader_rerun/` |

### Gap 1: a reader cannot detect a stopped or restarted writer

**Cause.** Layout version 1 had only the writer's PID and a "stopped" state. A killed writer never set "stopped", and
a PID cannot tell a dead writer from a reused PID. A reader had no way to find a new writer of the same name.

**Change.** Layout version 2 (README.md "Shared memory"; DESIGN.md section 5):
- a random instance id per writer start;
- a heartbeat (`CLOCK_MONOTONIC`, every 50 ms, own thread);
- a "replaced" state: a new writer marks an old object of the same name and wakes its readers before it removes the
  name.

`ShmReader::next` returns `WriterStopped`, `WriterNotAlive` (heartbeat older than 1 s, checked at least every 200 ms
while no pair arrives) or `WriterReplaced`, and `reattach()` attaches again by name. `d435_pair_reader` reattaches
automatically and reports the attaches and each instance (id, pid, received, missing, gaps, first-pair delay, end).

**Tests.**
- Unit test `shm_v2_killed_writer_detected_and_new_writer_followed`. A helper writer process is SIGKILLed; detection
  took 1.006 s (normal build, Jetson), 1.075 s (TSan), 1.018 s (ASan), 1.011 s (Mac). A new writer starts, and the
  same reader object gets pair 0 2-3 ms after its publish, with 60 pairs and 0 gaps. A clean stop gives
  `WriterStopped` and removes the object.
- `shm_v2_replaced_writer_followed`: a second writer starts while the first runs. The reader gets `WriterReplaced`,
  follows the new writer from pair 0, and the old writer's clean stop leaves the new name alone.
- Jetson (`gap_tests.sh kill`): capture with a reader attached, SIGKILL 20 s after each start, restart 3 s later;
  5 kills, then a 30 s run that stops cleanly.

| Writer | End | Detected after SIGKILL | First pair (seq) | First pair delay | Received | Missing / gaps |
|---|---|---|---|---|---|---|
| 1 | not alive | 1.194 s | 0 | 2.6 ms | 1062 | 0 / 0 |
| 2 | not alive | 0.995 s | 0 | 2.0 ms | 1069 | 0 / 0 |
| 3 | not alive | 0.996 s | 0 | 2.2 ms | 1059 | 0 / 0 |
| 4 | not alive | 0.997 s | 0 | 2.4 ms | 1057 | 0 / 0 |
| 5 | not alive | 1.002 s | 0 | 2.0 ms | 1062 | 0 / 0 |
| 6 | stopped cleanly | - | 0 | 2.7 ms | 2240 | 0 / 0 |

Reader: 6 attaches, 6 different instance ids, exit code 0. Writers 2-6 printed "replaced old shared-memory object".
After the clean stop: `shm_left=0`, no process left (`gap/kill_1/meta.txt`).

**Result: pass.**

### Gap 2: a real hardware reset test

**Change.**
- Option `--test-hw-reset SEQ:FIRST:INTERVAL` calls `rs2::device::hardware_reset()` on the streaming camera. This works
  without sudo (RSUSB backend).
- The watchdog restarts the camera after 2 s. The restart releases the old handles, finds the camera by serial, and
  repeats all startup checks (`open=full` in the events).
- `query_devices`, `open`, `start` and the reset catch every exception.
- The status line shows each camera's condition (`reset`, `restarting`, `absent`, `settling`, `ok`).
- `[event]` lines carry both clocks, both cameras' drop and overflow counters, and RSS.

**Test.** `gap_tests.sh hwreset`: one 480 s run with 10 resets of A, then 10 of B, and a reader attached. The kernel
log (`kernel_mono.log`) shows each reset 1.08-1.29 s after the request, in one of two forms:
- 7 resets: "USB disconnect" and then "new SuperSpeed USB device number N", that is, a new address.
- 13 resets: "reset SuperSpeed USB device number N", where the USB core reset the port with the same address and
  uvcvideo probed the device again (DEVIATIONS 46).

| Criterion | Value | Limit | |
|---|---|---|---|
| Program does not stop or deadlock | exit 0, "duration reached", 20/20 resets sent, 0 reset failures, 0 recovery timeouts, 0 callback errors | | pass |
| Pairs resume after re-enumeration | A 7.96-9.13 s, B 7.61-9.10 s (includes the 5 s settle) | <= 15 s | pass |
| Other camera during the resets | B during A's resets: 0 drops, 0 overflows. A during B's resets: 0 drops, 0 overflows | 0 | pass |
| RSS | 72,980 kB after the first recovery, 61,652 kB at the end (-15.5 %) | <= +5 % | pass |
| Pair order, no frame twice | 26,018 pairs: 0 sequence errors, 0 time-order errors, 0 duplicate frames | 0 | pass |
| Status line shows the condition | 20 status lines with `reset` or `settling` | | yes |
| Reader | 1 attach, 26,018 received, 0 missing | | |

Per reset (request -> kernel line -> first pair of the new epoch): `gap/gap_verdict.json`, `gap2_hwreset_1.per_reset`.

Observations:
- **A trial run before the formal test had 4 dropped frames on camera A during camera B's reset**
  (`gap/gap2_trial/stdout.log`: `dropA` changes from 0 to 4 between B's `hw_reset_sent` and `restarted`). The drops
  fall between B's reset command and B's USB disconnect (kernel 61330.25 s), before the program does anything on A. The
  formal run of 20 resets had none. The cause is not known; both cameras are on the same hub (2-1.1 and 2-1.3).
- In the 13 port-reset cases the kernel logged "usbfs: process ... (d435_capture) did not claim interface 0 before
  use", 1,365 lines in total, each burst lasting 1-2 s after the reset. The SDK prints errors during each reset
  ("uvc streamer watchdog triggered", "failed to claim usb interface ... BUSY" for the other, open camera). None of
  them stopped the program.
- The manual cable test `tests/acceptance/manual_unplug.sh` is written but was not run: it needs a person at the Jetson.

**Result: pass** (the cable test was not run).

### Gap 3: output delay

**Change.** `PairRecord` has `publish_mono_ns` and `publish_real_ns`, taken just before the copy into shared memory.
The output delay of a frame is publish `CLOCK_REALTIME` minus its SDK global timestamp. It is collected for the older
and the newer frame in 0.1 ms bins over 0-60 ms, separately for usual pairs and for pairs with
`| |dt| - P/2 | <= 0.5 ms`. It appears:
- in the status line (`out ...`);
- in the final counters and JSON (`output_delay`);
- in `--delay-hist-out` (`delay_hist.csv` in every run directory).

**Result** (long run, 45 min, 242,178 pairs; `t2v2/long_1/stdout.log`, `delay_hist.csv`):

| Pairs | Count | Older frame p50 / p99 / max | Newer frame p50 / p99 / max |
|---|---|---|---|
| Usual | 218,990 | 12.1 / 15.0 / 21.4 ms | 9.8 / 11.1 / 20.7 ms |
| Within 0.5 ms of half a period | 23,188 | 20.8 / 21.8 / 23.5 ms | 15.5 / 16.5 / 18.3 ms |

Your estimate:
- **"10-16 ms in usual operation" holds approximately.** The older frame is 12.1 ms at p50 and 15.0 ms at p99. The
  newer frame is 9.8 ms at p50, slightly below 10 ms, and 11.1 ms at p99. Values change with the phase between the
  cameras: older p50 was 14.3-14.4 ms in runs with |dt| about 4.5 ms (`t2v2/reader_3`, `gap/version_check`).
- **"About 11 ms more near half a period" does not hold as stated.** The increase at p50 is +8.7 ms for the older frame
  and +5.7 ms for the newer frame. Near half a period the matcher waits for the next frame of the earlier stream, which
  arrives about half a period (5.6 ms) after the newer frame; that matches the +5.7 ms.

The delay ends when the pair is handed to the writer; the copy into shared memory adds about 0.3 ms.

### Gap 4: one unpaired frame at startup in 12 of 20 start cycles

**Cause, with evidence.**
- In the first series (`test_logs/gap4_startstop_series2_analysis.log`) the unpaired frame was always camera B's
  first eligible frame, in exactly the 12 cycles where B's frame was later than A's, and in none of the 7 with B
  earlier.
- The camera that settles second starts with frame y0. Its partner from the first camera had arrived just before, and
  was discarded as `partner_settling` because y0's camera was not yet settled. Then y0 had no partner within the
  tolerance.
- The new unit test failed on the old engine (`test_logs/gap4_before_fix_unit.log`).

**Change** (DESIGN.md 4.2): frames whose partner camera is still settling are held for up to 4 frames or 3 periods,
and a frame older than the other camera's first eligible frame that finds no partner is counted `partner_settling`.
Unit test `matcher_g_first_pair_after_settle` covers start offsets of ±0.06 s and ±0.5 s, 8 phase offsets, and a
restart of B, with 0 unpaired frames.

**Test** (regression start/stop, 20 cycles; `gap/gap_verdict.json`, `gap4_startstop`):
- No cycle had an unpaired frame at startup.
- 3 cycles had one unpaired frame each. In all three it was between two pairs in mid-run, where dt crossed half a
  period (a partner change):

| Cycle | Median dt | Closest dt to half a period | Unpaired | Where |
|---|---|---|---|---|
| startstop_06 | -5.268 ms | 0.003 ms | B 1 | between pairs 380/381, dt -5.561 -> +5.561 ms |
| startstop_10 | -5.485 ms | 0.002 ms | A 1 | between pairs 162/163, dt +5.562 -> -5.562 ms |
| startstop_20 | +4.810 ms | 0.010 ms | B 1 | between pairs 9/10, dt -5.554 -> +5.544 ms |

The criterion says "every cycle whose dt is more than 0.5 ms from half a period". In `startstop_20` dt started 0.13 ms
from half a period, crossed it, and drifted to +4.8 ms within the 5 s of pairing.
- If a cycle counts as near half a period when **any** of its pairs is within 0.5 ms (4 cycles), all 20 cycles pass.
- If it counts by the cycle's **median** dt (3 cycles), `startstop_20` fails.

The evaluator reports both readings (DEVIATIONS 56).

**Result:** the startup cause is closed (0 of 20 instead of 12 of 20). The criterion passes under the any-pair reading
and fails in 1 cycle under the median reading. The remaining unpaired frames come from the known half-period crossing
(the matcher can leave one frame unpaired there), which this task did not ask to change.

### Gap 5: sanitizer coverage of shared memory

**Change.** The seqlock-protected copies go through `seqlock_copy`. Under TSan it is a `no_sanitize("thread")`
volatile word loop and is the only code excluded from TSan. Tests that need TSan to see both sides attach the reader to
the writer's own mapping. New CMake option `CAPTURE_ASAN` adds AddressSanitizer and UndefinedBehaviorSanitizer (with
LeakSanitizer on Linux). The first TSan run reported a double lock and a race in the heartbeat thread. Both came from
`condition_variable::wait_for`, which GCC 11's libtsan cannot follow; the heartbeat now uses the futex `EventSignal`
instead (DEVIATIONS 42). No suppression file is used.

**Stress test** `shm_v2_stress_1_writer_3_readers`: 1 writer with no pause and 3 readers (no pause, 1 ms pause, 5 ms
pause with a 2 ms delay inside every 4th read) for 30 s. Every 64-bit word of both images and the record is checked
against a per-pair pattern. A negative control in the test confirms that a half-mixed image is detected.

| Build (where) | Tests | Sanitizer reports | Writer pairs in 30 s | Accepted reads A/B/C | Reads that overlapped a write (discarded) | Accepted with mixed data | Log |
|---|---|---|---|---|---|---|---|
| normal (Jetson) | 20/20 | - | 84,600 | 43,958 / 13,901 / 4,524 | 76,227 | 0 | `gap/unit/unit_normal.log` |
| TSan (Jetson, GCC 11) | 20/20 | 0 | 5,920 | 5,920 / 5,920 / 3,201 | 567 | 0 | `gap/unit/unit_tsan.log` |
| ASan + UBSan + LSan (Jetson) | 20/20 | 0 | 42,798 | 32,842 / 14,244 / 4,447 | 22,119 | 0 | `gap/unit/unit_asan.log` |
| UBSan (Mac) | 20/20 | 0 | 81,258 | 73,494 / 15,661 / 3,544 | 10,754 | 0 | `gap5_ubsan_mac.log` |
| normal (Mac) | 20/20 | - | 154,782 | 112,794 / 14,037 / 3,529 | 39,815 | 0 | `gap_unit_mac.log` |

ASan on the Mac hangs before `main` (`gap5_asan_mac_hang_sample.log`, DEVIATIONS 41). Two tests were changed so that
they keep their meaning in the slow sanitizer builds, with the same pass criteria (DEVIATIONS 43, 45).

**Result: pass.**

### Layout version 2: v1/v2 rejection

`gap/version_check/`:
- v2 reader on a v1 object (saved v1 `d435_capture`): "cannot attach to /d435_pairs: shared-memory layout version 1,
  but this reader supports only version 2 (writer is an older d435_capture)", exit 3.
- Saved v1 reader on a v2 object: "cannot attach to /d435_pairs: unsupported layout version or size", exit 3. The v1
  message does not name the versions; it is compiled into the v1 binary.

The unit test `shm_v2_rejects_v1_object` checks the v2 message.

### Regression (same criteria as the first task)

Series started 09:39 Jetson time (`t2v2/run.out`, `t2v2/evaluate.out`). Every run recorded `tegrastats` and new kernel
lines: 0 relevant kernel lines in all regression runs.

| Test | Result | Key values | Log |
|---|---|---|---|
| Unit tests, Mac | PASS | 20/20 | `gap_unit_mac.log` |
| Unit tests, Jetson | PASS | 20/20 (also TSan and ASan, see gap 5) | `gap/unit/` |
| Baseline 3 x 60 s | PASS | 0 drops, 0 overflows, 0 restarts; paired 99.96-100 %; cb99 0.30-0.43 ms | `t2v2/baseline_{1,2,3}` |
| Reader attached 3 x 60 s | **FAIL** | Reader got every pair (4941/4942/4943, 0 missing, 0 discarded reads). **`reader_1`: camera A callback p99 0.520 ms > 0.5 ms**, in its first 20 s (interval p99 0.56 and 0.52 ms, then 0.36 ms). Diagnostic rerun of 3 runs: PASS, cb99 0.32-0.43 ms (`t2v2_reader_rerun/`). | `t2v2/reader_{1,2,3}` |
| Reader stopped 60 s | PASS | 0 drops, 0 overflows, 100 % paired | `t2v2/reader_stopped_1` |
| Memory load 3 x 60 s | PASS | no crash or deadlock. cb99 up to 1.96 ms (R2 not met under load, as in the first task). `memload_2`: 2 dropped frames on B (the first task had none). | `t2v2/memload_{1,2,3}` |
| Start/stop 20 cycles | PASS | 20/20: pairs supplied, no stall, no restart, shm removed (gap 4 above) | `t2v2/startstop_01..20` |
| Long run 45 min | PASS | 242,178 pairs; 0 drops, 0 overflows, 0 restarts; paired 99.992 % / 99.993 %; 35 partner changes; cb99 0.36/0.37 ms | `t2v2/long_1` |
| Watchdog | PASS | A restarted once; pairs continued | `t2v2/watchdog_1` |
| Cleanup (rule 8) | PASS | after all 32 runs: no process and no shm object of mine | `t2v2/*/meta.txt` |

The `reader_1` failure is a callback-time overrun of camera A for about 20 s. The callback code did not change in this
task. The new per-process work is the heartbeat thread (one store every 50 ms) and the status-line and delay
statistics in the supervisor and matcher, not in the callback. The cause is not known; the failure is reported, not
retried away.

## Changes on the Jetson

- `~/capture`: copy of this source tree, the build `~/capture/build`, the ThreadSanitizer build `~/capture/build-tsan`, the
  AddressSanitizer/UBSan build `~/capture/build-asan` (gap closure), and `~/capture/capture.conf` (copied from
  `capture.conf.example`).
- `~/capture_logs`: all logs (builds, T1, smoke and try runs, `t2_series1`, `t2`). Gap closure: `v1_bin/` (copies of the
  version 1 `d435_capture` and `d435_pair_reader` for the rejection check), `gap2_trial/`, `gap/`, `unit/`, `t2v2/`,
  `t2v2_reader_rerun/`, `gap_run.out`, `t2v2_run.out`.
- Nothing else changed:
  - No sudo, no package changes, no firmware, kernel, reboot or clock change.
  - No system configuration was changed.
  - `rs_investigation/` was only read, and `rs_investigation/src/build/cpuload` was run.
- At the end no process of mine runs on the Jetson, and `/dev/shm` contains no object of mine. The remaining entries
  (`nvsci*`, `ipc_test*`, `itc_test*`) belong to root.
