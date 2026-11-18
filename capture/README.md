# d435_capture

Captures 848x480 Z16 depth at 90 fps from two free-running Intel RealSense D435i cameras on a Jetson Orin Nano, pairs the
frames by timestamp (mutual nearest, tolerance half a frame period, about 5.56 ms), and publishes the pairs in POSIX shared
memory for a second process. There is no hardware sync: the two frames of a pair are 0 to 5.6 ms apart in capture time.

Design: [DESIGN.md](DESIGN.md). Tests: [TEST_REPORT.md](TEST_REPORT.md). Deviations from the task: [DEVIATIONS.md](DEVIATIONS.md).

## Build

Requirements: CMake >= 3.16, a C++17 compiler, librealsense2 (only for `d435_capture`). No other libraries.

On the Jetson (librealsense 2.59.1 in `~/.local`):

```
cd ~/capture
cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/.local -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -C build -j6
```

Targets: `d435_capture` (needs librealsense2), `d435_pair_reader` (example reader), `capture_tests` (unit tests),
library `capture_core` (camera-independent parts). Warnings are on (`-Wall -Wextra`) and treated as errors
(`-DCAPTURE_WERROR=OFF` to relax).

Without librealsense (for example on macOS) only the core library, the tests and the reader are built:

```
cmake -S . -B build-mac -G Ninja -DCMAKE_DISABLE_FIND_PACKAGE_realsense2=ON && ninja -C build-mac
./build-mac/capture_tests            # all unit tests; ./capture_tests <substring> runs a subset
```

Sanitizer builds: `-DCAPTURE_TSAN=ON` (ThreadSanitizer) or `-DCAPTURE_ASAN=ON` (AddressSanitizer + UndefinedBehaviorSanitizer,
with LeakSanitizer on Linux). Use them on Linux; on this Mac both runtimes hang or crash at startup (DEVIATIONS.md 1 and 41),
UBSan alone works there.

## Configuration

`capture.conf` (default name, in the working directory; `--config FILE` for another), `key=value`, `#` comments:

```
A=234322070408          # serial number of camera A (also accepted: serial_a)
B=234322070144          # serial number of camera B (also accepted: serial_b)
# optional: sink=shm|stats  shm_name=/d435_pairs  status_interval=10  pair_csv=FILE  hist_out=FILE  delay_hist_out=FILE
```

`capture.conf.example` is a template. The format of `rs_investigation/cams.conf` is accepted as is.

## Run

```
./build/d435_capture                       # until Ctrl-C / SIGTERM
./build/d435_capture --duration 60 --pair-csv pairs.csv --final-json final.json
./build/d435_pair_reader                   # in another terminal
```

| Option | Default | Meaning |
|---|---|---|
| `--config FILE` | `capture.conf` | configuration file |
| `--sink shm\|stats` | `shm` | shared memory, or a statistics sink that only counts |
| `--shm-name NAME` | `/d435_pairs` | POSIX shared-memory name |
| `--duration S` | 0 (no limit) | stop after S seconds (same clean shutdown as SIGINT) |
| `--status-interval S` | 10 | status line period |
| `--pair-csv FILE` | off | one CSV line per pair, written by a separate non-critical thread |
| `--hist-out FILE` | `latency_histograms.csv` | latency histograms (0.1 ms bins, 0-40 ms), written at shutdown |
| `--delay-hist-out FILE` | `output_delay_histograms.csv` | output delay histograms (0.1 ms bins, 0-60 ms), written at shutdown |
| `--final-json FILE` | off | final counters as JSON |
| `--affinity LIST` | off | e.g. `matcher=2,supervisor=3,csv=3,callback_a=4,callback_b=5` |
| `--test-ignore CAM:AT:FOR` | off | test only: the callback of camera CAM ignores frames from AT s after start for FOR s |
| `--test-hw-reset SEQ:FIRST:INTERVAL` | off | test only: real hardware resets (`rs2::device::hardware_reset`) of the cameras in SEQ (e.g. `AAAAABBBBB`), the first FIRST s after start, each next one INTERVAL s after the previous camera delivers pairs again |

Startup refuses to run (exit code 3, message on stderr) if a camera is missing, its `usb_type_descriptor` does not start
with `3.`, its firmware is older than 5.17.3.10, an option does not read back as set, or 848x480 Z16 @90 is not
available. Options set and read back on both cameras: `global_time_enabled=1`, `inter_cam_sync_mode=0`,
`enable_auto_exposure=1`, `frames_queue_size=16`.

### Output

```
[status +40s] A[ok]: 89.97fps drop=0 ovf=0 fill=4/4 cb99=0.30 cbmax=0.31ms lat50=9.7 lat99=10.2ms | B[settling]: ... |
  pairs=1591 unpaired=312/331 dt[min/med/max]=3.22/3.24/3.26ms partner_changes=0 restarts=1/1 settled=1/1 P=11.1283ms
  | out old50/99=12.9/13.5 new50/99=9.7/10.2ms near_half n=0 old50/99=nan/nanms
```

Per camera: condition in brackets (`ok`; `reset`: hardware reset requested; `restarting`; `absent`: not found by serial,
retried every 1 s; `restart_failed`; `settling`: restarted, no pair yet), frame rate, dropped frames (frame-counter gaps), ring overflows, highest ring fill (interval/run), callback time
p99/max, delivery latency p50/p99 (arrival `CLOCK_REALTIME` minus SDK global timestamp). Matcher: pairs, unpaired frames
(A/B), dt = t(B) - t(A) min/median/max in the interval, partner changes, camera restarts, settle state, measured period.
Output delay in the interval (`out`): publish time (`CLOCK_REALTIME` when the matcher hands the pair to the sink) minus the
SDK global timestamp of the older (`old`) and of the newer (`new`) frame of the pair, p50/p99, for usual pairs and
separately for pairs whose |dt| is within 0.5 ms of half a period (`near_half`, count and older-frame p50/p99; `nan` when
there was none). Watchdog restarts and frame-rate warnings are printed as `[watchdog ...]` / `[warning ...]` lines.
Condition changes of a camera are printed as machine-readable `[event ...]` lines (`frames_lost`, `restarted`,
`pairs_resumed`, `hw_reset_request`, ...) with both clocks, the epoch, the drop and overflow counters of both cameras, the
number of pairs and the RSS. At shutdown the final counters are printed (and written as JSON with `--final-json`),
including the output delay p50/p99/max per category and, with `--test-hw-reset`, the reset counters and RSS values.

Pairing starts 5 s after both streams started (settle period) and when both cameras deliver Global Time timestamps.
A frame without a partner after 3 frame periods is discarded and counted as unpaired.

## Shared memory (layout version 2)

Object: POSIX shared memory `/d435_pairs` (default), created by `d435_capture`, removed at its clean shutdown. Size
`512 + 4 * (512 + 2 * 814080)` = 6,515,200 bytes. All values little-endian, naturally aligned. Fields marked *atomic* are
written by the writer with release semantics while readers run; load them with acquire semantics (aligned 32/64-bit
loads on ARM64 and x86-64).

Version 2 adds writer liveness (instance id, heartbeat, "replaced" state) and the publish time in the pair record. Bytes
0-111 have the same meaning as in version 1, so a reader of either version can read `magic` and `layout_version` and
reject the other version: a version 2 reader prints "shared-memory layout version 1, but this reader supports only
version 2 (writer is an older d435_capture)"; the version 1 reader prints "unsupported layout version or size".

### Header (offset 0, 512 bytes)

| Offset | Type | Field | Value / meaning |
|---|---|---|---|
| 0 | u64 | magic | `0x5249415035333444` (bytes `D435PAIR`); written last (release) when the header is ready |
| 8 | u32 | layout_version | 2 |
| 12 | u32 | header_size | 512 |
| 16 | u32 | slot_count | 4 |
| 20 | u32 | image_width | 848 |
| 24 | u32 | image_height | 480 |
| 28 | u32 | bytes_per_pixel | 2 |
| 32 | u64 | image_bytes | 814080 |
| 40 | u64 | slot_stride | 1628672 |
| 48 | u64 | slots_offset | 512 |
| 56 | u64 | total_size | 6515200 (the object may be larger, rounded to pages) |
| 64 | i32 | writer_pid | information only; not a liveness signal (PIDs are reused, and a PID may be in another namespace) |
| 68 | u32 atomic | writer_state | 1 running, 2 stopped cleanly, 3 replaced by a new writer |
| 72 | u64 atomic | pairs_published | number of pairs written; pair n is available when `pairs_published > n` |
| 80 | u32 atomic | notify_word | incremented after each pair and at each state change; futex word (`FUTEX_WAIT`, not private) |
| 84 | u32 | record_offset | 64 (record offset within a slot) |
| 88 | u32 | image_offset | 512 (image A offset within a slot) |
| 92 | u32 | record_size | 128 |
| 96 | f64 | frame_period_ms | measured frame period (valid once `pairs_published > 0`) |
| 104 | f64 | tolerance_ms | pairing tolerance (half the period) |
| 112 | CameraInfo[2] | cam | 96 bytes each: A at 112, B at 208 |
| 304 | u64 | writer_instance_id | random, non-zero, different for every start of the writer |
| 312 | u64 atomic | heartbeat_mono_ns | `CLOCK_MONOTONIC` (ns) of the last heartbeat |
| 320 | u64 atomic | heartbeat_count | +1 per heartbeat |
| 328 | u32 | heartbeat_interval_ms | 50: the writer updates the heartbeat at least this often (own thread, not the capture path) |
| 332 | u32 | liveness_timeout_ms | 1000: heartbeat older than this means the writer is not alive (killed or hung) |
| 336 | i64 | writer_start_real_ns | `CLOCK_REALTIME` at creation (information) |
| 344 | u64 atomic | replaced_by_instance | instance id of the writer that replaced this one (state 3) |
| 352 | | reserved | to 512 |

CameraInfo (96 bytes): `char serial[32]` (NUL-terminated), `f32 depth_scale` (metres per unit) at 32, `u32 width` 36,
`u32 height` 40, `f32 ppx` 44, `f32 ppy` 48, `f32 fx` 52, `f32 fy` 56, `i32 distortion_model` 60 (rs2_distortion),
`f32 coeffs[5]` 64, reserved 84-95.

### Slots (offset `512 + k * 1628672`, k = 0..3; pair n is in slot n mod 4)

| Offset in slot | Type | Field |
|---|---|---|
| 0 | u64 atomic | seq: odd = being written; `2n + 2` = slot holds complete pair n |
| 64 | PairRecord | 128 bytes |
| 512 | u16[480][848] | depth image of camera A (multiply by depth_scale for metres; 0 = no data) |
| 512 + 814080 | u16[480][848] | depth image of camera B |

PairRecord (128 bytes): `u64 pair_seq` at 0 (0, 1, 2, ... for each writer instance), `f64 dt_ms` (t(B) - t(A)) at 8, then
two CamRecords (A at 16, B at 64), each 48 bytes: `f64 global_ts_ms` (SDK global timestamp, ms since 1970 in the Jetson's
`CLOCK_REALTIME` base) 0, `i64 hw_ts_us` (metadata frame timestamp, camera clock) 8, `i64 sensor_ts_us` 16,
`i64 arrival_mono_ns` (`CLOCK_MONOTONIC` at callback entry) 24, `i64 frame_counter` 32, `i32 exposure_us` 40,
`i32 domain` 44 (2 = Global Time). Missing metadata is -1. Then `i64 publish_mono_ns` at 112 and `i64 publish_real_ns`
at 120: `CLOCK_MONOTONIC` / `CLOCK_REALTIME` when the matcher handed the pair to the writer (just before the copy into the
slot). `publish_real_ns / 1e6 - global_ts_ms` is the output delay of that frame.

### Attach sequence

1. `fd = shm_open(name, O_RDONLY)`. Missing: no writer yet; retry later (the example reader retries every 100 ms).
2. `fstat(fd)`; if the size is below 512 the writer is still creating the object: close and retry.
3. `mmap(size, PROT_READ, MAP_SHARED)`, `close(fd)`.
4. Load `magic` (acquire). Not `0x5249415035333444`: header not ready yet; unmap and retry.
5. `layout_version` must be 2. Any other value: reject with a clear message (do not retry; it is another program version).
6. Check `header_size == 512`, `slot_count >= 1`, `record_size >= 128`, `slots_offset + slot_count * slot_stride <= size`.
7. Remember `instance = writer_instance_id`. Start with `next = pairs_published` (new pairs only) or
   `max(0, pairs_published - slot_count)` (oldest available).

The writer creates the object before the cameras start and publishes pair 0 after the settle period (about 5 s later), so a
reader that attaches while the writer starts gets every pair from pair 0.

### Read loop

- `w = notify_word` (acquire), then `pub = pairs_published` (acquire). In this order: a pair published after the first
  load also changes `notify_word`, so no notification is lost.
- If `writer_state == 3` (replaced): stop using this mapping; the pairs `next .. pub-1` are not read (missing); go to
  "Reattach".
- If `next < pub`: if `pub - next > slot_count`, the pairs `next .. pub-slot_count-1` are gone (missing); set
  `next = pub - slot_count`. Read slot `next % slot_count`: `s1 = seq` (acquire). If `s1 != 2*next + 2`, pair `next` was
  overwritten: missing, `next++`. Else copy the record and the images, then an acquire fence, then `s2 = seq`. If
  `s2 != s1` the writer changed the slot during the copy: discard the read (missing), `next++`. Else the copy is pair
  `next`: use it, `next++`.
- Else (nothing new): if `writer_state == 2` the writer stopped cleanly (every pair was read): exit, or go to "Reattach"
  to wait for the next writer. Otherwise do the liveness check; if alive, wait:
  `futex(&notify_word, FUTEX_WAIT, w, timeout)` with a timeout of at most 200 ms, so that the liveness check runs at least
  every 200 ms while no pair comes. (No polling while pairs arrive; the futex wakes the reader for every pair.)

### Liveness check

The writer is alive when `writer_state == 1` and `now - heartbeat_mono_ns <= liveness_timeout_ms * 1e6`, where `now` is
`CLOCK_MONOTONIC` of the reader (same host, same clock). A heartbeat thread of the writer updates `heartbeat_mono_ns`
every 50 ms independently of the cameras, so a writer whose cameras deliver nothing (restart, unplugged cable) is still
alive. A writer killed with SIGKILL, or hung, is detected 1.0-1.25 s later (1 s timeout + up to 200 ms until the next
check). `writer_pid` is not used: it cannot tell a dead writer from a reused PID.

### Reattach

When the writer is not alive, was replaced, or stopped and the reader wants to follow the next writer: keep the old
mapping, and every 100 ms run the attach sequence again on the same name. Accept the new object only if its
`writer_instance_id` differs from the old instance; then unmap the old mapping and continue with the new one (`pair_seq`
starts at 0 again). If the name still leads to the old instance (the old writer died and no new writer exists yet), keep
waiting; if the old writer is alive again (it was only stopped, for example by a debugger), continue with the old mapping.

A new writer that finds an object of the same name (left by a writer that crashed, or a writer that still runs) first sets
`replaced_by_instance` and `writer_state = 3` in the old object and wakes its readers (`notify_word += 1`, futex wake), then
removes the name and creates its own object. A version 1 object is set to `writer_state = 2` instead (version 1 readers
know only "stopped"). An old writer that was replaced does not remove the name when it stops.

### Example reader

`tools/pair_reader.cpp` (`d435_pair_reader`; class `ShmReader` in `src/shm_sink.*`) implements all of this. It reattaches
automatically after "not alive" and "replaced" (with `--follow` also after a clean stop), and reports the number of
attaches and, for each writer instance, the instance id, writer pid, pairs received, first and last sequence number,
missing numbers, gaps in the received numbers, discarded reads, the delay of its first pair, and how it ended
(`--json FILE` for a machine-readable summary, `--slow-ms N` to simulate a slow consumer, `--reattach-wait S` (default 30)
for how long to wait for a new writer).

The writer never waits for readers. A reader that is too slow loses pairs (it sees them as missing) but never gets a mixed
or partial pair, and never slows down the capture.

## Tests

- Unit tests (no camera): `capture_tests` — ring buffer (10 million items, two threads), pairing engine (synthetic cases
  a-g), threaded pipeline, shared-memory sink with slow and stopped readers, writer liveness (killed writer, new writer,
  replaced writer, clean stop; helper writer processes are the test binary started with `--helper-shm-writer`), version 1
  rejection, a 1-writer/3-reader stress test (`CAP_STRESS_S`, default 30 s), frame-rate monitor.
- Acceptance tests on the Jetson (two cameras): `tests/acceptance/t2_all.sh [test ...]`, evaluated with
  `tests/acceptance/evaluate.py ~/capture_logs/t2`. Each run records `tegrastats` and new kernel log lines.
- Gap-closure tests on the Jetson: `tests/acceptance/gap_tests.sh [kill] [hwreset]` (writer SIGKILL and restart with a
  reader attached; 10 + 10 hardware resets), evaluated with `tests/acceptance/evaluate_gaps.py ~/capture_logs/gap
  [REGRESSION_LOGROOT]` (also evaluates the start/stop cycles and the output delay of a regression series).
- Manual cable test: `tests/acceptance/manual_unplug.sh A|B [N]` (prompts to pull and replug the cable N times; same
  evaluation and criteria as the hardware-reset test).
- `tests/acceptance/sync_to_jetson.sh` (developer helper on the Mac): copies the sources to `~/capture` and builds.

## Files

| Path | Content |
|---|---|
| `src/common.hpp`, `frame_types.hpp` | constants, clocks, frame metadata and ring slot |
| `src/spsc_ring.hpp` | lock-free SPSC ring of preallocated slots |
| `src/event_signal.*` | futex-based wakeup (matcher), shared futex helpers (shared memory) |
| `src/producer.hpp` | work of the SDK callback: copy, publish, signal, statistics |
| `src/pairing_engine.*` | mutual-nearest pairing, settle, timeouts, partner changes (pure logic) |
| `src/timestamp_source.hpp` | `TimestampSource` interface, SDK global-time implementation |
| `src/matcher.*` | matcher thread |
| `src/pair_sink.hpp`, `shm_layout.hpp`, `shm_sink.*` | `PairSink`, statistics sink, shared-memory layout, writer and reader |
| `src/camera.*` | librealsense: startup checks, options, callback, restart |
| `src/status.*`, `src/fps_monitor.hpp` | supervisor: watchdog, hardware-reset test, recovery events, frame-rate check, status line, final report |
| `src/config.*`, `src/main.cpp` | configuration and program |
| `tools/pair_reader.cpp` | example reader |
| `tests/` | unit tests and acceptance scripts |
| `test_logs/` | copies of the test logs referenced in TEST_REPORT.md |
