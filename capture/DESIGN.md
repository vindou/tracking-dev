# d435_capture: design

Software-paired depth capture from two free-running Intel RealSense D435i cameras on a Jetson Orin Nano.
First version written before the code; this version agrees with the code as delivered (deviations: DEVIATIONS.md).

Measured basis: `rs_investigation/report/report.md` (main body). Values used as requirements:

| Value | Source (report section) | Use in this design |
|---|---|---|
| Frame period 11.128 ms (89.87 fps) on both cameras | exp 3 | Expected value; the real period is measured at runtime from hardware timestamps |
| Pairing tolerance = half a period (about 5.56 ms) | exp 6 | Mutual-nearest pairing tolerance |
| 0-7 frames in Hardware Clock domain at start; global-time error up to 7.8 ms in the first 3 s | exp 1, exp 6 | Settle: discard until Global Time and 5 s after stream start |
| Partner flips when the phase is near half a period | exp 6 | Matcher must stay correct; counts partner changes |
| Arrival gaps up to 31 ms under load, then back-to-back frames | exp 4d | 8 ring slots per camera |
| memcpy of 814 KB in the callback: p99 0.29-0.36 ms with two cameras | exp 8 | Copy in the callback into a preallocated slot |
| Callbacks of A and B run on different SDK threads and can overlap; one camera's own depth callbacks never overlap | exp 8 | One SPSC ring per camera; nothing written by both callbacks |
| Sync mode 4 without trigger: 0 frames, no error | exp 7 | Watchdog |

## 1. Threads

```
SDK thread A: Camera::on_frame -> FrameProducer::deliver -> ring A --\
                                                                      >-- EventSignal --> matcher thread -> PairSink (shm)
SDK thread B: Camera::on_frame -> FrameProducer::deliver -> ring B --/                        |
                                                                                              +--> CSV queue -> CSV thread
supervisor thread: watchdog / restart, frame-rate check, status line (reads atomics and a matcher snapshot)
main thread: startup, sigtimedwait(SIGINT, SIGTERM) or --duration, ordered shutdown, final report
```

| Thread | Owner | Does | Must not |
|---|---|---|---|
| Callback A, callback B | librealsense (one per sensor) | Read timestamp, domain, metadata, `CLOCK_MONOTONIC` and `CLOCK_REALTIME`; count counter gaps; copy depth into a ring slot; publish; signal the matcher; update atomic histograms | Allocate, lock, do I/O, block |
| Matcher | ours | Sleeps on the `EventSignal` (futex); drains both rings into the pairing engine; gives pairs to the sink; pops ring slots in FIFO order | Poll. It wakes on a callback signal or at the next timeout deadline only |
| Supervisor | ours | Every 50 ms: watchdog and frame-rate check; restarts one camera (also after a hardware reset or an unplugged cable: find by serial, all startup checks); hardware-reset test option; recovery events; every 10 s: status line | Touch ring contents |
| Shared-memory heartbeat | ours (`ShmSink`) | Every 50 ms: `heartbeat_mono_ns`, `heartbeat_count` in the shared header (gap 1) | Anything else; it is not the callback or the matcher, so the heartbeat continues while cameras restart |
| CSV writer (optional) | ours | Drains a 4096-entry SPSC queue of pair records into the CSV file | - |
| Main | ours | Startup, waits for a signal or the run duration, shuts down in order | - |

SIGINT and SIGTERM are blocked in all threads (the mask is set first in `main`, before any thread or `rs2::context`
exists) and accepted by `sigtimedwait` in the main thread. No real-time priorities. Optional CPU affinity (`--affinity`), off
by default; the SDK callback threads pin themselves on their first call when asked.

The matcher publishes its counters into a snapshot under a mutex that it only `try_lock`s (it never waits for the status
thread). The callbacks only touch atomics.

## 2. Ring buffer (`spsc_ring.hpp`, one per camera)

- Fixed capacity 8 slots, allocated at startup. `DepthSlot` = `FrameMeta` + 848 x 480 x 2 bytes (814,080) of depth,
  64-byte aligned.
- `head` (written only by the producer) and `tail` (written only by the consumer) are monotonically increasing 64-bit
  atomics on separate cache lines; slot index = counter mod capacity.
- Producer: `claim()` returns the slot at `head` if `head - tail < capacity` (tail loaded with acquire), else `nullptr` and
  the overflow counter is incremented: the producer discards the frame and never waits. `publish()` stores `head + 1` with
  release order and updates the highest fill level (run and interval).
- Consumer: `available() = head (acquire) - tail`; `peek(i)` returns the i-th unread slot without consuming it (the matcher
  looks at several frames before it decides); `pop()` stores `tail + 1` with release order. The consumer pops in FIFO order
  only.
- After a camera restart the producer runs on another SDK thread; the SDK has stopped the old callback before the new one
  starts, so there is still one producer at a time.

## 3. Event signal (`event_signal.*`)

`EventSignal`: a 32-bit atomic counter. `notify()`: increment (release), then `futex(FUTEX_WAKE_PRIVATE)`; never blocks,
no lock. Consumer: `seen = value(); drain everything; wait(seen, timeout)`. `wait` uses `futex(FUTEX_WAIT_PRIVATE)` and
returns at once if the counter has changed since `seen`, so no notification is lost. macOS (unit tests): dispatch semaphore.
The shared-memory notification uses the same counter idea with a process-shared futex (`FUTEX_WAKE` / `FUTEX_WAIT`).

## 4. Matcher

### 4.1 Input (`matcher.cpp`)
For each newly published ring slot the matcher builds a `FrameInfo`: camera, epoch (restart generation, written into the
slot by the callback), pairing time from the `TimestampSource`, whether that time is usable, hardware timestamp and frame
counter (period measurement and partner-change detection only), arrival time (`CLOCK_MONOTONIC`) and a pointer to the slot.
It keeps per camera the number of slots already given to the engine ("mirrored").

`TimestampSource` (`timestamp_source.hpp`): `bool pairing_time_ms(cam, meta, t_ms)`. First implementation:
`SdkGlobalTimestampSource` returns the SDK frame timestamp and accepts the frame only in the Global Time domain. A later own
clock model replaces it without changes to the matcher.

### 4.2 Pairing engine (`pairing_engine.*`, pure logic, unit-tested without a camera)
Per camera a FIFO queue (capacity 16 >= ring capacity) of `FrameInfo`. Invariant: every pushed frame leaves the engine
exactly once, in push order per camera, as part of a pair (`on_pair`) or as a discard with a reason (`on_discard`). The
matcher pops the corresponding ring slot on each of these calls, so ring slots are popped in FIFO order (an internal check
counts violations as `order_errors`; it is 0 in all tests).

**Settle.** Per camera and epoch: settle start = arrival of the first frame of the epoch. A camera becomes *settled* with the
first frame that has a usable time (Global Time) and arrived >= 5 s after the settle start; at that moment its frame period
is frozen. A frame is *eligible* when it belongs to the current epoch, its time is usable, its camera was settled when it
arrived, and the other camera is settled. Front frames that are not eligible are discarded with reason `settle`, `domain`
(no Global Time after settle), `partner_settling` or `restart`.
A new epoch of a camera (restart) discards that camera's queued frames (`restart`) and the other camera's queued frames
(`partner_settling`), resets the partner-change reference, and starts the settle period again for that camera.

*First pair after settle (gap 4).* The camera that settles second starts with its first eligible frame y0. Its partner
from the other camera may have arrived *before* y0's camera settled; such a frame used to be discarded as
`partner_settling` at once, and then y0 had no partner within the tolerance and was counted `unpaired` (the cause of the
one unpaired frame in 12 of 20 start/stop cycles: exactly the cycles in which B's frame was later than A's). Now:
(1) a front frame that would be discarded as `partner_settling` is held while the queue holds at most `hold_max_frames`
(4) frames and it waited less than the timeout (3 P), so that it can still pair with the other camera's first eligible
frame; (2) when the other camera has its first eligible frame (time `first_t`), a frame older than `first_t` that finds no
partner is counted `partner_settling`, not `unpaired` (it is older than anything the other camera can pair with).
Unit test `matcher_g_first_pair_after_settle`: start offsets of +-0.06 s and +-0.5 s, eight phase offsets, and a
restart of B; 0 unpaired.

**Frame period.** Per camera from metadata hardware timestamps and frame counters: `(hw_last - hw_first) / (counter_last -
counter_first)` over frames from 1 s after the stream start until the camera settles (at least 90 counter steps). The
engine period P is the mean of both cameras; tolerance = P / 2. Before both are measured (and if hardware timestamps are
missing) the expected 11.128 ms is used.

**Decision** (repeated while progress is possible; `step()`):
1. A front frame that is not eligible leaves with its reason.
2. Only one camera has a frame: if it waited >= 3 P since its arrival, it leaves as `unpaired`; else wait.
3. Both fronts eligible. x = the front with the earlier time (stream X), y = the other front, d = t(y) - t(x) >= 0,
   m = 0.5 ms (ambiguity margin):
   - d > P/2: x is farther than the tolerance from every remaining frame of Y -> x `unpaired`.
   - d <= P/2 - m: any later X frame is at least about P after x, so it is farther from y than x -> pair (x, y).
   - P/2 - m < d <= P/2: X's next frame x1 decides. If x1 exists and is eligible: t(x1) - t(y) < d -> x `unpaired` (y prefers
     x1); else pair (x, y). If x1 exists but is not eligible, pair (x, y). If x1 does not exist: wait until x waited 3 P,
     then pair (x, y).

Each queue is ordered by time and earlier frames have already left, so the earliest remaining frame of the other stream is
the only candidate for x; together with the check of x1 this is the mutual-nearest rule. Pairs leave in time order on both
streams, and a frame is used at most once (it leaves the queue when used). Pair differences are always within +-P/2.

**Timeout deadline.** `next_deadline_ns()` = earliest front arrival + 3 P. The matcher sleeps until that deadline or the next
callback signal.

**Partner change.** For each pair k = counter(B) - counter(A). Frame counters do not renumber on dropped frames, so k is
constant while the alignment of the two streams is constant. A change of k between consecutive pairs of the same epochs is
counted. Frame counters are never used to pair.

### 4.3 Output
`PairSink` (`pair_sink.hpp`): `on_pair(record, depthA, depthB)`, `on_period(period, tolerance)` (once, before the first
pair). Implementations: `StatsSink` (counts, checks the sequence, discards) and `ShmSink`. The record: pair sequence number,
dt = t(B) - t(A), per camera global, hardware and sensor timestamps, arrival time (`CLOCK_MONOTONIC`), frame counter,
exposure and domain, and the publish time (`CLOCK_MONOTONIC` and `CLOCK_REALTIME`, read just before `on_pair`). After the
sink returns, the matcher pops both ring slots and (optionally) queues the record with both frames' epochs for CSV.

**Output delay (gap 3).** For each pair: publish `CLOCK_REALTIME` minus the SDK global timestamp of the older and of the
newer frame. Four atomic histograms (0.1 ms bins, 0-60 ms): {usual, near half a period} x {older, newer}; "near half" means
`| |dt| - P/2 | <= 0.5 ms`, where the matcher may wait for the next frame of the earlier stream before it decides (decision
rule 3, third case). The status line shows interval p50/p99, the final report and JSON p50/p99/max, and the histograms
are written to `--delay-hist-out` at shutdown. The delay ends before the copy into shared memory (about 0.3 ms more until
a reader can see the pair).

## 5. Shared memory (`shm_layout.hpp`, `shm_sink.*`; offsets, attach sequence and liveness check in README.md)

```
offset 0      ShmHeader (512 bytes): magic, version 2, sizes, writer pid/state, pairs_published, notify_word,
              period, tolerance, CameraInfo[2] (serial, depth scale, intrinsics),
              writer_instance_id, heartbeat_mono_ns, heartbeat_count, heartbeat interval / liveness timeout,
              writer start time, replaced_by_instance
offset 512    slot 0 .. slot 3 (slot_stride = 512 + 2 * 814080 bytes)
slot:  +0     uint64 seq (atomic; odd = being written, 2n+2 = holds pair n)
       +64    PairRecord (128 bytes, with publish_mono_ns / publish_real_ns)
       +512   depth image A, then depth image B
```

Layout version 2 (gaps 1 and 3) keeps bytes 0-111 of version 1 unchanged, so either reader version reads magic and
version and rejects the other.

**Writer (never waits).** A random non-zero instance id per start. Created with `O_CREAT|O_EXCL`. If an object of the same
name exists (a writer that crashed, or one that still runs), the new writer maps it, sets `replaced_by_instance` and
`writer_state = 3` (replaced; a version 1 object gets `2`), increments `notify_word` and wakes its readers, and only then
unlinks the name and creates its own object. Sized, mapped, every page touched (`memset`), header filled, magic stored
last. Pair n -> slot n mod 4: `seq = 2n+1` (relaxed), release fence, copy record and both images, `seq = 2n+2` (release),
`pairs_published = n+1` (release), `notify_word += 1`, `futex(FUTEX_WAKE, all)`. A heartbeat thread stores
`CLOCK_MONOTONIC` into `heartbeat_mono_ns` every 50 ms (it sleeps on a futex `EventSignal`, so the stop is immediate). At
shutdown: stop the heartbeat, `writer_state = stopped` (unless replaced), notify, unmap, `shm_unlink` (not if replaced:
the name belongs to the new writer).

**Reader** (`ShmReader`, used by `d435_pair_reader`): load `notify_word`, then `pairs_published`. If the state is
"replaced", return `WriterReplaced` at once (unread pairs of the old writer are counted missing; a replaced writer may still
publish, and a reader slower than it would otherwise never move to the new writer). For pair n: `s1 = seq` (acquire); if
`s1 != 2n+2` the pair was overwritten (missing). Copy record and images, acquire fence, `s2 = seq`; if `s2 != s1` the slot
changed during the read: discard (counted as discarded read and missing). More than 4 pairs behind: skip to the oldest still
present and count the rest missing. Nothing new: `WriterStopped` if the writer stopped (after all its pairs), else
`WriterNotAlive` if the heartbeat is older than 1 s, else `futex(FUTEX_WAIT, notify_word)` for at most 200 ms (timed wait
only for the liveness check; a pair or a state change wakes it at once). `reattach()` maps the name again and accepts it
only if the instance id differs. A slow or stopped reader has no effect on the writer (unit test
`shm_slow_and_stopped_reader_do_not_slow_writer`, acceptance tests `reader`, `reader_stopped`).

**Sanitizers (gap 5).** The record and image copies on both sides go through `seqlock_copy`. In the ThreadSanitizer build
it is a `no_sanitize("thread")` volatile word loop: the overlap of a reader's copy with a writer's copy is the seqlock's
intended race, detected by the sequence counter and discarded; it is the only code excluded from TSan. The unit tests that
need TSan to see both sides attach the reader to the writer's own mapping (`ShmReader::attach_in_process`); TSan does not
relate accesses through two different mappings of the same object.

## 6. Camera handling (`camera.*`)

- Startup: both devices are found by serial in one query, before either is opened (with the RSUSB backend, creating a device
  object for a camera that is already open fails). Checks: `usb_type_descriptor` starts with "3.", firmware >= 5.17.3.10.
  Options set and read back (mismatch -> refuse to start): `global_time_enabled=1`, `inter_cam_sync_mode=0`,
  `enable_auto_exposure=1`, `frames_queue_size=16`. Profile 848x480 Z16 90 fps, depth scale, intrinsics.
- Callback (`on_frame`): see section 1; exceptions are caught and counted.
- Restart: `stop()`, `close()`, new epoch, `open()` of the same profile, `start()`. If the open fails (the device
  re-enumerated), or after a hardware reset, release the old handles, find the device again by serial (skipping devices
  that cannot be opened) and repeat all startup checks. Not found: the restart fails ("absent") and is retried every 1 s.
  `query_devices`, `open`, `start` and the hardware reset catch every exception (SDK errors must not stop the program).
- Hardware reset (test, gap 2): `rs2::device::hardware_reset()` while the camera streams. The device either disconnects and
  re-enumerates with a new USB address, or the USB core resets the port and re-initialises it with the same address (both
  seen on the Jetson; TEST_REPORT.md).

## 7. Watchdog, frame rate, status (`status.*`, supervisor thread)

- Watchdog per camera: no frame within 2 s after (re)start, or no frame for more than 10 P -> restart of that camera only.
  A failed restart is retried every 1 s. The other camera continues; the engine restarts the settle period for the
  restarted camera (new epoch in its slots).
- Dropped frames: the callback adds gaps in the frame counter within one epoch.
- Frame rate (`fps_monitor.hpp`): once per second the supervisor gives the monitor the callback frame count and the arrival
  time of the last frame. The rate is (frames since the sample about 2 s earlier) / (time between the last frames of the
  two samples), so it does not suffer from counting whole frames per window (a 1 s count has +-1 fps resolution at 90 fps
  and gave false warnings in the first acceptance series). Outside 89.87 +- 0.5 fps for more than 2 s -> one warning per
  episode; reset at a camera restart.
- Status line every 10 s (interval values from histogram differences): per camera condition, fps, dropped, overflows,
  highest fill, callback p99/max, latency p50/p99; matcher pairs, unpaired, dt min/median/max, partner changes, restarts,
  settle state, P; output delay p50/p99 (usual and near half a period).
- Recovery (gap 2): a camera is "recovering" from the first missing frames (or the reset request) until the first pair
  with a frame of its new epoch. Conditions: `reset` (reset requested; no watchdog restart for 2 s, so that the device can
  disconnect), `restarting`, `absent`, `restart_failed`, `settling`, `ok`. Each change is an `[event]` line with both clocks,
  the epoch, both cameras' drop/overflow counters, pairs and RSS (`/proc/self/statm`).
- Hardware-reset test (`--test-hw-reset SEQ:FIRST:INTERVAL`): one reset at a time; the next one INTERVAL s after the
  previous camera delivered pairs again (or after 120 s without recovery, counted as a recovery timeout).
- Histograms (atomic bins, lock-free `add`): latency 0.1 ms bins 0-40 ms (+ under/overflow), callback time 10 us bins
  0-5 ms, dt 0.01 ms bins -6..+6 ms. Latency = arrival `CLOCK_REALTIME` - SDK global timestamp. The latency histograms are
  written to a CSV file at shutdown.

## 8. Shutdown (`main.cpp`)

SIGINT/SIGTERM (or `--duration`): stop the supervisor (no more restarts) -> both callbacks stop taking frames at the same
instant (`set_stopping`; later frames are counted as `after_stop`) -> stop and join the matcher (frames still waiting in it
are left pending) -> stop and close both sensors -> stop and join the CSV writer -> final report (stdout, JSON,
histograms) -> close the shared memory (stop the heartbeat thread, writer state stopped, notify, unmap, unlink).
