#!/usr/bin/env bash
# Tracker tests on the Jetson (no viewer): unit tests in three builds, replay tests, 3 x 60 s live runs, capture
# restart during tracking. Logs: $LOGROOT/<test>/ (tegrastats, new kernel lines, program logs, eval.json).
#   run_jetson.sh [unit] [replay] [live] [caprestart]
source "$(dirname "$0")/common.sh"
TESTS=${*:-unit replay live caprestart}
EV="python3 $(dirname "$0")/evaluate_live.py"

t_unit() {
    local d=$LOGROOT/unit; mkdir -p "$d"
    for v in build build-tsan build-asan; do
        n=$(echo $v | sed "s/build-*//"); n=${n:-normal}
        TRK_STRESS_S=30 TSAN_OPTIONS="halt_on_error=0" ASAN_OPTIONS="detect_leaks=1" timeout 5400 $TRK_DIR/$v/trk_tests > "$d/unit_$n.log" 2>&1
        echo "unit $n: rc=$? $(grep -E 'tests,' "$d/unit_$n.log") sanitizer reports: $(grep -cE 'WARNING: ThreadSanitizer|ERROR: AddressSanitizer|runtime error|LeakSanitizer' "$d/unit_$n.log")"
    done
}

t_replay() {
    INIT=track bash "$(dirname "$0")/replay_test.sh"
    INIT=search bash "$(dirname "$0")/replay_test.sh"
}

# 3 x 60 s live: capture + tracker + pose dump. Calculation time, pair use, effect on the capture, static object.
t_live() {
    for r in 1 2 3; do
        local d=$LOGROOT/live_$r; rm -rf "$d"; mon_start "$d"
        $CAP --config $CONF --duration 75 --final-json "$d/capture.json" --hist-out "$d/capture_latency.csv" \
             --delay-hist-out "$d/capture_delay.csv" > "$d/capture.log" 2>&1 &
        local cp=$!
        sleep 1
        $B/trk_tracker --config $TCONF --duration 70 --final-json "$d/tracker.json" > "$d/tracker.log" 2>&1 &
        local tp=$!
        sleep 9   # tracker attached, capture settled (pairs start 5 s after the streams)
        $B/trk_posedump --out "$d/poses.csv" --duration 60 > "$d/posedump.log" 2>&1
        wait $tp; echo "tracker_rc=$?" >> "$d/meta.txt"
        wait $cp; echo "capture_rc=$?" >> "$d/meta.txt"
        mon_stop "$d"
        $EV "$d" > "$d/eval.out"
        echo "live_$r: $(python3 -c "import json;e=json.load(open('$d/eval.json'));print('pair_use',e.get('pair_use_pct'),'calc',e.get('calc_all_ms'),'capture',e.get('capture_counters'),'states',e.get('state_fraction'))")"
    done
}

# Capture restart during tracking: the capture program is stopped (SIGTERM) after 20 s and started again 3 s later;
# the tracker (same process) attaches again and continues.
t_caprestart() {
    local d=$LOGROOT/caprestart_1; rm -rf "$d"; mon_start "$d"
    $CAP --config $CONF --duration 20 --final-json "$d/capture1.json" --hist-out /dev/null --delay-hist-out /dev/null > "$d/capture1.log" 2>&1 &
    local c1=$!
    sleep 1
    $B/trk_tracker --config $TCONF --duration 50 --final-json "$d/tracker.json" > "$d/tracker.log" 2>&1 &
    local tp=$!
    $B/trk_posedump --out "$d/poses.csv" --duration 48 > "$d/posedump.log" 2>&1 &
    local pp=$!
    wait $c1
    sleep 3
    $CAP --config $CONF --duration 22 --final-json "$d/capture2.json" --hist-out /dev/null --delay-hist-out /dev/null > "$d/capture2.log" 2>&1 &
    local c2=$!
    wait $c2; wait $pp; wait $tp; echo "tracker_rc=$?" >> "$d/meta.txt"
    mon_stop "$d"
    $EV "$d" > "$d/eval.out"
    grep -E "attach|detach|final" "$d/tracker.log"
    python3 - "$d/poses.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
inst = {}
for r in rows:
    inst.setdefault(r["capture_instance"], []).append(int(r["pose_seq"]))
w = {r["writer_instance"] for r in rows}
seqs = [int(r["pose_seq"]) for r in rows]
gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
print(f"poses {len(rows)}; tracker instances {len(w)} (one = no tracker restart); capture instances seen {len(inst)}; "
      f"pose_seq gaps {gaps}; poses per capture instance {[len(v) for v in inst.values()]}")
print("RESULT", "PASS" if len(w) == 1 and len(inst) == 2 and gaps == 0 else "FAIL")
PY
}

for t in $TESTS; do echo "=== $(date +%T) $t"; "t_$t"; done
echo "=== $(date +%T) JETSON_TESTS_DONE"
