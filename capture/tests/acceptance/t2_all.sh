#!/usr/bin/env bash
# T2 acceptance tests on the Jetson (two cameras). About 110 min in total (45 min long run).
#   t2_all.sh [test ...]     tests: baseline reader reader_stopped memload startstop watchdog long  (default: all, long last)
# Logs: $LOGROOT/<test>_<n>/ ; evaluation: python3 evaluate.py $LOGROOT
source "$(dirname "$0")/common.sh"
TESTS=${*:-baseline reader reader_stopped memload startstop watchdog long}

t_baseline() {
    for r in 1 2 3; do
        local d=$LOGROOT/baseline_$r; rm -rf "$d"; mon_start "$d"
        capture "$d" --sink shm --duration 60 --pair-csv "$d/pairs.csv"
        mon_stop "$d"; echo "baseline_$r rc=$(sed -n 's/capture_rc=//p' "$d/meta.txt")"
    done
}

t_long() {
    local d=$LOGROOT/long_1; rm -rf "$d"; mon_start "$d"
    capture "$d" --sink shm --duration 2700 --pair-csv "$d/pairs.csv"
    mon_stop "$d"; echo "long_1 rc=$(sed -n 's/capture_rc=//p' "$d/meta.txt")"
}

t_reader() {
    for r in 1 2 3; do
        local d=$LOGROOT/reader_$r; rm -rf "$d"; mon_start "$d"
        "$READER" --shm-name $SHM --wait-attach 15 --json "$d/reader.json" > "$d/reader.log" 2>&1 &
        local rp=$!
        capture "$d" --sink shm --duration 60 --pair-csv "$d/pairs.csv"
        wait $rp; echo "reader_rc=$?" >> "$d/meta.txt"
        mon_stop "$d"; echo "reader_$r rc=$(sed -n 's/capture_rc=//p' "$d/meta.txt")"
    done
}

t_reader_stopped() {
    local d=$LOGROOT/reader_stopped_1; rm -rf "$d"; mon_start "$d"
    "$READER" --shm-name $SHM --wait-attach 15 --json "$d/reader.json" > "$d/reader.log" 2>&1 &
    local rp=$!
    ( sleep 10; kill -STOP $rp; echo "reader SIGSTOP at +10 s" >> "$d/meta.txt" ) &
    local sp=$!
    capture "$d" --sink shm --duration 60 --pair-csv "$d/pairs.csv"
    wait $sp
    grep -q State:.*T /proc/$rp/status && echo "reader_was_stopped_at_end=1" >> "$d/meta.txt"
    kill -KILL $rp; wait $rp 2>/dev/null
    mon_stop "$d"; echo "reader_stopped_1 rc=$(sed -n 's/capture_rc=//p' "$d/meta.txt")"
}

t_memload() {
    for r in 1 2 3; do
        local d=$LOGROOT/memload_$r; rm -rf "$d"; mon_start "$d"
        "$CPULOAD" 4 70 > "$d/cpuload.log" 2>&1 &
        local lp=$!
        sleep 2
        timeout 120 "$CAP" --config "$CONF" --final-json "$d/final.json" --hist-out "$d/latency_hist.csv" --sink shm \
            --delay-hist-out "$d/delay_hist.csv" \
            --duration 60 --pair-csv "$d/pairs.csv" > "$d/stdout.log" 2>&1
        echo "capture_rc=$?" >> "$d/meta.txt"    # 124 = timeout (deadlock)
        wait $lp
        mon_stop "$d"; echo "memload_$r rc=$(sed -n 's/capture_rc=//p' "$d/meta.txt")"
    done
}

t_startstop() {
    for i in $(seq -w 1 20); do
        local d=$LOGROOT/startstop_$i; rm -rf "$d"; mon_start "$d"
        timeout 60 "$CAP" --config "$CONF" --final-json "$d/final.json" --hist-out "$d/latency_hist.csv" --sink shm \
            --delay-hist-out "$d/delay_hist.csv" --pair-csv "$d/pairs.csv" --duration 10 > "$d/stdout.log" 2>&1
        echo "capture_rc=$?" >> "$d/meta.txt"
        mon_stop "$d"; echo "startstop_$i rc=$(sed -n 's/capture_rc=//p' "$d/meta.txt") $(grep shm_left "$d/meta.txt")"
    done
}

t_watchdog() {
    local d=$LOGROOT/watchdog_1; rm -rf "$d"; mon_start "$d"
    capture "$d" --sink shm --duration 60 --pair-csv "$d/pairs.csv" --test-ignore A:20:1
    mon_stop "$d"; echo "watchdog_1 rc=$(sed -n 's/capture_rc=//p' "$d/meta.txt")"
}

for t in $TESTS; do
    echo "=== $(date +%T) $t"
    "t_$t"
done
echo "=== $(date +%T) T2_DONE"
