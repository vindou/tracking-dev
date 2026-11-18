#!/usr/bin/env bash
# Gap-closure acceptance tests on the Jetson (two cameras).
#   gap_tests.sh [test ...]      tests: kill hwreset   (default: both)
#     kill     gap 1: reader attached; the writer is SIGKILLed 20 s after each start, then restarted; 5 kills, then a
#              last run that stops cleanly (about 3 min)
#     hwreset  gap 2: 10 hardware resets of camera A, then 10 of camera B, during one run (about 8 min)
# Logs: $LOGROOT/<test>_1/ ; evaluation: python3 evaluate_gaps.py $LOGROOT
source "$(dirname "$0")/common.sh"
LOGROOT=${LOGROOT_GAP:-$HOME/capture_logs/gap}
mkdir -p "$LOGROOT"
TESTS=${*:-kill hwreset}

t_kill() {
    local d=$LOGROOT/kill_1; rm -rf "$d"; mon_start "$d"
    "$READER" --shm-name $SHM --wait-attach 30 --reattach-wait 30 --quiet --json "$d/reader.json" > "$d/reader.log" 2>&1 &
    local rp=$!
    for i in 1 2 3 4 5 6; do
        local dur=0
        [ $i = 6 ] && dur=30       # last run: clean stop -> reader exits, object removed
        "$CAP" --config "$CONF" --final-json "$d/final_$i.json" --hist-out "$d/latency_hist_$i.csv" \
            --delay-hist-out "$d/delay_hist_$i.csv" --sink shm --duration $dur --pair-csv "$d/pairs_$i.csv" \
            > "$d/stdout_$i.log" 2>&1 &
        local cp=$!
        echo "start_$i real_ns=$(date +%s%N) pid=$cp" >> "$d/kills.txt"
        if [ $i -lt 6 ]; then
            sleep 20
            echo "kill_$i real_ns=$(date +%s%N) pid=$cp" >> "$d/kills.txt"
            kill -KILL $cp
            wait $cp 2>/dev/null
            echo "kill_$i rc=$?" >> "$d/kills.txt"
            sleep 3               # the reader detects "not alive" before the next writer starts
        else
            wait $cp
            echo "capture_rc=$?" >> "$d/meta.txt"
        fi
    done
    wait $rp; echo "reader_rc=$?" >> "$d/meta.txt"
    mon_stop "$d"; echo "kill_1 rc=$(sed -n 's/capture_rc=//p' "$d/meta.txt") reader_rc=$(sed -n 's/reader_rc=//p' "$d/meta.txt")"
}

t_hwreset() {
    local d=$LOGROOT/hwreset_1; rm -rf "$d"; mon_start "$d"
    "$READER" --shm-name $SHM --wait-attach 15 --quiet --json "$d/reader.json" > "$d/reader.log" 2>&1 &
    local rp=$!
    timeout 900 "$CAP" --config "$CONF" --final-json "$d/final.json" --hist-out "$d/latency_hist.csv" \
        --delay-hist-out "$d/delay_hist.csv" --sink shm --duration 480 --pair-csv "$d/pairs.csv" \
        --test-hw-reset AAAAAAAAAABBBBBBBBBB:20:10 > "$d/stdout.log" 2>&1
    echo "capture_rc=$?" >> "$d/meta.txt"    # 124 = timeout (deadlock)
    wait $rp; echo "reader_rc=$?" >> "$d/meta.txt"
    mon_stop "$d"; echo "hwreset_1 rc=$(sed -n 's/capture_rc=//p' "$d/meta.txt")"
}

for t in $TESTS; do
    echo "=== $(date +%T) $t"
    "t_$t"
done
echo "=== $(date +%T) GAP_DONE"
