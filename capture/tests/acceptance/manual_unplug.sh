#!/usr/bin/env bash
# Manual cable test (gap 2), same pass criteria as the hardware-reset test:
#   manual_unplug.sh A|B [N] [DURATION_S]
# Runs d435_capture with a reader attached. When the script says so, pull the USB cable of the named camera, wait about
# 3 s, and plug it in again; repeat N times (default 3), waiting each time until the script says pairs resumed.
# Then run: python3 evaluate_gaps.py $LOGROOT  (the run is evaluated like the hardware-reset run, using the
# frames_lost / pairs_resumed events and the kernel's USB disconnect / new device lines).
source "$(dirname "$0")/common.sh"
LOGROOT=${LOGROOT_GAP:-$HOME/capture_logs/gap}
CAMLETTER=${1:?usage: manual_unplug.sh A|B [N] [DURATION_S]}
N=${2:-3}
DUR=${3:-$((60 + N * 60))}
case $CAMLETTER in A|B) ;; *) echo "camera must be A or B"; exit 2 ;; esac
d=$LOGROOT/unplug_$CAMLETTER; rm -rf "$d"; mon_start "$d"
echo "unplug_camera=$CAMLETTER unplugs_planned=$N" >> "$d/meta.txt"
"$READER" --shm-name $SHM --wait-attach 15 --quiet --json "$d/reader.json" > "$d/reader.log" 2>&1 &
rp=$!
timeout $((DUR + 120)) "$CAP" --config "$CONF" --final-json "$d/final.json" --hist-out "$d/latency_hist.csv" \
    --delay-hist-out "$d/delay_hist.csv" --sink shm --duration "$DUR" --pair-csv "$d/pairs.csv" > "$d/stdout.log" 2>&1 &
cp=$!
echo "capture running for $DUR s; waiting 20 s before the first unplug"
sleep 20
for i in $(seq 1 "$N"); do
    before=$(grep -c "cam=$CAMLETTER what=pairs_resumed" "$d/stdout.log")
    echo ">>> unplug $i/$N: pull the cable of camera $CAMLETTER now, wait about 3 s, plug it in again"
    while [ "$(grep -c "cam=$CAMLETTER what=pairs_resumed" "$d/stdout.log")" -le "$before" ]; do
        kill -0 $cp 2>/dev/null || { echo "capture ended"; break 2; }
        sleep 1
    done
    echo "    pairs resumed: $(grep "cam=$CAMLETTER what=pairs_resumed" "$d/stdout.log" | tail -1 | grep -o 'lost_for_ms=[0-9]*')"
    sleep 10
done
echo "waiting for the end of the run"
wait $cp; echo "capture_rc=$?" >> "$d/meta.txt"
wait $rp; echo "reader_rc=$?" >> "$d/meta.txt"
mon_stop "$d"
echo "done: $d   (evaluate: python3 $(dirname "$0")/evaluate_gaps.py $LOGROOT)"
