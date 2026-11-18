#!/usr/bin/env bash
# Phase 6: a synthetic file through the full chain (replay -> pair buffer -> tracker -> pose buffer -> CSV) gives the
# same poses as the offline run of the tracker core on the same file (within 0.1 mm).
source "$(dirname "$0")/common.sh"
d=$LOGROOT/replay_synth_${INIT:-track}; rm -rf "$d"; mon_start "$d"
F=$DATA/synth_replay.trk
$B/trk_synth --out $F --frames 600 --pose -0.05,0.02,0.45,0.9,0.1,0.3,0.3 --vel 0.05,0,0 --angvel 5,-5,10 \
    --offset-ms 3 --noise-mm 1 --seed 11 --stl $TRK_DIR/../models/test_object.stl > "$d/synth.log" 2>&1
POSE0=-0.05,0.02,0.45,0.9,0.1,0.3,0.3
INIT=${INIT:-track}   # track: both runs start in track state at POSE0 (deterministic); search: both start in search
if [ "$INIT" = track ]; then OFF_INIT="--force-track $POSE0"; LIVE_INIT="--initial-pose $POSE0"; else OFF_INIT=""; LIVE_INIT=""; fi
$B/trk_offline --in $F --out "$d/offline.csv" --config $TCONF $OFF_INIT > "$d/offline.log" 2>&1
$B/trk_tracker --config $TCONF --duration 14 --final-json "$d/tracker.json" $LIVE_INIT > "$d/tracker.log" 2>&1 &
tp=$!
$B/trk_posedump --out "$d/live.csv" --duration 13 > "$d/posedump.log" 2>&1 &
pp=$!
sleep 1
$B/trk_replay --in $F --start-delay 1 > "$d/replay.log" 2>&1
wait $pp; wait $tp; echo "tracker_rc=$?" >> "$d/meta.txt"
mon_stop "$d"
python3 "$(dirname "$0")/compare_poses.py" "$d/offline.csv" "$d/live.csv" --tol-mm 0.1 | tee "$d/compare.out"
