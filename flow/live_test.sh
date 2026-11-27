#!/bin/bash
# Live test of the path flow -> hint buffer -> tracker on the Jetson, without cameras: trk_scenes shows scenes of the
# evaluation set in the pair buffer (0.7 s empty, then the scene for 2 s), the tracker runs as in the demo, and
# the inference process proposes while the tracker searches. Runs over the same scenes ("arms"):
#   hints     with the inference process trk_flow (flow/live/, C++)
#   hintspy   with flow/live.py (the Python version of the same process)
#   nohints   the tracker alone
#   flow/live_test.sh [STEP=8] [COUNT=0 (all)] [ARMS="hints nohints"] [ENGINES=frozen (a folder of ~/flow_probe)]
# Needs: tracker/tools/sync_to_jetson.sh, flow/live/build_jetson.sh, the engines in ~/flow_probe/frozen, scenes.bin in
# ~/flow_probe/frozen/trt_eval.
# Results: flow/runs/live/{scenes,poses}_<arm>.csv, tracker_<arm>.log, flow_<arm>.log; table: python -m flow.live_eval
set -e
cd "$(dirname "$0")/.."
STEP=${1:-8}; COUNT=${2:-0}; ARMS=${3:-"hints nohints"}; ENGINES=${4:-frozen}
scp -q flow/live.py flow/trt_run.py jetson:flow_probe/frozen/
ssh jetson "STEP=$STEP COUNT=$COUNT ARMS='$ARMS' ENGINES=$ENGINES bash -s" <<'REMOTE'
D=~/flow_probe/frozen/live; mkdir -p $D
cd ~/tracker
for arm in $ARMS; do
  opt=""; [ $arm = nohints ] && opt="--no-hints"
  build/trk_tracker --config config/tracker.json $opt --final-json $D/tracker_$arm.json > $D/tracker_$arm.log 2>&1 &
  TRK=$!
  build/trk_posedump --out $D/poses_$arm.csv --wait 30 > $D/posedump_$arm.log 2>&1 &
  DUMP=$!
  FLOW=""
  if [ $arm != nohints ]; then
    if [ $arm = hintspy ]; then ( cd ~/flow_probe/frozen && exec python3 -u live.py --tracker ~/tracker --engines ~/flow_probe/$ENGINES ) > $D/flow_$arm.log 2>&1 &
    else ~/flow_live/build/trk_flow --config ~/tracker/config/tracker.json --engines ~/flow_probe/$ENGINES > $D/flow_$arm.log 2>&1 &
    fi
    FLOW=$!
    for i in $(seq 60); do grep -q "hint buffer" $D/flow_$arm.log && break; sleep 1; done
  fi
  build/trk_scenes --scenes ~/flow_probe/frozen/trt_eval/scenes.bin --log $D/scenes_$arm.csv --step $STEP --count $COUNT --hold 2.0 --blank 0.7 | tail -1
  sleep 0.5
  [ -n "$FLOW" ] && kill -INT $FLOW
  kill -INT $TRK; wait $TRK 2>/dev/null
  kill -INT $DUMP 2>/dev/null; wait $DUMP 2>/dev/null
  [ -n "$FLOW" ] && wait $FLOW 2>/dev/null
  echo "$arm: $(grep -c . $D/poses_$arm.csv) pose rows; $(grep 'hints: records' $D/tracker_$arm.log)"
done
REMOTE
mkdir -p flow/runs/live
rsync -a jetson:flow_probe/frozen/live/ flow/runs/live/
