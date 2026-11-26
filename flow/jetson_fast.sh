#!/bin/bash
# Timing of the rearranged network (flow.export.FastNet) on the Jetson:
#   flow/jetson_fast.sh start                       one evaluation with every arrangement (after flow.export --fast)
#   flow/jetson_fast.sh start stage2f:stage2 ...    whole stages (after flow.export --fast-stages NAME); graph:inputs
#   flow/jetson_fast.sh fetch                       copy fast.tsv, fast.log and the exported outputs back
# The run on the Jetson is detached (flow/jetson_fast_remote.sh); fast.done appears when it has finished.
set -e
cd "$(dirname "$0")/.."
OUT=flow/runs/export
if [ "$1" = start ]; then
  shift
  rsync -a --exclude 'trt_*' --exclude 'run_*' --exclude 'tune.*' --exclude 'fast.*' $OUT/ jetson:flow_probe/frozen/
  scp -q flow/jetson_fast_remote.sh jetson:flow_probe/frozen/
  ssh -f jetson "cd ~/flow_probe/frozen && setsid nohup bash jetson_fast_remote.sh $* > fast.out 2>&1 < /dev/null &"
  echo started
else
  rsync -a --include 's*/' --include 'parts/' --include 'trt_*.json' --include 'run_*.log' --include 'fast.*' --exclude '*' jetson:flow_probe/frozen/ $OUT/
  cat $OUT/fast.log; cat $OUT/fast.tsv
fi
