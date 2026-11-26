#!/bin/bash
# The two timing experiments on the Jetson (after python -m flow.export and python -m flow.export --parts, and one
# run of flow/jetson_time.sh for the level-3 engines):
#   flow/jetson_tune.sh start      copy the graphs and start flow/jetson_tune_remote.sh there (detached)
#   flow/jetson_tune.sh fetch      copy tune.tsv, tune.log, the profiles and the exported outputs back
set -e
cd "$(dirname "$0")/.."
OUT=flow/runs/export
if [ "$1" = start ]; then
  rsync -a --exclude 'trt_*' --exclude 'run_*' --exclude 'tune.*' $OUT/ jetson:flow_probe/frozen/
  scp -q flow/jetson_tune_remote.sh jetson:flow_probe/frozen/
  ssh jetson 'cd ~/flow_probe/frozen && nohup bash jetson_tune_remote.sh > tune.out 2>&1 < /dev/null & echo started'
else
  rsync -a --include 's*/' --include 'parts/' --include 'trt_*.json' --include 'run_*.log' --include 'tune.*' --include 'profile_*.json' --include 'layers_*.json' --exclude '*' jetson:flow_probe/frozen/ $OUT/
  cat $OUT/tune.log
fi
