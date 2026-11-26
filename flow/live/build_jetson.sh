#!/bin/bash
# Copies flow/live to the Jetson (~/flow_live) and builds trk_flow there against ~/tracker (run
# tracker/tools/sync_to_jetson.sh first). Nothing is installed; TensorRT and CUDA are those of JetPack.
set -e
cd "$(dirname "$0")"
rsync -rlp --exclude build ./ jetson:flow_live/
ssh jetson 'cd ~/flow_live && find . -path ./build -prune -o -type f -exec touch {} + && cmake -S . -B build -DTRACKER_DIR=$HOME/tracker > build.log 2>&1 && make -C build -j4 trk_flow 2>&1 | grep -E "error|warning|Built target trk_flow" ; ls -la build/trk_flow'
