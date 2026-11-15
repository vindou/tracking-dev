#!/usr/bin/env bash
# Experiment 1: timestamp domain and metadata with global_time_enabled 0 and 1 (10 s each).
cd ~/rs_investigation; SER=$(sed -n "s/^A=//p" cams.conf)
for gt in 0 1; do
  src/run_capture.sh exp1_gt${gt}_256x144_90 --cam A:$SER --w 256 --h 144 --fps 90 --seconds 10 --global-time $gt
  src/run_capture.sh exp1_gt${gt}_848x480_10 --cam A:$SER --w 848 --h 480 --fps 10 --seconds 10 --global-time $gt
done
python3 src/analyze.py summary logs/exp1_*/ --skip 0
