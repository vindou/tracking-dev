#!/usr/bin/env bash
# Experiment 3: single-camera baseline, 3 x 60 s per mode, global time ON (SDK default), auto exposure (default).
# Camera A only (USB2: 848x480@90 unavailable; 256x144@90 is the only 90 fps depth mode).
cd ~/rs_investigation
SER=$(sed -n 's/^A=//p' cams.conf)
for mode in "256 144 90" "480 270 60" "848 480 10"; do
  set -- $mode
  for r in 1 2 3; do
    src/run_capture.sh exp3_A_${1}x${2}_${3}_r$r --cam A:$SER --w $1 --h $2 --fps $3 --seconds 60 --global-time 1
    sleep 3
  done
done
python3 src/analyze.py summary logs/exp3_*/ --skip 2 > logs/exp3_summary.txt
for mode in 256x144_90 480x270_60 848x480_10; do echo "### $mode"; python3 src/analyze.py agg logs/exp3_A_${mode}_r*/; done > logs/exp3_agg.txt
echo EXP3_DONE
