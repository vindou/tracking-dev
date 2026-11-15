#!/usr/bin/env bash
# Experiment 5: exposure vs frame rate, camera A at 256x144 Z16 90 fps (only 90 fps mode on USB2), 3 x 20 s per setting.
cd ~/rs_investigation
SER=$(sed -n 's/^A=//p' cams.conf)
for e in auto 1000 5000 8000 10000 10500 11000 11500 12000 15000; do
  for r in 1 2 3; do
    src/run_capture.sh exp5_A_exp${e}_r$r --cam A:$SER --w 256 --h 144 --fps 90 --seconds 20 --global-time 1 --exposure $e
    sleep 2
  done
done
# restore SDK default (auto exposure) so later runs start from the default
src/run_capture.sh exp5_restore_auto --cam A:$SER --w 256 --h 144 --fps 90 --seconds 2 --exposure auto
python3 src/analyze.py summary logs/exp5_A_*/ --skip 2 > logs/exp5_summary.txt
echo EXP5_DONE
