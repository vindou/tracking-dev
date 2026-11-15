#!/usr/bin/env bash
# Exp 3b (rule 9 follow-up): is delivery latency set by frame period or by frame size?
# Same resolution, different fps. 3 x 15 s each, camera A, global time on, fixed manual exposure 1 ms.
cd ~/rs_investigation
SER=$(sed -n 's/^A=//p' cams.conf)
for mode in "480 270 6" "480 270 15" "480 270 30" "480 270 60" "848 480 6" "848 480 8" "848 480 10"; do
  set -- $mode
  for r in 1 2 3; do
    src/run_capture.sh exp3b_A_${1}x${2}_${3}_r$r --cam A:$SER --w $1 --h $2 --fps $3 --seconds 15 --global-time 1 --exposure 1000
    sleep 2
  done
done
src/run_capture.sh exp3b_restore_auto --cam A:$SER --w 256 --h 144 --fps 90 --seconds 2 --exposure auto
python3 src/analyze.py summary logs/exp3b_A_*/ --skip 3 > logs/exp3b_summary.txt
echo EXP3B_DONE
