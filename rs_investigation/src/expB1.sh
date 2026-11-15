#!/usr/bin/env bash
# Revised exp 1 (camera B, USB3): timestamp domain and metadata at the target mode 848x480 Z16 @90, global time 0 and 1, 3 x 15 s.
source ~/rs_investigation/src/lib_B.sh
for r in 1 2 3; do for gt in 0 1; do
  src/run_capture.sh expB1_gt${gt}_r$r --cam B:$SER --w 848 --h 480 --fps 90 --seconds 15 --global-time $gt; sleep 2
done; done
python3 src/analyze.py summary logs/expB1_*/ --skip 0 > logs/expB1_summary.txt
echo EXPB1_DONE
