#!/usr/bin/env bash
# Revised exp 6 (one camera): camera clock vs Jetson clock and global-time error, 848x480 @90, 3 x 15 min, memcpy callback.
# Camera-to-camera offset/drift needs two cameras: OPEN.
source ~/rs_investigation/src/lib_B.sh
for r in 1 2 3; do cap expB6_15min_r$r --w 848 --h 480 --fps 90 --seconds 900 --cb memcpy; done
python3 src/analyze.py summary logs/expB6_*/ --skip 2 > logs/expB6_summary.txt
python3 src/clock_drift.py logs/expB6_*/ > logs/expB6_drift.txt
python3 src/globaltime_error.py logs/expB6_*/ > logs/expB6_globaltime_error.txt
echo EXPB6_DONE
