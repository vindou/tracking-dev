#!/usr/bin/env bash
# Revised exp 4 (one camera stands in for two; NOT a two-camera test), 848x480 @90, memcpy callback, 3 x 60 s each:
#  a) depth + IR1 + IR2 (Y8): 2x the USB payload of one depth stream, 3 callbacks per frame period
#  b) depth only + cpuload 4 threads (8 MB memcpy loops)    c) a + cpuload 4 threads
#  d) 30 start/stop cycles x 5 s, alternating depth-only / depth+IR (stall rate)
source ~/rs_investigation/src/lib_B.sh
for r in 1 2 3; do cap expB4a_ir_r$r --w 848 --h 480 --fps 90 --seconds 60 --cb memcpy --ir 1; done
for r in 1 2 3; do capl 4 expB4b_load4_r$r --w 848 --h 480 --fps 90 --seconds 60 --cb memcpy; done
for r in 1 2 3; do capl 4 expB4c_ir_load4_r$r --w 848 --h 480 --fps 90 --seconds 60 --cb memcpy --ir 1; done
for i in $(seq -w 1 30); do
  if [ $((10#$i % 2)) = 0 ]; then x=1; else x=0; fi
  LRS_LOG_LEVEL=WARN cap expB4d_stall_${i}_ir$x --w 848 --h 480 --fps 90 --seconds 5 --ir $x
done
python3 src/analyze.py summary logs/expB4*/ --skip 2 > logs/expB4_summary.txt
for g in a_ir b_load4 c_ir_load4; do echo "### $g"; python3 src/analyze.py agg logs/expB4${g}_r*/; done > logs/expB4_agg.txt
echo EXPB4_DONE
