#!/usr/bin/env bash
# Revised exp 3 (camera B, USB3): single-camera baseline at the target 848x480 Z16 @90, 3 x 60 s, log callback.
# 3m (rule 9, smoke test showed 11.128 ms period): frame period vs mode, depth only, 3 x 15 s each.
source ~/rs_investigation/src/lib_B.sh
for r in 1 2 3; do cap expB3_848x480_90_r$r --w 848 --h 480 --fps 90 --seconds 60; done
for m in "848 480 90" "640 480 90" "480 270 90" "256 144 90" "848 480 60" "848 480 30"; do set -- $m
  for r in 1 2 3; do cap expB3m_${1}x${2}_${3}_r$r --w $1 --h $2 --fps $3 --seconds 15; done; done
python3 src/analyze.py summary logs/expB3_*/ logs/expB3m_*/ --skip 2 > logs/expB3_summary.txt
python3 src/analyze.py agg logs/expB3_848x480_90_r*/ > logs/expB3_agg.txt
echo EXPB3_DONE
