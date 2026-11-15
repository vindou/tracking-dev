#!/usr/bin/env bash
# Revised exp 5 (camera B, USB3): fps vs exposure at 848x480 @90, 3 x 15 s per setting, then auto restored.
# 5s (repeat of A follow-up F2 on USB3): open 848x480 @6 -> @30 -> @90, manual 1 ms, 3 sequences x 10 s.
source ~/rs_investigation/src/lib_B.sh
for e in auto 1000 5000 8000 9000 10000 10100 10500 12000 15000; do
  for r in 1 2 3; do cap expB5_exp${e}_r$r --w 848 --h 480 --fps 90 --seconds 15 --exposure $e; done; done
for r in 1 2 3; do for f in 6 30 90; do cap expB5s_seq_${f}_r$r --w 848 --h 480 --fps $f --seconds 10 --exposure 1000; done; done
cap expB5_restore_auto --w 848 --h 480 --fps 90 --seconds 3 --exposure auto
python3 src/analyze.py summary logs/expB5*/ --skip 2 > logs/expB5_summary.txt
echo EXPB5_DONE
