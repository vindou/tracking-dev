#!/usr/bin/env bash
# Rule-9 follow-up: fw517_B5_expauto_r1-3 had 0 Hardware Clock frames at start (all other runs 1-7).
# Reproduce the original order (848x480@30 run, then auto runs) and compare with auto / no-option / manual runs after a 90 fps run.
source ~/rs_investigation/src/lib_B.sh
for r in 1 2 3; do
  cap fw517d_pre30_r$r      --w 848 --h 480 --fps 30 --seconds 5
  cap fw517d_auto_after30_r$r --w 848 --h 480 --fps 90 --seconds 5 --exposure auto
  cap fw517d_auto_after90_r$r --w 848 --h 480 --fps 90 --seconds 5 --exposure auto
  cap fw517d_noopt_r$r      --w 848 --h 480 --fps 90 --seconds 5
  cap fw517d_man5000_r$r    --w 848 --h 480 --fps 90 --seconds 5 --exposure 5000
done
cap fw517d_restore_auto --w 848 --h 480 --fps 90 --seconds 3 --exposure auto
echo FW517D_DONE
