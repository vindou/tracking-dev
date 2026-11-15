#!/usr/bin/env bash
# Re-check after flashing camera B from FW 5.16.0.1 to 5.17.3.10 (logs/fw_update/). Same methods as expB1/B3/B5.
#  - inventory; frame period in 6 modes (3 x 15 s, as expB3m); baseline 848x480@90 (3 x 60 s, as expB3);
#  - exposure: auto, 10000, 10100 us (3 x 15 s, as expB5); auto restored at the end.
source ~/rs_investigation/src/lib_B.sh
src/build/rs_probe inventory > logs/fw517_inventory.txt 2>&1
for r in 1 2 3; do cap fw517_B3_848x480_90_r$r --w 848 --h 480 --fps 90 --seconds 60; done
for m in "848 480 90" "640 480 90" "480 270 90" "256 144 90" "848 480 60" "848 480 30"; do set -- $m
  for r in 1 2 3; do cap fw517_B3m_${1}x${2}_${3}_r$r --w $1 --h $2 --fps $3 --seconds 15; done; done
for e in auto 10000 10100; do for r in 1 2 3; do cap fw517_B5_exp${e}_r$r --w 848 --h 480 --fps 90 --seconds 15 --exposure $e; done; done
cap fw517_restore_auto --w 848 --h 480 --fps 90 --seconds 3 --exposure auto
python3 src/analyze.py summary logs/fw517_*/ --skip 2 > logs/fw517_summary.txt
python3 src/analyze.py agg logs/fw517_B3_848x480_90_r*/ > logs/fw517_B3_agg.txt
echo FW517_DONE
