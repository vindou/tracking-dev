#!/usr/bin/env bash
# Rule-9 follow-ups after exp 3/5. Camera A, global time on.
cd ~/rs_investigation
SER=$(sed -n 's/^A=//p' cams.conf)
cap(){ local n=$1; shift; src/run_capture.sh $n --cam A:$SER --global-time 1 "$@"; sleep 2; }

# F1: auto exposure: resolution or fps? (3 x 10 s each)
for r in 1 2 3; do
  for m in "480 270 6" "480 270 15" "480 270 30" "640 360 30" "640 480 30" "640 480 15" "848 480 10" "256 144 90"; do
    set -- $m; cap f1_ae_${1}x${2}_${3}_r$r --w $1 --h $2 --fps $3 --seconds 10 --exposure auto
  done
done
# F2: reproduce exp3b order: 848x480 6 -> 8 -> 10, manual 1 ms (3 sequences x 10 s)
for r in 1 2 3; do
  for f in 6 8 10; do cap f2_seq_848x480_${f}_r$r --w 848 --h 480 --fps $f --seconds 10 --exposure 1000; done
done
# F3: fine exposure threshold at 90 fps (3 x 10 s each)
for e in 10100 10200 10300 10400; do
  for r in 1 2 3; do cap f3_exp${e}_r$r --w 256 --h 144 --fps 90 --seconds 10 --exposure $e; done
done
# F4: start-up stall rate: 30 start/stop cycles x 5 s, 256x144@90, alternating auto / manual 10500
for i in $(seq -w 1 30); do
  if [ $((10#$i % 2)) = 0 ]; then e=auto; else e=10500; fi
  LRS_LOG_LEVEL=WARN cap f4_stall_${i}_$e --w 256 --h 144 --fps 90 --seconds 5 --exposure $e
done
cap f_restore_auto --w 256 --h 144 --fps 90 --seconds 3 --exposure auto
echo FOLLOWUPS_DONE
