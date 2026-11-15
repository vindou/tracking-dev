#!/usr/bin/env bash
# Exp 3c (rule 9 follow-up): why did 848x480 @8/@10 with manual exposure 1 ms stream at 6 fps (exp3b)?
# Each case 10 s, run in this order, 3 repetitions of the whole sequence.
cd ~/rs_investigation
SER=$(sed -n 's/^A=//p' cams.conf)
run(){ src/run_capture.sh exp3c_${1}_r$r --cam A:$SER --w 848 --h 480 --fps $2 --seconds 10 --global-time 1 ${3:+--exposure $3}; sleep 2; }
for r in 1 2 3; do
  run a_10_auto        10 auto     # as exp3 (worked)
  run b_10_man1000     10 1000     # as exp3b (failed)
  run c_10_man5000     10 5000
  run d_10_man1000again 10 1000
  run e_10_noset       10          # do not touch exposure options at all
  run f_10_auto_again  10 auto
done
echo EXP3C_DONE
