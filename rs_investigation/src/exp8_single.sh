#!/usr/bin/env bash
# Exp 8 (partial, single camera): per-frame callback cost at the target frame size 848x480 Z16 (USB2 limits fps to 10).
cd ~/rs_investigation; SER=$(sed -n "s/^A=//p" cams.conf)
for cb in log memcpy keep; do
  for r in 1 2 3; do
    src/run_capture.sh exp8_A_848x480_10_${cb}_r$r --cam A:$SER --w 848 --h 480 --fps 10 --seconds 60 --global-time 1 --cb $cb; sleep 2
  done
done
for cb in log memcpy keep; do echo "### $cb"; python3 src/analyze.py summary logs/exp8_A_848x480_10_${cb}_r*/ > /dev/null; python3 src/analyze.py agg logs/exp8_A_848x480_10_${cb}_r*/; done > logs/exp8_single_agg.txt
echo EXP8_DONE
