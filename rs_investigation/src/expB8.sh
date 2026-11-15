#!/usr/bin/env bash
# Revised exp 8 (camera B, USB3): callback cost at the target 848x480 Z16 @90, log / memcpy / keep, 3 x 60 s each.
source ~/rs_investigation/src/lib_B.sh
for cb in log memcpy keep; do for r in 1 2 3; do cap expB8_${cb}_r$r --w 848 --h 480 --fps 90 --seconds 60 --cb $cb; done; done
for cb in log memcpy keep; do echo "### $cb"; python3 src/analyze.py summary logs/expB8_${cb}_r*/ > /dev/null; python3 src/analyze.py agg logs/expB8_${cb}_r*/; done > logs/expB8_agg.txt
echo EXPB8_DONE
