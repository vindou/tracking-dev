#!/usr/bin/env bash
# Re-run the whole single-camera investigation (about 75 min). Prerequisites: deps/build_deps.sh done, udev rule installed,
# camera A serial in ~/rs_investigation/cams.conf (A=SERIAL).
set -e
cd ~/rs_investigation/src
./build.sh
./exp0.sh
./exp1.sh
./exp3.sh
./exp3b_latency.sh
./exp5.sh
./exp3c_fps_anomaly.sh
./exp_followups.sh
./exp8_single.sh
cd ~/rs_investigation
python3 src/clock_drift.py logs/exp3_A_*/ > logs/exp6pre_camera_vs_host_drift.txt
python3 src/globaltime_error.py logs/exp3_A_*/ > logs/exp6pre_globaltime_error.txt
python3 src/plots.py
