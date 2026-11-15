#!/usr/bin/env bash
# Experiment 0 + 2: system and camera inventory.
cd ~/rs_investigation; export LD_LIBRARY_PATH=$HOME/.local/lib
{ cat /etc/nv_tegra_release; uname -a; nvpmodel -q; for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(basename $c) $(cat $c/cpufreq/scaling_governor)"; done; lsusb; lsusb -t; } > logs/exp0_system_rerun.txt 2>&1
src/build/rs_probe inventory > logs/exp0_exp2_sdk_inventory_rerun.txt 2>&1
echo "wrote logs/exp0_system_rerun.txt logs/exp0_exp2_sdk_inventory_rerun.txt"
