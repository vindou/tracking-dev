#!/usr/bin/env bash
# Revised single-camera investigation on camera B (USB3), about 100 min. Exp 7 (sync modes) is not run: left open.
cd ~/rs_investigation/src && ./build.sh
{ cat /etc/nv_tegra_release; uname -a; nvpmodel -q; lsusb; lsusb -t; } > ../logs/exp0_B_system.txt 2>&1
LD_LIBRARY_PATH=$HOME/.local/lib build/rs_probe inventory > ../logs/exp0_B_sdk_inventory.txt 2>&1
./expB1.sh && ./expB3.sh && ./expB4.sh && ./expB5.sh && ./expB8.sh && ./expB6.sh
cd ~/rs_investigation && python3 src/plots_B.py
echo ALL_B_DONE
