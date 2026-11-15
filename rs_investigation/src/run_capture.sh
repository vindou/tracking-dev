#!/usr/bin/env bash
# Run one timed capture with system monitoring.
# Usage: run_capture.sh RUN_NAME <rs_probe capture args, without --out>
# Output: ~/rs_investigation/logs/RUN_NAME/{frames.csv,stdout.txt,tegrastats.txt,kernel_new.txt,meta.txt}
set -uo pipefail
R=~/rs_investigation
NAME=$1; shift
D=$R/logs/$NAME
mkdir -p "$D"
export LD_LIBRARY_PATH=$HOME/.local/lib:${LD_LIBRARY_PATH:-}

{
  echo "run=$NAME"
  echo "cmd=$R/src/build/rs_probe capture $* --out $D/frames.csv"
  echo "start_local=$(date -Is)  (NOTE: Jetson wall clock is wrong, ~2023)"
  echo "uptime=$(cut -d' ' -f1 /proc/uptime)"
  echo "nvpmodel=$(nvpmodel -q 2>/dev/null | head -1)"
  echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "tj_start_mC=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)"
  echo "usb:"; lsusb -t | sed 's/^/  /'
} > "$D/meta.txt" 2>&1

CURSOR=$(journalctl -k -n 0 --show-cursor --no-pager 2>/dev/null | sed -n 's/^-- cursor: //p')
tegrastats --interval 200 --logfile "$D/tegrastats.txt" &
TS_PID=$!

"$R/src/build/rs_probe" capture "$@" --out "$D/frames.csv" > "$D/stdout.txt" 2>&1
RC=$?

kill $TS_PID 2>/dev/null; wait $TS_PID 2>/dev/null
if [ -n "$CURSOR" ]; then journalctl -k --after-cursor="$CURSOR" --no-pager > "$D/kernel_new.txt" 2>&1
else journalctl -k --since "-$(( ${SECS:-600} + 30 ))s" --no-pager > "$D/kernel_new.txt" 2>&1; fi
{
  echo "rc=$RC"
  echo "end_local=$(date -Is)"
  echo "uptime_end=$(cut -d' ' -f1 /proc/uptime)"
  echo "tj_end_mC=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)"
  echo "kernel_new_lines=$(grep -v -c '^-- ' "$D/kernel_new.txt")"
} >> "$D/meta.txt"
echo "[$NAME] rc=$RC frames_lines=$(($(wc -l < "$D/frames.csv" 2>/dev/null || echo 1)-1)) kernel_new=$(grep -v -c '^-- ' "$D/kernel_new.txt")"
exit $RC
