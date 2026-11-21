# Shared helpers for the tracker acceptance tests on the Jetson (sourced). Each run directory gets tegrastats.log,
# kernel_new.log / kernel_mono.log (new kernel lines), meta.txt.
set -u
TRK_DIR=${TRK_DIR:-$HOME/tracker}
CAP_DIR=${CAP_DIR:-$HOME/capture}
B=$TRK_DIR/build
CAP=$CAP_DIR/build/d435_capture
CONF=$CAP_DIR/capture.conf
TCONF=${TCONF:-$TRK_DIR/config/tracker.json}
LOGROOT=${LOGROOT:-$HOME/tracker_logs}
DATA=${DATA:-$HOME/tracker_data}
mkdir -p "$LOGROOT" "$DATA"
mon_start() {
    local d=$1; mkdir -p "$d"
    journalctl -k -n 0 --show-cursor --no-pager 2>/dev/null | sed -n 's/^-- cursor: //p' > "$d/.cursor"
    tegrastats --interval 1000 --logfile "$d/tegrastats.log" > /dev/null 2>&1 < /dev/null & echo $! > "$d/.tegra_pid"
    echo "start_uptime_s=$(cut -d' ' -f1 /proc/uptime)" > "$d/meta.txt"
}
mon_stop() {
    local d=$1
    kill "$(cat "$d/.tegra_pid")" 2>/dev/null; wait "$(cat "$d/.tegra_pid")" 2>/dev/null
    local cur; cur=$(cat "$d/.cursor")
    if [ -n "$cur" ]; then
        journalctl -k --after-cursor="$cur" --no-pager > "$d/kernel_new.log" 2>&1
        journalctl -k --after-cursor="$cur" --no-pager -o short-monotonic > "$d/kernel_mono.log" 2>&1
    fi
    { echo "end_uptime_s=$(cut -d' ' -f1 /proc/uptime)"
      echo "shm_left=$(ls /dev/shm | grep -cE '^(trk_|d435_pairs)')"
      echo "own_processes_left=$(pgrep -u "$USER" -f 'build/(trk_|d435_)' | wc -l)"; } >> "$d/meta.txt"
    rm -f "$d/.cursor" "$d/.tegra_pid"
}
