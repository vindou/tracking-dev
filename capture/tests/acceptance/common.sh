# Shared helpers for the T2 acceptance tests (run on the Jetson). Sourced by t2_*.sh.
# Each run directory gets: stdout.log, final.json, tegrastats.log, kernel_new.log, meta.txt (+ test-specific files).
set -u
CAP_DIR=${CAP_DIR:-$HOME/capture}
CAP=${CAP:-$CAP_DIR/build/d435_capture}
READER=${READER:-$CAP_DIR/build/d435_pair_reader}
CONF=${CONF:-$CAP_DIR/capture.conf}
LOGROOT=${LOGROOT:-$HOME/capture_logs/t2}
CPULOAD=${CPULOAD:-$HOME/rs_investigation/src/build/cpuload}   # read-only use (rule 6)
SHM=/d435_pairs
mkdir -p "$LOGROOT"

# mon_start DIR / mon_stop DIR: tegrastats (1 s) and new kernel lines for the duration of a run
mon_start() {
    local d=$1
    mkdir -p "$d"
    journalctl -k -n 0 --show-cursor --no-pager 2>/dev/null | sed -n 's/^-- cursor: //p' > "$d/.cursor"
    tegrastats --interval 1000 --logfile "$d/tegrastats.log" &
    echo $! > "$d/.tegrastats_pid"
    {
        echo "start_uptime_s=$(cut -d' ' -f1 /proc/uptime)"
        echo "usb:"; lsusb -t | sed 's/^/  /'
    } > "$d/meta.txt"
}
mon_stop() {
    local d=$1
    kill "$(cat "$d/.tegrastats_pid")" 2>/dev/null
    wait "$(cat "$d/.tegrastats_pid")" 2>/dev/null
    local cur
    cur=$(cat "$d/.cursor")
    if [ -n "$cur" ]; then
        journalctl -k --after-cursor="$cur" --no-pager > "$d/kernel_new.log" 2>&1
        journalctl -k --after-cursor="$cur" --no-pager -o short-monotonic > "$d/kernel_mono.log" 2>&1
    else : > "$d/kernel_new.log"; : > "$d/kernel_mono.log"; fi
    {
        echo "end_uptime_s=$(cut -d' ' -f1 /proc/uptime)"
        if [ -e /dev/shm${SHM} ]; then echo "shm_left=1"; else echo "shm_left=0"; fi
        echo "own_processes_left=$(pgrep -u "$USER" -f 'build/d435_capture|build/d435_pair_reader|build/cpuload' | wc -l)"
    } >> "$d/meta.txt"
    rm -f "$d/.cursor" "$d/.tegrastats_pid"
}

# capture DIR ARGS...: runs the capture program in the foreground; rc in meta.txt
capture() {
    local d=$1; shift
    local t0=$(date +%s.%N)
    "$CAP" --config "$CONF" --final-json "$d/final.json" --hist-out "$d/latency_hist.csv" \
        --delay-hist-out "$d/delay_hist.csv" "$@" > "$d/stdout.log" 2>&1
    local rc=$?
    echo "capture_rc=$rc" >> "$d/meta.txt"
    echo "capture_wall_s=$(awk "BEGIN{print $(date +%s.%N) - $t0}")" >> "$d/meta.txt"
    return $rc
}
