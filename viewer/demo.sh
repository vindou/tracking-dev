#!/usr/bin/env bash
# Demo (Mac): starts the capture program, the tracker and the telemetry bridge on the Jetson through SSH, starts the
# viewer here, prints the URL, and on Ctrl-C stops everything cleanly (viewer, then bridge, tracker, capture with
# SIGTERM; SIGKILL only for a process that has not stopped after 10 s) and checks that nothing is left.
#
#   viewer/demo.sh            (environment: JETSON=jetson host alias, JETSON_IP=192.168.55.1, HTTP_PORT=8080)
set -u
JETSON=${JETSON:-jetson}
JETSON_IP=${JETSON_IP:-192.168.55.1}
HTTP_PORT=${HTTP_PORT:-8080}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$ROOT/.venv/bin/python
RUN=/tmp/trk_demo
LOG=$ROOT/viewer/demo_logs/$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"

echo "starting capture, tracker and bridge on $JETSON ..."
# one simple command per program, so that $! is the program's own process id
ssh "$JETSON" "bash -s" <<EOF
mkdir -p $RUN
cd $RUN
rm -f *.pid *.log
nohup ~/capture/build/d435_capture --config ~/capture/capture.conf > capture.log 2>&1 < /dev/null &
echo \$! > capture.pid
sleep 1
nohup ~/tracker/build/trk_tracker --config ~/tracker/config/tracker.json > tracker.log 2>&1 < /dev/null &
echo \$! > tracker.pid
nohup ~/tracker/build/trk_bridge --config ~/tracker/config/tracker.json > bridge.log 2>&1 < /dev/null &
echo \$! > bridge.pid
sleep 1
for p in capture tracker bridge; do
    if kill -0 \$(cat \$p.pid) 2>/dev/null; then echo "  \$p running (pid \$(cat \$p.pid))"; else echo "  \$p NOT running:"; tail -3 \$p.log; fi
done
EOF

"$PY" "$ROOT/viewer/viewer.py" --host "$JETSON_IP" --http-port "$HTTP_PORT" --csv "$LOG/poses.csv" \
    --stats-json "$LOG/viewer_stats.json" > "$LOG/viewer.log" 2>&1 &
VPID=$!

stopped=0
stop_all() {
    [ $stopped = 1 ] && return
    stopped=1
    echo; echo "stopping ..."
    kill -TERM $VPID 2>/dev/null; wait $VPID 2>/dev/null   # (a background job of a script ignores SIGINT)
    ssh "$JETSON" "cd $RUN && for p in bridge tracker capture; do
        pid=\$(cat \$p.pid 2>/dev/null); [ -n \"\$pid\" ] || continue
        kill -TERM \$pid 2>/dev/null
        for i in \$(seq 1 100); do kill -0 \$pid 2>/dev/null || break; sleep 0.1; done
        if kill -0 \$pid 2>/dev/null; then kill -KILL \$pid; echo \"  \$p did not stop: killed\"; else echo \"  \$p stopped\"; fi
      done
      left=\$(pgrep -u \$USER -f 'build/(trk_|d435_)' | wc -l); shm=\$(ls /dev/shm | grep -cE '^(trk_|d435_pairs)')
      echo \"  processes left on the Jetson: \$left, shared-memory objects left: \$shm\"
      tail -3 tracker.log; tail -1 bridge.log" | tee "$LOG/stop.log"
    scp -q "$JETSON:$RUN/*.log" "$LOG/" 2>/dev/null
    echo "logs: $LOG"
}
trap 'stop_all; exit 0' INT TERM

sleep 2
echo
echo "viewer: http://localhost:$HTTP_PORT   (Ctrl-C stops the demo)"
wait $VPID
stop_all
