#!/usr/bin/env bash
# Link and viewer tests, run on the Mac: the Jetson programs are started over SSH, the viewer runs here.
#   link_tests.sh [link] [pause] [nobridge] [kill] [long]
# Jetson logs: ~/tracker_logs/<test>/ ; Mac logs: tracker/logs/<test>/
set -u
JETSON=${JETSON:-jetson}
IP=${JETSON_IP:-192.168.55.1}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$ROOT/.venv/bin/python
VIEWER=$ROOT/viewer/viewer.py
MLOG=$ROOT/tracker/logs
TESTS=${*:-link pause nobridge kill long}

# jet "commands": runs on the Jetson with the test helpers loaded
jet() { ssh "$JETSON" "bash -s" <<EOF
source ~/tracker/tests/acceptance/common.sh
$1
EOF
}
start_bg() {   # start_bg DIR NAME COMMAND...  (on the Jetson, in the background, pid in DIR/NAME.pid)
    local d=$1 n=$2; shift 2
    jet "cd $d && nohup $* > $n.log 2>&1 < /dev/null & echo \\\$! > $d/$n.pid"
}
stop_bg() { jet "kill -$2 \\\$(cat $1/$3.pid) 2>/dev/null; for i in \\\$(seq 1 100); do kill -0 \\\$(cat $1/$3.pid) 2>/dev/null || break; sleep 0.1; done"; }
viewer() {   # viewer MDIR DURATION [extra]
    "$PY" "$VIEWER" --host "$IP" --http-port 8090 --duration "$2" --csv "$1/viewer_poses.csv" --stats-json "$1/viewer_stats.json" > "$1/viewer.log" 2>&1
}
fetch() { scp -q "$JETSON:$1/*" "$2/" 2>/dev/null; }

l_link() {
    for r in 1 2 3; do
        local J=~/tracker_logs/link_$r M=$MLOG/link_$r; rm -rf "$M"; mkdir -p "$M"
        jet "rm -rf $J; mon_start $J"
        start_bg $J capture "\$CAP --config \$CONF --duration 80 --final-json $J/capture.json --hist-out /dev/null --delay-hist-out /dev/null"
        sleep 1
        start_bg $J tracker "\$B/trk_tracker --config \$TCONF --duration 76 --final-json $J/tracker.json"
        start_bg $J bridge "\$B/trk_bridge --config \$TCONF --duration 77 --final-json $J/bridge.json"
        sleep 8
        viewer "$M" 60
        sleep 12
        jet "mon_stop $J"
        fetch $J "$M"
        python3 - "$M" <<'PY' | tee "$M/result.txt"
import json, sys
m = sys.argv[1]
v = json.load(open(f"{m}/viewer_stats.json")); b = json.load(open(f"{m}/bridge.json")); t = json.load(open(f"{m}/tracker.json"))
inst = list(v["instances"].values())[0] if v["instances"] else {"received": 0, "first": 0, "last": -1, "gaps": 0, "out_of_order": 0}
span = inst["last"] - inst["first"] + 1
pct = 100.0 * inst["received"] / span if span > 0 else 0
print(f"viewer received {inst['received']} of {span} poses published while connected ({pct:.2f} %), gaps {inst['gaps']}, "
      f"out of order {inst['out_of_order']}; bridge dropped poses {b['poses_dropped']} clouds {b['clouds_dropped']}; "
      f"tracker cycles {t['cycles']}; connects {v['connects']}")
print("RESULT", "PASS" if pct >= 99.0 and inst["out_of_order"] == 0 else "FAIL")
PY
    done
}

# Viewer stopped (SIGSTOP) for 10 s while connected, then continued: tracker cycle time unchanged; bridge counts drops.
l_pause() {
    local J=~/tracker_logs/viewer_pause M=$MLOG/viewer_pause; rm -rf "$M"; mkdir -p "$M"
    jet "rm -rf $J; mon_start $J"
    start_bg $J capture "\$CAP --config \$CONF --duration 75 --final-json $J/capture.json --hist-out /dev/null --delay-hist-out /dev/null"
    sleep 1
    start_bg $J tracker "\$B/trk_tracker --config \$TCONF --duration 70 --final-json $J/tracker.json"
    start_bg $J bridge "\$B/trk_bridge --config \$TCONF --duration 71 --final-json $J/bridge.json"
    start_bg $J posedump "\$B/trk_posedump --out $J/poses.csv --duration 68"
    sleep 8
    viewer "$M" 55 &
    local vp=$!
    sleep 3
    local vpid; vpid=$(pgrep -f "viewer.py --host $IP --http-port 8090" | head -1)
    sleep 17
    date +%s.%N > "$M/stop_time"; jet "date +%s%N > $J/stop_real_ns; cat /proc/uptime > $J/stop_uptime"
    kill -STOP "$vpid"; sleep 10; kill -CONT "$vpid"
    jet "date +%s%N > $J/cont_real_ns"
    wait $vp
    sleep 18
    jet "mon_stop $J"
    fetch $J "$M"
    python3 - "$M" <<'PY' | tee "$M/result.txt"
import csv, json, sys
m = sys.argv[1]
rows = list(csv.DictReader(open(f"{m}/poses.csv")))
t_stop = int(open(f"{m}/stop_real_ns").read()); t_cont = int(open(f"{m}/cont_real_ns").read())
# pose publish times are CLOCK_MONOTONIC; the window is placed by the order of the poses around the wall-clock marks
pub = [int(r["publish_time_mono_ns"]) for r in rows]
calc = [float(r["t_total_us"]) + float(r["t_publish_us"]) for r in rows]
up = float(open(f"{m}/stop_uptime").read().split()[0]) * 1e9
i0 = next(i for i, p in enumerate(pub) if p >= up); i1 = next((i for i, p in enumerate(pub) if p >= up + (t_cont - t_stop)), len(pub))
def st(v):
    v = sorted(v); return v[len(v) // 2] / 1000, v[int(0.99 * (len(v) - 1))] / 1000
b = json.load(open(f"{m}/bridge.json")); v = json.load(open(f"{m}/viewer_stats.json"))
before, during, after = st(calc[:i0]), st(calc[i0:i1]), st(calc[i1:])
print(f"tracker calculation time p50/p99 (ms): before {before[0]:.2f}/{before[1]:.2f}, viewer stopped {during[0]:.2f}/{during[1]:.2f}, after {after[0]:.2f}/{after[1]:.2f}")
print(f"bridge: poses dropped {b['poses_dropped']}, clouds dropped {b['clouds_dropped']}, poses sent {b['poses_sent']}; viewer connects {v['connects']}, poses {v['poses']}")
ok = during[1] <= max(before[1], after[1]) * 1.2 + 0.2 and (b["poses_dropped"] + b["clouds_dropped"]) > 0
print("RESULT", "PASS" if ok else "FAIL", "(p99 while stopped within 20 % + 0.2 ms of the other windows; drops counted)")
PY
}

# Viewer started before the bridge: it waits and connects when the bridge starts.
l_nobridge() {
    local J=~/tracker_logs/viewer_first M=$MLOG/viewer_first; rm -rf "$M"; mkdir -p "$M"
    jet "rm -rf $J; mon_start $J"
    viewer "$M" 45 &
    local vp=$!
    sleep 15
    start_bg $J capture "\$CAP --config \$CONF --duration 35 --hist-out /dev/null --delay-hist-out /dev/null"
    sleep 1
    start_bg $J tracker "\$B/trk_tracker --config \$TCONF --duration 30"
    start_bg $J bridge "\$B/trk_bridge --config \$TCONF --duration 31 --final-json $J/bridge.json"
    wait $vp
    sleep 6
    jet "mon_stop $J"
    fetch $J "$M"
    grep -E "connected" "$M/viewer.log" | head -3 | tee "$M/result.txt"
    python3 -c "import json;v=json.load(open('$M/viewer_stats.json'));print('viewer connects',v['connects'],'poses',v['poses']);print('RESULT','PASS' if v['connects']>=1 and v['poses']>0 else 'FAIL')" | tee -a "$M/result.txt"
}

# Tracker killed with SIGKILL and started again; bridge and viewer continue. Synthetic replay keeps the tracker in
# track state.
l_kill() {
    local J=~/tracker_logs/tracker_kill M=$MLOG/tracker_kill; rm -rf "$M"; mkdir -p "$M"
    jet "rm -rf $J; mon_start $J"
    start_bg $J replay "\$B/trk_replay --in \$DATA/synth_replay.trk --loops 8 --start-delay 1"
    start_bg $J tracker1 "\$B/trk_tracker --config \$TCONF --initial-pose -0.05,0.02,0.45,0.9,0.1,0.3,0.3"
    start_bg $J bridge "\$B/trk_bridge --config \$TCONF --duration 50 --final-json $J/bridge.json"
    sleep 2
    viewer "$M" 45 &
    local vp=$!
    sleep 18
    stop_bg $J KILL tracker1
    jet "echo killed \\\$(date +%s%N) >> $J/meta.txt; ls /dev/shm | grep trk_ >> $J/meta.txt"
    sleep 3
    start_bg $J tracker2 "\$B/trk_tracker --config \$TCONF --duration 20"
    wait $vp
    sleep 8
    jet "stop_bg() { :; }; kill -TERM \\\$(cat $J/replay.pid) 2>/dev/null; sleep 2; mon_stop $J"
    fetch $J "$M"
    python3 - "$M" <<'PY' | tee "$M/result.txt"
import json, sys
m = sys.argv[1]
v = json.load(open(f"{m}/viewer_stats.json")); b = json.load(open(f"{m}/bridge.json"))
inst = v["instances"]
print(f"viewer: connects {v['connects']} (1 = no reconnect needed), tracker instances seen {len(inst)}: " +
      ", ".join(f"{k}: {x['received']} poses, gaps {x['gaps']}" for k, x in inst.items()))
print(f"bridge: attaches {b['attaches']}, poses read {b['poses_read']}, sent {b['poses_sent']}")
ok = v["connects"] == 1 and len(inst) == 2 and all(x["received"] > 100 for x in inst.values()) and b["attaches"] == 2
print("RESULT", "PASS" if ok else "FAIL")
PY
}

# Long run: 30 min live with bridge and viewer; RSS of tracker, bridge (Jetson) and viewer (Mac) every 30 s.
l_long() {
    local J=~/tracker_logs/long_30min M=$MLOG/long_30min; rm -rf "$M"; mkdir -p "$M"
    local DUR=${LONG_S:-1800}
    jet "rm -rf $J; mon_start $J"
    start_bg $J capture "\$CAP --config \$CONF --duration $((DUR + 30)) --final-json $J/capture.json --hist-out /dev/null --delay-hist-out /dev/null"
    sleep 1
    start_bg $J tracker "\$B/trk_tracker --config \$TCONF --duration $((DUR + 20)) --final-json $J/tracker.json"
    start_bg $J bridge "\$B/trk_bridge --config \$TCONF --duration $((DUR + 22)) --final-json $J/bridge.json"
    sleep 8
    viewer "$M" "$DUR" &
    local vp=$!
    sleep 3
    local vpid; vpid=$(pgrep -f "viewer.py --host $IP --http-port 8090" | head -1)
    echo "t_s,tracker_rss_kb,bridge_rss_kb,viewer_rss_kb" > "$M/rss.csv"
    local t=0
    while kill -0 $vp 2>/dev/null; do
        local r; r=$(jet "ps -o rss= -p \\\$(cat $J/tracker.pid) | tr -d ' '; ps -o rss= -p \\\$(cat $J/bridge.pid) | tr -d ' '" | tr '\n' ',')
        echo "$t,${r}$(ps -o rss= -p "$vpid" 2>/dev/null | tr -d ' ')" >> "$M/rss.csv"
        sleep 30; t=$((t + 30))
    done
    sleep 30
    jet "mon_stop $J"
    fetch $J "$M"
    python3 - "$M" <<'PY' | tee "$M/result.txt"
import csv, json, sys
m = sys.argv[1]
rows = [r for r in csv.DictReader(open(f"{m}/rss.csv")) if all(r[k] for k in ("tracker_rss_kb", "bridge_rss_kb", "viewer_rss_kb"))]
ref = next(r for r in rows if int(r["t_s"]) >= 120)      # after the start-up (2 min)
end = rows[-1]
ok = True
for k in ("tracker_rss_kb", "bridge_rss_kb", "viewer_rss_kb"):
    g = 100.0 * (int(end[k]) / int(ref[k]) - 1)
    print(f"{k}: {ref[k]} kB at {ref['t_s']} s -> {end[k]} kB at {end['t_s']} s ({g:+.1f} %)")
    ok = ok and g <= 5.0
t = json.load(open(f"{m}/tracker.json")); b = json.load(open(f"{m}/bridge.json")); v = json.load(open(f"{m}/viewer_stats.json"))
c = json.load(open(f"{m}/capture.json"))
print(f"tracker: cycles {t['cycles']}, pairs used {t['pairs_used']} of {t['capture_published']}, skipped {t['pairs_skipped']}, "
      f"cycle p99 {t['cycle_p99_ms']} ms; bridge: poses read/sent/dropped {b['poses_read']}/{b['poses_sent']}/{b['poses_dropped']}; "
      f"viewer: poses {v['poses']}, clouds {v['clouds']}, connects {v['connects']}")
print(f"capture: A dropped {c['A']['dropped']} overflows {c['A']['overflows']}, B dropped {c['B']['dropped']} overflows {c['B']['overflows']}")
for f in ("tracker.log", "bridge.log"):
    print(f, "ended normally" if "final:" in open(f"{m}/{f}").read() else "NO final line (crash?)")
print("RESULT", "PASS" if ok and "final:" in open(f"{m}/tracker.log").read() and "final:" in open(f"{m}/bridge.log").read() else "FAIL")
PY
}

for t in $TESTS; do echo "=== $(date +%T) $t"; "l_$t"; done
echo "=== $(date +%T) LINK_TESTS_DONE"
