#!/bin/bash
# Why the networks take longer in the live test than when trk_flow runs alone: every 8th scene of the evaluation set
# through trk_flow on the Jetson (scenes mode, --probe) under one condition at a time, two repetitions.
#   flow/jetson_contention.sh start     run detached on the Jetson (about 35 min; ~/flow_probe/contention/done)
#   flow/jetson_contention.sh table     fetch to flow/runs/contention and print the table
# Conditions (trk_flow is otherwise alone, back to back):
#   base                         nothing else
#   gapN                         N ms pause after every scene (the GPU and the core idle, as between live proposals)
#   cpu1, cpu5, cpu6             that many busy loops (no memory traffic); cpu1 on core 5
#   mem2                         two loops copying 64 MB arrays (memory traffic)
#   scenes                       trk_scenes publishing pairs
#   tracker                      trk_scenes and trk_tracker --no-hints (always searching)
#   tracker_pin                  the same, both on cores 0-3 and trk_flow on cores 4-5
#   tracker_gap700               tracker with a 700 ms pause (closest to the live test)
#   gap700_pin5                  700 ms pause, trk_flow on core 5
#   gap700_pin5_warm             the same with a busy loop at nice 19 on core 4 (keeps the CPU clock of that cluster up
#                                while the GPU still idles in the pauses)
# No root is needed: the clocks are only read.
set -e
cd "$(dirname "$0")/.."
if [ "$1" = table ]; then
  mkdir -p flow/runs/contention
  rsync -a jetson:flow_probe/contention/ flow/runs/contention/
  exec .venv/bin/python - <<'P'
import csv, glob, numpy as np
from pathlib import Path
cat = np.load("flow/data/eval2/meta.npz")["cat"]
rows = {}
for f in sorted(glob.glob("flow/runs/contention/*_r[0-9].csv")):
    name, rep = Path(f).stem.rsplit("_r", 1)
    r = [x for x in csv.DictReader(open(f)) if str(cat[int(x["scene"])]) != "clean"]      # four positions: the 128-hypothesis engines
    g = lambda k: np.array([float(x[k]) for x in r])
    rows.setdefault(name, []).append(dict(net=np.median(g("net_ms")), p90=np.percentile(g("net_ms"), 90), s1=np.median(g("net1_ms")),
        prep=np.median(g("prep_ms") + g("cloud_ms")), gb=np.median(g("gpu_hz_before")) / 1e6, ga=np.median(g("gpu_hz_after")) / 1e6,
        cb=np.median(g("cpu_khz_before")) / 1e3, ca=np.median(g("cpu_khz_after")) / 1e3, mig=np.mean(g("core_before") != g("core_after"))))
print(f"{'condition':18s} networks ms, median per repetition (p90) | position stage | cloud+prep | GPU MHz before -> after | CPU MHz before -> after | changed core")
key = lambda n: (0, 0) if n == "base" else (1, int(n[3:])) if n.startswith("gap") and n[3:].isdigit() else (2, 0)
for name in sorted(rows, key=lambda n: (key(n), n)):
    v = rows[name]
    print(f"{name:18s} " + " ".join(f"{x['net']:6.1f}" for x in v) + f"  ({np.mean([x['p90'] for x in v]):6.1f}) | {np.mean([x['s1'] for x in v]):5.1f} | {np.mean([x['prep'] for x in v]):5.1f} | "
          f"{np.mean([x['gb'] for x in v]):5.0f} -> {np.mean([x['ga'] for x in v]):5.0f} | {np.mean([x['cb'] for x in v]):5.0f} -> {np.mean([x['ca'] for x in v]):5.0f} | {100 * np.mean([x['mig'] for x in v]):3.0f} %")
P
fi
ssh -f jetson "setsid nohup bash -s > ~/flow_probe/contention.out 2>&1 < /dev/null <<'REMOTE' &
D=~/flow_probe/contention; rm -rf \$D; mkdir -p \$D
F=~/flow_live/build/trk_flow; T=~/tracker
flow() {   # name, prefix command (or -), options
  local name=\$1 pre=\$2; shift 2; [ \"\$pre\" = - ] && pre=''
  \$pre \$F --config \$T/config/tracker.json --engines ~/flow_probe/tanh --scenes ~/flow_probe/frozen/trt_eval/scenes.bin --step 8 --probe --out \$D/\$name.bin \"\$@\" > \$D/\$name.log 2>&1
  mv \$D/\$name.bin.times.csv \$D/\$name.csv; rm -f \$D/\$name.bin; sleep 3
}
PIDS=''
burn() { local pre=\$1; \$pre bash -c 'while :; do :; done' & PIDS=\"\$PIDS \$!\"; }
memload() { python3 -c 'import numpy as np
a = np.ones(8 << 20)
while True: b = a.copy()' & PIDS=\"\$PIDS \$!\"; }
track() {  # prefix command (or -): tracker without hints and the scene source, as in flow/live_test.sh
  local pre=\$1; [ \"\$pre\" = - ] && pre=''
  ( cd \$T && exec \$pre build/trk_tracker --config config/tracker.json --no-hints ) > \$D/tracker.log 2>&1 & PIDS=\"\$PIDS \$!\"
  sleep 2
  ( cd \$T && exec \$pre build/trk_scenes --scenes ~/flow_probe/frozen/trt_eval/scenes.bin --log \$D/scenes_log.csv --step 8 --hold 2.0 --blank 0.7 ) > /dev/null 2>&1 & PIDS=\"\$PIDS \$!\"
  sleep 3
}
scenes_only() { ( cd \$T && exec build/trk_scenes --scenes ~/flow_probe/frozen/trt_eval/scenes.bin --log \$D/scenes_log.csv --step 8 --hold 2.0 --blank 0.7 ) > /dev/null 2>&1 & PIDS=\"\$PIDS \$!\"; sleep 3; }
stop() { for p in \$PIDS; do kill -INT \$p 2>/dev/null; done; sleep 1; for p in \$PIDS; do kill -9 \$p 2>/dev/null; done; wait 2>/dev/null; PIDS=''; sleep 2; }
cond() {
  case \$1 in
    base) flow base_r\$2 - ;;
    gap*_pin5_warm) burn 'nice -n 19 taskset -c 4'; flow \$1_r\$2 'taskset -c 5' --gap-ms 700; stop ;;
    gap*_pin5) flow \$1_r\$2 'taskset -c 5' --gap-ms 700 ;;
    gap*) flow \$1_r\$2 - --gap-ms \${1#gap} ;;
    cpu1) burn 'taskset -c 5'; flow \$1_r\$2 -; stop ;;
    cpu5) for i in 1 2 3 4 5; do burn ''; done; flow \$1_r\$2 -; stop ;;
    cpu6) for i in 1 2 3 4 5 6; do burn ''; done; flow \$1_r\$2 -; stop ;;
    mem2) memload; memload; sleep 2; flow \$1_r\$2 -; stop ;;
    scenes) scenes_only; flow \$1_r\$2 -; stop ;;
    tracker) track -; flow \$1_r\$2 -; stop ;;
    tracker_pin) track 'taskset -c 0-3'; flow \$1_r\$2 'taskset -c 4-5'; stop ;;
    tracker_gap700) track -; flow \$1_r\$2 - --gap-ms 700; stop ;;
  esac
  echo \"\$1 r\$2 \$(date +%H:%M:%S)\" >> \$D/progress
}
L='base gap20 gap50 gap100 gap300 gap700 cpu1 cpu5 cpu6 mem2 scenes tracker tracker_pin tracker_gap700 gap700_pin5 gap700_pin5_warm'
for c in \$L; do cond \$c 1; done
for c in \$(echo \$L | tr ' ' '\\n' | tac); do cond \$c 2; done
touch \$D/done
REMOTE"
echo started
