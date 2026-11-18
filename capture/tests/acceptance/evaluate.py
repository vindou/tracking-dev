#!/usr/bin/env python3
"""Evaluates the T2 acceptance runs (log analysis only; standard library).

    evaluate.py LOGROOT            -> prints one line per run and per test, writes LOGROOT/verdict.json

Pass criteria are the ones of the task (TEST_REPORT.md lists them). Values come from final.json (written by
d435_capture), pairs.csv (independent check of every pair), reader.json, meta.txt and kernel_new.log.
"""
import csv
import glob
import json
import os
import re
import sys

KERNEL_PAT = re.compile(r"xhci|reset|error|-71|-110|throttl|over.?current|disconnect|cannot enable", re.I)
KERNEL_BENIGN = re.compile(r"Found UVC|Unknown video format|input: Intel|uvcvideo|hid-generic|input,hidraw", re.I)


def meta(d):
    m = {}
    p = os.path.join(d, "meta.txt")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line and not line.startswith(" "):
                k, v = line.strip().split("=", 1)
                m[k] = v
    return m


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def kernel_relevant(d):
    p = os.path.join(d, "kernel_new.log")
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p) if not l.startswith("-- ")]
    return [l.strip() for l in lines if KERNEL_PAT.search(l) and not KERNEL_BENIGN.search(l)]


def tegra(d):
    p = os.path.join(d, "tegrastats.log")
    if not os.path.exists(p):
        return None
    cpu, tj = [], []
    for line in open(p):
        m = re.search(r"CPU \[([^\]]*)\]", line)
        if m:
            vals = [int(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
            if vals:
                cpu.append(sum(vals) / len(vals))
        m = re.search(r"tj@([\d.]+)C", line)
        if m:
            tj.append(float(m.group(1)))
    return {"cpu_mean": round(sum(cpu) / len(cpu), 1) if cpu else None, "tj_max": max(tj) if tj else None}


def pairs_csv_check(d, tol):
    """Independent check of every pair: |dt| <= tolerance, pair_seq without gaps, frame counters of each camera strictly
    increasing within an epoch (no frame in two pairs, pairs in time order)."""
    p = os.path.join(d, "pairs.csv")
    if not os.path.exists(p):
        return None
    n = 0
    over_tol = 0
    seq_err = 0
    order_err = 0
    max_abs = 0.0
    last = None
    restarts_seen = 0
    for row in csv.DictReader(open(p)):
        n += 1
        dt = float(row["dt_ms"])
        max_abs = max(max_abs, abs(dt))
        if tol is not None and abs(dt) > tol + 1e-9:
            over_tol += 1
        seq = int(row["pair_seq"])
        ta, tb = float(row["a_global_ts_ms"]), float(row["b_global_ts_ms"])
        ca, cb = int(row["a_frame_counter"]), int(row["b_frame_counter"])
        if last is not None:
            if seq != last["seq"] + 1:
                seq_err += 1
            if not (ta > last["ta"] and tb > last["tb"]):
                order_err += 1
            if ca <= last["ca"] or cb <= last["cb"]:
                restarts_seen += 1   # counter restart: only allowed after a camera restart
        last = {"seq": seq, "ta": ta, "tb": tb, "ca": ca, "cb": cb}
    return {"pairs": n, "over_tolerance": over_tol, "max_abs_dt_ms": round(max_abs, 4), "seq_errors": seq_err,
            "time_order_errors": order_err, "counter_restarts": restarts_seen}


def run_values(d):
    f = load_json(os.path.join(d, "final.json"))
    m = meta(d)
    v = {"run": os.path.basename(d), "rc": int(m.get("capture_rc", -1)), "shm_left": int(m.get("shm_left", -1)),
         "own_processes_left": int(m.get("own_processes_left", -1)), "final": f is not None}
    if f:
        tol = f["matcher"]["tolerance_ms"]
        v.update({
            "run_s": f["run_s"], "pairs": f["matcher"]["pairs"], "tolerance_ms": tol,
            "dt_min_ms": f["matcher"]["dt_min_ms"], "dt_max_ms": f["matcher"]["dt_max_ms"],
            "partner_changes": f["matcher"]["partner_changes"], "order_errors": f["matcher"]["order_errors"],
            "period_ms": f["matcher"]["period_ms"],
        })
        for c in ("A", "B"):
            x = f[c]
            elig = x["eligible"]
            v[c] = {k: x[k] for k in ("frames", "dropped", "overflows", "max_fill", "restarts", "unpaired", "cb_p99_ms",
                                      "cb_max_ms", "lat_p50_ms", "lat_p99_ms", "fps_warnings", "eligible", "paired",
                                      "period_ms", "callback_errors", "ignored", "after_stop", "timeout_decisions")}
            v[c]["paired_pct"] = round(100.0 * x["paired"] / elig, 3) if elig else 0.0
            v[c]["fps"] = round(x["frames"] / f["run_s"], 3)
        v["csv"] = pairs_csv_check(d, tol)
    v["kernel_relevant"] = kernel_relevant(d)
    v["tegra"] = tegra(d)
    r = load_json(os.path.join(d, "reader.json"))
    if r:
        v["reader"] = r
    return v


def baseline_checks(v):
    """Baseline criteria: 0 dropped, 0 overflows, 0 restarts, >= 99% of eligible frames paired, all pair differences
    within the tolerance. Plus: clean exit, shared memory removed."""
    c = []
    ok_final = v["final"]
    c.append(("exit code 0", v["rc"] == 0))
    c.append(("final counters written", ok_final))
    if ok_final:
        for cam in ("A", "B"):
            x = v[cam]
            c.append((f"{cam} dropped == 0 ({x['dropped']})", x["dropped"] == 0))
            c.append((f"{cam} overflows == 0 ({x['overflows']})", x["overflows"] == 0))
            c.append((f"{cam} restarts == 0 ({x['restarts']})", x["restarts"] == 0))
            c.append((f"{cam} paired >= 99% of eligible ({x['paired_pct']}%)", x["paired_pct"] >= 99.0))
            # R2 budget (no other load): 99% of callbacks < 0.5 ms. cb_p99_ms is the upper edge of a 10 us bin.
            c.append((f"{cam} callback p99 < 0.5 ms ({x['cb_p99_ms']:.3f} ms)", x["cb_p99_ms"] < 0.5))
        within = v["dt_min_ms"] >= -v["tolerance_ms"] - 1e-9 and v["dt_max_ms"] <= v["tolerance_ms"] + 1e-9
        c.append((f"all dt within +-{v['tolerance_ms']:.4f} ms (min {v['dt_min_ms']}, max {v['dt_max_ms']})", within))
        if v["csv"]:
            c.append((f"csv: {v['csv']['pairs']} pairs, 0 over tolerance, 0 seq/order errors, no frame reused",
                      v["csv"]["over_tolerance"] == 0 and v["csv"]["seq_errors"] == 0 and v["csv"]["time_order_errors"] == 0
                      and v["csv"]["counter_restarts"] == 0 and v["csv"]["pairs"] == v["pairs"]))
        c.append((f"order_errors == 0 ({v['order_errors']})", v["order_errors"] == 0))
    c.append(("shared memory removed", v["shm_left"] == 0))
    return c


def evaluate(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*_*"))):
        if os.path.isdir(d):
            runs.setdefault(re.sub(r"_\d+$", "", os.path.basename(d)), []).append(d)
    verdict = {}
    for test, dirs in runs.items():
        results = []
        for d in dirs:
            v = run_values(d)
            if test in ("baseline", "long"):
                checks = baseline_checks(v)
            elif test == "reader":
                checks = baseline_checks(v)
                r = v.get("reader")
                checks.append(("reader result written", r is not None))
                if r:
                    expected = v.get("pairs", -1) - r["start_seq"]
                    checks.append((f"reader missing == 0 ({r['missing']}), seq gaps == 0 ({r['seq_gaps']})",
                                   r["missing"] == 0 and r["seq_gaps"] == 0))
                    checks.append((f"reader received all pairs after attach ({r['received']} of {expected})",
                                   r["received"] == expected))
                    checks.append(("reader saw writer stop", r["writer_stopped"]))
            elif test == "reader_stopped":
                checks = baseline_checks(v)   # "no effect on the capture counters" = baseline criteria hold
                checks.append(("reader was stopped (SIGSTOP) during the run", meta(d).get("reader_was_stopped_at_end") == "1"))
            elif test == "memload":
                # lost frames are permitted; criteria: no crash, no deadlock (timeout 120 s for a 60 s run)
                checks = [("exit code 0 (no crash; 124 would be a timeout/deadlock)", v["rc"] == 0),
                          ("final counters written", v["final"]),
                          ("shared memory removed", v["shm_left"] == 0)]
                if v["final"]:
                    checks.append((f"all dt within tolerance", v["dt_min_ms"] >= -v["tolerance_ms"] - 1e-9
                                   and v["dt_max_ms"] <= v["tolerance_ms"] + 1e-9))
            elif test == "startstop":
                checks = [("exit code 0 (no stall: finished within the timeout)", v["rc"] == 0),
                          ("final counters written", v["final"]),
                          ("shared memory removed", v["shm_left"] == 0)]
                if v["final"]:
                    checks.append((f"pairs supplied ({v['pairs']})", v["pairs"] > 0))
                    checks.append((f"frames from both cameras ({v['A']['frames']}/{v['B']['frames']})",
                                   v["A"]["frames"] > 0 and v["B"]["frames"] > 0))
                    checks.append((f"no camera restart (stall) ({v['A']['restarts']}/{v['B']['restarts']})",
                                   v["A"]["restarts"] == 0 and v["B"]["restarts"] == 0))
            elif test == "watchdog":
                checks = [("exit code 0", v["rc"] == 0), ("final counters written", v["final"]),
                          ("shared memory removed", v["shm_left"] == 0)]
                if v["final"]:
                    checks.append((f"camera A restarted ({v['A']['restarts']})", v["A"]["restarts"] >= 1))
                    checks.append((f"camera B not restarted ({v['B']['restarts']})", v["B"]["restarts"] == 0))
                    checks.append((f"frames ignored by the test option ({v['A']['ignored']})", v["A"]["ignored"] > 0))
                    after = pairs_after_restart(d)
                    checks.append((f"pairs continue after the restart ({after})", after is not None and after > 0))
            else:
                checks = [("unknown test", False)]
            passed = all(ok for _, ok in checks)
            results.append({"values": v, "checks": [[n, ok] for n, ok in checks], "pass": passed})
        verdict[test] = {"runs": results, "pass": all(r["pass"] for r in results)}
    return verdict


def pairs_after_restart(d):
    """Number of pairs whose camera-A frame counter restarted (pairs from the new stream of camera A)."""
    p = os.path.join(d, "pairs.csv")
    if not os.path.exists(p):
        return None
    last = None
    after = 0
    restarted = False
    for row in csv.DictReader(open(p)):
        ca = int(row["a_frame_counter"])
        if last is not None and ca < last:
            restarted = True
        if restarted:
            after += 1
        last = ca
    return after


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/capture_logs/t2")
    verdict = evaluate(root)
    for test, t in verdict.items():
        print(f"== {test}: {'PASS' if t['pass'] else 'FAIL'}")
        for r in t["runs"]:
            v = r["values"]
            line = f"  {v['run']:<18} {'PASS' if r['pass'] else 'FAIL'} rc={v['rc']}"
            if v["final"]:
                line += (f" run={v['run_s']:.0f}s pairs={v['pairs']} chg={v['partner_changes']}"
                         f" dt=[{v['dt_min_ms']:.3f},{v['dt_max_ms']:.3f}]")
                for c in ("A", "B"):
                    x = v[c]
                    line += (f" | {c}: fps={x['fps']} drop={x['dropped']} ovf={x['overflows']} fill={x['max_fill']}"
                             f" rst={x['restarts']} paired={x['paired_pct']}% unp={x['unpaired']}"
                             f" cb99={x['cb_p99_ms']:.3f} cbmax={x['cb_max_ms']:.3f} lat50/99={x['lat_p50_ms']}/{x['lat_p99_ms']}")
            if v.get("tegra"):
                line += f" | cpu={v['tegra']['cpu_mean']}% tj={v['tegra']['tj_max']}"
            if v.get("kernel_relevant") is not None:
                line += f" kern={len(v['kernel_relevant'])}"
            print(line)
            for n, ok in r["checks"]:
                if not ok:
                    print(f"      FAILED: {n}")
    with open(os.path.join(root, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)


if __name__ == "__main__":
    main()
