#!/usr/bin/env python3
"""Evaluates a live (or replay) run directory (standard library): tracker poses (posedump CSV), tracker final JSON,
capture final JSON, tegrastats, kernel lines. Prints one block and writes RUN/eval.json.
    evaluate_live.py RUN_DIR
Calculation time of a cycle = t_total_us (pair reader to tracking filter) + t_publish_us (publish of the pose and,
every 4th to 6th cycle, the cloud); measured on the Jetson in every cycle."""
import csv, json, math, os, re, statistics, sys


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))] if v else float("nan")


def tegra(d):
    p = os.path.join(d, "tegrastats.log")
    cpu, tj, ram = [], [], []
    if os.path.exists(p):
        for line in open(p):
            m = re.search(r"CPU \[([^\]]*)\]", line)
            if m:
                vals = [int(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
                if vals:
                    cpu.append(sum(vals) / len(vals))
            m = re.search(r"tj@([\d.]+)C", line)
            if m:
                tj.append(float(m.group(1)))
            m = re.search(r"RAM (\d+)/(\d+)MB", line)
            if m:
                ram.append(int(m.group(1)))
    return {"cpu_mean_pct": round(statistics.mean(cpu), 1) if cpu else None, "tj_max_c": max(tj) if tj else None,
            "ram_used_mb_max": max(ram) if ram else None}


def kernel(d):
    p = os.path.join(d, "kernel_new.log")
    if not os.path.exists(p):
        return None
    lines = [l.strip() for l in open(p) if not l.startswith("-- ")]
    bad = [l for l in lines if re.search(r"xhci|reset|error|-71|-110|disconnect|over.?current", l, re.I)
           and not re.search(r"Found UVC|Unknown video format|input: Intel|uvcvideo|hid-generic|usbfs", l, re.I)]
    return {"new_lines": len(lines), "usb_error_lines": len(bad), "examples": bad[:5]}


def main():
    d = sys.argv[1]
    out = {"run": os.path.basename(d)}
    rows = list(csv.DictReader(open(os.path.join(d, "poses.csv")))) if os.path.exists(os.path.join(d, "poses.csv")) else []
    if rows:
        calc = [float(r["t_total_us"]) + float(r["t_publish_us"]) for r in rows]
        track = [r for r in rows if r["state"] == "1"]
        ctrack = [float(r["t_total_us"]) + float(r["t_publish_us"]) for r in track]
        out["poses"] = len(rows)
        out["state_fraction"] = {n: round(sum(1 for r in rows if r["state"] == s) / len(rows), 4) for s, n in (("0", "search"), ("1", "track"), ("2", "lost"))}
        out["valid_fraction"] = round(sum(1 for r in rows if r["valid"] == "1") / len(rows), 4)
        out["calc_all_ms"] = {"p50": q(calc, .5) / 1000, "p99": q(calc, .99) / 1000, "max": max(calc) / 1000,
                              "le_5ms_pct": 100 * sum(c <= 5000 for c in calc) / len(calc), "le_2ms_pct": 100 * sum(c <= 2000 for c in calc) / len(calc)}
        if ctrack:
            out["calc_track_ms"] = {"cycles": len(ctrack), "p50": q(ctrack, .5) / 1000, "p99": q(ctrack, .99) / 1000, "max": max(ctrack) / 1000,
                                    "le_5ms_pct": 100 * sum(c <= 5000 for c in ctrack) / len(ctrack), "le_2ms_pct": 100 * sum(c <= 2000 for c in ctrack) / len(ctrack)}
        delay = [(int(r["publish_time_mono_ns"]) - int(r["meas_time_mono_ns"])) / 1e6 for r in rows]
        out["delay_capture_to_publish_ms"] = {"p50": q(delay, .5), "p99": q(delay, .99)}
        vt = [r for r in track if r["valid"] == "1"]
        if len(vt) > 2:
            P = [[float(r[c]) for c in ("px", "py", "pz")] for r in vt]
            mean = [statistics.mean(c) for c in zip(*P)]
            sd = [statistics.pstdev(c) * 1000 for c in zip(*P)]
            # orientation spread: angle of each quaternion to the first one
            q0 = [float(vt[0][c]) for c in ("qw", "qx", "qy", "qz")]
            ang = []
            for r in vt:
                q1 = [float(r[c]) for c in ("qw", "qx", "qy", "qz")]
                dd = abs(sum(a * b for a, b in zip(q0, q1)))
                ang.append(2 * math.degrees(math.acos(min(1.0, dd))))
            out["static"] = {"valid_track_poses": len(vt), "mean_position_m": [round(m, 5) for m in mean],
                             "position_sd_mm": [round(s, 3) for s in sd], "orientation_sd_deg": round(statistics.pstdev(ang), 3),
                             "rmse_mm_mean": round(1000 * statistics.mean(float(r["rmse"]) for r in vt), 3),
                             "inlier_ratio_mean": round(statistics.mean(float(r["inlier_ratio"]) for r in vt), 4)}
    for name in ("tracker.json", "capture.json", "bridge.json", "viewer_stats.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            try:
                out[name.replace(".json", "")] = json.load(open(p))
            except ValueError:
                pass
    t = out.get("tracker")
    if t and t.get("capture_published"):
        out["pair_use_pct"] = round(100.0 * t["pairs_used"] / t["capture_published"], 3)
    c = out.get("capture")
    if c:
        out["capture_counters"] = {k: {f: c[k][f] for f in ("dropped", "overflows", "restarts", "callback_errors")} for k in ("A", "B") if k in c}
    out["tegrastats"] = tegra(d)
    out["kernel"] = kernel(d)
    print(json.dumps({k: v for k, v in out.items() if k not in ("capture", "tracker", "bridge", "viewer_stats")}, indent=1))
    json.dump(out, open(os.path.join(d, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
