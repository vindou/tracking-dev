#!/usr/bin/env python3
"""Evaluates the gap-closure runs (log analysis only; standard library). Runs on the Jetson.

    evaluate_gaps.py GAPROOT [REGROOT]

GAPROOT: kill_1/ (gap 1), hwreset_1/ and unplug_A|B/ (gap 2) from gap_tests.sh and manual_unplug.sh.
REGROOT: regression runs from t2_all.sh: startstop_*/ (gap 4) and long_1/ (gap 3).
Prints one block per test and writes GAPROOT/gap_verdict.json. Pass criteria are the ones of the task.
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys

NEAR_HALF_MS = 0.5
EVENT_RE = re.compile(r"^\[event \+([\d.]+)s\] cam=(\S+) what=(\S+) mono_ns=(\d+) real_ns=(\d+) epoch=(\d+) "
                      r"dropA=(\d+) ovfA=(\d+) dropB=(\d+) ovfB=(\d+) pairs=(\d+) rss_kb=(\d+)(.*)$")
KMONO_RE = re.compile(r"^\[\s*([\d.]+)\]")


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def meta(d, name="meta.txt"):
    m = {}
    p = os.path.join(d, name)
    if os.path.exists(p):
        for line in open(p):
            if "=" in line and not line.startswith(" "):
                k, v = line.strip().split("=", 1)
                m[k] = v
    return m


def events(path):
    ev = []
    for line in open(path):
        m = EVENT_RE.match(line.strip())
        if m:
            ev.append({"t": float(m.group(1)), "cam": m.group(2), "what": m.group(3), "mono_ns": int(m.group(4)),
                       "real_ns": int(m.group(5)), "epoch": int(m.group(6)), "dropA": int(m.group(7)),
                       "ovfA": int(m.group(8)), "dropB": int(m.group(9)), "ovfB": int(m.group(10)),
                       "pairs": int(m.group(11)), "rss_kb": int(m.group(12)), "extra": m.group(13).strip()})
    return ev


def kernel_mono_lines(d):
    """Kernel lines of the run with CLOCK_MONOTONIC-like times (s): kernel_mono.log, or journalctl over the run's
    uptime window."""
    p = os.path.join(d, "kernel_mono.log")
    lines = []
    if os.path.exists(p) and os.path.getsize(p) > 0:
        lines = open(p).read().splitlines()
    else:
        m = meta(d)
        try:
            t0, t1 = float(m["start_uptime_s"]), float(m["end_uptime_s"])
            out = subprocess.run(["journalctl", "-k", "--no-pager", "-o", "short-monotonic",
                                  "--since", "-%d s" % int(10 ** 7)], capture_output=True, text=True).stdout
            for l in out.splitlines():
                mm = KMONO_RE.match(l)
                if mm and t0 - 1 <= float(mm.group(1)) <= t1 + 1:
                    lines.append(l)
            with open(p, "w") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass
    res = []
    for l in lines:
        mm = KMONO_RE.match(l)
        if mm:
            res.append((float(mm.group(1)), l))
    return res


def read_csv(p):
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p)))


def pair_order_check(rows):
    """Pairs in time order and no frame in two pairs."""
    seq_err = order_err = dup = 0
    last_ts = [-1.0, -1.0]
    for i, r in enumerate(rows):
        if int(r["pair_seq"]) != i:
            seq_err += 1
        for c, p in enumerate("ab"):
            ts = float(r[p + "_global_ts_ms"])
            if i > 0 and ts <= last_ts[c]:
                order_err += 1
            last_ts[c] = ts
    # a frame (same epoch and frame counter, or same arrival time) in two pairs
    for p in "ab":
        ec = [(r[p + "_epoch"], r[p + "_frame_counter"]) for r in rows]
        dup += len(ec) - len(set(ec))
        arr = [r[p + "_arrival_mono_ns"] for r in rows]
        dup += len(arr) - len(set(arr))
    return {"pairs": len(rows), "seq_errors": seq_err, "time_order_errors": order_err, "duplicate_frames": dup}


# ------------------------------------------------------------------------------------------------ gap 1
def eval_kill(d):
    r = load_json(os.path.join(d, "reader.json"))
    if not r:
        return {"pass": False, "error": "reader.json missing"}
    kills, starts = {}, {}
    for line in open(os.path.join(d, "kills.txt")):
        m = re.match(r"(start|kill)_(\d+) real_ns=(\d+)", line)
        if m:
            (starts if m.group(1) == "start" else kills)[int(m.group(2))] = int(m.group(3))
    inst = r["instances"]
    res = {"attaches": r["attaches"], "instances": len(inst), "cycles": []}
    ok = r["attaches"] == 6 and len(inst) == 6 and len({i["instance_id"] for i in inst}) == 6
    for k in range(1, 7):
        i = inst[k - 1] if k - 1 < len(inst) else None
        c = {"writer": k}
        if i is None:
            c["error"] = "no instance"
            ok = False
            res["cycles"].append(c)
            continue
        c.update({"instance_id": i["instance_id"], "received": i["received"], "first_seq": i["first_seq"],
                  "missing": i["missing"], "seq_gaps": i["seq_gaps"], "end": i["end"]})
        # first pair of this writer: seq 0 in its CSV (publish_real_ns), compared with the reader's first receive
        rows = read_csv(os.path.join(d, "pairs_%d.csv" % k))
        pub0 = int(rows[0]["publish_real_ns"]) if rows and rows[0]["pair_seq"] == "0" else None
        if i["first_seq"] == 0:
            pub0 = i["first_publish_real_ns"]
        c["first_pair_delay_s"] = round((i["first_recv_real_ns"] - pub0) / 1e9, 4) if pub0 and i["received"] else None
        c["pass_follow"] = (c["first_pair_delay_s"] is not None and c["first_pair_delay_s"] <= 2.0 and
                            i["missing"] == 0 and i["seq_gaps"] == 0)
        if k <= 5:
            c["detect_s"] = round((i["end_real_ns"] - kills[k]) / 1e9, 3) if k in kills else None
            c["pass_detect"] = i["end"] == "writer_not_alive" and c["detect_s"] is not None and 0 <= c["detect_s"] <= 2.0
        else:
            c["pass_detect"] = i["end"] == "writer_stopped"
        if k >= 2:
            so = open(os.path.join(d, "stdout_%d.log" % k)).read()
            c["replaced_note"] = "replaced old shared-memory object" in so
        ok = ok and c["pass_follow"] and c["pass_detect"]
        res["cycles"].append(c)
    m = meta(d)
    res["shm_left_after_clean_stop"] = m.get("shm_left")
    res["last_capture_rc"] = m.get("capture_rc")
    ok = ok and m.get("shm_left") == "0" and m.get("capture_rc") == "0"
    res["pass"] = bool(ok)
    return res


# ------------------------------------------------------------------------------------------------ gap 2
def eval_reset(d, label):
    so = os.path.join(d, "stdout.log")
    if not os.path.exists(so):
        return {"pass": False, "error": "stdout.log missing"}
    ev = events(so)
    f = load_json(os.path.join(d, "final.json"))
    m = meta(d)
    text = open(so).read()
    kern = kernel_mono_lines(d)
    rows = read_csv(os.path.join(d, "pairs.csv"))
    first_pair_by_epoch = {}
    for r in rows:
        for c in "ab":
            k = (c.upper(), int(r[c + "_epoch"]))
            if k not in first_pair_by_epoch:
                first_pair_by_epoch[k] = int(r["publish_mono_ns"])
    # one loss per reset request (hwreset) or per frames_lost event (unplug)
    trig = [e for e in ev if e["what"] == "hw_reset_request"]
    if not trig:
        trig = [e for e in ev if e["what"] == "frames_lost"]
    resets = []
    ok = True
    for n, e in enumerate(trig):
        t_req = e["mono_ns"] / 1e9
        t_next = trig[n + 1]["mono_ns"] / 1e9 if n + 1 < len(trig) else 1e18
        # The kernel shows the reset in one of two forms: "USB disconnect" + "new SuperSpeed USB device number N"
        # (re-enumeration with a new address), or "reset SuperSpeed USB device number N" (the USB core resets the
        # port and re-initialises the device with the same address; uvcvideo probes it again). The re-enumeration time
        # is the "new ... device" or "reset ... device" line.
        first = next(((t, l) for t, l in kern if t_req - 0.05 <= t < min(t_next, t_req + 30)
                      and ("USB disconnect" in l or "reset SuperSpeed USB device" in l)), None)
        if first and "USB disconnect" in first[1]:
            disc = first
            new = next(((t, l) for t, l in kern if t >= disc[0] and t < min(t_next, t_req + 30)
                        and "new SuperSpeed USB device" in l), None)
            form = "disconnect+new_device" if new else "disconnect_only"
        elif first:
            disc, new, form = None, first, "port_reset"
        else:
            disc, new, form = None, None, "none"
        rst = next((x for x in ev if x["cam"] == e["cam"] and x["what"] == "restarted" and x["mono_ns"] >= e["mono_ns"]
                    and x["mono_ns"] / 1e9 < t_next), None)
        res_ev = next((x for x in ev if x["cam"] == e["cam"] and x["what"] == "pairs_resumed"
                       and x["mono_ns"] >= e["mono_ns"] and x["mono_ns"] / 1e9 < t_next), None)
        r = {"n": n + 1, "cam": e["cam"], "t_request_s": round(e["t"], 3), "kernel_form": form}
        r["kernel_disconnect"] = disc[1].split("kernel: ", 1)[-1] if disc else None
        r["kernel_new_device"] = new[1].split("kernel: ", 1)[-1] if new else None
        r["reenum_after_request_s"] = round(new[0] - t_req, 3) if new else None
        r["restart_epoch"] = rst["epoch"] if rst else None
        r["restart_open"] = rst["extra"] if rst else None
        fp = first_pair_by_epoch.get((e["cam"], rst["epoch"])) if rst else None
        r["first_pair_after_reenum_s"] = round(fp / 1e9 - new[0], 3) if (fp and new) else None
        r["first_pair_after_request_s"] = round(fp / 1e9 - t_req, 3) if fp else None
        r["pairs_resumed_event"] = bool(res_ev)
        r["pass"] = (new is not None and r["first_pair_after_reenum_s"] is not None and
                     r["first_pair_after_reenum_s"] <= 15.0)
        ok = ok and r["pass"]
        resets.append(r)
    # other camera: drops and overflows during each phase (phase = from the first reset of a camera to the first
    # reset of the other camera, or to the end of the run)
    phases = []
    cams_in_order = []
    for e in trig:
        if not cams_in_order or cams_in_order[-1][0] != e["cam"]:
            cams_in_order.append((e["cam"], e))
    for i, (cam, e0) in enumerate(cams_in_order):
        other = "B" if cam == "A" else "A"
        if i + 1 < len(cams_in_order):
            e1 = cams_in_order[i + 1][1]
            d_drop = e1["drop" + other] - e0["drop" + other]
            d_ovf = e1["ovf" + other] - e0["ovf" + other]
        elif f:
            d_drop = f[other]["dropped"] - e0["drop" + other]
            d_ovf = f[other]["overflows"] - e0["ovf" + other]
        else:
            d_drop = d_ovf = None
        ph = {"reset_camera": cam, "other_camera": other, "other_drops": d_drop, "other_overflows": d_ovf,
              "pass": d_drop == 0 and d_ovf == 0}
        ok = ok and ph["pass"]
        phases.append(ph)
    res = {"label": label, "resets": len(resets), "per_reset": resets, "phases": phases}
    rec = [r["first_pair_after_reenum_s"] for r in resets if r["first_pair_after_reenum_s"] is not None]
    res["kernel_forms"] = {f: sum(1 for r in resets if r["kernel_form"] == f) for f in
                           ("disconnect+new_device", "port_reset", "disconnect_only", "none")}
    res["first_pair_after_reenum_s_max"] = max(rec) if rec else None
    res["first_pair_after_reenum_s_min"] = min(rec) if rec else None
    # program did not stop or deadlock
    res["capture_rc"] = m.get("capture_rc")
    res["clean_end"] = "stopping (duration reached)" in text and "shared memory /d435_pairs removed" in text
    ok = ok and m.get("capture_rc") == "0" and res["clean_end"]
    if f:
        p = f.get("process", {})
        res["rss_kb_after_first_recovery"] = p.get("rss_kb_after_first_recovery")
        res["rss_kb_end"] = p.get("rss_kb_end")
        if label.startswith("hwreset"):
            first_rec = res["rss_kb_after_first_recovery"]
        else:   # unplug: first pairs_resumed event
            pr = [e for e in ev if e["what"] == "pairs_resumed"]
            first_rec = pr[0]["rss_kb"] if pr else None
            res["rss_kb_after_first_recovery"] = first_rec
        res["rss_growth_pct"] = round(100.0 * (res["rss_kb_end"] / first_rec - 1), 2) if first_rec else None
        res["pass_rss"] = res["rss_growth_pct"] is not None and res["rss_growth_pct"] <= 5.0
        ok = ok and res["pass_rss"]
        res["hw_resets_requested"] = p.get("hw_resets_requested")
        res["hw_reset_failures"] = p.get("hw_reset_failures")
        res["hw_recovery_timeouts"] = p.get("hw_recovery_timeouts")
        res["callback_errors"] = [f["A"]["callback_errors"], f["B"]["callback_errors"]]
        if label.startswith("hwreset"):
            ok = ok and p.get("hw_resets_requested") == 20 and p.get("hw_reset_failures") == 0 and \
                p.get("hw_recovery_timeouts") == 0
    else:
        ok = False
    res["status_lines_with_condition"] = len(re.findall(r"\[status [^\n]*(?:\[reset\]|\[restarting\]|\[absent\]|\[settling\])",
                                                        text))
    oc = pair_order_check(rows)
    res["pair_check"] = oc
    ok = ok and oc["seq_errors"] == 0 and oc["time_order_errors"] == 0 and oc["duplicate_frames"] == 0
    rd = load_json(os.path.join(d, "reader.json"))
    if rd:
        res["reader"] = {"attaches": rd["attaches"], "received": rd["received"], "missing": rd["missing"],
                         "seq_gaps": rd["seq_gaps"]}
    res["pass"] = bool(ok and len(resets) > 0 and all(r["pass"] for r in resets))
    return res


# ------------------------------------------------------------------------------------------------ gap 4
def eval_startstop(reg):
    cyc = []
    ok = True
    dirs = sorted(glob.glob(os.path.join(reg, "startstop_*")))
    for d in dirs:
        f = load_json(os.path.join(d, "final.json"))
        if not f:
            cyc.append({"run": os.path.basename(d), "error": "final.json missing"})
            ok = False
            continue
        P = f["matcher"]["period_ms"]
        rows = read_csv(os.path.join(d, "pairs.csv"))
        dts = sorted(float(r["dt_ms"]) for r in rows)
        dt = dts[len(dts) // 2] if dts else f["matcher"]["dt_median_ms"]
        # two readings of "cycle whose dt is within 0.5 ms of half a period": the median dt of the cycle, or any pair
        near = abs(abs(dt) - P / 2) <= NEAR_HALF_MS
        min_dist = min((abs(abs(float(r["dt_ms"])) - P / 2) for r in rows), default=None)
        near_any = min_dist is not None and min_dist <= NEAR_HALF_MS
        un = [f["A"]["unpaired"], f["B"]["unpaired"]]
        # where an unpaired frame was: frame-counter gaps between consecutive pairs (same epoch), with dt there
        where = []
        for cam in "ab":
            for i in range(1, len(rows)):
                p0, p1 = rows[i - 1], rows[i]
                if p0[cam + "_epoch"] == p1[cam + "_epoch"] and int(p1[cam + "_frame_counter"]) != int(p0[cam + "_frame_counter"]) + 1:
                    where.append({"cam": cam.upper(), "between_pairs": [i - 1, i], "dt_before_ms": float(p0["dt_ms"]),
                                  "dt_after_ms": float(p1["dt_ms"])})
        c = {"run": os.path.basename(d), "dt_median_ms": round(dt, 3), "near_half": near,
             "min_dist_to_half_ms": round(min_dist, 3) if min_dist is not None else None, "near_half_any_pair": near_any,
             "unpaired_A": un[0], "unpaired_B": un[1], "unpaired_between_pairs": where,
             "partner_settling_A": f["A"]["partner_settling_discards"],
             "partner_settling_B": f["B"]["partner_settling_discards"], "pairs": f["matcher"]["pairs"],
             "partner_changes": f["matcher"]["partner_changes"]}
        c["pass_median_reading"] = near or (un[0] == 0 and un[1] == 0)
        c["pass"] = near_any or (un[0] == 0 and un[1] == 0)
        ok = ok and c["pass"]
        cyc.append(c)
    return {"cycles": len(cyc), "near_half_cycles": sum(1 for c in cyc if c.get("near_half")),
            "near_half_any_pair_cycles": sum(1 for c in cyc if c.get("near_half_any_pair")),
            "pass_median_reading": all(c.get("pass_median_reading") for c in cyc) and len(cyc) == 20,
            "cycles_with_unpaired": sum(1 for c in cyc if c.get("unpaired_A", 0) + c.get("unpaired_B", 0) > 0),
            "per_cycle": cyc, "pass": bool(ok and len(cyc) == 20)}


# ------------------------------------------------------------------------------------------------ gap 3
def hist_quantile(rows, col, q):
    tot = sum(int(r[col]) for r in rows)
    if tot == 0:
        return None
    acc = 0
    for r in rows:
        acc += int(r[col])
        if acc >= q * tot:
            return float(r["bin_hi_ms"]) if r["bin_hi_ms"] != "inf" else float("inf")
    return None


def eval_delay(d):
    p = os.path.join(d, "delay_hist.csv")
    if not os.path.exists(p):
        return {"error": "delay_hist.csv missing"}
    rows = [r for r in csv.DictReader(l for l in open(p) if not l.startswith("#"))]
    out = {}
    for cat in ("usual", "near_half"):
        o = {"pairs": sum(int(r["older_" + cat]) for r in rows)}
        for which in ("older", "newer"):
            for q in (0.5, 0.99):
                o["%s_p%d_ms" % (which, int(q * 100))] = hist_quantile(rows, which + "_" + cat, q)
        out[cat] = o
    u, n = out["usual"], out["near_half"]
    out["estimate_usual_10_16ms"] = (u["older_p50_ms"] is not None and 10 <= u["newer_p50_ms"] and
                                     u["older_p99_ms"] <= 16)
    if n["pairs"] and u["older_p50_ms"] is not None:
        out["near_half_extra_older_p50_ms"] = round(n["older_p50_ms"] - u["older_p50_ms"], 1)
        out["near_half_extra_newer_p50_ms"] = round(n["newer_p50_ms"] - u["newer_p50_ms"], 1)
    return out


def main():
    gap = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/capture_logs/gap")
    reg = sys.argv[2] if len(sys.argv) > 2 else None
    verdict = {}
    if os.path.isdir(os.path.join(gap, "kill_1")):
        verdict["gap1_kill"] = eval_kill(os.path.join(gap, "kill_1"))
    for d in sorted(glob.glob(os.path.join(gap, "hwreset_*")) + glob.glob(os.path.join(gap, "unplug_*"))):
        verdict["gap2_" + os.path.basename(d)] = eval_reset(d, os.path.basename(d))
    if reg:
        verdict["gap4_startstop"] = eval_startstop(reg)
        if os.path.isdir(os.path.join(reg, "long_1")):
            verdict["gap3_delay_long"] = eval_delay(os.path.join(reg, "long_1"))
    for k, v in verdict.items():
        print("== %s: %s" % (k, "PASS" if v.get("pass") else ("FAIL" if "pass" in v else "REPORT")))
        print(json.dumps(v, indent=1))
    with open(os.path.join(gap, "gap_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)


if __name__ == "__main__":
    main()
