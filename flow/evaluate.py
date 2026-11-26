"""Learned proposals against the classical search on the evaluation scenes, both through the tracker core.

    python -m flow.evaluate [--eval flow/data/eval] [--propose-args "--device mps"]

1. flow.propose writes the hypotheses of every scene (hints.bin).
2. tracker/build-mac/trk_hint_eval runs the tracker core on every scene: classical search alone (baseline.csv, made
   once) and with the hypotheses (hints.csv). Same ICP, same fit check.
3. A pose is counted correct if the tracker accepted it and it is within 10 mm and 10 degrees of the truth, wrong if
   accepted and farther, none if nothing was accepted (from a hypothesis: in the first cycle; classical: in 180
   cycles, 2 s of pairs).
Writes <eval>/results.json and prints the table. flow.compare gives the paired comparison of two result folders."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import scene

ROOT = Path(__file__).resolve().parents[1]
PERIOD_MS = 11.1283
OK_MM, OK_DEG = 10.0, 10.0


def verdict(r):
    if r["valid"] != "1":
        return "none"
    return "correct" if float(r["err_mm"]) < OK_MM and float(r["err_deg"]) < OK_DEG else "wrong"


def read(path, mode):
    return {int(r["scene"]): r for r in csv.DictReader(open(path)) if r["mode"] == mode}


def summarize(ev):
    ev = Path(ev)
    meta = np.load(ev / "meta.npz")
    fl = np.load(ev / "flow.npz")
    base, hint = read(ev / "baseline.csv", "baseline"), read(ev / "hints.csv", "hints")
    scenes = []
    for i, cat in enumerate(meta["cat"]):
        b, h = base[i], hint[i]
        hv = verdict(h)
        scenes.append(dict(
            scene=i, cat=str(cat),
            classical=verdict(b), classical_cycles=int(b["cycles"]), classical_ms=float(b["compute_ms"]),
            classical_err_mm=float(b["err_mm"]), classical_err_deg=float(b["err_deg"]),
            classical_pose=[float(b[k]) for k in ("px", "py", "pz", "qw", "qx", "qy", "qz")],
            # the hint path alone: accepted in the first cycle from a flow hypothesis
            flow=hv if h["by_hint"] == "1" else "none",
            # hints first, the classical search after them if none passes
            flow_fallback=hv, flow_cycles=int(h["cycles"]), flow_icp_ms=float(h["first_cycle_ms"]),
            flow_net_ms=1e3 * float(fl["t_flow"][i]), flow_prep_ms=1e3 * float(fl["t_prep"][i]),
            flow_err_mm=float(h["err_mm"]), flow_err_deg=float(h["err_deg"]), hints_tried=int(h["hints_tried"]),
            flow_pose=[float(h[k]) for k in ("px", "py", "pz", "qw", "qx", "qy", "qz")],
            hyp_good=int(fl["n_good"][i]), hyp_first_good_rank=int(fl["first_good_rank"][i])))
    table = {}
    for cat in list(scene.CATEGORIES) + ["all"]:
        rs = [s for s in scenes if cat in (s["cat"], "all")]
        row = {"n": len(rs)}
        for m in ("classical", "flow", "flow_fallback"):
            for v in ("correct", "wrong", "none"):
                row[f"{m}_{v}"] = sum(s[m] == v for s in rs)
        cc = [s["classical_cycles"] * PERIOD_MS for s in rs if s["classical"] == "correct"]
        row["classical_time_ms_p50"] = float(np.median(cc)) if cc else None
        ff = [s["flow_net_ms"] + s["flow_prep_ms"] + s["flow_icp_ms"] for s in rs if s["flow"] == "correct"]
        row["flow_time_ms_p50"] = float(np.median(ff)) if ff else None
        ce = [s["classical_err_mm"] for s in rs if s["classical"] == "correct"]
        fe = [s["flow_err_mm"] for s in rs if s["flow"] == "correct"]
        row["classical_err_mm_p50"] = float(np.median(ce)) if ce else None
        row["flow_err_mm_p50"] = float(np.median(fe)) if fe else None
        table[cat] = row
    json.dump({"table": table, "scenes": scenes, "ok_mm": OK_MM, "ok_deg": OK_DEG}, open(ev / "results.json", "w"))
    return table


def markdown(table):
    L = ["| scenes | n | classical: correct / wrong / none | learned + ICP: correct / wrong / none | learned, then classical | classical: pairs until the pose (p50) |",
         "|---|---|---|---|---|---|"]
    for cat, r in table.items():
        L.append(f"| {cat} | {r['n']} | {r['classical_correct']} / {r['classical_wrong']} / {r['classical_none']} | "
                 f"{r['flow_correct']} / {r['flow_wrong']} / {r['flow_none']} | "
                 f"{r['flow_fallback_correct']} / {r['flow_fallback_wrong']} / {r['flow_fallback_none']} | "
                 f"{'-' if r['classical_time_ms_p50'] is None else round(r['classical_time_ms_p50'] / PERIOD_MS)} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="flow/data/eval")
    ap.add_argument("--propose-args", default="", help="arguments for flow.propose, e.g. '--no-verifier' or '--device mps'")
    ap.add_argument("--skip-propose", action="store_true", help="use the hints.bin already in the folder (made on another machine)")
    a = ap.parse_args()
    ev = (ROOT / a.eval).resolve()
    tool = ROOT / "tracker/build-mac/trk_hint_eval"
    if not a.skip_propose:
        subprocess.run([sys.executable, "-m", "flow.propose", "--eval", str(ev)] + a.propose_args.split(), check=True, cwd=ROOT)
    if not (ev / "baseline.csv").exists():
        subprocess.run([tool, "--scenes", ev / "scenes.bin", "--out", ev / "baseline.csv"], check=True, cwd=ROOT / "tracker")
    subprocess.run([tool, "--scenes", ev / "scenes.bin", "--hints", ev / "hints.bin", "--baseline", "0", "--out", ev / "hints.csv"],
                   check=True, cwd=ROOT / "tracker")
    print(markdown(summarize(ev)))


if __name__ == "__main__":
    main()
