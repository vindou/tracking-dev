#!/usr/bin/env python3
"""Table of tracker CPU benchmark runs.

    cpu_bench_table.py DIR REF NAME [NAME ...]

A run NAME is two trk_hint_eval runs on the same scenes (tools/cpu_bench_run.sh):
  NAME_hints.csv  [NAME_hints_cycles.csv]    the hypotheses of the scene tried in the first cycle (--baseline 0 --max-cycles 1)
  NAME_search.csv [NAME_search_cycles.csv]   the classical search without time slicing (config/bench_noslice.json): one
                                             cycle is one whole search round, so the work per cycle does not depend on timing
Per run: elapsed time of a hint cycle and of a search round (median, 99 %, worst) and the CPU time of the loop thread
(on a loaded machine the CPU time is the one to read; with hint threads it leaves out the pool's work), the verdicts (correct = within 10 mm and
10 degrees / wrong / none), and the comparison with REF scene by scene: "identical" if every scene has the same
verdict and the same pose within 1e-6 m and 1e-6 in the quaternion, else the number of scenes whose verdict changed
and the number whose pose changed. Needs only the Python standard library."""
import csv
import os
import statistics as st
import sys


def load(d, name, kind):
    f = f"{d}/{name}_{kind}.csv"
    rows = {r["scene"]: r for r in csv.DictReader(open(f)) if r["mode"] == ("hints" if kind == "hints" else "baseline")}
    c = f"{d}/{name}_{kind}_cycles.csv"
    cyc = list(csv.DictReader(open(c))) if os.path.exists(c) else []
    return rows, cyc


def verdict(r, kind):
    if r["valid"] == "0" or (kind == "hints" and r["by_hint"] != "1"):
        return "none"
    return "correct" if float(r["err_mm"]) < 10 and float(r["err_deg"]) < 10 else "wrong"


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def times(cyc, kind):
    """Elapsed time and, if recorded, CPU time of the loop thread: median / 99 % / worst."""
    sel = [c for c in cyc if (int(c["hints_tried"]) > 0 if kind == "hints" else True)]
    out = []
    for col in ("ms", "cpu_ms"):
        v = [float(c[col]) for c in sel if c.get(col) not in (None, "")]
        out.append(f"{st.median(v):7.2f} /{q(v, .99):7.2f} /{max(v):8.2f}" if v else "      -  /     -  /      - ")
    return "   ".join(out)


def main():
    d, ref, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    for kind, title in (("hints", "cycle that tries the hypotheses"), ("search", "one search round (no time slicing)")):
        if not os.path.exists(f"{d}/{ref}_{kind}.csv"):
            continue
        R, _ = load(d, ref, kind)
        print(f"\n{title}: elapsed ms median / 99 % / worst, then CPU ms of the loop thread; verdicts correct / wrong / none; against {ref}")
        for n in [ref] + names:
            if not os.path.exists(f"{d}/{n}_{kind}.csv"):
                continue
            rows, cyc = load(d, n, kind)
            v = [verdict(r, kind) for r in rows.values()]
            changed = moved = 0
            worst = 0.0
            for k, r in rows.items():
                a = R[k]
                va, vr = verdict(a, kind), verdict(r, kind)
                if va != vr:
                    changed += 1
                elif vr != "none":
                    dd = max(abs(float(a[c]) - float(r[c])) for c in ("px", "py", "pz", "qw", "qx", "qy", "qz"))
                    worst = max(worst, dd)
                    moved += dd > 1e-6
            same = "identical" if changed == 0 and moved == 0 else f"verdict changed in {changed}, pose moved in {moved} of {len(rows)} scenes (largest {worst:.1e})"
            print(f"  {n:18s} {times(cyc, kind)}   {v.count('correct'):5d} /{v.count('wrong'):4d} /{v.count('none'):4d}   {same}")


if __name__ == "__main__":
    main()
