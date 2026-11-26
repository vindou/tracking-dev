"""Table of the live test (flow/live_test.sh): per scene, the first pose the tracker accepted while the scene was
shown, how long after the first pair of the scene it was published, and whether it is correct (within 10 mm and 10
degrees of the truth), for the run with the hint buffer and the run without.

    python -m flow.live_eval [--dir flow/runs/live] [--meta flow/data/eval2/meta.npz]"""
import argparse
import csv
from pathlib import Path

import numpy as np

from . import propose, scene


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load(d, arm, cats):
    """Per scene of the run: kind, verdict of the first accepted pose, ms from the first pair of the scene to its
    publication, by_hint, verdict of the last pose of the scene; and the tracker cycle times in search state."""
    scenes = list(csv.DictReader(open(d / f"scenes_{arm}.csv")))
    P = list(csv.DictReader(open(d / f"poses_{arm}.csv")))
    meas = np.array([int(r["meas_time_mono_ns"]) for r in P])
    out = []
    for s in scenes:
        T = np.array([float(s[f"T{i}"]) for i in range(16)]).reshape(4, 4)
        a, b = np.searchsorted(meas, [int(s["first_capture_mono_ns"]) - 1000, int(s["last_capture_mono_ns"]) + 1000])
        rows = P[a:b]

        def verdict(r):
            ep = 1e3 * np.linalg.norm(np.array([float(r[k]) for k in ("icp_px", "icp_py", "icp_pz")]) - T[:3, 3])
            R = quat_R([float(r[k]) for k in ("icp_qw", "icp_qx", "icp_qy", "icp_qz")])
            er = np.degrees(np.arccos(np.clip((np.trace(R @ T[:3, :3].T) - 1) / 2, -1, 1)))
            return "correct" if ep < 10 and er < 10 else "wrong"
        ok = [r for r in rows if r["valid"] == "1" and r["state"] == "1"]
        first = ok[0] if ok else None
        out.append(dict(scene=int(s["scene"]), cat=cats[int(s["scene"])], first=verdict(first) if first else "none",
                        ms=(int(first["publish_time_mono_ns"]) - int(s["first_capture_mono_ns"])) / 1e6 if first else np.nan,
                        by_hint=int(first["by_hint"]) if first else 0,
                        last=verdict(rows[-1]) if rows and rows[-1]["valid"] == "1" and rows[-1]["state"] == "1" else "none"))
    search = np.array([float(r["t_total_us"]) for r in P if r["state"] == "0"]) / 1e3
    hint_cycles = np.array([float(r["t_total_us"]) for r in P if int(r["hints_tried"]) > 0]) / 1e3
    return out, search, hint_cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="flow/runs/live")
    ap.add_argument("--meta", default="flow/data/eval2/meta.npz")
    a = ap.parse_args()
    d = propose.ROOT / a.dir
    cats = [str(c) for c in np.load(propose.ROOT / a.meta)["cat"]]
    cnt = lambda rs, key: " / ".join(str(sum(r[key] == k for r in rs)) for k in ("correct", "wrong", "none"))
    res = {}
    for arm in ("nohints", "hintspy", "hints"):
        if not (d / f"scenes_{arm}.csv").exists():
            continue
        rs, search, hc = res[arm] = load(d, arm, cats)
        t = np.array([r["ms"] for r in rs if r["first"] != "none"])
        print(f"\n{arm}: {len(rs)} scenes; first accepted pose correct / wrong / none: {cnt(rs, 'first')}; at the end of the scene: {cnt(rs, 'last')}")
        print(f"  time from the first pair of the scene to the published pose, ms: median {np.median(t):.0f}, p90 {np.percentile(t, 90):.0f}, max {t.max():.0f}"
              f"   accepted from a hypothesis: {sum(r['by_hint'] for r in rs)}")
        print(f"  tracker cycle in search state, ms: p50 {np.median(search):.2f}, p99 {np.percentile(search, 99):.2f}, max {search.max():.1f}"
              + (f"   cycles that tried hypotheses: {len(hc)}, p50 {np.median(hc):.1f}, max {hc.max():.1f}" if len(hc) else ""))
        for cat in scene.CATEGORIES:
            k = [r for r in rs if r["cat"] == cat]
            tk = np.array([r["ms"] for r in k if r["first"] != "none"])
            print(f"  {cat:11s} {cnt(k, 'first'):>14s}   median {np.median(tk) if len(tk) else float('nan'):5.0f} ms   by hypothesis {sum(r['by_hint'] for r in k)}")
    if "nohints" in res and "hints" in res:
        A, B = res["nohints"][0], res["hints"][0]
        ch = [(x["first"], y["first"]) for x, y in zip(A, B) if x["first"] != y["first"]]
        print("\nchanges without -> with hints: " + ", ".join(f"{x}->{y} {ch.count((x, y))}" for x, y in sorted(set(ch))))
        both = [(x["ms"], y["ms"]) for x, y in zip(A, B) if x["first"] == "correct" and y["first"] == "correct"]
        if both:
            both = np.array(both)
            print(f"scenes correct in both ({len(both)}): median time {np.median(both[:, 0]):.0f} ms without, {np.median(both[:, 1]):.0f} ms with")


if __name__ == "__main__":
    main()
