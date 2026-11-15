#!/usr/bin/env python3
"""Camera hardware clock vs host CLOCK_MONOTONIC: rate difference in ppm, per run.
Uses the lower envelope (5th percentile per 1 s bin) of (arrival_mono - hw_ts) to remove delivery jitter,
then a linear fit. Also: settling time of global time (|global_latency - final median| < 0.5 ms for good).
Usage: clock_drift.py RUN_DIR [...]"""
import sys, numpy as np, pandas as pd
WRAP = 2**32
print("%-26s %6s %10s %9s %13s %16s" % ("run", "dur_s", "drift_ppm", "fit_rms_us", "gt_settle_s", "gt_vs_hw_ppm"))
for run in sys.argv[1:]:
    d = pd.read_csv(f"{run}/frames.csv")
    if len(d) < 100: continue
    hw = d["md_Frame Timestamp"].to_numpy(float)
    hw = hw + np.concatenate([[0], np.cumsum(np.where(np.diff(hw) < -WRAP / 2, WRAP, 0))])
    t = (d.arrival_mono_ns.to_numpy() - d.arrival_mono_ns.iloc[0]) / 1e9
    off = d.arrival_mono_ns.to_numpy() / 1e3 - hw            # us
    bins = np.floor(t).astype(int)
    bx, by = [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() > 3: bx.append(t[m].mean()); by.append(np.percentile(off[m], 5))
    bx, by = np.array(bx), np.array(by)
    p = np.polyfit(bx, by, 1); rms = np.sqrt(np.mean((by - np.polyval(p, bx)) ** 2))
    # global time: settle time and its own rate vs hw clock
    g = d.domain == "Global Time"
    lat = d.arrival_real_ns / 1e6 - d.ts_ms
    final = lat[g & (t > t[-1] - 10)].median()
    bad = g & ((lat - final).abs() > 1.5)
    settle = t[bad.to_numpy()].max() if bad.any() else 0.0
    gs = g.to_numpy() & (t > max(settle, 3))
    q = np.polyfit(hw[gs] / 1e6, d.ts_ms.to_numpy()[gs] / 1e3 - hw[gs] / 1e6, 1)[0] * 1e6 if gs.sum() > 100 else np.nan
    name = run.rstrip("/").split("/")[-1]
    print("%-26s %6.0f %10.1f %9.1f %13.2f %16.1f" % (name, t[-1], p[0] * 1e6 / 1e6 * 1, rms, settle, q))
