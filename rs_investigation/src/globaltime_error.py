#!/usr/bin/env python3
"""Global-time estimator error: ts_global - (a + b*hw_ts), with a,b fitted on frames t>10 s (ideal linear clock map).
Reports: settle time (last time |err|>1 ms), |err| p50/p99/max after settle and after 10 s, and the slope b in ppm.
Usage: globaltime_error.py RUN_DIR [...]   (writes RUN_DIR/globaltime_error.csv)"""
import sys, numpy as np, pandas as pd
WRAP = 2**32
print("%-26s %8s %9s %9s %9s %9s %9s" % ("run", "settle_s", "slope_ppm", "err_p50", "err_p99", "err_max", "max<3s"))
for run in sys.argv[1:]:
    d = pd.read_csv(f"{run}/frames.csv")
    if len(d) < 100: continue
    hw = d["md_Frame Timestamp"].to_numpy(float)
    hw = (hw + np.concatenate([[0], np.cumsum(np.where(np.diff(hw) < -WRAP / 2, WRAP, 0))])) / 1e3  # ms
    t = (d.arrival_mono_ns.to_numpy() - d.arrival_mono_ns.iloc[0]) / 1e9
    g = (d.domain == "Global Time").to_numpy()
    ts = d.ts_ms.to_numpy()
    fit = g & (t > 10)
    b, a = np.polyfit(hw[fit] - hw[0], ts[fit], 1)
    err = np.where(g, ts - (a + b * (hw - hw[0])), np.nan)
    big = g & (np.abs(err) > 1.0)
    settle = t[big].max() if big.any() else 0.0
    after = g & (t > 10)
    e = np.abs(err[after])
    early = np.nanmax(np.abs(err[g & (t < 3)])) if (g & (t < 3)).any() else np.nan
    pd.DataFrame(dict(t_s=t, domain=d.domain, err_ms=err)).to_csv(f"{run}/globaltime_error.csv", index=False)
    print("%-26s %8.2f %9.1f %9.3f %9.3f %9.3f %9.2f" % (run.rstrip('/').split('/')[-1], settle, (b - 1) * 1e6,
          np.percentile(e, 50), np.percentile(e, 99), e.max(), early))
