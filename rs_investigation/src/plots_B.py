#!/usr/bin/env python3
"""Plots for the revised camera-B (USB3) experiments. Run from ~/rs_investigation. Writes report/plots/B_*.png"""
import glob, json, re, collections, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "report/plots"; os.makedirs(OUT, exist_ok=True)
C = ["#2a6fdb", "#e8833a", "#3aa776", "#c94f7c"]

# B6: global time error over 15 min
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True, sharey=True)
for i, r in enumerate((1, 2, 3)):
    e = pd.read_csv(f"logs/expB6_15min_r{r}/globaltime_error.csv").dropna()
    e = e[e.t_s > 3]
    ax[i].plot(e.t_s, e.err_ms, ",", color=C[i]); ax[i].axhline(1, color="k", lw=.6, ls="--"); ax[i].axhline(-1, color="k", lw=.6, ls="--")
    ax[i].set_ylabel(f"run {r}\nerror (ms)"); ax[i].set_ylim(-2.5, 2.5)
ax[0].set_title("Exp 6 (camera B, 848x480@90): global timestamp minus linear fit of hardware timestamp (t > 3 s)")
ax[-1].set_xlabel("time in run (s)"); fig.tight_layout(); fig.savefig(f"{OUT}/B_exp6_globaltime_error_15min.png", dpi=110); plt.close(fig)

# B5: fps vs manual exposure
pts = collections.defaultdict(list)
for f in glob.glob("logs/expB5_exp*_r*/summary.json"):
    r = json.load(open(f)); m = re.search(r"exp(\d+)_r", r["run"])
    if m: pts[int(m.group(1))].append(r["cams"]["B"]["mean_fps_arrival"])
fig, ax = plt.subplots(figsize=(7, 4.5))
for e, v in sorted(pts.items()): ax.plot([e / 1000] * len(v), v, "o", color=C[0])
ax.axvline(1000 / 90, color="k", ls="--", lw=.8, label="90 fps frame period (11.11 ms)")
ax.set_xlabel("manual exposure (ms)"); ax.set_ylabel("measured fps"); ax.set_ylim(50, 95); ax.legend(fontsize=8)
ax.set_title("Exp 5 (camera B): 848x480 @90 requested, fps vs manual exposure")
fig.tight_layout(); fig.savefig(f"{OUT}/B_exp5_fps_vs_exposure.png", dpi=110); plt.close(fig)

# B4: arrival-interval distribution of the depth stream under each load condition
cases = [("expB3_848x480_90_r", "depth only (exp 3)"), ("expB4a_ir_r", "depth + IR1 + IR2"),
         ("expB4b_load4_r", "depth + 4 load threads"), ("expB4c_ir_load4_r", "depth + IR + 4 load threads")]
fig, ax = plt.subplots(figsize=(9, 5))
for (p, lbl), col in zip(cases, C):
    ia = []
    for r in (1, 2, 3):
        d = pd.read_csv(f"logs/{p}{r}/frames.csv"); d = d[d.cam == "B"]
        t = d.arrival_mono_ns.to_numpy(); t = t[(t - t[0]) > 2e9]; ia.append(np.diff(t) / 1e6)
    ax.hist(np.concatenate(ia), bins=np.arange(0, 30, .25), histtype="step", color=col, label=lbl, log=True)
ax.axvline(1000 / 89.87, color="k", lw=.8, ls="--")
ax.set_xlabel("depth frame arrival interval (ms)"); ax.set_ylabel("count (log), 3 x 60 s pooled"); ax.legend(fontsize=8)
ax.set_title("Exp 4 (one camera, proxy load): arrival interval at 848x480@90")
fig.tight_layout(); fig.savefig(f"{OUT}/B_exp4_arrival_interval_load.png", dpi=110); plt.close(fig)
print("written", sorted(f for f in os.listdir(OUT) if f.startswith("B_")))
