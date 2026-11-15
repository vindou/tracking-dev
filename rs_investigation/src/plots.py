#!/usr/bin/env python3
"""Plots for the report. Usage: plots.py  (run from ~/rs_investigation). Writes report/plots/*.png"""
import glob, json, os, re, collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'report/plots'
os.makedirs(OUT, exist_ok=True)
C = ['#2a6fdb', '#e8833a', '#3aa776', '#c94f7c']


def load(run):
    d = pd.read_csv(f'logs/{run}/frames.csv')
    d['t_s'] = (d.arrival_mono_ns - d.arrival_mono_ns.iloc[0]) / 1e9
    return d


# 1. Exp 3: arrival-interval distribution and latency over time, one panel per mode (run r1)
modes = [('256x144_90', 90), ('480x270_60', 60), ('848x480_10', 10)]
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for j, (m, fps) in enumerate(modes):
    for r, col in zip((1, 2, 3), C):
        d = load(f'exp3_A_{m}_r{r}')
        d = d[d.t_s >= 2]
        ia = np.diff(d.arrival_mono_ns.values) / 1e6
        ax[0, j].hist(ia, bins=120, histtype='step', color=col, label=f'run {r}', log=True)
        g = d[d.domain == 'Global Time']
        ax[1, j].plot(g.t_s, g.arrival_real_ns / 1e6 - g.ts_ms, '.', ms=1, color=col, label=f'run {r}')
    ax[0, j].axvline(1000 / fps, color='k', lw=0.8, ls='--')
    ax[0, j].set_title(f'Exp 3, {m.replace("_", " @")} fps: arrival interval')
    ax[0, j].set_xlabel('ms'); ax[0, j].set_ylabel('count (log)'); ax[0, j].legend(fontsize=8)
    ax[1, j].set_title('delivery latency (CLOCK_REALTIME arrival - global ts)')
    ax[1, j].set_xlabel('time in run (s)'); ax[1, j].set_ylabel('ms')
fig.tight_layout(); fig.savefig(f'{OUT}/exp3_intervals_latency.png', dpi=110); plt.close(fig)

# 2. Exp 3b: latency vs frame period
rows = []
for f in glob.glob('logs/exp3b_A_*/summary.json'):
    r = json.load(open(f)); c = r['cams'].get('A')
    if not c: continue
    m = re.search(r'(\d+)x(\d+)_(\d+)_r', r['run'])
    rows.append((m.group(1) + 'x' + m.group(2), c['interval_hw_ms']['p50'], c['latency_real_minus_ts_ms']['p50']))
for f in glob.glob('logs/exp3_A_*/summary.json'):
    r = json.load(open(f)); c = r['cams']['A']
    m = re.search(r'(\d+)x(\d+)_(\d+)_r', r['run'])
    rows.append((m.group(1) + 'x' + m.group(2) + ' (exp3)', c['interval_hw_ms']['p50'], c['latency_real_minus_ts_ms']['p50']))
df = pd.DataFrame(rows, columns=['res', 'period', 'lat'])
fig, ax = plt.subplots(figsize=(7, 5))
for i, (res, g) in enumerate(df.groupby('res')):
    ax.plot(g.period, g.lat, 'o', color=C[i % 4], label=res, alpha=0.7)
x = np.linspace(0, 170, 10)
ax.plot(x, 0.88 * x, 'k--', lw=0.8, label='0.88 x period')
ax.set_xlabel('measured frame period (ms)'); ax.set_ylabel('delivery latency p50 (ms)')
ax.set_title('Exp 3b: latency scales with frame period, not resolution'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/exp3b_latency_vs_period.png', dpi=110); plt.close(fig)

# 3. Exp 5 (+F3): fps vs manual exposure
pts = collections.defaultdict(list)
for f in glob.glob('logs/exp5_A_exp*/summary.json') + glob.glob('logs/f3_exp*/summary.json'):
    r = json.load(open(f)); c = r['cams'].get('A')
    m = re.search(r'exp(\d+|auto)_r', r['run'])
    if not c or m.group(1) == 'auto': continue
    pts[int(m.group(1))].append(c['mean_fps_arrival'])
fig, ax = plt.subplots(figsize=(7, 4.5))
for e, v in sorted(pts.items()):
    ax.plot([e / 1000] * len(v), v, 'o', color=C[0])
ax.axvline(1000 / 90, color='k', ls='--', lw=0.8, label='frame period at 90 fps (11.11 ms)')
ax.set_xlabel('manual exposure (ms)'); ax.set_ylabel('measured fps'); ax.set_ylim(50, 95)
ax.set_title('Exp 5: 256x144 @90 requested, fps vs manual exposure'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/exp5_fps_vs_exposure.png', dpi=110); plt.close(fig)

# 4. Exp 1: domain switch at stream start (global time on)
fig, ax = plt.subplots(figsize=(7, 4))
for r, col in zip((1, 2, 3), C):
    d = load(f'exp3_A_256x144_90_r{r}').iloc[:15]
    ax.plot(d.t_s * 1000, (d.domain == 'Global Time').astype(int) + (r - 2) * 0.03, 'o-', color=col, label=f'exp3 256x144@90 run {r}')
ax.set_yticks([0, 1]); ax.set_yticklabels(['Hardware Clock', 'Global Time'])
ax.set_xlabel('ms after first frame'); ax.set_title('Exp 1: timestamp domain of first frames (global_time_enabled=1)')
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/exp1_domain_switch.png', dpi=110); plt.close(fig)
print('plots written to', OUT, os.listdir(OUT))

# 5. Global-time estimator error vs ideal linear map of the hw clock (from globaltime_error.py)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for j, m in enumerate(["256x144_90", "480x270_60"]):
    for r, col in zip((1, 2, 3), C):
        p = f"logs/exp3_A_{m}_r{r}/globaltime_error.csv"
        if not os.path.exists(p): continue
        e = pd.read_csv(p).dropna()
        ax[j].plot(e.t_s, e.err_ms, ".", ms=1, color=col, label=f"run {r}")
    ax[j].set_ylim(-3, 10); ax[j].axhline(0, color="k", lw=0.5)
    lbl = m.replace("_", " @"); ax[j].set_title(f"Global time error, {lbl} fps"); ax[j].set_xlabel("time in run (s)"); ax[j].set_ylabel("global ts - linear(hw ts)  [ms]")
    ax[j].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/exp6pre_globaltime_error.png", dpi=110); plt.close(fig)
print("plot 5 done")
