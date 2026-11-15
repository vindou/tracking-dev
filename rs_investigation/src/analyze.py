#!/usr/bin/env python3
"""Analysis of rs_probe per-frame CSV logs.

  analyze.py summary RUN_DIR [RUN_DIR ...] [--skip 2] [--nominal-fps 90]
      Per-run, per-camera metrics -> RUN_DIR/summary.json, printed as a table.
  analyze.py agg RUN_DIR [RUN_DIR ...]
      Median and worst over runs (reads summary.json files).
"""
import json, re, sys, os
import numpy as np
import pandas as pd

WRAP_US = 2**32


def unwrap_us(x):
    x = np.asarray(x, dtype=np.float64)
    d = np.diff(x)
    jumps = np.where(d < -WRAP_US / 2, WRAP_US, 0)
    return x - x[0] + np.concatenate([[0], np.cumsum(jumps)]) + x[0]


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float('nan')


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[~np.isnan(a)]
    if not len(a):
        return None
    return dict(n=int(len(a)), mean=float(a.mean()), p50=pct(a, 50), p99=pct(a, 99), max=float(a.max()), min=float(a.min()))


def parse_tegrastats(path):
    if not os.path.exists(path):
        return None
    cores, tj, gpu = [], [], []
    for line in open(path):
        m = re.search(r'CPU \[([^\]]*)\]', line)
        if m:
            vals = []
            for c in m.group(1).split(','):
                mm = re.match(r'(\d+)%@(\d+)', c)
                vals.append(float(mm.group(1)) if mm else np.nan)
            cores.append(vals)
        m = re.search(r'tj@([\d.]+)C', line)
        if m:
            tj.append(float(m.group(1)))
    if not cores:
        return None
    c = np.array(cores)
    return dict(samples=len(c), core_mean=[round(float(v), 1) for v in np.nanmean(c, 0)],
                core_max=[float(v) for v in np.nanmax(c, 0)], total_mean=round(float(np.nanmean(c)), 1),
                tj_max=max(tj) if tj else None)


def kernel_events(path):
    if not os.path.exists(path):
        return None
    lines = [l.rstrip() for l in open(path) if not l.startswith('-- ')]
    # RSUSB detaches/re-probes uvcvideo on open/close: 'Found UVC', 'Unknown video format', 'input: ...' are expected.
    pat = re.compile(r'xhci|reset|error|-71|-110|throttl|over.?current|disconnect|not accepting|cannot enable', re.I)
    hits = [l for l in lines if pat.search(l)]
    reprobe = sum(1 for l in lines if re.search(r'Found UVC|Unknown video format|input: Intel', l))
    return dict(total=len(lines), relevant=len(hits), uvc_reprobe_lines=reprobe, sample=hits[:10])


def per_cam(df, skip_s, nominal_fps):
    out = {}
    t0 = df.arrival_mono_ns.min()
    df = df.assign(t_s=(df.arrival_mono_ns - t0) / 1e9)
    out['domain_counts'] = df.domain.value_counts().to_dict()
    out['md_available'] = {c[3:]: round(float(df[c].notna().mean()), 4) for c in df.columns if c.startswith('md_')}
    out['frames_total'] = int(len(df))
    s = df[df.t_s >= skip_s]
    out['frames_after_skip'] = int(len(s))
    if len(s) < 3:
        return out
    dur = (s.arrival_mono_ns.iloc[-1] - s.arrival_mono_ns.iloc[0]) / 1e9
    out['duration_s'] = round(dur, 3)
    out['mean_fps_arrival'] = (len(s) - 1) / dur
    # frame counter gaps
    fc_col = 'md_Frame Counter' if 'md_Frame Counter' in s and s['md_Frame Counter'].notna().all() else 'frame_number'
    fc = s[fc_col].to_numpy(dtype=np.int64)
    d = np.diff(fc)
    out['counter_source'] = fc_col
    out['dropped'] = int(np.sum(d[d > 1] - 1))
    out['gap_events'] = int(np.sum(d > 1))
    out['counter_nonmonotonic'] = int(np.sum(d < 1))
    out['gaps_detail'] = [dict(t_s=round(float(s.t_s.iloc[i + 1]), 3), missing=int(d[i] - 1)) for i in np.where(d > 1)[0][:20]]
    # intervals
    ia = np.diff(s.arrival_mono_ns.to_numpy()) / 1e6
    out['interval_arrival_ms'] = stats(ia)
    if 'md_Frame Timestamp' in s and s['md_Frame Timestamp'].notna().all():
        hw = unwrap_us(s['md_Frame Timestamp'].to_numpy())
        ih = np.diff(hw) / 1e3
        out['interval_hw_ms'] = stats(ih)
        span = (hw[-1] - hw[0]) / 1e6
        out['mean_fps_hw'] = (fc[-1] - fc[0]) / span if span > 0 else None
    out['interval_ts_ms'] = stats(np.diff(s.ts_ms.to_numpy()))
    # delivery latency: arrival (CLOCK_REALTIME) - frame timestamp, valid only when ts is global/system (epoch ms)
    if s.ts_ms.iloc[0] > 1e12:
        out['latency_real_minus_ts_ms'] = stats(s.arrival_real_ns.to_numpy() / 1e6 - s.ts_ms.to_numpy())
    if 'md_Actual Exposure' in s and s['md_Actual Exposure'].notna().any():
        out['exposure_us'] = stats(s['md_Actual Exposure'].to_numpy())
    if 'md_Actual Fps' in s and s['md_Actual Fps'].notna().any():
        v = s['md_Actual Fps'].dropna()
        out['actual_fps_md'] = {str(k): int(n) for k, n in v.value_counts().items()}
    out['cb_us'] = stats(s.cb_ns.to_numpy() / 1e3)
    if "overlap" in s:
        out["cb_overlap_frames"] = int((s["overlap"] > 0).sum())
    if (s.push_ok >= 0).any():
        out['push_fail'] = int((s.push_ok == 0).sum())
    if nominal_fps:
        out['expected_frames'] = int(round(dur * nominal_fps)) + 1
    return out


def summary(run_dir, skip_s=2.0, nominal_fps=None):
    df = pd.read_csv(os.path.join(run_dir, 'frames.csv'))
    res = dict(run=os.path.basename(run_dir.rstrip('/')), cams={})
    for cam, g in df.groupby('cam'):
        res['cams'][cam] = per_cam(g.sort_values('idx'), skip_s, nominal_fps)
    res['tegrastats'] = parse_tegrastats(os.path.join(run_dir, 'tegrastats.txt'))
    res['kernel'] = kernel_events(os.path.join(run_dir, 'kernel_new.txt'))
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    return res


def fmt(st, k='p50'):
    return '-' if not st else f"{st[k]:.3f}"


def print_row(res):
    for cam, c in res['cams'].items():
        print(f"{res['run']:<34} {cam} fps={c.get('mean_fps_arrival', float('nan')):7.3f} drop={c.get('dropped', '-'):>4} "
              f"gapev={c.get('gap_events', '-'):>3} hwI p50/p99/max={fmt(c.get('interval_hw_ms'))}/{fmt(c.get('interval_hw_ms'), 'p99')}/"
              f"{fmt(c.get('interval_hw_ms'), 'max')} arrI p99/max={fmt(c.get('interval_arrival_ms'), 'p99')}/{fmt(c.get('interval_arrival_ms'), 'max')} "
              f"lat p50/p99={fmt(c.get('latency_real_minus_ts_ms'))}/{fmt(c.get('latency_real_minus_ts_ms'), 'p99')} "
              f"exp={fmt(c.get('exposure_us'), 'mean')} cb p99={fmt(c.get('cb_us'), 'p99')}us dom={list(c['domain_counts'])}")
    t = res.get('tegrastats')
    k = res.get('kernel')
    print(f"{'':<34}   cpu_mean={t and t['total_mean']} cores={t and t['core_mean']} tj_max={t and t['tj_max']} "
          f"kernel_relevant={k and k['relevant']}")


def agg(dirs):
    rows = [json.load(open(os.path.join(d, 'summary.json'))) for d in dirs]
    cams = sorted({c for r in rows for c in r['cams']})
    for cam in cams:
        cs = [r['cams'][cam] for r in rows if cam in r['cams']]
        def col(f):
            return np.array([f(c) for c in cs], dtype=float)
        metrics = {
            'mean_fps': (col(lambda c: c.get('mean_fps_arrival', np.nan)), 'min'),
            'dropped': (col(lambda c: c.get('dropped', np.nan)), 'max'),
            'hw_interval_p99_ms': (col(lambda c: (c.get('interval_hw_ms') or {}).get('p99', np.nan)), 'max'),
            'hw_interval_max_ms': (col(lambda c: (c.get('interval_hw_ms') or {}).get('max', np.nan)), 'max'),
            'arrival_interval_p50_ms': (col(lambda c: (c.get('interval_arrival_ms') or {}).get('p50', np.nan)), 'max'),
            'arrival_interval_p99_ms': (col(lambda c: (c.get('interval_arrival_ms') or {}).get('p99', np.nan)), 'max'),
            'arrival_interval_max_ms': (col(lambda c: (c.get('interval_arrival_ms') or {}).get('max', np.nan)), 'max'),
            'latency_p50_ms': (col(lambda c: (c.get('latency_real_minus_ts_ms') or {}).get('p50', np.nan)), 'max'),
            'latency_p99_ms': (col(lambda c: (c.get('latency_real_minus_ts_ms') or {}).get('p99', np.nan)), 'max'),
            'latency_max_ms': (col(lambda c: (c.get('latency_real_minus_ts_ms') or {}).get('max', np.nan)), 'max'),
            'cb_p50_us': (col(lambda c: (c.get('cb_us') or {}).get('p50', np.nan)), 'max'),
            'cb_p99_us': (col(lambda c: (c.get('cb_us') or {}).get('p99', np.nan)), 'max'),
            'cb_max_us': (col(lambda c: (c.get('cb_us') or {}).get('max', np.nan)), 'max'),
        }
        print(f"camera {cam}: n_runs={len(cs)}")
        for k, (v, worst) in metrics.items():
            if np.all(np.isnan(v)):
                continue
            w = np.nanmin(v) if worst == 'min' else np.nanmax(v)
            print(f"  {k:<26} median={np.nanmedian(v):10.3f}  worst={w:10.3f}  runs={np.round(v, 3).tolist()}")
    cpu = [r['tegrastats']['total_mean'] for r in rows if r.get('tegrastats')]
    if cpu:
        print(f"  cpu_total_mean%            median={np.median(cpu):.1f} worst={max(cpu):.1f}")
    print(f"  kernel relevant lines per run: {[r['kernel'] and r['kernel']['relevant'] for r in rows]}")


if __name__ == '__main__':
    args = sys.argv[1:]
    skip, nom = 2.0, None
    if '--skip' in args:
        i = args.index('--skip'); skip = float(args[i + 1]); del args[i:i + 2]
    if '--nominal-fps' in args:
        i = args.index('--nominal-fps'); nom = float(args[i + 1]); del args[i:i + 2]
    if args[0] == 'summary':
        for d in args[1:]:
            print_row(summary(d, skip, nom))
    elif args[0] == 'agg':
        agg(args[1:])
