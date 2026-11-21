"""Presentation charts as SVG (no plotting library: only numpy is permitted). Run with the repo's .venv python.
usage: make_figs.py REPO OUTDIR PLOTDATA"""
import csv, sys, html
import numpy as np

REPO, OUT, PD = sys.argv[1], sys.argv[2], sys.argv[3]
LOGS = f"{REPO}/tracker/logs"

# reference palette (light), validated: dataviz/scripts/validate_palette.js
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FONT = "system-ui, -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"


def esc(s):
    return html.escape(str(s))


class Fig:
    def __init__(self, w, h, title, subtitle=None, note=None, note_dy=16):
        self.w, self.h, self.items = w, h, []
        self.text(24, 36, title, 20, INK, weight=600)
        if subtitle:
            self.text(24, 60, subtitle, 13, INK2)
        if note:
            self.text(24, h - note_dy, note, 11, MUTED)

    def text(self, x, y, s, size=12, fill=INK2, anchor="start", weight=400, rotate=None):
        tr = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                          f'font-weight="{weight}" font-family="{FONT}"{tr}>{esc(s)}</text>')

    def raw(self, s):
        self.items.append(s)

    def save(self, name):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
               f'<rect width="100%" height="100%" fill="{SURF}"/>' + "".join(self.items) + "</svg>")
        open(f"{OUT}/{name}.svg", "w").write(svg)


class Ax:
    """Plot area at (x0, y0) size (w, h) inside a Fig, linear scales."""

    def __init__(self, fig, x0, y0, w, h, xlim, ylim, xticks, yticks, xlabel=None, ylabel=None, xfmt="{:g}",
                 yfmt="{:g}", grid_y=True):
        self.f, self.x0, self.y0, self.w, self.h, self.xl, self.yl = fig, x0, y0, w, h, xlim, ylim
        if grid_y:
            for v in yticks:
                y = self.Y(v)
                fig.raw(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        for v in yticks:
            fig.text(x0 - 8, self.Y(v) + 4, yfmt.format(v), 11, MUTED, "end")
        fig.raw(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" stroke="{AXIS}" stroke-width="1"/>')
        for v in xticks:
            x = self.X(v)
            fig.raw(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0 + h}" y2="{y0 + h + 4}" stroke="{AXIS}"/>')
            fig.text(x, y0 + h + 18, xfmt.format(v), 11, MUTED, "middle")
        if xlabel:
            fig.text(x0 + w / 2, y0 + h + 38, xlabel, 12, INK2, "middle")
        if ylabel:
            fig.text(x0 - 52, y0 + h / 2, ylabel, 12, INK2, "middle", rotate=-90)

    def X(self, v):
        return self.x0 + (v - self.xl[0]) / (self.xl[1] - self.xl[0]) * self.w

    def Y(self, v):
        return self.y0 + self.h - (v - self.yl[0]) / (self.yl[1] - self.yl[0]) * self.h

    def line(self, xs, ys, color, width=2.0, dash=None):
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.f.raw(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
                   f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def steps(self, xs, ys, color, width=2):
        # step line: value ys[i] holds from xs[i] to xs[i+1]
        px, py = [], []
        for i in range(len(ys)):
            px += [xs[i], xs[i + 1]]
            py += [ys[i], ys[i]]
        self.line(px, py, color, width)

    def bars(self, edges, counts, color, gap=1.0):
        base = self.Y(self.yl[0])
        for i, c in enumerate(counts):
            if c <= 0:
                continue
            xa, xb = self.X(edges[i]) + gap / 2, self.X(edges[i + 1]) - gap / 2
            yt = self.Y(min(c, self.yl[1]))
            r = min(2.0, (xb - xa) / 2)
            self.f.raw(f'<path d="M{xa:.1f},{base:.1f} L{xa:.1f},{yt + r:.1f} Q{xa:.1f},{yt:.1f} {xa + r:.1f},{yt:.1f} '
                       f'L{xb - r:.1f},{yt:.1f} Q{xb:.1f},{yt:.1f} {xb:.1f},{yt + r:.1f} L{xb:.1f},{base:.1f} Z" fill="{color}"/>')

    def vline(self, v, label=None, color=INK2, dash="4 3", label_y=None, anchor="start"):
        x = self.X(v)
        self.f.raw(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{self.y0}" y2="{self.y0 + self.h}" stroke="{color}" '
                   f'stroke-width="1.2" stroke-dasharray="{dash}"/>')
        if label:
            dx = 4 if anchor == "start" else -4
            self.f.text(x + dx, label_y if label_y else self.y0 + 12, label, 11, color, anchor)

    def hline(self, v, label=None, color=INK2, dash="4 3", anchor="end"):
        y = self.Y(v)
        self.f.raw(f'<line x1="{self.x0}" x2="{self.x0 + self.w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" '
                   f'stroke-width="1.2" stroke-dasharray="{dash}"/>')
        if label:
            self.f.text(self.x0 + self.w - 4 if anchor == "end" else self.x0 + 4, y - 6, label, 11, color, anchor)

    def band(self, a, b, color, opacity=0.12):
        xa, xb = self.X(a), self.X(b)
        self.f.raw(f'<rect x="{xa:.1f}" y="{self.y0}" width="{xb - xa:.1f}" height="{self.h}" fill="{color}" '
                   f'fill-opacity="{opacity}"/>')


def legend(fig, x, y, entries):
    for i, (label, color) in enumerate(entries):
        yy = y + i * 20
        fig.raw(f'<line x1="{x}" x2="{x + 18}" y1="{yy}" y2="{yy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        fig.text(x + 26, yy + 4, label, 12, INK2)


def rows(path):
    return list(csv.DictReader(open(path)))


def pct(v, p):
    v = sorted(v)
    return v[int((len(v) - 1) * p)]


# ---------------------------------------------------------------- 1. cycle time by state
def fig_cycle_time():
    rep = [float(r["t_total_us"]) / 1000 for r in rows(f"{LOGS}/jetson/replay_synth_track/live.csv")
           if r["state"] == "1"]
    live = [r for d in ("link_1", "link_2", "link_3") for r in rows(f"{LOGS}/{d}/viewer_poses.csv")]
    srch = [float(r["t_total_us"]) / 1000 for r in live if r["track_state"] == "0"]
    trk = [float(r["t_total_us"]) / 1000 for r in live if r["track_state"] == "1"]
    series = [("Track state, test object (synthetic replay)", rep, S1),
              ("Search state, live", srch, S2),
              ("Track state, live, locked on static scenery (not the object)", trk, S3)]
    W, H = 960, 640
    f = Fig(W, H, "Calculation time per cycle on the Jetson Orin Nano",
            "Share of cycles per 0.25 ms bin. Pair period 11.1 ms; limit 5 ms (99 % of track cycles), goal 2 ms.",
            "Sources: tracker/logs/jetson/replay_synth_track/live.csv; tracker/logs/link_{1,2,3}/viewer_poses.csv "
            "(3 x 60 s live). Cycles above 20 ms are counted in the last bin.")
    edges = np.arange(0, 20.25, 0.25)
    top, ph, gap = 92, 128, 46
    for i, (name, v, c) in enumerate(series):
        y0 = top + i * (ph + gap)
        cnt, _ = np.histogram(np.clip(v, 0, 19.99), edges)
        share = cnt / len(v) * 100
        ymax = 50 if i == 0 else 10
        ax = Ax(f, 90, y0, 820, ph, (0, 20), (0, ymax), list(range(0, 21, 2)) if i == 2 else [],
                [0, ymax / 2, ymax], xlabel="milliseconds per cycle" if i == 2 else None, yfmt="{:g} %")
        ax.bars(edges, share, c, gap=1.0)
        ax.vline(2, "2 ms goal" if i == 0 else None, INK2, label_y=y0 + 12)
        ax.vline(5, "5 ms limit" if i == 0 else None, CRIT, label_y=y0 + 12)
        f.text(90, y0 - 10, name, 13, INK, weight=600)
        stat = (f"n = {len(v):,}  ·  median {pct(v, .5):.2f} ms  ·  p99 {pct(v, .99):.2f} ms  ·  "
                f"<= 5 ms: {sum(x <= 5 for x in v) / len(v) * 100:.1f} %  ·  <= 2 ms: {sum(x <= 2 for x in v) / len(v) * 100:.0f} %")
        f.text(910, y0 - 10, stat, 11, INK2, "end")
    f.save("cycle_time_by_state")


# ---------------------------------------------------------------- 2. time to track
def fig_time_to_track():
    r = rows(f"{PD}/acquisition.csv")
    t = [float(x["time_to_track_s"]) for x in r]
    W, H = 960, 460
    f = Fig(W, H, "Time to track: 100 random object poses, unknown start (synthetic)",
            "Time from the first pair until the tracker reports a valid pose in track state. Criterion: 90 % within 1 s.",
            "Bars: Mac run of tests/test_tracker_synth.cpp (data in presentation/figures/data/acquisition.csv). The search "
            "is time-sliced (8 ms per cycle), so a slower CPU needs more cycles.")
    edges = np.arange(0, 1.0001, 0.02)
    cnt, _ = np.histogram(t, edges)
    ymax = int(np.ceil(cnt.max() / 10.0) * 10)
    ax = Ax(f, 90, 96, 820, 270, (0, 1.0), (0, ymax), [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
            list(range(0, ymax + 1, 10)), "seconds to track", "poses", xfmt="{:.1f}")
    ax.bars(edges, cnt, S1, gap=2)
    ax.vline(1.0, "1 s criterion", CRIT, anchor="end", label_y=110)
    ax.vline(float(np.median(t)), None, INK2)
    f.text(ax.X(0.40) + 6, 160, f"Mac median {np.median(t):.2f} s", 11, INK2)
    ax.vline(0.60, "Jetson median 0.60 s (p90 0.68, max 0.70)", S2, dash="2 3", label_y=138)
    ok = sum(1 for x in r if float(x["time_to_track_s"]) <= 1.0 and float(x["pos_err_mm"]) <= 5)
    wrong = sum(1 for x in r if float(x["pos_err_mm"]) > 5 or float(x["rot_err_deg"]) > 5)
    f.text(910, 76, f"{ok} of 100 tracked within 1 s with the right pose; {wrong} wrong poses accepted", 12, INK, "end",
           weight=600)
    f.save("time_to_track")


# ---------------------------------------------------------------- 3. memory over 30 min
def fig_memory():
    r3 = rows(f"{LOGS}/long_30min/rss.csv")
    t3 = [int(x["t_s"]) for x in r3]
    W, H = 960, 500
    f = Fig(W, H, "Resident memory over the 30-minute live run (criterion: growth <= 5 %)",
            "Attempt 2 failed: one step per process between 150 and 180 s (kernel huge-page merging), flat afterwards. "
            "Attempt 3: huge pages off for both programs.",
            "Attempt 3: tracker/logs/long_30min/rss.csv (30 s samples). Attempt 2: values recorded from its rss.csv before "
            "it was overwritten (constant to 150 s,", note_dy=26)
    f.text(24, H - 10, "constant from 180 s to the end; tracker/DEVIATIONS.md item 47).", 11, MUTED)
    specs = [("trk_tracker", "tracker_rss_kb", 15252, 16020, (14800, 16400), [14800, 15200, 15600, 16000, 16400]),
             ("trk_bridge", "bridge_rss_kb", 6488, 8388, (6000, 8800), [6000, 6700, 7400, 8100, 8800])]
    for i, (name, key, a2a, a2b, yl, yt) in enumerate(specs):
        x0 = 100 + i * 440
        ax = Ax(f, x0, 130, 360, 260, (0, 1800), yl, [0, 600, 1200, 1800], yt, "seconds into the run",
                "resident memory (kB)" if i == 0 else None, yfmt="{:,.0f}")
        f.text(x0, 118, name, 13, INK, weight=600)
        ref3 = int(next(x for x in r3 if int(x["t_s"]) >= 120)[key])
        if i == 0:
            ax.hline(ref3 * 1.05, None, CRIT)
            f.text(x0 + 150, ax.Y(ref3 * 1.05) + 16, "+5 % limit (value at 120 s)", 11, CRIT, "middle")
        else:
            ax.hline(ref3 * 1.05, "+5 % limit (value at 120 s)", CRIT)
        ax.line([0, 165, 165, 1770], [a2a, a2a, a2b, a2b], S2, 2)
        ax.line(t3, [int(x[key]) for x in r3], S1, 2.5)
        g2 = (a2b / a2a - 1) * 100
        g3 = (int(r3[-1][key]) / ref3 - 1) * 100
        f.text(x0 + 356, ax.Y(a2b) + (16 if i == 0 else -8), f"attempt 2: {g2:+.1f} %", 11, INK2, "end")
        f.text(x0 + 356, ax.Y(int(r3[-1][key])) + 16, f"attempt 3: {g3:+.1f} %", 11, INK2, "end")
    legend(f, 560, 80, [("attempt 2 (fail)", S2)])
    legend(f, 740, 80, [("attempt 3 (pass)", S1)])
    f.save("memory_30min")


# ---------------------------------------------------------------- 4. moving object error
def fig_moving():
    W, H = 960, 440
    f = Fig(W, H, "Moving object: filtered position error (synthetic, 0.2 m/s and 30 deg/s, 1 mm depth noise)",
            "Camera time offset between the two images of a pair: 0 ms and 5.6 ms (half a frame, the worst case "
            "without hardware sync). Criterion: RMS <= 3 mm.",
            "Data: presentation/figures/data/moving_offset_*.csv (Mac run of tests/test_tracker_synth.cpp). "
            "The first 10 cycles (filter start-up) are excluded from the RMS.")
    ax = Ax(f, 90, 100, 820, 250, (0, 1.5), (0, 3.5), [0, .25, .5, .75, 1.0, 1.25, 1.5], [0, 1, 2, 3],
            "seconds", "position error (mm)")
    ax.band(0, 10 * 11.1283e-3, MUTED, 0.15)
    ax.hline(3.0, "3 mm RMS limit", CRIT)
    for off, c in (("0.0", S1), ("5.6", S2)):
        r = rows(f"{PD}/moving_offset_{off}.csv")
        ax.line([float(x["time_s"]) for x in r], [float(x["filt_err_mm"]) for x in r], c, 2)
        e = np.array([float(x["filt_err_mm"]) for x in r][10:])
        rms = np.sqrt(np.mean(e ** 2))
        valid = sum(int(x["valid"]) for x in r)
        legend_y = 380 if off == "0.0" else 400
        legend(f, 640, legend_y, [(f"offset {off} ms: RMS {rms:.2f} mm, {valid}/{len(r)} valid", c)])
    f.text(ax.X(0.06), 116, "start-up", 10, MUTED, "middle")
    f.save("moving_object_error")


# ---------------------------------------------------------------- 5. removed and put back
def fig_removed():
    r = rows(f"{PD}/removed_run_0.csv")
    t = [float(x["time_s"]) for x in r] + [float(r[-1]["time_s"]) + 0.0111]
    W, H = 960, 330
    f = Fig(W, H, "Object removed for 2 s, then put back (synthetic): track -> lost -> search -> track",
            "One of 5 runs; in all 5, no pose was marked valid while the object was absent or wrong by more than 5 mm / 5 deg.",
            "Data: presentation/figures/data/removed_run_0.csv (Mac run of tests/test_tracker_synth.cpp).")
    x0, w = 150, 760
    ax = Ax(f, x0, 90, w, 170, (0, t[-1]), (0, 3), [0, 1, 2, 3, 4], [], "seconds", grid_y=False)

    def runs(vals):
        out, s = [], 0
        for i in range(1, len(vals) + 1):
            if i == len(vals) or vals[i] != vals[s]:
                out.append((t[s], t[i], vals[s]))
                s = i
        return out

    lanes = [("object", [int(x["present"]) for x in r], {1: (S1, "present"), 0: (AXIS, "absent")}),
             ("state", [int(x["track_state"]) for x in r], {1: (GOOD, "TRACK"), 2: (SERIOUS, "LOST"), 0: (WARN, "SEARCH")}),
             ("pose valid", [int(x["valid"]) for x in r], {1: (GOOD, "valid"), 0: (AXIS, "")})]
    for li, (name, vals, cmap) in enumerate(lanes):
        y = 96 + li * 54
        f.text(x0 - 12, y + 26, name, 12, INK2, "end")
        for a, b, v in runs(vals):
            col, lab = cmap[v]
            xa, xb = ax.X(a), ax.X(b)
            f.raw(f'<rect x="{xa + 1:.1f}" y="{y}" width="{max(xb - xa - 2, 1):.1f}" height="38" rx="4" fill="{col}"/>')
            if lab and xb - xa > 60:
                fg = INK if col in (WARN, AXIS) else "#ffffff"
                f.text((xa + xb) / 2, y + 24, lab, 12, fg, "middle", weight=600)
    f.save("removed_and_put_back")


fig_cycle_time()
fig_time_to_track()
fig_memory()
fig_moving()
fig_removed()
print("ok")
