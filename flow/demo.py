"""Viser demo: the classical search of the tracker against flow hypotheses + the same ICP, on the evaluation scenes.

    python -m flow.evaluate            # once: hypotheses, tracker runs, results.json
    python -m flow.demo [--port 8090]  # then open http://localhost:8090

Per scene it shows the cloud of the depth pair, the true pose (green wireframe), the pose the classical search
accepted (red), and the flow in three labelled phases: (1) every hypothesis moving from its start to its end point
(thin, coloured by agreement group); (2) the flow end point closest to the pose the tracker accepted (yellow), with
its error; (3) the tracker's ICP moving from that end point to the accepted pose (blue), with the error after ICP.
The ICP motion is drawn as an interpolation between the two poses (the ICP iterations are not recorded).
Two 2-D slices of the learned field around the true pose are plotted at the bottom (colour = size of the predicted
remaining motion, viridis; arrows = its direction in the slice). They are precomputed for every flow time
(flow.precompute_fields, cache in <eval>/fields/) and change with the steps of the playback, with the hypotheses
projected onto them. Text and plots are floating cards; the viser panel on the right holds only the settings.
Playback is paced by real time (network time on the Jetson from the measured time per evaluation, ICP time as
measured on this Mac) at a selectable fraction: "real time" plays the roughly quarter second as it would run. The table and the verdicts are read from results.json
(tracker runs); only the animation is computed live, with the same seeds as the evaluation."""
import argparse
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
from plotly.colors import sample_colorscale
import torch
import trimesh
import viser

from . import evaluate, geom, infer, scene

ROOT = Path(__file__).resolve().parents[1]
FILTERS = ("all scenes", "classical fails, flow succeeds", "classical succeeds, flow fails", "both succeed", "both fail")
GREEN, RED, BLUE, GREY = (40, 190, 80), (225, 60, 50), (40, 110, 235), (150, 150, 150)
YELLOW = (240, 190, 20)
SPEEDS = {"real time": 1.0, "1/4 of real time": 0.25, "1/10 of real time": 0.1, "1/25 of real time": 0.04,
          "1/50 of real time": 0.02}
REFINE = 2      # end-point corrections after the Euler steps (infer.propose default)
MARK = {"correct": "✅ correct", "wrong": "❌ WRONG pose accepted", "none": "⚠️ no pose"}


CAM_COL = ((0, 165, 160), (210, 70, 165))                 # points and frustum of camera 1 (D435i_1), camera 2 (D435i_2)
CAM_NAME = ("camera 1 (D435i_1, -Y side)", "camera 2 (D435i_2, +Y side)")
CARD = ("position:fixed;z-index:20;pointer-events:none;background:rgba(255,255,255,0.94);border-radius:8px;"
        "box-shadow:0 1px 6px rgba(0,0,0,0.25);padding:10px 14px;font-size:13px;line-height:1.42;color:#1c1c1c;")
VIRIDIS = sample_colorscale("Viridis", [i / 255 for i in range(256)])


def md_html(md):
    """The small markdown subset used here ('## ' / '### ' headings, **bold**, tables) as HTML."""
    out, rows = [], []
    flush = lambda: out.append('<table style="border-collapse:collapse;margin:4px 0">' + "".join(rows) + "</table>")
    for line in md.split("\n"):
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(set(c) <= set("-") for c in cells):
                rows.append("<tr>" + "".join(f'<td style="padding:1px 12px 1px 0">{c}</td>' for c in cells) + "</tr>")
            continue
        if rows:
            flush()
            rows = []
        if line.startswith("### "):
            out.append(f'<div style="font-size:1.2em;font-weight:700;margin:2px 0 4px">{line[4:]}</div>')
        elif line.startswith("## "):
            out.append(f'<div style="font-size:1.35em;font-weight:700;margin:0 0 4px">{line[3:]}</div>')
        elif line.strip():
            out.append(f'<div style="margin-bottom:5px">{line}</div>')
    if rows:
        flush()
    return "".join(out)


def swatch(rgb, text):
    return f'<span style="color:rgb{tuple(rgb)};font-weight:700">■ {text}</span>'


def field_svg(pid, z, u, v, zmax, half, unit, title, xlabel, ylabel, circle=None, dots=None, size=230):
    """One slice of the field as SVG: cells coloured by z (viridis, 0..zmax), a white arrow per cell in the direction
    (u, v), a cross at the reference, optionally a dotted circle (radius in axis units) and dots (axis units)."""
    G = z.shape[0]
    L, T, cell = 46, 22, size / G
    span = size - cell                                            # cell centres run from the first to the last cell
    X = lambda x: L + cell / 2 + (x + half) / (2 * half) * span
    Y = lambda y: T + size - cell / 2 - (y + half) / (2 * half) * span
    g = np.linspace(-half, half, G)
    zi = np.clip(z / max(zmax, 1e-9) * 255, 0, 255).astype(int)
    n = np.hypot(u, v)
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{L + size + 62}" height="{T + size + 36}" style="font:10px sans-serif">',
         f'<defs><marker id="ah{pid}" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">'
         f'<path d="M0,0 L5,2.5 L0,5 z" fill="white"/></marker>'
         f'<linearGradient id="cb{pid}" x1="0" y1="1" x2="0" y2="0">'
         + "".join(f'<stop offset="{k / 8}" stop-color="{VIRIDIS[int(k / 8 * 255)]}"/>' for k in range(9)) + "</linearGradient></defs>",
         f'<text x="{L}" y="12" font-size="11" font-weight="700">{title}</text>']
    for r in range(G):
        for c in range(G):
            e.append(f'<rect x="{X(g[c]) - cell / 2:.1f}" y="{Y(g[r]) - cell / 2:.1f}" width="{cell + 0.4:.1f}" height="{cell + 0.4:.1f}" '
                     f'fill="{VIRIDIS[zi[r, c]]}"/>')
    for r in range(G):
        for c in range(G):
            if n[r, c] > 1e-9:
                dx, dy = u[r, c] / n[r, c] * cell * 0.38, -v[r, c] / n[r, c] * cell * 0.38
                e.append(f'<line x1="{X(g[c]) - dx:.1f}" y1="{Y(g[r]) - dy:.1f}" x2="{X(g[c]) + dx:.1f}" y2="{Y(g[r]) + dy:.1f}" '
                         f'stroke="white" stroke-width="0.9" marker-end="url(#ah{pid})"/>')
    if circle:
        e.append(f'<circle cx="{X(0):.1f}" cy="{Y(0):.1f}" r="{circle / (2 * half) * span:.1f}" fill="none" stroke="white" '
                 'stroke-width="1.6" stroke-dasharray="4 3"/>')
    e.append(f'<path d="M{X(0) - 5},{Y(0) - 5} L{X(0) + 5},{Y(0) + 5} M{X(0) - 5},{Y(0) + 5} L{X(0) + 5},{Y(0) - 5}" stroke="white" stroke-width="2"/>')
    if dots is not None:
        for x, y in dots:
            if abs(x) <= half and abs(y) <= half:
                e.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="#ff3b6b" stroke="white" stroke-width="1"/>')
    for val in (-half, 0, half):
        e.append(f'<text x="{X(val):.1f}" y="{T + size + 12}" text-anchor="middle">{val:g}</text>')
        e.append(f'<text x="{L - 4}" y="{Y(val) + 3:.1f}" text-anchor="end">{val:g}</text>')
    e.append(f'<rect x="{L}" y="{T}" width="{size}" height="{size}" fill="none" stroke="#666" stroke-width="0.6"/>')
    e.append(f'<text x="{L + size / 2}" y="{T + size + 27}" text-anchor="middle">{xlabel} ({unit})</text>')
    e.append(f'<text transform="translate(11,{T + size / 2}) rotate(-90)" text-anchor="middle">{ylabel} ({unit})</text>')
    bx = L + size + 10
    e.append(f'<rect x="{bx}" y="{T}" width="9" height="{size}" fill="url(#cb{pid})"/>')
    e.append(f'<text x="{bx + 12}" y="{T + 8}">{zmax:.0f} {unit}</text><text x="{bx + 12}" y="{T + size}">0</text>')
    e.append("</svg>")
    return "".join(e)


def pose_T(v):
    return geom.make_T(geom.R_from_quat(np.array(v[3:])), np.array(v[:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flow/runs/a/model.pt")
    ap.add_argument("--eval", default="flow/data/eval")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--view-distance", type=float, default=1.0, help="camera distance factor of the automatic view")
    ap.add_argument("--centers", type=int, default=4)
    ap.add_argument("--orient", type=int, default=32)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    ev = ROOT / a.eval
    res = json.load(open(ev / "results.json"))
    scenes = res["scenes"]
    meta = np.load(ev / "meta.npz")
    n_px = scene.W * scene.H
    raw = np.memmap(ev / "scenes.bin", dtype=np.uint8, mode="r")
    rec = 128 + 4 * n_px
    dev = torch.device(a.device)
    net = infer.load_model(ROOT / a.model, dev)
    rig = scene.Rig()

    def load(i):
        blk = raw[24 + i * rec: 24 + (i + 1) * rec]
        depth = [np.frombuffer(blk[128 + c * 2 * n_px: 128 + (c + 1) * 2 * n_px].tobytes(), "<u2").reshape(scene.H, scene.W) for c in range(2)]
        labels = [np.unpackbits(meta["labels"][i][c])[:n_px].reshape(scene.H, scene.W) for c in range(2)]
        return meta["T"][i], depth, labels

    server = viser.ViserServer(host="0.0.0.0", port=a.port, verbose=False)
    sc, gui = server.scene, server.gui
    gui.configure_theme(control_width="large", dark_mode=False)
    sc.set_up_direction("+z")
    rig_stl = ROOT / "models/rig.stl"
    if rig_stl.exists():
        m = trimesh.load(rig_stl)
        sc.add_mesh_simple("/rig", np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32), color=(120, 120, 125), opacity=0.5)
    for c, T in enumerate(rig.T_rig_cam):
        sc.add_camera_frustum(f"/cam{c}", fov=2 * np.arctan(scene.H / 2 / scene.INTR[c][0]), aspect=scene.W / scene.H, scale=0.05,
                              wxyz=geom.quat_from_R(T[:3, :3]), position=T[:3, 3], color=CAM_COL[c], line_width=2.5)
        sc.add_label(f"/cam{c}/label", CAM_NAME[c].split(" (")[0])
    Vc = (rig.V - rig.centroid).astype(np.float32)     # centred on the centroid: a hypothesis is (q, p)
    V0, F = rig.V.astype(np.float32), rig.F
    h_truth = sc.add_mesh_simple("/truth", V0, F, color=GREEN, wireframe=True)
    h_classical = sc.add_mesh_simple("/classical", V0, F, color=RED, opacity=0.55, visible=False)
    h_flow = sc.add_mesh_simple("/flow", V0, F, color=BLUE, opacity=0.85, visible=False)
    h_end = sc.add_mesh_simple("/flow_end", V0, F, color=YELLOW, opacity=0.8, visible=False)
    h_end_wire = sc.add_mesh_simple("/flow_end_wire", V0, F, color=YELLOW, wireframe=True, visible=False)
    h_hyp = [sc.add_mesh_simple(f"/hyp/{k}", Vc, F, color=GREY, opacity=0.25, visible=False) for k in range(a.centers * a.orient)]
    state = {"list": list(range(len(scenes))), "cloud": None, "seeds": None, "gen": 0}
    lock = threading.Lock()

    # Reading panes (floating cards on the left and at the bottom: text and field plots) are separate from the viser
    # control panel on the right, which holds only the settings.
    class Pane:
        def __init__(self, style, order):
            self.style, self.order, self.parts = style, order, {}
            self.h = gui.add_html("")

        def set(self, key, html):
            self.parts[key] = html
            self.h.content = f'<div style="{CARD}{self.style}">' + "".join(self.parts.get(k, "") for k in self.order) + "</div>"

    class Section:                      # .content = markdown, as the gui markdown elements had
        def __init__(self, pane, key, wrap="{}"):
            self.pane, self.key, self.wrap = pane, key, wrap

        content = property(lambda self: "", lambda self, md: self.pane.set(self.key, self.wrap.format(md_html(md))))

    info = Pane("left:12px;top:12px;width:400px;", ("intro", "phase", "verdict"))
    plots = Pane("left:438px;bottom:12px;", ("plots",))
    rule = '<div style="border-top:1px solid #ccc;margin:8px 0 6px"></div>'
    phase = Section(info, "phase", rule + '<div style="height:205px;overflow:hidden">{}</div>')
    verdict = Section(info, "verdict", rule + "{}")
    info.set("intro", md_html(
        "## Finding the object: classical search vs flow\nOne depth pair per scene, tracker in search state. Both methods end in "
        "the same ICP and fit check of the tracker. Counts are **correct / wrong / none**: accepted within "
        f"{res['ok_mm']:.0f} mm and {res['ok_deg']:.0f}° of the truth / accepted elsewhere / nothing accepted in 2 s.\n"
        + evaluate.markdown(res["table"], compact=True)) +
        f'<div>Points: {swatch(CAM_COL[0], CAM_NAME[0])}, {swatch(CAM_COL[1], CAM_NAME[1])}; the camera outlines have the same '
        f'colours.<br>{swatch(GREEN, "truth (wireframe)")} {swatch(RED, "classical result")} {swatch(YELLOW, "flow end point")} '
        f'{swatch(BLUE, "after ICP + fit check")}<br>Thin shapes: flow hypotheses ({swatch(BLUE, "largest agreeing group")}, '
        f'{swatch((235, 160, 40), "other groups")}, {swatch(GREY, "alone")}).</div>')

    with gui.add_folder("Scene"):
        dd_cat = gui.add_dropdown("Clutter", options=("all",) + tuple(scene.CATEGORIES), initial_value="all")
        dd_fil = gui.add_dropdown("Show", options=FILTERS, initial_value=FILTERS[1])
        sl = gui.add_slider("Scene", min=0, max=1, step=1, initial_value=0)
        b_prev, b_next = gui.add_button("Previous scene"), gui.add_button("Next scene")
    with gui.add_folder("Playback"):
        b_play = gui.add_button("Play flow")
        dd_speed = gui.add_dropdown("Speed", options=tuple(SPEEDS), initial_value="1/25 of real time")
    with gui.add_folder("Display"):
        c_truth = gui.add_checkbox("True pose", initial_value=True)
        c_classical = gui.add_checkbox("Classical result", initial_value=True)
        c_flow = gui.add_checkbox("Flow end point and result", initial_value=True)
        c_hyp = gui.add_checkbox("Flow hypotheses", initial_value=True)
        c_field = gui.add_checkbox("Flow field plots", initial_value=True)
        c_label = gui.add_checkbox("Colour cloud by true object / clutter", initial_value=False)
        c_follow = gui.add_checkbox("Move the view to the object", initial_value=True)

    VIEW = np.array([0.20, -0.29, -0.17])                   # camera offset from the object (automatic view)
    TIMES = infer.field_times()

    def draw_field(frame, step):
        """The two field slices at flow-time frame `frame` (index into TIMES), with the hypotheses at trajectory state
        `step` projected onto each slice."""
        last = state.get("last")
        if last is None or not c_field.value:
            plots.h.content = ""
            return
        T, h, fr = last
        c = T[:3, :3] @ rig.centroid + T[:3, 3]
        dp = dr = None
        if "traj_q" in h and len(h["q"]):
            step = min(step, len(h["traj_p"]) - 1)
            dp = (h["traj_p"][step] - c)[:, list(infer.FIELD_AXES)] * 1000
            q_ref = torch.from_numpy(geom.quat_from_R(T[:3, :3]).astype(np.float32))
            tq = torch.from_numpy(h["traj_q"][step]).float()
            near = np.linalg.norm(h["traj_p"][step] - c, axis=1) < 0.05          # rotation dots: hypotheses at the object
            dr = np.degrees(geom.qlog(geom.qmul(geom.qconj(q_ref).expand_as(tq), tq)).numpy())[near][:, list(infer.FIELD_AXES)]
        t = TIMES[frame]
        a_ = field_svg("p", fr["pos_mm"][frame], fr["pos_u"][frame], fr["pos_v"][frame], float(fr["pos_mm"].max()), 100, "mm",
                       f"Translation field, t = {t:.3g}", "rig x", "rig y", dots=dp)
        b_ = field_svg("r", fr["rot_deg"][frame], fr["rot_u"][frame], fr["rot_v"][frame], float(fr["rot_deg"].max()), 170, "deg",
                       f"Rotation field, t = {t:.3g}", "about object x", "about object y", circle=70, dots=dr)
        cap = ('<div style="width:650px;font-size:11.5px;line-height:1.35">Two 2-D slices of the learned field through the true pose '
               '(white cross). Colour: size of the motion the network predicts is still needed; arrows: its direction in the slice. '
               'Left: rotation at the truth, position varied. Right: position at the truth, rotation varied (dotted circle: 70°, the '
               'training range). <span style="color:#ff3b6b;font-weight:700">●</span> this scene\'s hypotheses, projected onto the '
               'slice (they are not in it: each also has an error in the other four dimensions).</div>')
        plots.set("plots", f'<div style="display:flex;gap:6px">{a_}{b_}</div>{cap}')

    def matches(s):
        if dd_cat.value != "all" and s["cat"] != dd_cat.value:
            return False
        c, f = s["classical"] == "correct", s["flow"] == "correct"
        return {FILTERS[0]: True, FILTERS[1]: not c and f, FILTERS[2]: c and not f, FILTERS[3]: c and f, FILTERS[4]: not c and not f}[dd_fil.value]

    def describe(s, n):
        cl, fw = s["classical"], s["flow"]
        t_c = f"after {s['classical_cycles']} cycles ({s['classical_cycles'] * evaluate.PERIOD_MS / 1000:.2f} s of pairs)"
        e_c = f", error {s['classical_err_mm']:.1f} mm / {s['classical_err_deg']:.1f}°" if cl != "none" else ""
        e_f = f", error {s['flow_err_mm']:.1f} mm / {s['flow_err_deg']:.1f}°" if fw != "none" else ""
        t_f = f"ICP and fit check of {s['hints_tried']} hypotheses in one cycle ({s['flow_icp_ms']:.0f} ms on this Mac)"
        return (f"**Scene {s['scene']}** ({s['cat']}), {n} shown\n\n"
                f"**Classical search:** {MARK[cl]} {t_c}{e_c}\n\n"
                f"**Flow + ICP:** {MARK[fw]}{e_f}; {t_f}\n\n"
                f"Flow hypotheses within 15 mm / 15° before ICP: {s['hyp_good']} of {a.centers * a.orient}")

    def err(Ta, Tb):
        return 1000 * np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]), geom.rot_angle_deg(Ta[:3, :3], Tb[:3, :3])

    def set_pose(handle, q, t):
        handle.wxyz, handle.position = q, t

    def animate(gen, h):
        alive = lambda: state["gen"] == gen
        groups, T_true, T_acc, ok = h["groups"], state["T_true"], state["T_flow"], state["flow_ok"]
        col = {}
        for gi, g in enumerate(groups):
            for j in g:
                col[j] = BLUE if gi == 0 and len(g) > 1 else ((235, 160, 40) if len(g) > 1 else GREY)
        K = len(h["q"])
        # Real durations: network time on the Jetson from the measured 5.2 ms per evaluation of 32 hypotheses (position
        # stage: 5 evaluations of the seeds; rotation flow: one evaluation of K hypotheses per step); ICP and fit check
        # as measured in the tracker run of this scene on this Mac. The animation runs at `rate` times that speed.
        rate = SPEEDS[dd_speed.value]
        ms_pos = infer.JETSON_MS_PER_32 * 5 * len(h["seeds"]) / 32
        ms_step = infer.JETSON_MS_PER_32 * K / 32
        ms_icp = state["icp_ms"]
        clock = lambda ms: f"\n\n**Real time elapsed: {ms:.0f} ms** (playback: {dd_speed.value})"

        def play(ms, draw):
            """Draws draw(w), w in (0, 1], over ms of real time at the playback rate. False if the scene changed."""
            dur = ms / 1000 / rate
            n = max(1, int(round(dur * 30)))
            for f in range(1, n + 1):
                if not alive():
                    return False
                draw(f / n)
                time.sleep(dur / n)
            return True

        hold = lambda sec: time.sleep(0 if rate >= 1 else sec)     # reading pauses (not real time), none at real time
        with server.atomic():
            h_end.visible = h_end_wire.visible = h_flow.visible = False
            for k, hh in enumerate(h_hyp):
                hh.visible = k < K and c_hyp.value
                if k < K:
                    hh.color = col.get(k, GREY)
        # phase 0: position stage (not drawn: the hypotheses start at the positions it found)
        phase.content = (f"### 0 · Position stage\n{len(h['seeds'])} seeds on the cloud, 5 network evaluations: {len(h['centers'])} "
                         f"candidate position(s). Not drawn; the shapes start there." + clock(0))
        draw_field(0, 0)
        if not play(ms_pos, lambda w: None):
            return
        # phase 1: the flow (Euler steps, then end-point corrections)
        tq, tp = h["traj_q"], h["traj_p"]
        n_euler = len(tq) - 1 - REFINE
        for step in range(len(tq) - 1):
            phase.content = (f"### 1 · Flow: Euler step {step + 1} of {n_euler}" if step < n_euler else
                             f"### 1 · Flow: end-point correction {step - n_euler + 1} of {REFINE}") + \
                (f"\n{K} hypotheses, each moved by the network's predicted remaining motion ({ms_step:.1f} ms per step on the "
                 "Jetson). No ICP yet.") + clock(ms_pos + step * ms_step)
            draw_field(min(step, len(TIMES) - 1), step)       # the field the network is evaluated on in this step
            q0, q1 = tq[step], tq[step + 1]
            q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)

            def draw(w, q0=q0, q1=q1, p0=tp[step], p1=tp[step + 1]):
                q = (1 - w) * q0 + w * q1
                q /= np.linalg.norm(q, axis=-1, keepdims=True)
                pp = (1 - w) * p0 + w * p1
                with server.atomic():
                    for k in range(K):
                        set_pose(h_hyp[k], q[k], pp[k])

            if not play(ms_step, draw):
                return
        ms_flow = ms_pos + (len(tq) - 1) * ms_step
        draw_field(len(TIMES) - 1, len(tq) - 1)
        hold(0.5)
        if not alive() or K == 0:
            return
        if not ok:      # nothing passed the fit check: leave the largest agreeing group on screen
            keep = set(groups[0]) if groups else set()
            with server.atomic():
                for k in range(K):
                    h_hyp[k].visible = c_hyp.value and k in keep
            best = min(err(h["T"][k], T_true) for k in range(K))
            phase.content = ("### 2 · Tracker: no hypothesis passed the fit check\nThe shapes left are the largest agreeing group "
                             f"of flow end points. Closest flow end point to the truth: {best[0]:.1f} mm / {best[1]:.1f}°.")
            return
        # phase 2: the flow end point that the accepted pose came from (the closest one to it)
        j = min(range(K), key=lambda k: err(h["T"][k], T_acc)[0] / 10 + err(h["T"][k], T_acc)[1] / 10)
        T_end = h["T"][j]
        e_end, e_acc = err(T_end, T_true), err(T_acc, T_true)
        q_end, q_acc = geom.quat_from_R(T_end[:3, :3]), geom.quat_from_R(T_acc[:3, :3])
        with server.atomic():
            for k in range(K):
                h_hyp[k].visible = False
            set_pose(h_end, q_end, T_end[:3, 3])
            set_pose(h_end_wire, q_end, T_end[:3, 3])
            h_end.visible = c_flow.value
        phase.content = ("### 2 · Where the flow ends (yellow)\nThe flow end point closest to the pose the tracker accepted: "
                         f"**{e_end[0]:.1f} mm / {e_end[1]:.1f}°** from the truth. This is the output of the network alone." + clock(ms_flow))
        hold(1.8)
        if not alive():
            return
        # phase 3: ICP and fit check of the tracker, drawn as a straight interpolation to the accepted pose
        phase.content = ("### 3 · Tracker: ICP from the flow end points, then the fit check (blue)\nClassical point-to-plane ICP "
                         f"closes the remaining gap ({ms_icp:.0f} ms for all hypotheses, this Mac). Yellow outline = where the flow ended."
                         + clock(ms_flow))
        with server.atomic():
            h_end.visible = False
            h_end_wire.visible = c_flow.value
            set_pose(h_flow, q_end, T_end[:3, 3])
            h_flow.visible = c_flow.value
        if np.dot(q_end, q_acc) < 0:
            q_acc = -q_acc
        def draw_icp(w):
            w = 0.5 - 0.5 * np.cos(np.pi * w)
            q = (1 - w) * q_end + w * q_acc
            set_pose(h_flow, q / np.linalg.norm(q), (1 - w) * T_end[:3, 3] + w * T_acc[:3, 3])

        if not play(ms_icp, draw_icp):
            return
        phase.content = (f"### Result\n| | error to the truth |\n|---|---|\n| flow end point (yellow outline) | {e_end[0]:.1f} mm / {e_end[1]:.1f}° |\n"
                         f"| after the tracker's ICP (blue) | **{e_acc[0]:.1f} mm / {e_acc[1]:.1f}°** |\n\n"
                         + ("The accepted pose is wrong: it passed the tracker's fit check." if state["flow_verdict"] != "correct"
                            else "ICP refined the flow end point." if e_acc[0] < e_end[0] - 0.5 or e_acc[1] < e_end[1] - 0.5
                            else "The flow end point was already this accurate; ICP and the fit check confirmed it.")
                         + f"\n\n**Real time: {ms_flow:.0f} ms network (Jetson, from the measured time per evaluation) + {ms_icp:.0f} ms "
                           f"ICP (this Mac) = {ms_flow + ms_icp:.0f} ms.** Classical search on this scene: "
                           f"{state['classical_text']}.")

    def show(replay_only=False):
        with lock:
            state["gen"] += 1
            gen = state["gen"]
            lst = state["list"]
            if not lst:
                verdict.content = "No scene matches this selection."
                return
            s = scenes[lst[min(int(sl.value), len(lst) - 1)]]
            T, depth, labels = load(s["scene"])
            if not replay_only:
                P, C, L = rig.cloud(depth, labels=labels)
                state["cloud"] = (P, C, L)
            P, C, L = state["cloud"]
            idx = np.random.default_rng(0).choice(len(P), min(len(P), 60000), replace=False)
            colr = np.where(L[idx, None] > 0, [[250, 170, 40]], [[130, 130, 140]]) if c_label.value else \
                np.where(C[idx, None] > 0, [list(CAM_COL[1])], [list(CAM_COL[0])])
            sc.add_point_cloud("/cloud", P[idx], colr.astype(np.uint8), point_size=0.0022, point_shape="circle")
            h_truth.wxyz, h_truth.position, h_truth.visible = geom.quat_from_R(T[:3, :3]), T[:3, 3], c_truth.value
            Tc = pose_T(s["classical_pose"])
            h_classical.wxyz, h_classical.position = geom.quat_from_R(Tc[:3, :3]), Tc[:3, 3]
            h_classical.visible = s["classical"] != "none" and c_classical.value
            Tf = pose_T(s["flow_pose"])
            h_flow.wxyz, h_flow.position = geom.quat_from_R(Tf[:3, :3]), Tf[:3, 3]
            h_flow.visible = h_end.visible = h_end_wire.visible = False
            state["flow_ok"] = s["flow"] != "none"
            state["flow_verdict"], state["T_true"], state["T_flow"] = s["flow"], np.array(T), Tf
            state["icp_ms"] = s["flow_icp_ms"]
            state["classical_text"] = (f"{s['classical_cycles'] * evaluate.PERIOD_MS:.0f} ms of pairs, "
                                       + {"correct": "correct", "wrong": "wrong pose", "none": "no pose"}[s["classical"]])
            phase.content = "### Computing the flow ..."
            verdict.content = describe(s, len(lst))
            c = T[:3, :3] @ rig.centroid + T[:3, 3]
            state["center"] = c
            server.initial_camera.position = c + a.view_distance * VIEW
            server.initial_camera.look_at = c
            if c_follow.value and not replay_only:
                for cl in server.get_clients().values():
                    aim(cl)
            h = infer.propose(net, rig, depth, dev, k=a.k, centers=a.centers, n_orient=a.orient, seed=s["scene"], trajectory=True)
            if not infer.field_cache_path(ev, s["scene"]).exists():
                phase.content = ("### Computing the flow field of this scene ...\nNot in the cache yet (about 10 s, once). "
                                 "`python -m flow.precompute_fields` fills the cache for all scenes.")
            fr = infer.cached_field_frames(net, rig, ev, s["scene"], P, C, np.array(T), dev)
            state["last"] = (np.array(T), h, fr)
        animate(gen, h)

    def aim(cl):
        c = state.get("center")
        if c is not None:
            with cl.atomic():
                cl.camera.position = c + a.view_distance * VIEW
                cl.camera.look_at = c
                cl.camera.up_direction = (0.0, 0.0, 1.0)

    def connected(cl):
        time.sleep(0.6)       # after the client's own initial camera message
        aim(cl)
        show(replay_only=True)

    def refilter(_=None):
        state["list"] = [s["scene"] for s in scenes if matches(s)]
        sl.max = max(1, len(state["list"]) - 1)
        sl.value = 0
        show()

    def step(d):
        n = len(state["list"])
        if n:
            sl.value = (int(sl.value) + d) % n

    dd_cat.on_update(refilter)
    dd_fil.on_update(refilter)
    sl.on_update(lambda _: show())
    b_prev.on_click(lambda _: step(-1))
    b_next.on_click(lambda _: step(1))
    b_play.on_click(lambda _: show(replay_only=True))
    dd_speed.on_update(lambda _: show(replay_only=True))
    for cb in (c_truth, c_classical, c_flow, c_hyp, c_field, c_label):
        cb.on_update(lambda _: show(replay_only=True))
    server.on_client_connect(connected)
    state["list"] = [s["scene"] for s in scenes if matches(s)]
    sl.max = max(1, len(state["list"]) - 1)
    show()
    print(f"demo: http://localhost:{a.port}", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
