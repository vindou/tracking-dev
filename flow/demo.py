"""Viser demo: the classical search of the tracker against the learned proposals (flow + verifier), on the
evaluation scenes. Both end in the same ICP and fit check of the tracker.

    python -m flow.evaluate --eval flow/data/eval_demo                   # once: hypotheses, tracker runs, results.json
    python -m flow.demo [--port 8090]                                     # then open http://localhost:8090

Per scene it shows the cloud of the depth pair, the true pose (green wireframe), the pose the classical search
accepted (red), and the learned proposals in labelled phases:
  1. flow: every hypothesis moving from its start to its end point (RK4 steps of the learned field);
  2. verifier: each hypothesis is refined on its own points and scored; those below the minimum score disappear,
     the rest are coloured by score; one per agreeing group is handed to the tracker;
  3. tracker: its ICP and fit check, drawn as a motion from the handed hypothesis closest to the accepted pose
     (yellow) to the accepted pose (blue). The ICP iterations are not recorded: the motion is an interpolation.
Two 2-D slices of the learned velocity field around the true pose are plotted at the bottom (colour = speed,
arrows = direction a hypothesis moves), at the flow time of the current step, with the hypotheses projected onto them.
They are cached per scene in <eval>/fields/.

The table and the verdicts are read from results.json (tracker runs). The animation is computed live with the
same seeds; it reproduces the evaluated hypotheses when results.json was made on this machine with the same
--device (the default folder eval_demo is), not for hypotheses made on another machine.
Playback is paced by the time the network and verifier took for this scene on this machine, at a selectable
fraction; that time is not the time on the Jetson, which has not been measured for this model."""
import argparse
import base64
import io
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plotly.colors import sample_colorscale
import torch
import trimesh
import viser

from . import evaluate, geom, model, propose, scene

ROOT = Path(__file__).resolve().parents[1]
FILTERS = ("all scenes", "classical fails, learned succeeds", "classical succeeds, learned fails", "both succeed", "both fail",
           "learned accepts a wrong pose")
GREEN, RED, BLUE, GREY = (40, 190, 80), (225, 60, 50), (40, 110, 235), (150, 150, 150)
YELLOW, ORANGE = (240, 190, 20), (235, 140, 40)
SPEEDS = {"real time": 1.0, "1/2 of real time": 0.5, "1/5 of real time": 0.2, "1/10 of real time": 0.1, "1/25 of real time": 0.04}
MARK = {"correct": "✅ correct", "wrong": "❌ WRONG pose accepted", "none": "⚠️ no pose"}
FIELD_GRID = 17                 # estimates per axis of a field slice
FIELD_POS_HALF = 0.10           # m: half-width of the translation slice
FIELD_ROT_HALF = 170.0          # degrees: half-width of the rotation slice
FIELD_AXES = (0, 1)             # the slices use axes x and y (rig axes for positions, object axes for rotations)


CAM_COL = ((0, 165, 160), (210, 70, 165))                 # points and frustum of camera 1 (D435i_1), camera 2 (D435i_2)
CAM_NAME = ("camera 1 (D435i_1, -Y side)", "camera 2 (D435i_2, +Y side)")
CARD = ("position:fixed;z-index:20;pointer-events:none;background:rgba(255,255,255,0.94);border-radius:8px;"
        "box-shadow:0 1px 6px rgba(0,0,0,0.25);padding:10px 14px;font-size:13px;line-height:1.42;color:#1c1c1c;")
VIRIDIS = sample_colorscale("Viridis", [i / 255 for i in range(256)])
VIRIDIS_RGB = np.array([[int(v) for v in re.findall(r"\d+", c)[:3]] for c in VIRIDIS], np.uint8)


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


def field_svg(pid, z, u, v, zmax, half, unit, title, xlabel, ylabel, dots=None, size=230):
    """One slice of the field as SVG: cells coloured by z (viridis, 0..zmax), a white arrow per cell in the direction
    (u, v), a cross at the reference, optionally dots (axis units)."""
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
    # the cells as one G x G image (no seams between cells), first row at the bottom
    png = io.BytesIO()
    Image.fromarray(VIRIDIS_RGB[zi[::-1]], "RGB").save(png, "PNG")
    e.append(f'<image x="{L}" y="{T}" width="{size}" height="{size}" preserveAspectRatio="none" style="image-rendering:pixelated" '
             f'href="data:image/png;base64,{base64.b64encode(png.getvalue()).decode()}"/>')
    for r in range(G):
        for c in range(G):
            if n[r, c] > 1e-9:
                dx, dy = u[r, c] / n[r, c] * cell * 0.38, -v[r, c] / n[r, c] * cell * 0.38
                e.append(f'<line x1="{X(g[c]) - dx:.1f}" y1="{Y(g[r]) - dy:.1f}" x2="{X(g[c]) + dx:.1f}" y2="{Y(g[r]) + dy:.1f}" '
                         f'stroke="white" stroke-width="0.9" marker-end="url(#ah{pid})"/>')
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



@torch.no_grad()
def field_frames(net, table, rig, P, C, T_ref, dev, steps):
    """Two 2-D slices of the learned velocity field around T_ref, at the flow time at the start of each solver step
    (t = 1, ..., 1 / steps; hypotheses move from t = 1 to t = 0):
      translation slice: rotation at the reference, centroid estimate moved in the rig x-y plane;
      rotation slice: position at the reference, estimate rotated about the object x and y axes.
    Arrays [steps, G, G] (row = second axis): pos_mm / rot_deg = speed (mm and degrees per unit of flow time) and
    pos_u, pos_v / rot_u, rot_v = the direction a hypothesis moves, in the slice."""
    G, (i, j) = FIELD_GRID, FIELD_AXES
    N = G * G
    c = T_ref[:3, :3] @ rig.centroid + T_ref[:3, 3]
    X, cam, _ = propose.point_sets(P, C, c[None], np.random.default_rng(0), dev, torch.Generator(device=dev).manual_seed(0))
    X, cam = X.expand(2 * N, -1, -1), cam.expand(2 * N, *cam.shape[1:])
    g = np.linspace(-FIELD_POS_HALF, FIELD_POS_HALF, G)
    A, B = np.meshgrid(g, g, indexing="xy")
    off = np.zeros((2 * N, 3), np.float32)
    off[:N, i], off[:N, j] = A.ravel(), B.ravel()
    g = np.radians(np.linspace(-FIELD_ROT_HALF, FIELD_ROT_HALF, G))
    A, B = np.meshgrid(g, g, indexing="xy")
    rv = np.zeros((2 * N, 3), np.float32)
    rv[N:, i], rv[N:, j] = A.ravel(), B.ravel()
    q_ref = torch.from_numpy(geom.quat_from_R(T_ref[:3, :3]).astype(np.float32)).to(dev)[None].expand(2 * N, -1)
    q = geom.qmul(q_ref, model.qexp(torch.from_numpy(rv).to(dev)))
    p = torch.from_numpy(off).to(dev)
    out = {k: [] for k in ("pos_mm", "pos_u", "pos_v", "rot_deg", "rot_u", "rot_v")}
    h = 1.0 / steps
    for s in range(steps):
        f = model.field(net, table, X, cam, q, p, model.T_END * (1 - s * h))[0]
        v = -(geom.qmat(q) @ f[:, 3:, None]).squeeze(-1) * model.POS_SCALE                 # motion per unit time, rig frame (m)
        q2 = geom.qmul(q, model.qexp(-h * f[:, :3]))                                       # after one step of the rotation
        d = (geom.qlog(geom.qmul(geom.qconj(q_ref), q2)) - torch.from_numpy(rv).to(dev)).cpu().numpy()
        d[np.linalg.norm(d, axis=1) > 2.8] = 0                                             # the chart wraps
        v, w = v.cpu().numpy(), f[:, :3].norm(dim=-1).rad2deg().cpu().numpy()
        out["pos_mm"].append(1000 * np.linalg.norm(v[:N], axis=1))
        out["pos_u"].append(v[:N, i])
        out["pos_v"].append(v[:N, j])
        out["rot_deg"].append(w[N:])
        out["rot_u"].append(d[N:, i])
        out["rot_v"].append(d[N:, j])
    return {k: np.stack(v).reshape(-1, G, G).astype(np.float32) for k, v in out.items()}


def cached_field_frames(net, table, rig, ev, scene_index, P, C, T_ref, dev, steps):
    f = Path(ev) / "fields" / f"scene_{scene_index:04d}.npz"
    if f.exists():
        z = np.load(f)
        if z["pos_mm"].shape[:2] == (steps, FIELD_GRID):
            return {k: z[k] for k in z.files}
    fr = field_frames(net, table, rig, P, C, T_ref, dev, steps)
    f.parent.mkdir(exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **fr)
    tmp.replace(f)
    return fr


def results_table(table):
    L = ["| scenes | classical | flow + verifier |", "|---|---|---|"]
    for cat, r in table.items():
        L.append(f"| {cat} ({r['n']}) | {r['classical_correct']} / {r['classical_wrong']} / {r['classical_none']} | "
                 f"**{r['flow_correct']}** / {r['flow_wrong']} / {r['flow_none']} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=propose.MODEL)
    ap.add_argument("--verifier", default=propose.VERIFIER)
    ap.add_argument("--min-score", type=float, default=propose.MIN_SCORE)
    ap.add_argument("--table", default=propose.TABLE)
    ap.add_argument("--eval", default="flow/data/eval_demo")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--hints", type=int, default=16)
    ap.add_argument("--view-distance", type=float, default=1.0, help="camera distance factor of the automatic view")
    ap.add_argument("--centers", type=int, default=4)
    ap.add_argument("--orient", type=int, default=32)
    ap.add_argument("--solver-steps", type=int, default=propose.SOLVER_STEPS)
    ap.add_argument("--show", default=FILTERS[1], choices=FILTERS, help="the selection of scenes shown first")
    ap.add_argument("--speed", default="1/5 of real time", choices=tuple(SPEEDS))
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
    net, ver, table = propose.load(dev, a.model, a.verifier, a.table)
    rig = scene.Rig()
    STEPS = a.solver_steps

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
    state = {"list": list(range(len(scenes))), "cloud": None, "gen": 0}
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

    class Section:                      # .content = markdown
        def __init__(self, pane, key, wrap="{}"):
            self.pane, self.key, self.wrap = pane, key, wrap

        content = property(lambda self: "", lambda self, md: self.pane.set(self.key, self.wrap.format(md_html(md))))

    info = Pane("left:12px;top:12px;width:410px;", ("intro", "phase", "verdict"))
    plots = Pane("left:448px;bottom:12px;", ("plots",))
    rule = '<div style="border-top:1px solid #ccc;margin:8px 0 6px"></div>'
    phase = Section(info, "phase", rule + '<div style="height:215px;overflow:hidden">{}</div>')
    verdict = Section(info, "verdict", rule + "{}")
    info.set("intro", md_html(
        "## Finding the object: classical search vs learned proposals\nOne depth pair per scene, tracker in search state. The "
        "network is trained on synthetic data only. Both methods end in the same ICP and fit check of the tracker. Counts are "
        f"**correct / wrong / none**: accepted within {res['ok_mm']:.0f} mm and {res['ok_deg']:.0f}° of the truth / accepted "
        "elsewhere / nothing accepted.\n" + results_table(res["table"])) +
        f'<div>Points: {swatch(CAM_COL[0], CAM_NAME[0])}, {swatch(CAM_COL[1], CAM_NAME[1])}; the camera outlines have the same '
        f'colours.<br>{swatch(GREEN, "truth (wireframe)")} {swatch(RED, "classical result")} {swatch(YELLOW, "handed to the tracker")} '
        f'{swatch(BLUE, "after ICP + fit check")}<br>Thin shapes: hypotheses ({swatch(GREY, "during the flow")}; after the verifier '
        f'{swatch(BLUE, "kept")}, {swatch(ORANGE, "kept, lower score")}; rejected ones disappear).</div>')

    with gui.add_folder("Scene"):
        dd_cat = gui.add_dropdown("Clutter", options=("all",) + tuple(scene.CATEGORIES), initial_value="all")
        dd_fil = gui.add_dropdown("Show", options=FILTERS, initial_value=a.show)
        sl = gui.add_slider("Scene", min=0, max=1, step=1, initial_value=0)
        b_prev, b_next = gui.add_button("Previous scene"), gui.add_button("Next scene")
    with gui.add_folder("Playback"):
        b_play = gui.add_button("Play")
        dd_speed = gui.add_dropdown("Speed", options=tuple(SPEEDS), initial_value=a.speed)
    with gui.add_folder("Display"):
        c_truth = gui.add_checkbox("True pose", initial_value=True)
        c_classical = gui.add_checkbox("Classical result", initial_value=True)
        c_flow = gui.add_checkbox("Learned result", initial_value=True)
        c_hyp = gui.add_checkbox("Hypotheses", initial_value=True)
        c_field = gui.add_checkbox("Flow field plots", initial_value=True)
        c_label = gui.add_checkbox("Colour cloud by true object / clutter", initial_value=False)
        c_follow = gui.add_checkbox("Move the view to the object", initial_value=True)

    VIEW = np.array([0.20, -0.29, -0.17])                   # camera offset from the object (automatic view)

    def draw_field(frame, step):
        """The two field slices of solver step `frame`, with the hypotheses at trajectory state `step` projected on them."""
        last = state.get("last")
        if last is None or not c_field.value:
            plots.h.content = ""
            return
        T, h, fr = last
        c = T[:3, :3] @ rig.centroid + T[:3, 3]
        dp = dr = None
        if "traj_q" in h and len(h["q"]):
            step = min(step, len(h["traj_p"]) - 1)
            dp = (h["traj_p"][step] - c)[:, list(FIELD_AXES)] * 1000
            q_ref = torch.from_numpy(geom.quat_from_R(T[:3, :3]).astype(np.float32))
            tq = torch.from_numpy(h["traj_q"][step]).float()
            near = np.linalg.norm(h["traj_p"][step] - c, axis=1) < 0.05          # rotation dots: hypotheses at the object
            dr = np.degrees(geom.qlog(geom.qmul(geom.qconj(q_ref).expand_as(tq), tq)).numpy())[near][:, list(FIELD_AXES)]
        frame = min(frame, STEPS - 1)
        t = 1 - frame / STEPS
        a_ = field_svg("p", fr["pos_mm"][frame], fr["pos_u"][frame], fr["pos_v"][frame], float(fr["pos_mm"].max()), 100, "mm",
                       f"Translation field, t = {t:.3g}", "rig x", "rig y", dots=dp)
        b_ = field_svg("r", fr["rot_deg"][frame], fr["rot_u"][frame], fr["rot_v"][frame], float(fr["rot_deg"].max()), 170, "deg",
                       f"Rotation field, t = {t:.3g}", "about object x", "about object y", dots=dr)
        cap = ('<div style="width:650px;font-size:11.5px;line-height:1.35">Two 2-D slices of the learned velocity field through the '
               'true pose (white cross); hypotheses move from t = 1 to t = 0. Colour: speed (per unit of flow time); arrows: the '
               'direction a hypothesis moves. Left: rotation at the truth, position varied. Right: position at the truth, rotation '
               'varied. <span style="color:#ff3b6b;font-weight:700">●</span> this scene\'s hypotheses, projected onto the slice '
               '(they are not in it: each also has an error in the other four dimensions).</div>')
        plots.set("plots", f'<div style="display:flex;gap:6px">{a_}{b_}</div>{cap}')

    def matches(s):
        if dd_cat.value != "all" and s["cat"] != dd_cat.value:
            return False
        c, f = s["classical"] == "correct", s["flow"] == "correct"
        return {FILTERS[0]: True, FILTERS[1]: not c and f, FILTERS[2]: c and not f, FILTERS[3]: c and f, FILTERS[4]: not c and not f,
                FILTERS[5]: s["flow"] == "wrong"}[dd_fil.value]

    def describe(s, n):
        cl, fw = s["classical"], s["flow"]
        t_c = f"after {s['classical_cycles']} cycles ({s['classical_cycles'] * evaluate.PERIOD_MS / 1000:.2f} s of pairs)"
        e_c = f", error {s['classical_err_mm']:.1f} mm / {s['classical_err_deg']:.1f}°" if cl != "none" else ""
        e_f = f", error {s['flow_err_mm']:.1f} mm / {s['flow_err_deg']:.1f}°" if fw != "none" else ""
        return (f"**Scene {s['scene']}** ({s['cat']}), {n} shown\n\n"
                f"**Classical search:** {MARK[cl]} {t_c}{e_c}\n\n"
                f"**Flow + verifier + ICP:** {MARK[fw]}{e_f}\n\n"
                f"Flow end points within 15 mm / 15° of the truth: {s['hyp_good']}")

    def err(Ta, Tb):
        return 1000 * np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]), geom.rot_angle_deg(Ta[:3, :3], Tb[:3, :3])

    def set_pose(handle, q, t):
        handle.wxyz, handle.position = q, t

    def lerp(q0, q1, p0, p1, w):
        q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
        q = (1 - w) * q0 + w * q1
        return q / np.linalg.norm(q, axis=-1, keepdims=True), (1 - w) * p0 + w * p1

    def animate(gen, h):
        alive = lambda: state["gen"] == gen
        T_true, T_acc, ok = state["T_true"], state["T_flow"], state["flow_ok"]
        K = len(h["q"])
        rate = SPEEDS[dd_speed.value]
        # Time of the network and verifier for this scene on this machine, divided over the phases by the number of
        # network evaluations: position stage (seeds x 4 per step), flow (K x 4 per step), verifier (ICP steps + score).
        n_pos, n_flow, n_ver = len(h["seeds"]) * 4 * STEPS, K * 4 * STEPS, K * 3
        ms_all = 1000 * h["t_flow"]
        ms_pos, ms_step, ms_ver = (ms_all * n / max(1, n_pos + n_flow + n_ver) for n in (n_pos, n_flow / STEPS, n_ver))
        ms_icp = state["icp_ms"]
        clock = lambda ms: f"\n\n**Time elapsed: {ms:.0f} ms** on this machine (playback: {dd_speed.value})"

        def play(ms, draw):
            """Draws draw(w), w in (0, 1], over ms of real time at the playback rate. False if the scene changed."""
            dur = max(ms, 60) / 1000 / rate
            n = max(1, int(round(dur * 30)))
            for f in range(1, n + 1):
                if not alive():
                    return False
                draw(f / n)
                time.sleep(dur / n)
            return True

        hold = lambda sec: time.sleep(sec if rate < 1 else 0.3 * sec)     # reading pauses (not part of the time shown)
        with server.atomic():
            h_end.visible = h_end_wire.visible = h_flow.visible = False
            for k, hh in enumerate(h_hyp):
                hh.visible = k < K and c_hyp.value
                if k < K:
                    hh.color = GREY
        if K == 0:
            phase.content = "### No seed\nThe cloud has no group of points to start from: no hypothesis."
            return
        # phase 0: position stage (not drawn: the hypotheses start at the positions it found)
        phase.content = (f"### 0 · Position stage\n{len(h['seeds'])} seeds on the cloud, each integrated through the flow: "
                         f"{len(h['centers'])} candidate position(s). Not drawn; the shapes start there, in {a.orient} orientations "
                         "each." + clock(0))
        draw_field(0, 0)
        tq, tp = h["traj_q"], h["traj_p"]
        with server.atomic():
            for k in range(K):
                set_pose(h_hyp[k], tq[0][k], tp[0][k])
        if not play(ms_pos, lambda w: None):
            return
        # phase 1: the flow
        for step in range(STEPS):
            phase.content = (f"### 1 · Flow: step {step + 1} of {STEPS}\n{K} hypotheses move along the learned velocity field "
                             f"(Runge-Kutta, 4 network evaluations per step). No ICP yet." + clock(ms_pos + step * ms_step))
            draw_field(step, step)

            def draw(w, s=step):
                q, pp = lerp(tq[s], tq[s + 1], tp[s], tp[s + 1], w)
                with server.atomic():
                    for k in range(K):
                        set_pose(h_hyp[k], q[k], pp[k])

            if not play(ms_step, draw):
                return
        ms_flow = ms_pos + STEPS * ms_step
        draw_field(STEPS - 1, STEPS)
        hold(0.5)
        if not alive():
            return
        # phase 2: verifier (refinement of each hypothesis on its own points, then the score)
        score, keep = h["score"], h["score"] >= a.min_score
        handed = propose.hint_order(h, a.hints)
        phase.content = (f"### 2 · Verifier: refine, then score\nEach hypothesis is refined on its own {model.N_POINTS} points "
                         "(6 point-to-plane steps), then a second small network scores the refined pose (0 to 1)." + clock(ms_flow))

        def draw_refine(w):
            q, pp = lerp(tq[-1], h["q"].astype(np.float32), tp[-1], h["p"].astype(np.float32), w)
            with server.atomic():
                for k in range(K):
                    set_pose(h_hyp[k], q[k], pp[k])

        if not play(ms_ver, draw_refine):
            return
        with server.atomic():
            for k in range(K):
                h_hyp[k].visible = c_hyp.value and bool(keep[k])
                h_hyp[k].color = BLUE if score[k] >= 0.5 + a.min_score / 2 else ORANGE
        ms_net = ms_flow + ms_ver
        best = min((err(h["T"][k], T_true) for k in range(K)), key=lambda e: e[0] / 10 + e[1] / 10)
        phase.content = (f"### 2 · Verifier: {int(keep.sum())} of {K} kept\nScore at least {a.min_score:g}: kept "
                         f"({int(keep.sum())}); below: rejected ({int((~keep).sum())}, removed). Kept hypotheses that agree "
                         f"(15 mm, 15°) form {len(handed)} group(s); the best of each is handed to the tracker, highest score "
                         f"first.\n\nClosest hypothesis to the truth, kept or not: {best[0]:.1f} mm / {best[1]:.1f}°." + clock(ms_net))
        hold(2.0)
        if not alive():
            return
        if not handed:
            phase.content = ("### Result: no pose from the learned proposals\nThe verifier rejected every hypothesis, so nothing "
                             f"is handed to the tracker and no pose is reported (closest hypothesis: {best[0]:.1f} mm / "
                             f"{best[1]:.1f}° from the truth). In the tracker the classical search continues in that case."
                             + clock(ms_net))
            return
        if not ok:
            with server.atomic():
                for k in range(K):
                    h_hyp[k].visible = c_hyp.value and k in handed
            phase.content = (f"### 3 · Tracker: none of the {len(handed)} handed hypotheses passed the fit check\nThe shapes left "
                             "are the handed hypotheses. No pose is reported; the classical search continues." + clock(ms_net))
            return
        # phase 3: the handed hypothesis the accepted pose came from (the closest one), then ICP and fit check
        j = min(handed, key=lambda k: err(h["T"][k], T_acc)[0] / 10 + err(h["T"][k], T_acc)[1] / 10)
        T_end = h["T"][j]
        e_end, e_acc = err(T_end, T_true), err(T_acc, T_true)
        q_end, q_acc = geom.quat_from_R(T_end[:3, :3]), geom.quat_from_R(T_acc[:3, :3])
        with server.atomic():
            for k in range(K):
                h_hyp[k].visible = False
            set_pose(h_end, q_end, T_end[:3, 3])
            set_pose(h_end_wire, q_end, T_end[:3, 3])
            h_end.visible = c_flow.value
        phase.content = (f"### 3 · Handed to the tracker (yellow)\nThe handed hypothesis closest to the pose the tracker accepted: "
                         f"score {score[j]:.2f}, **{e_end[0]:.1f} mm / {e_end[1]:.1f}°** from the truth. This is the output of the "
                         "learned part alone." + clock(ms_net))
        hold(1.8)
        if not alive():
            return
        phase.content = ("### 3 · Tracker: ICP on the full cloud, then the fit check (blue)\nThe tracker's own point-to-plane ICP "
                         f"and fit check decide ({ms_icp:.1f} ms for the handed hypotheses). Yellow outline = what was handed over."
                         + clock(ms_net))
        with server.atomic():
            h_end.visible = False
            h_end_wire.visible = c_flow.value
            set_pose(h_flow, q_end, T_end[:3, 3])
            h_flow.visible = c_flow.value

        def draw_icp(w):
            w = 0.5 - 0.5 * np.cos(np.pi * w)
            q, pp = lerp(q_end, q_acc, T_end[:3, 3], T_acc[:3, 3], w)
            set_pose(h_flow, q, pp)

        if not play(max(ms_icp, 150), draw_icp):
            return
        phase.content = (f"### Result\n| | error to the truth |\n|---|---|\n| handed hypothesis (yellow outline) | {e_end[0]:.1f} mm / {e_end[1]:.1f}° |\n"
                         f"| after the tracker's ICP (blue) | **{e_acc[0]:.1f} mm / {e_acc[1]:.1f}°** |\n\n"
                         + ("The accepted pose is wrong: it passed the verifier and the tracker's fit check."
                            if state["flow_verdict"] != "correct" else "The tracker's fit check accepted it.")
                         + f"\n\n**{ms_net:.0f} ms network and verifier + {ms_icp:.1f} ms ICP and fit check, on this machine** "
                           f"(not measured on the Jetson). Classical search on this scene: {state['classical_text']}.")

    def show(replay_only=False):
        with lock:
            state["gen"] += 1
            gen = state["gen"]
            lst = state["list"]
            if not lst:
                verdict.content = "No scene matches this selection."
                phase.content = ""
                return
            s = scenes[lst[min(int(sl.value), len(lst) - 1)]]
            T, depth, labels = load(s["scene"])
            if not replay_only or state["cloud"] is None:
                state["cloud"] = rig.cloud(depth, labels=labels)
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
            h_flow.visible = h_end.visible = h_end_wire.visible = False
            state["flow_ok"] = s["flow"] != "none"
            state["flow_verdict"], state["T_true"], state["T_flow"] = s["flow"], np.array(T), Tf
            state["icp_ms"] = s["flow_icp_ms"]
            state["classical_text"] = (f"{s['classical_cycles'] * evaluate.PERIOD_MS:.0f} ms of pairs, "
                                       + {"correct": "correct", "wrong": "wrong pose", "none": "no pose"}[s["classical"]])
            phase.content = "### Computing the hypotheses ..."
            verdict.content = describe(s, len(lst))
            c = T[:3, :3] @ rig.centroid + T[:3, 3]
            state["center"] = c
            server.initial_camera.position = c + a.view_distance * VIEW
            server.initial_camera.look_at = c
            if c_follow.value and not replay_only:
                for cl in server.get_clients().values():
                    aim(cl)
            h = propose.propose(net, table, rig, depth, dev, k=a.k, centers=a.centers, n_orient=a.orient, solver_steps=STEPS,
                                    solver="rk4", stage1_steps=STEPS, seed=s["scene"], trajectory=True, verifier=ver, min_score=a.min_score)
            fr = cached_field_frames(net, table, rig, ev, s["scene"], P, C, np.array(T), dev, STEPS)
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
