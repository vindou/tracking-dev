"""Shared style of the animations: one colour per concept, text helpers, a small 3-D view (projection done here in
numpy, drawn with 2-D manim objects), the object mesh, pose glyphs, quaternion helpers and the data of the scene.

Everything 3-D on screen is in metres in the rig frame of the tracker (or the body frame of a pose where a scene says
so) and comes from assets/scene.npz, which flow/anim/precompute.py writes from the real pipeline."""
from pathlib import Path

import numpy as np
from manim import *

HERE = Path(__file__).resolve().parent
config.media_dir = str(HERE / "media")
config.background_color = "#ffffff"
config.frame_rate = 30

# ---- one colour per concept, in every scene (chosen to stay apart on white, also for red-green colour blindness)
C_TRUE = "#009E73"       # the true pose (green)
C_EST = "#E07B00"        # a pose estimate / hypothesis (orange)
C_OBJ = "#0072B2"        # measured points on the object (blue)
C_CLUT = "#9AA0A6"       # measured points on clutter (grey)
C_ACC = "#4B2E9E"        # accepted by the tracker (indigo, heavy outline, with a tick)
C_REJ = "#D62728"        # rejected / wrong (red)
C_RES = "#B07A00"        # closest-surface vectors, residuals, velocity targets (ochre)
C_SEED = "#D81B9A"       # seeds and position candidates (magenta)
C_NET = "#00697A"        # network blocks (dark teal)
C_COND = "#9C5B2E"       # time and seed conditioning (brown)
C_TEXT = "#111111"
C_DIM = "#5F6368"
C_PANEL = "#FAFAF7"
C_RULE = "#111111"
SYNTH = "synthetic scenes only: nothing here was measured on real data"


def _latex_font():
    """New Computer Modern (the text face of LaTeX documents, with Greek and symbols) from the TeX installation,
    registered with Pango; a system serif if it is not there."""
    import subprocess
    import manimpango
    try:
        for f in ("NewCM10-Regular.otf", "NewCM10-Bold.otf", "NewCM10-Italic.otf"):
            path = subprocess.run(["kpsewhich", f], capture_output=True, text=True).stdout.strip()
            if not path or not manimpango.register_font(path):
                return "Times New Roman"
        return "NewComputerModern10"
    except Exception:
        return "Times New Roman"


FONT = _latex_font()
TEXT_SCALE = 1.08        # the serif face runs smaller than the sizes of the layouts were chosen for


def T(s, size=24, color=C_TEXT, weight=NORMAL, **kw):
    return Text(s, font=FONT, font_size=size * TEXT_SCALE, color=color, weight=weight, **kw)


def M(tex, size=24, color=C_TEXT):
    """A formula set by LaTeX (Computer Modern), sized like text of the same `size`."""
    return MathTex(tex, color=color, font_size=size * 1.25)


def title(s, sub=None, n=None):
    """A section heading as in a LaTeX article: number, bold title, a rule, and a one-line statement under it."""
    t = T(s if n is None else f"{n}   {s}", 30, weight=BOLD).to_corner(UL, buff=0.35)
    rule = Line(t.get_corner(DL) + DOWN * 0.1, t.get_corner(DL) + DOWN * 0.1 + RIGHT * max(t.width, 3.2), color=C_RULE, stroke_width=1.5)
    g = VGroup(t, rule)
    if sub is not None:
        g.add(T(sub, 20, C_DIM, slant=ITALIC).next_to(rule, DOWN, buff=0.1, aligned_edge=LEFT))
    return VGroup(BackgroundRectangle(g, color=config.background_color, fill_opacity=0.92, buff=0.12), g).set_z_index(30)


def legend(items, size=18):
    """items: (colour, label[, kind]); kind "dot" (default), "line" or "box". A framed column, placed by the scene."""
    rows = VGroup()
    for it in items:
        col, lab, kind = it if len(it) == 3 else (*it, "dot")
        if kind == "dot":
            sym = Dot(radius=0.07, color=col)
        elif kind == "line":
            sym = Line(LEFT * 0.16, RIGHT * 0.16, color=col, stroke_width=4)
        else:
            sym = Square(0.2, color=col, stroke_width=3)
        box = VGroup(Rectangle(width=0.36, height=0.2, stroke_width=0, fill_opacity=0), sym)
        rows.add(VGroup(box, T(lab, size)).arrange(RIGHT, buff=0.12))
    rows.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
    bg = SurroundingRectangle(rows, buff=0.14, color=C_RULE, stroke_width=1, fill_color="#ffffff", fill_opacity=0.95, corner_radius=0)
    return VGroup(bg, rows).set_z_index(30)


class Caption:
    """One line at the bottom of the frame under a thin rule, replaced as the scene goes on."""

    def __init__(self, scene, y=-3.55):
        self.scene, self.y, self.mob = scene, y, None
        self.rule = Line([-6.8, y + 0.33, 0], [6.8, y + 0.33, 0], color=C_RULE, stroke_width=1).set_z_index(49)
        self.band = Rectangle(width=14.3, height=0.95, stroke_width=0, fill_color=config.background_color, fill_opacity=1).move_to([0, y - 0.12, 0]).set_z_index(48)
        scene.add(self.band, self.rule)

    def show(self, s, wait=0.0, size=24, color=C_TEXT, run_time=0.5):
        new = T(s, size, color, line_spacing=0.8)
        if new.width > 13.4:
            new.scale_to_fit_width(13.4)
        new.move_to([0, self.y, 0]).set_z_index(50)
        if self.mob is None:
            self.scene.play(FadeIn(new), run_time=run_time)
        else:
            self.scene.play(FadeOut(self.mob), FadeIn(new), run_time=run_time)
        self.mob = new
        if wait:
            self.scene.wait(wait)


def note(s, size=18, color=C_DIM):
    """The standing caveat line (top right), in italics."""
    t = T(s, size, color, slant=ITALIC).to_corner(UR, buff=0.3)
    return VGroup(BackgroundRectangle(t, color=config.background_color, fill_opacity=0.92, buff=0.08), t).set_z_index(30)


# ---------------- quaternions (w, x, y, z), as flow/geom.py ----------------
def qmul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qconj(q):
    return np.asarray(q, float) * np.array([1, -1, -1, -1.0])


def qexp(w):
    w = np.asarray(w, float)
    a = np.linalg.norm(w, axis=-1, keepdims=True)
    return np.concatenate([np.cos(a / 2), np.sinc(a / 2 / np.pi) / 2 * w], -1)


def qlog(q):
    q = np.asarray(q, float)
    q = np.where(q[..., :1] < 0, -q, q)
    n = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    return np.where(n < 1e-9, 2 * q[..., 1:], 2 * np.arctan2(n, q[..., :1]) * q[..., 1:] / np.maximum(n, 1e-12))


def qmat(q):
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def qslerp(q0, q1, s):
    """The geodesic q0 exp(s log(q0^-1 q1))."""
    return qmul(q0, qexp(s * qlog(qmul(qconj(q0), q1))))


def qangle_deg(q0, q1):
    return np.degrees(2 * np.arccos(np.clip(np.abs((np.asarray(q0) * np.asarray(q1)).sum(-1)), 0, 1)))


def path_pose(tq, tp, s):
    """Pose at s in [0, 1] along stored solver states tq [K+1, 4], tp [K+1, 3]: geodesic between consecutive states."""
    K = len(tq) - 1
    u = min(max(s, 0.0), 1.0) * K
    k = min(int(u), K - 1)
    return qslerp(tq[k], tq[k + 1], u - k), (1 - (u - k)) * tp[k] + (u - k) * tp[k + 1]


# ---------------- data ----------------
_data = None


def data():
    global _data
    if _data is None:
        z = np.load(HERE / "assets/scene.npz")
        _data = {k: z[k] for k in z.files}
    return _data


# ---------------- 3-D view ----------------
class View:
    """Orthographic view of 3-D points (m): looks at `center` from azimuth `az` / elevation `el` (degrees; az = 0 looks
    along +z of the rig with -y up), `scale` screen units per metre, the centre drawn at `at` on the screen."""

    def __init__(self, center, az=30.0, el=20.0, scale=8.0, at=ORIGIN, up=(0, -1, 0)):
        self.center, self.az, self.el, self.scale = np.asarray(center, float), az, el, scale
        self.at, self.up0 = np.asarray(at, float), np.asarray(up, float)
        self.clip = (-7.2, 7.2, -3.2, 2.8)        # screen box outside which cloud points are not drawn

    def basis(self):
        up = self.up0 / np.linalg.norm(self.up0)
        f0 = np.array([0, 0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0, 0])
        f0 = f0 - up * (f0 @ up)
        f0 /= np.linalg.norm(f0)
        r0 = np.cross(f0, up)                     # right when looking along f0 with `up` up
        a, e = np.radians(self.az), np.radians(self.el)
        fwd = (np.cos(a) * f0 + np.sin(a) * r0) * np.cos(e) - np.sin(e) * up
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        return right, np.cross(right, fwd), fwd

    def __call__(self, X):
        r, u, _ = self.basis()
        d = (np.asarray(X, float) - self.center) * self.scale
        return np.stack([d @ r, d @ u, np.zeros(d.shape[:-1])], -1) + self.at

    def depth(self, X):
        return (np.asarray(X, float) - self.center) @ self.basis()[2]

    def copy(self, **kw):
        v = View(self.center, self.az, self.el, self.scale, self.at, self.up0)
        v.clip = self.clip
        for k, val in kw.items():
            setattr(v, k, np.asarray(val, float) if k in ("center", "at") else val)
        return v


def cloud(view, P, colors, size=3.0, opacity=1.0):
    """Points as a pixel cloud. colors: one colour or a list / array of colours per point."""
    m = PMobject(stroke_width=size * config.pixel_height / 480)      # size is in pixels of a 480-line frame
    P = np.asarray(P, float).reshape(-1, 3)
    if len(P) == 0:
        return m
    S0 = view(P)
    inside = (S0[:, 0] > view.clip[0]) & (S0[:, 0] < view.clip[1]) & (S0[:, 1] > view.clip[2]) & (S0[:, 1] < view.clip[3])
    if not inside.all() and not (isinstance(colors, str) or isinstance(colors, ManimColor)):
        colors = np.asarray(colors)[inside]
    P = P[inside]
    if len(P) == 0:
        return m
    order = np.argsort(-view.depth(P))
    if isinstance(colors, str) or isinstance(colors, ManimColor):
        rgba = np.tile(np.append(color_to_rgb(colors), opacity), (len(P), 1))
    else:
        rgba = np.concatenate([np.asarray(colors, float)[:, :3], np.full((len(P), 1), opacity)], 1)
    m.add_points(view(P)[order], rgbas=rgba[order])
    return m


def label_colors(L):
    """Object / clutter colours of labels L (1 = on the object)."""
    return np.where(np.asarray(L)[:, None] > 0.5, color_to_rgb(C_OBJ), color_to_rgb(C_CLUT))


# ---------------- the object mesh (models/test_object.stl, 74 triangles) ----------------
class Mesh:
    def __init__(self):
        d = data()
        self.V, self.F, self.c = d["V"].astype(float), d["F"].astype(int), d["centroid"].astype(float)
        tri = self.V[self.F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.n = n / np.linalg.norm(n, axis=1, keepdims=True)
        edges = {}
        for fi, f in enumerate(self.F):
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                edges.setdefault((min(a, b), max(a, b)), []).append(fi)
        # feature edges: where the surface bends (coplanar triangles of a flat face are not separated by a line)
        self.edges = [(a, b, fs) for (a, b), fs in edges.items() if len(fs) == 2 and self.n[fs[0]] @ self.n[fs[1]] < 0.95]

    def world(self, q, p):
        """Vertices and face normals of the mesh at pose (q, p = position of the model centroid)."""
        R = qmat(q)
        return (self.V - self.c) @ R.T + np.asarray(p, float), self.n @ R.T

    def mob(self, view, q, p, color=C_TRUE, fill=0.85, edge=None, edge_width=2.0, edge_opacity=1.0, hidden=False):
        """The object at pose (q, p). fill > 0: shaded faces, far to near, back faces removed; edges of visible faces
        on top. fill = 0: a wire outline (visible edges only unless hidden = True)."""
        W, n = self.world(q, p)
        _, up, fwd = view.basis()
        front = n @ fwd < -1e-6
        S = view(W)
        g = VGroup()
        if fill > 0:
            light = (-fwd + 0.6 * up + 0.3 * view.basis()[0])
            light /= np.linalg.norm(light)
            order = np.argsort(-view.depth(W[self.F].mean(1)))
            base = color_to_rgb(color)
            for fi in order:
                if not front[fi]:
                    continue
                sh = 0.35 + 0.65 * max(0.0, float(n[fi] @ light))
                col = rgb_to_color(base * (0.72 + 0.28 * sh) * (1 - 0.3 * sh) + 0.3 * sh)        # lit faces are tinted toward white
                g.add(Polygon(*S[self.F[fi]], stroke_width=0.6, stroke_color=col, stroke_opacity=fill, fill_color=col, fill_opacity=fill))
        ec = edge if edge is not None else (rgb_to_color(color_to_rgb(color) * 0.4) if fill > 0.5 else color)
        for a, b, fs in self.edges:
            if hidden or front[fs[0]] or front[fs[1]]:
                g.add(Line(S[a], S[b], stroke_width=edge_width, color=ec, stroke_opacity=edge_opacity))
        return g


_mesh = None


def mesh():
    global _mesh
    if _mesh is None:
        _mesh = Mesh()
    return _mesh


def triad(view, q, p, color=C_EST, length=0.05, width=3.0, opacity=1.0):
    """A pose as a frame: three axes from the model centroid, x long, y medium, z short (one colour: the concept)."""
    R = qmat(q)
    p = np.asarray(p, float)
    o = view(p)
    g = VGroup()
    for i, s in enumerate((1.0, 0.7, 0.45)):
        g.add(Line(o, view(p + R[:, i] * length * s), stroke_width=width, color=color, stroke_opacity=opacity))
    g.add(Dot(o, radius=0.012 * width, color=color, fill_opacity=opacity))
    return g


def axes_gizmo(view, origin, length=0.06, labels=("x", "y", "z"), color=C_DIM, size=16, R=None):
    """Labelled coordinate axes at `origin` (the rig frame unless a rotation R is given)."""
    R = np.eye(3) if R is None else R
    g = VGroup()
    o = view(origin)
    for i, lab in enumerate(labels):
        e = view(np.asarray(origin, float) + R[:, i] * length)
        g.add(Line(o, e, stroke_width=2, color=color))
        d = e - o
        lm = M(lab[1:-1], size, color) if lab.startswith("$") else T(lab, size, color, slant=ITALIC)
        g.add(lm.move_to(e + 0.2 * d / max(np.linalg.norm(d), 1e-6)))
    return g


def camera_glyph(view, T_rig_cam, size=0.035, color=C_DIM, depth=0.07):
    """A depth camera as a small frustum at its pose in the rig frame."""
    R, t = T_rig_cam[:3, :3], T_rig_cam[:3, 3]
    c = [t + R @ np.array([sx * size * 1.6, sy * size, depth]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    g = VGroup()
    S = view(np.array(c))
    o = view(t)
    for i in range(4):
        g.add(Line(o, S[i], stroke_width=2, color=color), Line(S[i], S[(i + 1) % 4], stroke_width=2, color=color))
    return g


def tick(color=C_ACC, size=0.3):
    return VMobject(color=color, stroke_width=6).set_points_as_corners([LEFT * 0.5 * size + UP * 0.05 * size, DOWN * 0.4 * size + LEFT * 0.1 * size, UP * 0.5 * size + RIGHT * 0.55 * size])


def cross_mark(color=C_REJ, size=0.26):
    return VGroup(Line(UL * size / 2, DR * size / 2), Line(UR * size / 2, DL * size / 2)).set_stroke(color, 6)


def block(label, w=1.8, h=0.7, color=C_NET, size=18, sub=None):
    """A box of a block diagram."""
    r = RoundedRectangle(width=w, height=h, corner_radius=0.05, color=color, stroke_width=2, fill_color=color, fill_opacity=0.09)
    t = T(label, size)
    if sub:
        t = VGroup(t, T(sub, size - 5, C_DIM)).arrange(DOWN, buff=0.05)
    if t.width > w - 0.15:
        t.scale_to_fit_width(w - 0.15)
    return VGroup(r, t.move_to(r))


def arrow(a, b, color=C_TEXT, width=2.5, tip=0.16, **kw):
    return Arrow(a, b, buff=0.06, stroke_width=width, color=color, tip_length=tip, max_tip_length_to_length_ratio=0.5, **kw)


def spike(view, q, p, color=C_EST, length=0.085, width=1.6, opacity=0.9):
    """A pose hypothesis drawn light enough for dozens at once: its x axis with a large dot at the tip and its y axis
    (shorter) with a small dot. Hypotheses that agree put their dots in the same place."""
    R = qmat(q)
    p = np.asarray(p, float)
    o, a, b = view(p), view(p + R[:, 0] * length), view(p + R[:, 1] * length * 0.6)
    return VGroup(Line(o, a, stroke_width=width, color=color, stroke_opacity=0.55 * opacity),
                  Line(o, b, stroke_width=width * 0.7, color=color, stroke_opacity=0.4 * opacity),
                  Dot(a, radius=0.055, color=color, fill_opacity=opacity), Dot(b, radius=0.03, color=color, fill_opacity=opacity))


def tracker_pose(row):
    """(q, centroid position) of a pose accepted by the tracker: row of trk_hint_eval as stored in scene.npz
    (position of the object origin and quaternion of rig-from-object)."""
    q, t = row[5:9], row[2:5]
    return q, t + qmat(q) @ data()["centroid"]


def ring(view, p, radius_m, color=C_SEED, width=2.0, opacity=1.0):
    """A sphere of the given radius around p, drawn as its outline circle."""
    return Circle(radius=radius_m * view.scale, color=color, stroke_width=width, stroke_opacity=opacity).move_to(view(p))


def live_cloud(fn):
    """A point cloud recomputed every frame by fn() (always_redraw cannot be used: the number of points may change)."""
    m = fn()

    def upd(mob):
        n = fn()
        mob.points, mob.rgbas = n.points, n.rgbas
    m.add_updater(upd)
    return m


def tag(s, size=15, color=C_TEXT, **kw):
    """A label with a dark backing, for text placed over a cloud."""
    t = T(s, size, color, **kw)
    return VGroup(BackgroundRectangle(t, color=config.background_color, fill_opacity=0.85, buff=0.06), t).set_z_index(25)
