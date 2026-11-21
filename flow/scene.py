"""Synthetic scenes of the rig: the test object (and clutter) seen by the two D435i, as depth images with a sensor
noise model, and the point cloud the flow module builds from a depth pair.

Rig, intrinsics and work volume are the tracker's (tracker/config/tracker.json, tracker/src/core/synth.cpp). Depth
images are 848 x 480 uint16 in units of 1 mm, 0 = no data, like the capture program's."""
import json
import struct
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from . import geom

ROOT = Path(__file__).resolve().parents[1]
W, H = 848, 480
INTR = [(431.372, 424.663, 234.116), (430.382, 424.649, 235.569)]   # f, ppx, ppy of camera A (D435i_1), B (D435i_2)
BASELINE = 0.050       # D435 stereo baseline, m (for the disparity noise model)
CLOUD_STRIDE = 2       # the flow module takes every 2nd pixel in rows and columns


class Rig:
    def __init__(self):
        cfg = json.load(open(ROOT / "tracker/config/tracker.json"))
        self.T_rig_cam = []
        for name in ("D435i_1", "D435i_2"):
            T = np.array(cfg["cameras"][name]["T_rig_from_camera_mm"], float)
            T[:3, 3] /= 1000.0
            self.T_rig_cam.append(T)
        self.wv_min = np.array(cfg["work_volume"]["min"])
        self.wv_max = np.array(cfg["work_volume"]["max"])
        m = trimesh.load(ROOT / "models/test_object.stl")
        self.V = np.asarray(m.vertices, np.float64)
        self.F = np.asarray(m.faces, np.uint32)
        with open(ROOT / "models/test_object.model", "rb") as f:
            assert f.read(8) == b"TRKMODEL"
            _, n = struct.unpack("<II", f.read(8))
            f.read(48)
            pn = np.frombuffer(f.read(48 * n), "<f8").reshape(n, 6)
        self.model_pts = pn[:, :3].copy()
        self.centroid = self.model_pts.mean(0)
        self.radius = float(np.linalg.norm(self.model_pts - self.centroid, axis=1).max())
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        self.dirs_cam = [np.stack([(u - c) / f, (v - d) / f, np.ones_like(u, float)], -1).reshape(-1, 3)
                         for f, c, d in INTR]

    def project(self, c, p):
        """Rig point -> (u, v, z) in camera c (nominal pose)."""
        T = self.T_rig_cam[c]
        q = (p - T[:3, 3]) @ T[:3, :3]
        f, cx, cy = INTR[c]
        return f * q[0] / q[2] + cx, f * q[1] / q[2] + cy, q[2]

    def sample_pose(self, rng, margin_px=70, zmin=0.27, zmax=0.72):
        """Random orientation; centroid inside the work volume and in view of both cameras at zmin..zmax."""
        while True:
            p = rng.uniform(self.wv_min + 0.05, self.wv_max - 0.05)
            ok = True
            for c in range(2):
                u, v, z = self.project(c, p)
                ok &= zmin < z < zmax and margin_px < u < W - margin_px and margin_px < v < H - margin_px
            if ok:
                R = geom.random_R(rng)
                return geom.make_T(R, p - R @ self.centroid)

    def cloud(self, depth, stride=CLOUD_STRIDE, labels=None):
        """Depth pair -> points in the rig frame inside the work volume (nominal camera poses), camera index per point
        (and the label of each point's pixel)."""
        P, C, L = [], [], []
        for c in range(2):
            d = depth[c][::stride, ::stride]
            m = d > 0
            z = d[m].astype(np.float64) * 0.001
            dirs = self.dirs_cam[c].reshape(H, W, 3)[::stride, ::stride][m]
            T = self.T_rig_cam[c]
            p = (dirs * z[:, None]) @ T[:3, :3].T + T[:3, 3]
            keep = np.all((p >= self.wv_min) & (p <= self.wv_max), axis=1)
            P.append(p[keep])
            C.append(np.full(int(keep.sum()), c, np.uint8))
            if labels is not None:
                L.append(labels[c][::stride, ::stride][m][keep])
        P, C = np.concatenate(P).astype(np.float32), np.concatenate(C)
        return (P, C, np.concatenate(L)) if labels is not None else (P, C)


# ---------------- clutter ----------------
def _place(mesh, R, t):
    mesh.apply_transform(geom.make_T(R, t))
    return mesh


def _axis_R(axis):
    """Rotation that takes z to the given axis."""
    a = axis / np.linalg.norm(axis)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, a)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if a[2] > 0 else np.diag([1.0, -1.0, -1.0])
    return geom.rotvec_R(v / s * np.arctan2(s, a[2]))


def clutter_hand(rig, rng, T):
    """A hand holding the object from above (the cameras look up from below): palm, fingers around the sides, forearm."""
    c = T[:3, :3] @ rig.centroid + T[:3, 3]
    out = []
    tilt = geom.rotvec_R(rng.normal(size=3) * 0.25)
    palm_c = c + np.array([rng.normal() * 0.015, rng.normal() * 0.015, rng.uniform(0.035, 0.07)])
    out.append(_place(trimesh.creation.box(extents=[rng.uniform(0.07, 0.10), rng.uniform(0.08, 0.11), rng.uniform(0.02, 0.035)]),
                      tilt @ geom.rotvec_R([0, 0, rng.uniform(0, np.pi)]), palm_c))
    a0 = rng.uniform(0, 2 * np.pi)
    for k in range(rng.integers(2, 6)):
        ang = a0 + (k * rng.uniform(0.35, 0.6) if k < 4 else np.pi + rng.normal() * 0.3)   # 4 fingers one side, thumb opposite
        r = rng.uniform(0.045, 0.075)
        base = c + np.array([r * np.cos(ang), r * np.sin(ang), rng.uniform(-0.02, 0.03)])
        axis = np.array([-0.3 * np.cos(ang), -0.3 * np.sin(ang), 1.0]) + rng.normal(size=3) * 0.15
        out.append(_place(trimesh.creation.capsule(height=rng.uniform(0.05, 0.085), radius=rng.uniform(0.007, 0.011), count=[8, 8]),
                          _axis_R(axis), base))
    arm_axis = np.array([rng.normal() * 0.6, rng.normal() * 0.6, 1.0])
    arm_axis /= np.linalg.norm(arm_axis)
    L = rng.uniform(0.2, 0.35)
    out.append(_place(trimesh.creation.cylinder(radius=rng.uniform(0.025, 0.04), height=L, sections=16),
                      _axis_R(arm_axis), palm_c + arm_axis * (L / 2 + 0.01)))
    return out


def clutter_rod(rig, rng, T):
    """A rod that carries the object (through the object's centre region, random direction)."""
    c = T[:3, :3] @ rig.centroid + T[:3, 3] + rng.normal(size=3) * 0.01
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    L = rng.uniform(0.3, 0.6)
    return [_place(trimesh.creation.cylinder(radius=rng.uniform(0.004, 0.009), height=L, sections=12), _axis_R(axis),
                   c + axis * rng.uniform(-0.3, 0.3) * L)]


def clutter_structures(rig, rng, T, n=None):
    """Static structure elsewhere in the work volume: boxes and cylinders, some of them about the size of the object."""
    c = T[:3, :3] @ rig.centroid + T[:3, 3]
    out = []
    for _ in range(rng.integers(1, 4) if n is None else n):
        for _try in range(20):
            p = rng.uniform(rig.wv_min, rig.wv_max)
            if 0.14 < np.linalg.norm(p - c) < 0.45:
                break
        if rng.random() < 0.6:
            m = trimesh.creation.box(extents=rng.uniform(0.04, 0.14, size=3) * (1 if rng.random() < 0.7 else 2))
        else:
            m = trimesh.creation.cylinder(radius=rng.uniform(0.02, 0.06), height=rng.uniform(0.05, 0.25), sections=16)
        out.append(_place(m, geom.random_R(rng), p))
    return out


def clutter_plane(rig, rng, T):
    """A large plate behind the object as the cameras see it (above it), slightly tilted."""
    c = T[:3, :3] @ rig.centroid + T[:3, 3]
    z = min(rig.wv_max[2] + 0.05, c[2] + rng.uniform(0.10, 0.35))
    return [_place(trimesh.creation.box(extents=[2.0, 1.2, 0.01]), geom.rotvec_R(rng.normal(size=3) * [0.12, 0.12, 0]),
                   np.array([0, 0, z]))]


CATEGORIES = {   # name -> clutter kinds
    "clean": (),
    "hand": ("hand",),
    "rod": ("rod",),
    "structures": ("structures", "plane"),
    "everything": ("hand", "structures", "plane"),
}
_CLUTTER = {"hand": clutter_hand, "rod": clutter_rod, "structures": clutter_structures, "plane": clutter_plane}


def random_clutter_kinds(rng):
    kinds = []
    if rng.random() < 0.45:
        kinds.append("hand")
    elif rng.random() < 0.3:
        kinds.append("rod")
    if rng.random() < 0.5:
        kinds.append("structures")
    if rng.random() < 0.3:
        kinds.append("plane")
    return tuple(kinds)


# ---------------- rendering and sensor model ----------------
def _shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    h, w = a.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def sensor_model(z, cosang, f, rng, sigma_d, low_frac=0.5):
    """Ideal depth (m, inf = nothing) -> uint16 mm with D435-like defects: disparity noise (sigma_z ~ z^2; part of it
    correlated over ~16 px), flying pixels and holes along depth edges, holes at grazing angles, dropout patches,
    random missing pixels, 1 mm quantization."""
    valid = np.isfinite(z) & (z > 0.17) & (z < 2.5)
    zz = np.where(valid, z, 10.0)
    # depth edges: neighbours that differ by more than 15 mm
    near = zz.copy()
    far = zz.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        s = _shift(zz, dy, dx, 10.0)
        near = np.minimum(near, s)
        far = np.maximum(far, s)
    edge = valid & ((far - near) > 0.015)
    if rng.random() < 0.5:   # 2 px wide edge band
        e2 = edge.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            e2 |= _shift(edge, dy, dx, False)
        edge = e2 & valid
    # disparity noise
    fb = f * BASELINE
    d = fb / zz
    coarse = rng.normal(size=(H // 16 + 2, W // 16 + 2))
    low = np.kron(coarse, np.ones((16, 16)))[rng.integers(0, 16):, rng.integers(0, 16):][:H, :W]
    d = d + sigma_d * (np.sqrt(1 - low_frac) * rng.normal(size=z.shape) + np.sqrt(low_frac) * low)
    zn = fb / np.maximum(d, 1e-3)
    # edge pixels: flying (between the two surfaces, or behind when there is no second surface), missing, or kept
    r = rng.random(z.shape)
    p_fly, p_hole = rng.uniform(0.1, 0.4), rng.uniform(0.2, 0.5)
    fly = edge & (r < p_fly)
    far_c = np.where(far > 5.0, near + 0.2, far)
    zn = np.where(fly, near + rng.random(z.shape) * (far_c - near), zn)
    valid &= ~(edge & (r > 1 - p_hole))
    # grazing angles
    valid &= ~((cosang < rng.uniform(0.12, 0.3)) & (rng.random(z.shape) < 0.8))
    # dropout patches and missing pixels
    for _ in range(rng.integers(0, 14)):
        cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        ry, rx = rng.uniform(3, 22), rng.uniform(3, 22)
        y0, y1, x0, x1 = max(0, int(cy - ry)), min(H, int(cy + ry) + 1), max(0, int(cx - rx)), min(W, int(cx + rx) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        valid[y0:y1, x0:x1] &= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 > 1
    valid &= rng.random(z.shape) > rng.uniform(0.0, 0.06)
    return np.where(valid, np.clip(np.rint(zn * 1000.0), 0, 65535), 0).astype(np.uint16)


def render(rig, rng, T_rig_obj, clutter=(), sigma_d=None, extrinsic_error=True):
    """Depth pair of a scene. Returns depth [2] (uint16 mm) and labels [2] (1 = pixel shows the object, before noise).
    Each camera is rendered from a slightly wrong pose (the tracker runs on uncalibrated CAD transforms)."""
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    Vw = (rig.V @ T_rig_obj[:3, :3].T + T_rig_obj[:3, 3]).astype(np.float32)
    obj_id = scene.add_triangles(o3d.core.Tensor(Vw), o3d.core.Tensor(rig.F))
    for m in clutter:
        scene.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, np.float32)), o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
    if sigma_d is None:
        sigma_d = rng.uniform(0.05, 0.22)
    depth, labels = [], []
    for c in range(2):
        T = rig.T_rig_cam[c]
        if extrinsic_error:
            T = geom.make_T(geom.rotvec_R(rng.normal(size=3) * np.radians(0.15)) @ T[:3, :3], T[:3, 3] + rng.normal(size=3) * 0.001)
        dirs = rig.dirs_cam[c] @ T[:3, :3].T
        rays = np.hstack([np.broadcast_to(T[:3, 3], dirs.shape), dirs]).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        z = ans["t_hit"].numpy().reshape(H, W).astype(np.float64)
        n = ans["primitive_normals"].numpy()
        cosang = np.abs(np.einsum("ij,ij->i", n, dirs) / np.linalg.norm(dirs, axis=1)).reshape(H, W)
        depth.append(sensor_model(z, cosang, INTR[c][0], rng, sigma_d))
        labels.append((ans["geometry_ids"].numpy().reshape(H, W) == obj_id).astype(np.uint8))
    return depth, labels


def make_scene(rig, rng, kinds=None):
    """One random scene: pose, clutter of the given kinds (None = random mix), depth pair, labels."""
    T = rig.sample_pose(rng)
    if kinds is None:
        kinds = random_clutter_kinds(rng)
    clutter = []
    for k in kinds:
        clutter += _CLUTTER[k](rig, rng, T)
    depth, labels = render(rig, rng, T, clutter)
    return T, depth, labels, clutter
