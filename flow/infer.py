"""Pose hypotheses for a depth pair: a position stage from seeds on the cloud, the rotation flow from a set of
orientations at the positions found, agreement between the results.

    python -m flow.infer --model flow/runs/a/model.pt --eval flow/data/eval [--k 32] [--centers 4] [--orient 32] [--hints 16]
writes <eval>/hints.bin for tracker/build-mac/trk_hint_eval and <eval>/flow.npz (hypotheses, timing)."""
import argparse
import struct
import time
from pathlib import Path

import numpy as np
import torch

from . import geom, sampler, scene
from .make_data import STORE_RADIUS, load_eval
from .model import FlowNet

MAX_PTS = 6144


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    net = FlowNet(ck.get("width", 1)).to(dev)
    net.load_state_dict(ck["model"])
    return net.eval()


def seeds_fps(P, k, rng, voxel=0.03, min_pts=4):
    """k start positions spread over the cloud: farthest-point selection on the centres of the occupied 3 cm cells."""
    key = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    cen = np.stack([np.bincount(inv, P[:, i]) for i in range(3)], 1) / cnt[:, None]
    cen = cen[cnt >= min_pts]
    if len(cen) == 0:
        return np.zeros((0, 3))
    sel = [int(rng.integers(len(cen)))]
    d = np.linalg.norm(cen - cen[sel[0]], axis=1)
    while len(sel) < min(k, len(cen)):
        sel.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(cen - cen[sel[-1]], axis=1))
    return cen[sel]


def group(q, p, res, tol_m=0.015, tol_deg=15.0):
    """Groups hypotheses that agree (within tol of the group's best member). Returns the groups, largest first (ties:
    smallest predicted remaining motion), each a list of indices with the best member first."""
    order = np.argsort(res)
    R = geom.qmat(torch.from_numpy(q)).numpy()
    groups = []
    for i in order:
        for g in groups:
            j = g[0]
            if np.linalg.norm(p[i] - p[j]) < tol_m and geom.rot_angle_deg(R[i], R[j]) < tol_deg:
                g.append(int(i))
                break
        else:
            groups.append([int(i)])
    groups.sort(key=lambda g: (-len(g), res[g[0]]))
    return groups


_orient = {}


def orientations(n):
    """n rotations spread evenly over SO(3) (farthest-point selection on random rotations), as quaternions."""
    if n not in _orient:
        rng = np.random.default_rng(5)
        q = rng.normal(size=(4000, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        sel = [0]
        d = 1 - np.abs(q @ q[0])
        while len(sel) < n:
            sel.append(int(np.argmax(d)))
            d = np.minimum(d, 1 - np.abs(q @ q[sel[-1]]))
        _orient[n] = (q[sel], float(np.degrees(2 * np.arccos(1 - d.max()))))     # set, covering radius (deg)
    return _orient[n]


def _crops(P, C, centers, rng):
    K = len(centers)
    pts = np.zeros((K, MAX_PTS, 3), np.float32)
    cam = np.zeros((K, MAX_PTS), np.int64)
    n = np.zeros(K, np.int64)
    for i, s in enumerate(centers):
        d = P - s.astype(np.float32)
        idx = np.flatnonzero(np.einsum("ij,ij->i", d, d) < STORE_RADIUS ** 2)
        if len(idx) > MAX_PTS:
            idx = rng.choice(idx, MAX_PTS, replace=False)
        pts[i, :len(idx)] = d[idx]
        cam[i, :len(idx)] = C[idx]
        n[i] = len(idx)
    return pts, cam, n


@torch.no_grad()
def propose(net, rig, depth, dev, k=32, centers=4, n_orient=32, steps=8, refine=2, seed=0, trajectory=False):
    """Depth pair -> pose hypotheses, in two stages.
      1. Position: k seeds spread over the cloud, each with a random rotation; the translation part of the flow moves
         each seed toward the object centroid it sees (5 evaluations, rotation held). The end points are grouped
         (30 mm); the `centers` groups with the most object points (segmentation head) are kept.
      2. Rotation: at each kept position the flow starts from n_orient orientations spread over SO(3) and runs
         `steps` Euler steps and `refine` end-point corrections on SO(3) x R^3.
    Returns a dict: cloud P, C; seeds, seed_end (stage 1); centers; q, p (final poses of stage 2: quaternion, centroid
    position in the rig frame); res (predicted remaining motion: rad + m / 0.1); objfrac; groups of agreeing
    hypotheses; T (4x4 T_rig_from_object); times; with trajectory = True also traj_q, traj_p [steps + refine + 1, K, .]."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    P, C = rig.cloud(depth)
    S = seeds_fps(P, k, rng)
    out = {"P": P, "C": C, "seeds": S, "seed_end": S, "centers": np.zeros((0, 3)), "q": np.zeros((0, 4)), "p": np.zeros((0, 3)),
           "res": np.zeros(0), "objfrac": np.zeros(0), "groups": [], "T": np.zeros((0, 4, 4)), "t_prep": 0.0, "t_flow": 0.0}
    if len(S) == 0:
        return out
    to = lambda x: torch.from_numpy(x).to(dev)
    # stage 1
    pts, cam, n = _crops(P, C, S, rng)
    t1 = time.perf_counter()
    valid = torch.arange(MAX_PTS, device=dev)[None] < to(n)[:, None]
    _, p1, _, _, of1 = sampler.integrate(net, to(pts), valid, to(cam), geom.qrand(len(S), dev, g), torch.zeros(len(S), 3, device=dev),
                                         steps=4, refine=1, generator=g, position_only=True)
    e = p1.cpu().numpy().astype(np.float64) + S
    of1 = of1.cpu().numpy()
    t2 = time.perf_counter()
    clusters = []
    for i in np.argsort(-of1):
        for cl in clusters:
            if np.linalg.norm(e[i] - e[cl[0]]) < 0.03:
                cl.append(i)
                break
        else:
            clusters.append([i])
    clusters.sort(key=lambda cl: -sum(of1[j] for j in cl))
    cen = np.stack([np.average(e[cl], axis=0, weights=of1[cl] + 1e-3) for cl in clusters[:centers]])
    # stage 2
    Q0, _ = orientations(n_orient)
    pts, cam, n = _crops(P, C, cen, rng)
    rep = np.repeat(np.arange(len(cen)), n_orient)
    K = len(rep)
    t3 = time.perf_counter()
    valid = torch.arange(MAX_PTS, device=dev)[None] < to(n[rep])[:, None]
    q0 = to(np.tile(Q0, (len(cen), 1)).astype(np.float32))
    r = sampler.integrate(net, to(pts[rep]), valid, to(cam[rep]), q0, torch.zeros(K, 3, device=dev), steps=steps, refine=refine,
                          trajectory=trajectory, generator=g)
    q, p, rr, rp, of = [x.cpu().numpy().astype(np.float64) for x in r[:5]]
    t4 = time.perf_counter()
    p = p + cen[rep]
    res = rr + rp / 0.1
    R = geom.qmat(torch.from_numpy(q)).numpy()
    T = np.tile(np.eye(4), (K, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = p - R @ rig.centroid
    out.update(seed_end=e, centers=cen, q=q, p=p, res=res, objfrac=of, groups=group(q, p, res), T=T,
               t_prep=(t1 - t0) + (t3 - t2), t_flow=(t2 - t1) + (t4 - t3))
    if trajectory:
        out["traj_q"] = np.stack([a.cpu().numpy() for a, _ in r[5]])
        out["traj_p"] = np.stack([b.cpu().numpy() for _, b in r[5]]) + cen[rep][None]
    return out


def hint_order(h, n):
    """The hypotheses to hand to the tracker: the best member of each group, largest group first."""
    return [g[0] for g in h["groups"]][:n]


# One evaluation of the network on the Jetson Orin Nano (TensorRT 10.3, fp16, 512 points per hypothesis), measured with
# trtexec: 2.8 ms for 16 hypotheses, 5.2 ms for 32 (flow/README.md).
JETSON_MS_PER_32 = 5.2


def jetson_network_ms(k=32, centers=4, n_orient=32, steps=8, refine=2):
    """Network time of one propose() call on the Jetson, from the measured time per evaluation (scaled linearly with
    the number of hypotheses, as measured between 16 and 32): the position stage and the rotation flow."""
    return JETSON_MS_PER_32 * (5 * k / 32 + (steps + refine) * centers * n_orient / 32)


def write_hints(ev, rig, proposer, n_hints):
    """Runs proposer(index, depth) -> hypotheses dict on every scene of the evaluation set ev; writes hints.bin (for
    trk_hint_eval) and flow.npz (diagnostics against the truth), prints a summary per kind of scene."""
    rows, blobs = [], []
    for i, cat, T, depth, _ in load_eval(ev):
        h = proposer(i, depth)
        sel = hint_order(h, n_hints)
        c = T[:3, :3] @ rig.centroid + T[:3, 3]
        ep = np.linalg.norm(h["p"] - c, axis=1) * 1000 if len(h["p"]) else np.zeros(0)
        er = np.array([geom.rot_angle_deg(Th[:3, :3], T[:3, :3]) for Th in h["T"]])
        good = (ep < 15) & (er < 15)
        rank = next((r for r, j in enumerate(sel) if good[j]), -1)
        rows.append(dict(scene=i, cat=cat, k=len(ep), n_good=int(good.sum()), first_good_rank=rank, n_groups=len(h["groups"]),
                         top_group=len(h["groups"][0]) if h["groups"] else 0,
                         best_mm=float(ep[good].min()) if good.any() else -1, best_deg=float(er[good].min()) if good.any() else -1,
                         seed_near=int((np.linalg.norm(h["seeds"] - c, axis=1) < 0.10).sum()) if len(ep) else 0,
                         center_mm=float(np.linalg.norm(h["centers"] - c, axis=1).min() * 1000) if len(ep) else -1,
                         t_prep=h["t_prep"], t_flow=h["t_flow"]))
        blobs.append(np.concatenate([np.stack([h["T"][j] for j in sel]).reshape(-1), np.zeros(16 * (n_hints - len(sel)))])
                     if sel else np.zeros(16 * n_hints))
        rows[-1]["n_hints"] = len(sel)
    ev = Path(ev)
    with open(ev / "hints.bin", "wb") as f:
        f.write(b"TRKHINTS" + struct.pack("<III", 1, len(rows), n_hints))
        for r, b in zip(rows, blobs):
            f.write(struct.pack("<I", r["n_hints"]) + b.astype("<f8").tobytes())
    np.savez(ev / "flow.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    for cat in scene.CATEGORIES:
        rs = [r for r in rows if r["cat"] == cat]
        print(f"{cat:11s} seed within 0.10 m {np.mean([r['seed_near'] > 0 for r in rs]):.2f}  position stage within 20 mm "
              f"{np.mean([0 <= r['center_mm'] < 20 for r in rs]):.2f}  any hypothesis within 15 mm / 15 deg "
              f"{np.mean([r['n_good'] > 0 for r in rs]):.2f}  in the {n_hints} hints {np.mean([r['first_good_rank'] >= 0 for r in rs]):.2f}  "
              f"first {np.mean([r['first_good_rank'] == 0 for r in rs]):.2f}  flow ms p50 {1e3 * np.median([r['t_flow'] for r in rs]):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flow/runs/a/model.pt")
    ap.add_argument("--eval", default="flow/data/eval")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--hints", type=int, default=16)
    ap.add_argument("--centers", type=int, default=4)
    ap.add_argument("--orient", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    net = load_model(a.model, dev)
    rig = scene.Rig()
    write_hints(a.eval, rig, lambda i, depth: propose(net, rig, depth, dev, k=a.k, centers=a.centers, n_orient=a.orient, steps=a.steps, seed=i),
                a.hints)


if __name__ == "__main__":
    main()


# ---------------- samples of the learned field (demo) ----------------
@torch.no_grad()
def field_eval(net, rig, P, C, T_ref, dev, t, pos_offsets, rotvecs, seed=0, chunk=256):
    """The learned field at pose estimates around a reference pose T_ref (4x4), for one cloud. Estimate i has the
    centroid at (reference centroid + pos_offsets[i]) (rig frame, m) and the rotation R_ref exp(rotvecs[i]).
    Returns, per estimate: pos_end (offset from the reference centroid of the centroid the network predicts),
    rotvec_end (rotation vector, relative to the reference, of the rotation it predicts), and the size of the predicted
    remaining motion: move_mm and turn_deg. The network output is the remaining motion xi at flow time t; the velocity
    of the flow is xi / (1 - t)."""
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    c = T_ref[:3, :3] @ rig.centroid + T_ref[:3, 3]
    q_ref = torch.from_numpy(geom.quat_from_R(T_ref[:3, :3]).astype(np.float32)).to(dev)
    pts, cam, n = _crops(P, C, c[None], rng)
    pts_t, cam_t = torch.from_numpy(pts).to(dev), torch.from_numpy(cam).to(dev)
    out = {"pos_end": [], "rotvec_end": [], "move_mm": [], "turn_deg": []}
    for i in range(0, len(pos_offsets), chunk):
        p = torch.from_numpy(np.asarray(pos_offsets[i:i + chunk], np.float32)).to(dev)
        B = len(p)
        qr = q_ref[None].repeat(B, 1)
        q = geom.qmul(qr, geom.qexp(torch.from_numpy(np.asarray(rotvecs[i:i + chunk], np.float32)).to(dev)))
        valid = (torch.arange(MAX_PTS, device=dev)[None] < int(n[0])).expand(B, -1)
        x, _, _ = sampler.crop_canonical(pts_t.expand(B, -1, -1), valid, cam_t.expand(B, -1), q, p, generator=g)
        xi, _ = net(x, torch.full((B,), float(t), device=dev))
        q_end, p_end = sampler.apply(q, p, xi, 1.0)
        out["pos_end"].append(p_end.cpu().numpy())
        out["rotvec_end"].append(geom.qlog(geom.qmul(geom.qconj(qr), q_end)).cpu().numpy())
        out["move_mm"].append((xi[:, 3:].norm(dim=-1) * sampler.POS_SCALE * 1000).cpu().numpy())
        out["turn_deg"].append(xi[:, :3].norm(dim=-1).rad2deg().cpu().numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}


FIELD_GRID = 17                 # estimates per axis of a field slice
FIELD_POS_HALF = 0.10           # m: half-width of the translation slice
FIELD_ROT_HALF = 170.0          # degrees: half-width of the rotation slice
FIELD_AXES = (0, 1)             # the slices use axes x and y (rig axes for positions, object axes for rotations)


def field_times(steps=8):
    """Flow times at which propose() evaluates the network in the rotation flow: the Euler steps, then t = 1 (the
    end-point corrections)."""
    return [i / steps for i in range(steps)] + [1.0]


def field_frames(net, rig, P, C, T_ref, dev, steps=8):
    """The two 2-D slices of the learned field around T_ref (demo), at every flow time of field_times().
      translation slice: rotation at the reference, centroid estimate moved in the rig x-y plane;
      rotation slice: position at the reference, estimate rotated about the object x and y axes.
    Returns arrays [times, G, G] (row = second axis): pos_mm / rot_deg (size of the predicted remaining motion) and
    pos_u, pos_v / rot_u, rot_v (its direction in the slice: m and rad; rotation: zero where the chart wraps)."""
    G, (i, j) = FIELD_GRID, FIELD_AXES
    N = G * G
    g = np.linspace(-FIELD_POS_HALF, FIELD_POS_HALF, G)
    A, B = np.meshgrid(g, g, indexing="xy")
    off = np.zeros((N, 3))
    off[:, i], off[:, j] = A.ravel(), B.ravel()
    g = np.radians(np.linspace(-FIELD_ROT_HALF, FIELD_ROT_HALF, G))
    A, B = np.meshgrid(g, g, indexing="xy")
    rv = np.zeros((N, 3))
    rv[:, i], rv[:, j] = A.ravel(), B.ravel()
    out = {k: [] for k in ("pos_mm", "pos_u", "pos_v", "rot_deg", "rot_u", "rot_v")}
    for t in field_times(steps):
        f = field_eval(net, rig, P, C, T_ref, dev, t, np.concatenate([off, np.zeros((N, 3))]), np.concatenate([np.zeros((N, 3)), rv]))
        d = f["pos_end"][:N] - off
        out["pos_mm"].append(f["move_mm"][:N])
        out["pos_u"].append(d[:, i])
        out["pos_v"].append(d[:, j])
        d = f["rotvec_end"][N:] - rv
        d[np.linalg.norm(d, axis=1) > 2.8] = 0
        out["rot_deg"].append(f["turn_deg"][N:])
        out["rot_u"].append(d[:, i])
        out["rot_v"].append(d[:, j])
    return {k: np.stack(v).reshape(-1, G, G).astype(np.float32) for k, v in out.items()}


def field_cache_path(ev, scene_index):
    return Path(ev) / "fields" / f"scene_{scene_index:04d}.npz"


def cached_field_frames(net, rig, ev, scene_index, P, C, T_ref, dev):
    """field_frames() from the cache of the evaluation set, computed and stored if it is not there."""
    f = field_cache_path(ev, scene_index)
    if f.exists():
        z = np.load(f)
        if z["pos_mm"].shape[1] == FIELD_GRID:
            return {k: z[k] for k in z.files}
    fr = field_frames(net, rig, P, C, T_ref, dev)
    f.parent.mkdir(exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **fr)
    tmp.replace(f)
    return fr
