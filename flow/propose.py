"""Pose hypotheses for a depth pair from the flow (flow/model.py) and the verifier (flow/verifier.py).

    python -m flow.propose --eval flow/data/eval [--no-verifier] [--min-score 0.85]
writes <eval>/hints.bin for tracker/build-mac/trk_hint_eval and <eval>/flow.npz (diagnostics against the truth).

The source of the flow is centred on the true pose, which is not known here, so the starts are not source samples:
  1. Position: k seeds spread over the cloud; from each, a start with a random rotation at the seed is integrated to
     t = 0 (4 Euler steps). The end positions are grouped (30 mm); the `centers` groups with the most object points (per-point output
     of the network at the end pose) are kept and become the seeds of stage 2.
  2. Pose: per seed, n_orient starts at the seed with the rotations of a set spread evenly over SO(3) (covering
     radius 63 degrees; the source has 34 degrees standard deviation), integrated to t = 0 (8 Euler steps).
  3. Verifier: every end pose is refined on its own points (verifier.icp) and scored; poses below the minimum score
     are dropped. Poses that agree (15 mm, 15 degrees) are grouped; the best of each group goes to the tracker,
     highest score first (TrackerCore::set_hints), which runs its own ICP and fit check."""
import argparse
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch

from . import geom, model, scene, verifier as V
from .make_data import MAX_PTS, STORE_RADIUS, load_eval

ROOT = Path(__file__).resolve().parents[1]
MODEL, VERIFIER, TABLE = "flow/runs/flow/model.pt", "flow/runs/verifier/model.pt", "flow/data/table.npy"
MIN_SCORE = 0.85
# The solver of the deployed configuration: Euler, 4 steps in the position stage and 2 in the pose stage (7
# evaluations with the object logits). In PyTorch, through the tracker, 2 pose steps equal 4 Runge-Kutta steps in both
# stages (33 evaluations) and 8, 6, 5, 4 and 3 Euler steps; through the Jetson engines they give fewer wrong poses
# and more scenes without one than 4 (flow/README.md).
SOLVER, SOLVER_STEPS, STAGE1_STEPS = "euler", 2, 4


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
    smallest res), each a list of indices with the best member (smallest res) first."""
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
    """n rotations spread evenly over SO(3) (farthest-point selection on random rotations), as quaternions, and the
    covering radius of the set (degrees)."""
    if n not in _orient:
        rng = np.random.default_rng(5)
        q = rng.normal(size=(4000, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        sel = [0]
        d = 1 - np.abs(q @ q[0])
        while len(sel) < n:
            sel.append(int(np.argmax(d)))
            d = np.minimum(d, 1 - np.abs(q @ q[sel[-1]]))
        _orient[n] = (q[sel], float(np.degrees(2 * np.arccos(1 - d.max()))))
    return _orient[n]


def point_sets(P, C, seeds, rng, dev, generator):
    """The condition of each seed: N_POINTS points of the cloud P (camera index C) within SET_RADIUS, relative to the
    seed. Returns points [K, N, 3], camera (-1 / +1) and the mask of real points."""
    K = len(seeds)
    pts = np.zeros((K, MAX_PTS, 3), np.float32)
    cam = np.zeros((K, MAX_PTS), np.int64)
    n = np.zeros(K, np.int64)
    for i, s in enumerate(seeds):
        d = P - s.astype(np.float32)
        idx = np.flatnonzero(np.einsum("ij,ij->i", d, d) < STORE_RADIUS ** 2)
        if len(idx) > MAX_PTS:
            idx = rng.choice(idx, MAX_PTS, replace=False)
        pts[i, :len(idx)] = d[idx]
        cam[i, :len(idx)] = C[idx]
        n[i] = len(idx)
    to = lambda x: torch.from_numpy(x).to(dev)
    valid = torch.arange(MAX_PTS, device=dev)[None] < to(n)[:, None]
    X, c, _, ok = model.select_points(to(pts), valid, to(cam), generator=generator)
    return X, c, ok


@torch.no_grad()
def race_solve(net, table, X, c, q, steps, schedule, score="inlier", verifier=None):
    """The pose-stage solve (Euler) with racing: after the steps named in schedule ((step, keep), ...) only the keep
    hypotheses with the highest score continue. score "inlier": share of the 512 points that lie within 5 mm of the
    model surface at the current pose and near it (computed from the features the network sees anyway);
    "verifier": the verifier on the current pose, without the refinement. Returns q, p of the survivors, their
    indices, and the number of hypothesis evaluations spent."""
    alive = torch.arange(len(q), device=q.device)
    p = torch.zeros(len(q), 3, device=q.device)
    h, done, spent = model.T_END / steps, 0, 0
    for step, keep in list(schedule) + [(steps, len(q))]:
        n = step - done
        if n > 0:
            q, p = model.integrate(net, table, X[alive], c[alive], q, p, model.T_END - done * h, model.T_END - step * h, n, solver="euler")
            spent += n * len(alive)
            done = step
        if step < steps and keep < len(alive):
            if score == "verifier":
                s = verifier(table, X[alive], c[alive], q, p)
            else:
                f = model.features(table, (X[alive] - p[:, None]) @ geom.qmat(q), c[alive])
                s = (f[..., 18] * (f[..., 11] < math.tanh(0.005 / model.RES_SCALE))).mean(1)
            top = torch.topk(s, keep).indices.sort().values
            alive, q, p = alive[top], q[top], p[top]
    return q, p, alive, spent


@torch.no_grad()
def propose(net, table, rig, depth, dev, k=32, centers=4, n_orient=32, solver_steps=SOLVER_STEPS, seed=0, trajectory=False,
            verifier=None, min_score=MIN_SCORE, solver=SOLVER, stage1_steps=STAGE1_STEPS, engine=None, race=None, race_score="inlier", step_power=1.0, prior=None,
            center_shift=None, dedup=None, pose_solve=None):
    """Depth pair -> hypotheses. Dictionary: q, p (centroid, rig frame), T (4x4 rig-from-object) of the K hypotheses
    (refined, if there is a verifier), score (verifier, or None), objfrac (share of points the network calls
    object), groups (see group()), seeds, seed_end, centers, timings, and with trajectory = True the states of the
    stage-2 flow (traj_q, traj_p: first = start, last = end of the flow, before the refinement).
    Experimental options (all off by default): step_power (unequal steps of the pose stage, model.integrate); prior
    (a position, rig frame: the position stage is skipped and this is the only position, e.g. the tracker's last
    pose); center_shift (a vector added to the positions of the position stage: positions kept from an earlier
    proposal after the object moved); dedup (mm and degrees: after the refinement, hypotheses within this of an
    earlier one are not scored); race (race_solve); pose_solve (a function (pts, cam, q) -> q, p that replaces the
    solve of the pose stage: flow/fewstep.py).
    With an engine the networks are not run here: engine.stage1(pts, cam, q) -> q, p, seg and
    engine.stage2(pts [centres, N, 3], cam, rep, q) -> refined q, p and scores (flow/trt_eval.py: TensorRT on the
    Jetson); objfrac of the hypotheses is then zero."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    P, C = rig.cloud(depth)
    S = seeds_fps(P, k, rng)
    out = {"P": P, "C": C, "seeds": S, "seed_end": S, "centers": np.zeros((0, 3)), "q": np.zeros((0, 4)), "p": np.zeros((0, 3)),
           "objfrac": np.zeros(0), "score": None, "min_score": min_score, "groups": [], "T": np.zeros((0, 4, 4)),
           "t_prep": 0.0, "t_flow": 0.0}
    if len(S) == 0:
        return out

    def objfrac(X, c, ok, q, p):
        seg = model.field(net, table, X, c, q, p, 0.0)[1]
        return (((torch.sigmoid(seg) > 0.5) & ok).float().sum(1) / ok.float().sum(1).clamp_min(1)).cpu().numpy()

    # stage 1
    X, c, ok = point_sets(P, C, S, rng, dev, g)
    t1 = time.perf_counter()
    qb = geom.qrand(len(S), dev, g)                   # starts: a random rotation at the seed
    if engine is not None:
        q1, p1, seg = engine.stage1(X, c, qb)
        of1 = (((seg > 0) & ok).float().sum(1) / ok.float().sum(1).clamp_min(1)).cpu().numpy()
    else:
        # the position stage starts far from the data and keeps Euler steps when the pose stage uses the multistep solver
        q1, p1 = model.generate(net, table, X, c, qb, torch.zeros(len(S), 3, device=dev), stage1_steps or solver_steps,
                                solver="euler" if solver == "ab2" else solver)
        of1 = objfrac(X, c, ok, q1, p1)
    e = p1.cpu().numpy().astype(np.float64) + S
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
    if prior is not None:
        cen = np.asarray(prior, np.float64)[None]
    if center_shift is not None:
        cen = cen + np.asarray(center_shift, np.float64)[None]
    # stage 2
    Q0, _ = orientations(n_orient)
    X, c, ok = point_sets(P, C, cen, rng, dev, g)
    rep = np.repeat(np.arange(len(cen)), n_orient)
    K = len(rep)
    t3 = time.perf_counter()
    qb = torch.from_numpy(np.tile(Q0, (len(cen), 1)).astype(np.float32)).to(dev)
    if engine is not None:
        qe, pe, score = engine.stage2(X, c, rep, qb)
        of, trajectory = np.zeros(K), False
    else:
        if pose_solve is not None:
            qe, pe = pose_solve(X[rep], c[rep], qb)
            out["evals"], trajectory = solver_steps * K, False
        elif race:        # racing over the hypotheses during the solve: the losers are dropped (rep, K: the survivors)
            qe, pe, alive, out["evals"] = race_solve(net, table, X[rep], c[rep], qb, solver_steps, race, race_score, verifier)
            rep, trajectory = rep[alive.cpu().numpy()], False
            K = len(rep)
        else:
            r = model.generate(net, table, X[rep], c[rep], qb, torch.zeros(K, 3, device=dev), solver_steps, trajectory=trajectory, solver=solver,
                               power=step_power)
            qe, pe = r[0], r[1]
            out["evals"] = solver_steps * K
        score = None
        if verifier is not None:        # refine every hypothesis on its points, then judge the refined pose
            qe, pe = V.icp(table, X[rep], qe, pe)
            if dedup:       # score one of every set of refined hypotheses that agree within dedup mm and degrees
                qn, pn = qe.cpu().numpy().astype(np.float64), pe.cpu().numpy().astype(np.float64) + cen[rep]
                cosmin, keep = np.cos(np.radians(dedup) / 2), []
                for i in range(K):
                    if not any(np.linalg.norm(pn[i] - pn[j]) < dedup * 1e-3 and abs(float(qn[i] @ qn[j])) > cosmin for j in keep):
                        keep.append(i)
                kt = torch.as_tensor(keep, device=dev)
                qe, pe, rep, trajectory = qe[kt], pe[kt], rep[keep], False
                K = len(rep)
            out["scored"] = K
            score = torch.sigmoid(verifier(table, X[rep], c[rep], qe, pe)).cpu().numpy()
        of = objfrac(X[rep], c[rep], ok[rep], qe, pe)
    q, p = qe.cpu().numpy().astype(np.float64), pe.cpu().numpy().astype(np.float64) + cen[rep]
    t4 = time.perf_counter()
    R = geom.qmat(torch.from_numpy(q)).numpy()
    T = np.tile(np.eye(4), (K, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = p - R @ rig.centroid
    out.update(seed_end=e, centers=cen, q=q, p=p, objfrac=of, groups=group(q, p, -of if score is None else -score), T=T,
               score=score, t_prep=(t1 - t0) + (t3 - t2), t_flow=(t2 - t1) + (t4 - t3))
    if trajectory:
        out["traj_q"] = np.stack([a.cpu().numpy() for a, _ in r[2]])
        out["traj_p"] = np.stack([b.cpu().numpy() for _, b in r[2]]) + cen[rep][None]
    return out


def hint_order(h, n):
    """The hypotheses to hand to the tracker: the best member of each group. With verifier scores: highest score
    first, and none below the minimum score; without: largest group first."""
    reps = [g[0] for g in h["groups"]]
    if h.get("score") is not None:
        reps = sorted((j for j in reps if h["score"][j] >= h["min_score"]), key=lambda j: -h["score"][j])
    return reps[:n]


DUMP_K = 128


def write_hints(ev, rig, proposer, n_hints, dump=None):
    """Runs proposer(index, depth) -> hypotheses dict on every scene of the evaluation set ev; writes hints.bin (for
    trk_hint_eval) and flow.npz (diagnostics against the truth), prints a summary per kind of scene."""
    rows, blobs = [], []
    for i, cat, T, depth, _ in load_eval(ev):
        h = proposer(i, depth)
        if dump is not None:        # every hypothesis of the scene, for flow.simulate
            dump.append({k: h[k] for k in ("q", "p", "T", "score", "traj_q", "traj_p", "centers") if h.get(k) is not None})
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
              f"first {np.mean([r['first_good_rank'] == 0 for r in rs]):.2f}  network ms p50 {1e3 * np.median([r['t_flow'] for r in rs]):.0f}")


def truth_pose(ev, i):
    """T_rig_from_object of scene i of an evaluation set (scenes.bin, format of trk_hint_eval)."""
    with open(Path(ev) / "scenes.bin", "rb") as f:
        f.seek(12)
        _, w, h = struct.unpack("<III", f.read(12))
        f.seek(24 + i * (128 + 4 * w * h))
        return np.frombuffer(f.read(128), "<f8").reshape(4, 4)


def load(dev, model_path=MODEL, verifier_path=VERIFIER, table_path=TABLE):
    """The flow network, the verifier (None for an empty path) and the closest-point table, on the device."""
    net = model.load(ROOT / model_path, dev)
    ver = V.load(ROOT / verifier_path, dev) if verifier_path else None
    return net, ver, torch.from_numpy(np.load(ROOT / table_path)).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--verifier", default=VERIFIER)
    ap.add_argument("--no-verifier", action="store_true", help="hand on the flow end poses, ordered by agreement")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE, help="hypotheses the verifier scores below this are not handed on")
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--eval", default="flow/data/eval")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--hints", type=int, default=16)
    ap.add_argument("--centers", type=int, default=4)
    ap.add_argument("--orient", type=int, default=32)
    ap.add_argument("--solver-steps", type=int, default=SOLVER_STEPS)
    ap.add_argument("--solver", default=SOLVER, choices=["rk4", "euler", "ab2"])
    ap.add_argument("--stage1-steps", type=int, default=STAGE1_STEPS, help="steps of the position stage (0: as --solver-steps)")
    ap.add_argument("--race", default="", help="racing in the pose stage: 'step:keep,...', e.g. 3:64,5:32 (Euler only)")
    ap.add_argument("--race-score", default="inlier", choices=["inlier", "verifier"])
    ap.add_argument("--act", default="", choices=["", "tanh_gelu", "relu"], help="replace the activation of the flow network (tanh_gelu: the tanh form of GELU, without retraining)")
    ap.add_argument("--step-power", type=float, default=1.0, help="pose stage: step sizes (1: equal; > 1: larger first)")
    ap.add_argument("--prior-offset", type=float, default=-1.0, help="mm: skip the position stage; the only position is the true one moved by this much in a random direction")
    ap.add_argument("--center-shift", type=float, default=0.0, help="mm: move the positions of the position stage by this much in a random direction")
    ap.add_argument("--dedup", type=float, default=0.0, help="mm and degrees: score one of every set of refined hypotheses that agree within this")
    ap.add_argument("--pose-run", default="", help="a checkpoint of flow.fewstep for the pose stage (with --solver-steps)")
    ap.add_argument("--dump", action="store_true", help="also write <eval>/hyp.pkl: all hypotheses and their flow states (flow.simulate)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    net, ver, table = load(dev, a.model, "" if a.no_verifier else a.verifier, a.table)
    if a.act:
        model.set_activation(net, a.act)
    pose_solve = None
    if a.pose_run:
        from . import fewstep
        pose_solve, name = fewstep.load_pose(ROOT / a.pose_run, net, table, dev, a.solver_steps)
        print("pose stage:", name, flush=True)
    rig = scene.Rig()
    dump = [] if a.dump else None
    stats = {"evals": [], "scored": []}

    def extra(i):       # the options that need the true pose or a random direction per scene
        kw = {}
        if a.prior_offset >= 0 or a.center_shift > 0:
            d = np.random.default_rng(900000 + i).normal(size=3)
            d /= np.linalg.norm(d)
            if a.prior_offset >= 0:
                T = truth_pose(ROOT / a.eval, i)
                kw["prior"] = T[:3, :3] @ rig.centroid + T[:3, 3] + d * a.prior_offset * 1e-3
            else:
                kw["center_shift"] = d * a.center_shift * 1e-3
        return kw

    def run(i, depth):
        h = propose(net, table, rig, depth, dev, k=a.k, centers=a.centers, n_orient=a.orient,
                    solver_steps=a.solver_steps, seed=i, verifier=ver, min_score=a.min_score,
                    solver=a.solver, stage1_steps=a.stage1_steps or None, trajectory=a.dump,
                    race=[tuple(int(v) for v in s.split(":")) for s in a.race.split(",")] if a.race else None,
                    race_score=a.race_score, step_power=a.step_power, dedup=a.dedup or None, pose_solve=pose_solve, **extra(i))
        for k in stats:
            stats[k].append(h.get(k, 0))
        return h
    write_hints(ROOT / a.eval, rig, run, a.hints, dump)
    print(f"pose stage: {np.mean(stats['evals']):.0f} hypothesis evaluations per scene; hypotheses scored {np.mean(stats['scored']):.1f}")
    if a.dump:
        import pickle
        pickle.dump(dump, open(ROOT / a.eval / "hyp.pkl", "wb"))


if __name__ == "__main__":
    main()
