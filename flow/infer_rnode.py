"""Pose hypotheses from the RNODE model (flow/rnode.py): base samples integrated backward to the data end.

    python -m flow.infer_rnode --model flow/runs/rnode/model.pt --eval flow/data/eval_rnode [--solver-steps 4]
writes <eval>/hints.bin and flow.npz like flow.infer (same two stages, same grouping, same number of hypotheses).

The base of the model is centred on the true pose, which is not known here, so the starts are not base samples:
  1. Position: k seeds spread over the cloud; from each, a start with a random rotation at the seed is integrated to
     t = 0. The end positions are grouped (30 mm); the `centers` groups with the most object points (segmentation
     output at the end pose) are kept and become the seeds of stage 2.
  2. Pose: per seed, n_orient starts at the seed with the rotations of the set spread evenly over SO(3) (covering
     radius 63 degrees; the base has 34 degrees standard deviation), integrated to t = 0."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import geom, infer, rnode, scene

ROOT = Path(__file__).resolve().parents[1]


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    rnode.SMOOTH = bool(ck.get("config", {}).get("smooth", False))      # the smooth-network variant (train_rnode --smooth)
    net = rnode.RNet(ck.get("width", 1)).to(dev)
    net.load_state_dict(ck["model"])
    return net.eval()


def _sets(P, C, seeds, rng, dev, generator):
    """The condition of each seed: N_POINTS points within SET_RADIUS, relative to the seed."""
    pts, cam, n = infer._crops(P, C, seeds, rng)
    to = lambda x: torch.from_numpy(x).to(dev)
    valid = torch.arange(pts.shape[1], device=dev)[None] < to(n)[:, None]
    X, c, _, ok = rnode.select_points(to(pts), valid, to(cam), generator=generator)
    return X, c, ok


@torch.no_grad()
def propose(net, table, rig, depth, dev, k=32, centers=4, n_orient=32, solver_steps=4, seed=0, trajectory=False):
    """Depth pair -> hypotheses; the same dictionary as infer.propose (res = minus the object fraction)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    P, C = rig.cloud(depth)
    S = infer.seeds_fps(P, k, rng)
    out = {"P": P, "C": C, "seeds": S, "seed_end": S, "centers": np.zeros((0, 3)), "q": np.zeros((0, 4)), "p": np.zeros((0, 3)),
           "res": np.zeros(0), "objfrac": np.zeros(0), "groups": [], "T": np.zeros((0, 4, 4)), "t_prep": 0.0, "t_flow": 0.0}
    if len(S) == 0:
        return out

    def objfrac(X, c, ok, q, p):
        seg = rnode.field(net, table, X, c, q, p, 0.0)[3]
        return (((torch.sigmoid(seg) > 0.5) & ok).float().sum(1) / ok.float().sum(1).clamp_min(1)).cpu().numpy()

    # stage 1
    X, c, ok = _sets(P, C, S, rng, dev, g)
    t1 = time.perf_counter()
    qb = geom.qrand(len(S), dev, g)                   # starts: a random rotation at the seed
    q1, p1 = rnode.generate(net, table, X, c, qb, torch.zeros(len(S), 3, device=dev), solver_steps)
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
    # stage 2
    Q0, _ = infer.orientations(n_orient)
    X, c, ok = _sets(P, C, cen, rng, dev, g)
    rep = np.repeat(np.arange(len(cen)), n_orient)
    K = len(rep)
    t3 = time.perf_counter()
    qb = torch.from_numpy(np.tile(Q0, (len(cen), 1)).astype(np.float32)).to(dev)
    r = rnode.generate(net, table, X[rep], c[rep], qb, torch.zeros(K, 3, device=dev), solver_steps, trajectory=trajectory)
    of = objfrac(X[rep], c[rep], ok[rep], r[0], r[1])
    q, p = r[0].cpu().numpy().astype(np.float64), r[1].cpu().numpy().astype(np.float64) + cen[rep]
    t4 = time.perf_counter()
    R = geom.qmat(torch.from_numpy(q)).numpy()
    T = np.tile(np.eye(4), (K, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = p - R @ rig.centroid
    out.update(seed_end=e, centers=cen, q=q, p=p, res=-of, objfrac=of, groups=infer.group(q, p, -of), T=T,
               t_prep=(t1 - t0) + (t3 - t2), t_flow=(t2 - t1) + (t4 - t3))
    if trajectory:      # stored from the base (first) to the data end (last), like infer.propose
        out["traj_q"] = np.stack([a.cpu().numpy() for a, _ in r[2]])
        out["traj_p"] = np.stack([b.cpu().numpy() for _, b in r[2]]) + cen[rep][None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flow/runs/rnode/model.pt")
    ap.add_argument("--table", default="flow/data/rnode_table.npy")
    ap.add_argument("--eval", default="flow/data/eval_rnode")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--hints", type=int, default=16)
    ap.add_argument("--centers", type=int, default=4)
    ap.add_argument("--orient", type=int, default=32)
    ap.add_argument("--solver-steps", type=int, default=4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    net = load_model(ROOT / a.model, dev)
    table = torch.from_numpy(np.load(ROOT / a.table)).to(dev)
    rig = scene.Rig()
    infer.write_hints(ROOT / a.eval, rig, lambda i, depth: propose(net, table, rig, depth, dev, k=a.k, centers=a.centers, n_orient=a.orient,
                                                                 solver_steps=a.solver_steps, seed=i), a.hints)


if __name__ == "__main__":
    main()
