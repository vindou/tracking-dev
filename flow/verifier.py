"""A learned check of pose hypotheses: is this pose, after refinement, the pose of the object?

Why: the errors left after flow + ICP are mostly wrong poses that pass the tracker's fit check (hand on the object,
heavy clutter). Pose papers that rank or reject hypotheses with a learned score report it as one of their larger
gains (GenPose energy ranking, FoundationPose pose ranking, MegaPose coarse scorer; ZePHyR for point clouds).

  icp()        point-to-plane refinement against the closest-point table (a stand-in for the tracker's ICP: the
               verifier has to judge poses where ICP has converged, right or wrong).
  Verifier     two PointNet branches: the cloud in the body frame of the pose with its residuals against the model
               (the features of the flow network), and points of the model with their distance to the nearest cloud
               point (how much of the model the cloud explains). One logit.
  train        negatives and positives come from the pipeline itself (as MegaPose): outputs of the trained flow from
               the kinds of start the inference uses, plus perturbed true poses, all refined with icp() and labelled by
               their error after refinement.

   python -m flow.verifier --flow flow/runs/rnode/model_fm.pt --run flow/runs/verifier"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geom, rnode, train_rnode as T
from .model import POS_SCALE, mlp

N_MODEL = 256
OK_MM, OK_DEG = 10.0, 10.0


def model_points(table, n=N_MODEL):
    """n points of the model surface with their normals [n, 6], from the closest-point table (farthest-point order)."""
    G = table.shape[0]
    a = (torch.arange(G, device=table.device) + 0.5) / G * 2 * rnode.TABLE_HALF - rnode.TABLE_HALF
    cen = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), -1)
    near = (table[..., :3] - cen).norm(dim=-1) < rnode.TABLE_HALF / G          # cells that contain the surface
    cand = table[near]
    cand = cand[torch.randperm(len(cand), generator=torch.Generator().manual_seed(0))[:20000].to(cand.device)]
    idx = [0]
    d = torch.full((len(cand),), float("inf"), device=cand.device)
    for _ in range(n - 1):
        d = torch.minimum(d, (cand[:, :3] - cand[idx[-1], :3]).pow(2).sum(-1))
        idx.append(int(d.argmax()))
    return cand[idx].clone()


@torch.no_grad()
def icp(table, pts, q, p, iters=6, d0=0.020, d1=0.008):
    """Point-to-plane Gauss-Newton steps of the pose (q, p) on the points pts [B, N, 3]; correspondences from the
    table, used within a distance that shrinks from d0 to d1. Returns q, p."""
    eye = 1e-6 * torch.eye(6, device=pts.device, dtype=pts.dtype)
    for i in range(iters):
        xm = (pts - p[:, None]) @ geom.qmat(q)
        e = rnode.lookup(table, xm)
        cp, n = e[..., :3], e[..., 3:]
        dmax = d0 + (d1 - d0) * i / max(1, iters - 1)
        w = ((cp - xm).norm(dim=-1) < dmax).to(pts.dtype)
        r = ((xm - cp) * n).sum(-1)
        A = torch.cat([torch.cross(xm, n, dim=-1), n], -1)                    # d r / d (rotation, translation), body frame
        H = torch.einsum("bn,bni,bnj->bij", w, A, A)
        H = H + eye * (1.0 + 1e3 * torch.diagonal(H, dim1=1, dim2=2).mean(-1))[:, None, None]      # damping relative to the scale
        g = torch.einsum("bn,bni,bn->bi", w, A, r)
        delta = torch.linalg.solve_ex(H.cpu().double(), g.cpu().double()[..., None]).result.squeeze(-1).to(pts)     # 6 x 6, float64: on the CPU (no float64 on MPS)
        usable = (w.sum(1) >= 12) & torch.isfinite(delta).all(-1) & (delta.norm(dim=-1) < 1.0)
        delta = torch.where(usable[:, None], delta, torch.zeros_like(delta))
        q = geom.qmul(q, rnode.qexp(delta[:, :3]))
        q = q / q.norm(dim=-1, keepdim=True)
        p = p + (geom.qmat(q) @ delta[:, 3:, None]).squeeze(-1)
    return q, p


class Verifier(nn.Module):
    def __init__(self, model_pts, w=1):
        super().__init__()
        self.register_buffer("model_pts", model_pts)
        b = 128 * w
        self.cloud1 = mlp([rnode.N_FEAT, 64 * w, 128 * w, b], norm=False)
        self.cloud2 = mlp([2 * b, 256 * w, 256 * w], norm=False)
        self.model1 = mlp([3 + 3 + 3, 64 * w, b], norm=False)
        self.model2 = mlp([2 * b, 256 * w], norm=False)
        self.head = nn.Sequential(mlp([4 * 256 * w, 256, 128]), nn.Linear(128, 1))

    def forward(self, table, pts, cam, q, p):
        R = geom.qmat(q)
        xm = (pts - p[:, None]) @ R
        f = self.cloud1(rnode.features(table, xm, cam))
        h = self.cloud2(torch.cat([f, f.max(1, keepdim=True).values.expand_as(f)], -1))
        # the model points in the frame of the cloud: distance and direction to the nearest cloud point, bounded
        mp, mn = self.model_pts[None, :, :3], self.model_pts[None, :, 3:]
        d, j = torch.cdist(mp.expand(len(xm), -1, -1), xm).min(dim=-1)
        near = torch.gather(xm, 1, j[..., None].expand(-1, -1, 3))
        v = (near - mp) / rnode.RES_SCALE
        dn = d[..., None] / rnode.RES_SCALE
        m = self.model1(torch.cat([mp.expand(len(xm), -1, -1) / rnode.SET_RADIUS, mn.expand(len(xm), -1, -1), v * torch.tanh(dn) / (dn + 1e-6)], -1))
        k = self.model2(torch.cat([m, m.max(1, keepdim=True).values.expand_as(m)], -1))
        return self.head(torch.cat([h.max(1).values, h.mean(1), k.max(1).values, k.mean(1)], -1)).squeeze(-1)


def errors(q, p, q1, p1):
    return (geom.qlog(geom.qmul(geom.qconj(q), q1)).norm(dim=-1).rad2deg(), (p - p1).norm(dim=-1) * 1000)


@torch.no_grad()
def candidates(flow, table, P, cam, q1, p1, per_scene, g=None):
    """per_scene candidate poses for each scene, refined, with labels. A quarter are perturbed true poses; half are
    flow outputs from a uniformly random rotation at a centre 25 mm (sigma) from the truth, as the second stage of the
    inference; a quarter are flow outputs from the base around the truth. Returns rep [M], q, p, label, and the
    errors before the refinement."""
    B, dev = len(q1), q1.device
    rep = torch.arange(B, device=dev).repeat_interleave(per_scene)
    M = len(rep)
    kind = torch.arange(M, device=dev) % 4                                    # 0: truth + noise, 1, 2: far start, 3: base
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    q_far = geom.qrand(M, dev, g)
    q_base, p_base = rnode.sample_base(q1[rep], p1[rep], g)
    qs = torch.where((kind == 3)[:, None], q_base, q_far)
    ps = torch.where((kind == 3)[:, None], p_base, p1[rep] + rn(M, 3) * 0.025)
    qf, pf = rnode.generate(flow, table, P[rep], cam[rep], qs, ps, 4, solver="euler")
    scale = torch.rand(M, 1, device=dev, generator=g)                          # spread of the perturbation: up to 12 deg, 12 mm
    qn = geom.qmul(q1[rep], rnode.qexp(rn(M, 3) * scale * 0.12))
    pn = p1[rep] + rn(M, 3) * scale * 0.007
    q0 = torch.where((kind == 0)[:, None], qn, qf)
    p0 = torch.where((kind == 0)[:, None], pn, pf)
    er0, ep0 = errors(q0, p0, q1[rep], p1[rep])
    q, p = icp(table, P[rep], q0, p0)
    er, ep = errors(q, p, q1[rep], p1[rep])
    return rep, q, p, ((er < OK_DEG) & (ep < OK_MM)).float(), (er0, ep0, er, ep)


def auc(score, label):
    order = torch.argsort(score)
    rank = torch.empty_like(order, dtype=torch.float)
    rank[order] = torch.arange(1, len(score) + 1, device=score.device, dtype=torch.float)
    n1, n0 = label.sum(), (1 - label).sum()
    return ((rank[label > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0).clamp_min(1)).item()


@torch.no_grad()
def validate(net, flow, table, d, dev, per_scene=8):
    net.eval()
    g = torch.Generator(device=dev).manual_seed(21)
    S, Y, E = [], [], []
    for i in range(0, 512, 32):
        P, cam, _, _, _, _, q1, p1 = T.make_batch(d, torch.arange(i, i + 32), dev, g)
        rep, q, p, y, err = candidates(flow, table, P, cam, q1, p1, per_scene, g)
        S.append(net(table, P[rep], cam[rep], q, p)), Y.append(y), E.append(torch.stack(err, 1))
    net.train()
    s, y, e = torch.cat(S), torch.cat(Y), torch.cat(E)
    pr = torch.sigmoid(s)
    out = dict(val_auc=auc(s, y), val_pos=y.mean().item(), val_bce=F.binary_cross_entropy_with_logits(s, y).item())
    # what a threshold does: share of the wrong poses rejected and of the right poses kept
    for th in (0.3, 0.5, 0.7):
        out[f"reject_wrong@{th}"] = ((pr < th) & (y < 0.5)).float().sum().item() / max(1.0, (y < 0.5).float().sum().item())
        out[f"keep_right@{th}"] = ((pr >= th) & (y > 0.5)).float().sum().item() / max(1.0, y.sum().item())
    # the hard cases: poses that are wrong although the refinement left a small residual are in here
    out["icp_right_before_wrong"] = (((e[:, 0] >= OK_DEG) | (e[:, 1] >= OK_MM)) & (y > 0.5)).float().mean().item()
    return out


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    net = Verifier(ck["model"]["model_pts"], ck.get("width", 1)).to(dev)
    net.load_state_dict(ck["model"])
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="flow/runs/rnode/model_fm.pt", help="the flow model whose outputs are judged")
    ap.add_argument("--run", default="flow/runs/verifier")
    ap.add_argument("--data", default="flow/data/train")
    ap.add_argument("--table", default="flow/data/rnode_table.npy")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=48, help="scenes per step")
    ap.add_argument("--per-scene", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    table = torch.from_numpy(np.load(a.table)).to(dev)
    flow = rnode.net_from_checkpoint(torch.load(a.flow, map_location=dev), dev)
    d = T.load(a.data)
    S = len(d["n"])
    net = Verifier(model_points(table)).to(dev)
    print(f"{sum(x.numel() for x in net.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    log = open(run / "log.jsonl", "a")
    acc, t0 = np.zeros(3), time.time()
    for step in range(1, a.steps + 1):
        P, cam, _, _, _, _, q1, p1 = T.make_batch(d, torch.randint(512, S, (a.batch,)), dev)
        rep, q, p, y, _ = candidates(flow, table, P, cam, q1, p1, a.per_scene)
        s = net(table, P[rep], cam[rep], q, p)
        loss = F.binary_cross_entropy_with_logits(s, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        sched.step()
        acc += [loss.item(), y.mean().item(), 1]
        if step % 100 == 0:
            r = dict(step=step, min=(time.time() - t0) / 60, bce=acc[0] / acc[2], pos=acc[1] / acc[2])
            acc[:] = 0
            if step % a.val_every == 0 or step == a.steps:
                r.update(validate(net, flow, table, d, dev))
                torch.save({"model": net.state_dict(), "width": 1, "step": step, "config": vars(a)}, run / "model.pt")
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
