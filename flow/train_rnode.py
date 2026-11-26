"""RNODE training of the pose flow (flow/rnode.py) on the synthetic shards. Runs on a CUDA machine without open3d /
trimesh: it needs only the shards and flow/data/rnode_table.npy (python -m flow.train_rnode --make-table, on the Mac).

    python -m flow.train_rnode --data flow/data/train --run flow/runs/rnode [--fm-steps 20000] [--steps 3000]
                               [--batch 64] [--solver-steps 4] [--lambda-k 0.01] [--lambda-j 0.01] [--exact]
--fm-steps: a flow-matching warm start of the same network before the RNODE phase (see the comment in main).
Loss (paper, Eq. 17): mean of [-log q(z(T)) - l(T) + lambda_K E(T) + lambda_J n(T)] / 6, plus 0.2 x the cross-entropy of
the object / clutter output at the data point (auxiliary, not part of RNODE). lambda 0 / 0 is the unregularized flow."""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import geom, rnode
from .model import POS_SCALE


def load(data, have=0):
    """data: one folder of shards or several separated by commas (the held-out scenes are the first of the first)."""
    files = [f for d in data.split(",") for f in sorted(glob.glob(f"{d}/shard_*.npz"))]
    if len(files) <= have:
        return None
    z = [np.load(f) for f in files]
    d = {k: torch.from_numpy(np.concatenate([x[k] for x in z])) for k in ("pts", "flag", "n", "q")}
    d["files"] = len(files)
    print(f"{len(files)} shards, {len(d['n'])} scenes", flush=True)
    return d


def make_batch(d, idx, dev, g=None, dequantize=True):
    """Data of the scenes idx: a seed near the true centroid, the points around the seed, and the data pose z(0) (the
    true pose with the dequantization noise), all relative to the seed. Also the true pose (q1, p1)."""
    pts = d["pts"][idx].to(dev).float()                       # relative to the true centroid
    flag = d["flag"][idx].to(dev)
    valid = torch.arange(pts.shape[1], device=dev)[None] < d["n"][idx].to(dev)[:, None]
    q1 = d["q"][idx].to(dev)
    B = len(idx)
    u = torch.randn(B, 3, device=dev, generator=g)
    seed = u / u.norm(dim=-1, keepdim=True) * torch.rand(B, 1, device=dev, generator=g).pow(1 / 3) * rnode.SEED_MAX
    P, cam, lab, ok = rnode.select_points(pts - seed[:, None], valid, (flag & 1).long(), ((flag >> 1) & 1).float(), generator=g)
    p1 = -seed
    q, p = q1, p1
    if dequantize:
        q = geom.qmul(q1, rnode.qexp(torch.randn(B, 3, device=dev, generator=g) * rnode.DEQ_ROT))
        p = p1 + torch.randn(B, 3, device=dev, generator=g) * rnode.DEQ_POS * POS_SCALE
    return P, cam, lab, ok, q, p, q1, p1


def validate(net, table, d, idx, dev, solver_steps, exact=False, samples=16):
    """Held-out scenes. nll / E / n of the data (nats per dimension; fixed noise), and generation: `samples` base
    samples per scene (around the true pose) integrated back to the data end; share of samples, and of scenes with
    at least one sample, within 10 degrees and 10 mm of the truth, and the same for the samples that start more than
    45 degrees from the truth."""
    net.eval()
    g = torch.Generator(device=dev).manual_seed(11)
    acc = {k: [] for k in ("nll", "E", "n", "hit", "any", "rot", "pos")}
    for i in range(0, len(idx), 64):
        P, cam, _, _, q, p, q1, p1 = make_batch(d, idx[i:i + 64], dev, g)
        B = len(q)
        eps = rnode.EXACT if exact else torch.randn(B, rnode.DIM, device=dev, generator=g)
        nll, E, n, _ = rnode.rnode_terms(net, table, P, cam, q, p, q1, p1, eps, solver_steps, create_graph=False)
        acc["nll"].append(nll.detach() / rnode.DIM)
        acc["E"].append(E.detach())
        acc["n"].append(n.detach())
        rep = torch.arange(B, device=dev).repeat_interleave(samples)
        qb, pb = rnode.sample_base(q1[rep], p1[rep], g)
        gen = [rnode.generate(net, table, P[r], cam[r], qb[j:j + 256], pb[j:j + 256], solver_steps)      # 256 at a time: memory
               for j, r in ((j, rep[j:j + 256]) for j in range(0, len(rep), 256))]
        qg, pg = torch.cat([x[0] for x in gen]), torch.cat([x[1] for x in gen])
        er = geom.qlog(geom.qmul(geom.qconj(qg), q1[rep])).norm(dim=-1).rad2deg()
        ep = (pg - p1[rep]).norm(dim=-1) * 1000
        far = geom.qlog(geom.qmul(geom.qconj(qb), q1[rep])).norm(dim=-1).rad2deg() > 45
        acc.setdefault("hit_far", []).append(((er < 10) & (ep < 10)).float()[far])
        hit = ((er < 10) & (ep < 10)).float().reshape(B, samples)
        acc["hit"].append(hit.mean(1))
        acc["any"].append(hit.max(1).values)
        acc["rot"].append(er)
        acc["pos"].append(ep)
    net.train()
    c = {k: torch.cat(v) for k, v in acc.items()}
    return {"val_nll": c["nll"].mean().item(), "val_E": c["E"].mean().item(), "val_n": c["n"].mean().item(),
            "gen_hit": c["hit"].mean().item(), "gen_any": c["any"].mean().item(), "gen_hit_from_over_45deg": c["hit_far"].mean().item(),
            "gen_rot_p50": c["rot"].median().item(), "gen_pos_p50": c["pos"].median().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="flow/data/train")
    ap.add_argument("--run", default="flow/runs/rnode")
    ap.add_argument("--table", default="flow/data/rnode_table.npy")
    ap.add_argument("--make-table", action="store_true")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--solver-steps", type=int, default=4)
    ap.add_argument("--lambda-k", type=float, default=0.01)
    ap.add_argument("--lambda-j", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--val", type=int, default=512)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0, help="gradient norm limit")
    ap.add_argument("--deq-rot", type=float, default=rnode.DEQ_ROT, help="dequantization noise of the rotation, rad")
    ap.add_argument("--deq-pos", type=float, default=rnode.DEQ_POS, help="dequantization noise of the position, POS_SCALE units")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--exact", action="store_true", help="exact trace and Frobenius norm (6 products) in place of Hutchinson")
    ap.add_argument("--fm-steps", type=int, default=0, help="flow-matching warm start before RNODE (0: none)")
    ap.add_argument("--fm-batch", type=int, default=256)
    ap.add_argument("--fm-lr", type=float, default=1e-3)
    ap.add_argument("--init", default="", help="checkpoint to start from")
    ap.add_argument("--accum", type=int, default=1, help="batches per RNODE update (gradients summed)")
    ap.add_argument("--encoder", default="pointnet", choices=("pointnet", "lpn", "litept_s", "litept_t", "litept_s_rig", "litept_t_rig"),
                    help="point encoder: PointNet-style (RNet), Lean Point Networks (LPNet), LitePT (litept.LitePT)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="pointnet encoder: the object / clutter output weights the pooling")
    ap.add_argument("--time-sampling", default="uniform", choices=("uniform", "logitnormal", "beta"),
                    help="flow-matching time: uniform; sigmoid of a standard normal (Esser et al. 2024); or more weight "
                         "at the base end, t = 1 - 0.999 (1 - s), s ~ Beta(1.5, 1) (Black et al. 2024, pi0)")
    ap.add_argument("--sigma-rot", type=float, default=rnode.SIGMA_ROT, help="width of the base around the truth, rad")
    ap.add_argument("--sigma-pos", type=float, default=rnode.SIGMA_POS, help="width of the base, POS_SCALE units")
    ap.add_argument("--source-uniform", type=float, default=0.0,
                    help="flow-matching phase: share of base samples with a uniformly random rotation (the starts of "
                         "the inference are spread over all rotations)")
    ap.add_argument("--loss-norm", action="store_true",
                    help="flow-matching loss with the rotation and position parts divided by the widths of the base")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay in the flow-matching phase (0: Adam)")
    ap.add_argument("--ema", default="0.999,0.9999", help="decays of the weight averages kept next to the model")
    ap.add_argument("--smooth", action="store_true", help="ablation: smooth pooling and table interpolation (rnode.SMOOTH)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.make_table:
        from . import scene
        np.save(a.table, rnode.build_table(scene.Rig()))
        print(a.table)
        return
    rnode.DEQ_ROT, rnode.DEQ_POS = a.deq_rot, a.deq_pos
    rnode.SMOOTH = a.smooth
    rnode.SIGMA_ROT, rnode.SIGMA_POS = a.sigma_rot, a.sigma_pos
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True          # TF32 products on the GPU (about 1e-3 relative; faster)
    torch.backends.cudnn.allow_tf32 = True
    table = torch.from_numpy(np.load(a.table)).to(dev)
    d = load(a.data)
    S = len(d["n"])
    val_idx = torch.arange(a.val)
    net = rnode.make_net(a.width, a.encoder, a.gate).to(dev)
    decays = [float(x) for x in a.ema.split(",") if x]
    ema = {str(dc): {k: v.detach().clone() for k, v in net.named_parameters()} for dc in decays}

    @torch.no_grad()
    def ema_update(step):
        for dc in decays:
            w = min(dc, (1 + step) / (10 + step))
            for k, v in net.named_parameters():
                ema[str(dc)][k].lerp_(v, 1 - w)

    @torch.no_grad()
    def with_weights(sd, fn):
        """fn() with the parameters replaced by sd, then restored."""
        keep = {k: v.detach().clone() for k, v in net.named_parameters()}
        for k, v in net.named_parameters():
            v.copy_(sd[k])
        try:
            return fn()
        finally:
            for k, v in net.named_parameters():
                v.copy_(keep[k])
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, device {dev}, RK4 steps {a.solver_steps}, "
          f"lambda_K {a.lambda_k}, lambda_J {a.lambda_j}", flush=True)
    cfg = vars(a)
    log = open(run / "log.jsonl", "a")
    t0 = time.time()

    def emit(r, phase, step, total):
        r.update(phase=phase, step=step, min=(time.time() - t0) / 60)
        if step % a.val_every == 0 or step == total:
            r.update(validate(net, table, d, val_idx, dev, a.solver_steps, a.exact))
            if phase == "fm":
                for dc in decays:
                    v = with_weights(ema[str(dc)], lambda: validate(net, table, d, val_idx, dev, a.solver_steps, a.exact))
                    r.update({f"ema{dc}_{k}": v[k] for k in ("val_nll", "gen_hit", "gen_rot_p50", "gen_pos_p50")})
            torch.save({"model": net.state_dict(), "ema": ema, "width": a.width, "step": step, "phase": phase, "config": cfg},
                       run / ("model_fm.pt" if phase == "fm" else "model.pt"))
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()

    def schedule(opt, total):
        return torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

    # ---- phase 1 (optional): flow-matching warm start. Not RNODE: a regression of f on the velocity of the geodesic
    # from the data to a base sample, without the solver. It gives the RNODE phase a field that already carries the
    # data to the base.
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["model"])
    if a.fm_steps:
        opt = (torch.optim.AdamW(net.parameters(), lr=a.fm_lr, weight_decay=a.weight_decay) if a.weight_decay > 0
               else torch.optim.Adam(net.parameters(), lr=a.fm_lr))
        sched = schedule(opt, a.fm_steps)
        acc = np.zeros(3)
        loss_scale = torch.tensor([a.sigma_rot] * 3 + [a.sigma_pos] * 3, device=dev) / math.sqrt(a.sigma_rot * a.sigma_pos)
        for step in range(1, a.fm_steps + 1):
            idx = torch.randint(a.val, S, (a.fm_batch,))
            P, cam, lab, ok, q, p, q1, p1 = make_batch(d, idx, dev)
            qb, pb = rnode.sample_base(q1, p1)
            if a.source_uniform > 0:
                uni = torch.rand(a.fm_batch, device=dev) < a.source_uniform
                qb = torch.where(uni[:, None], geom.qrand(a.fm_batch, dev), qb)
            if a.time_sampling == "logitnormal":
                t = torch.sigmoid(torch.randn(a.fm_batch, device=dev))
            elif a.time_sampling == "beta":
                t = 1 - 0.999 * (1 - torch.distributions.Beta(1.5, 1.0).sample((a.fm_batch,)).to(dev))
            else:
                t = torch.rand(a.fm_batch, device=dev)
            t = t * rnode.T_END
            qt, pt, target = rnode.fm_pair(q, p, qb, pb, t)
            f, _, _, seg = rnode.field(net, table, P, cam, qt, pt, t)
            l_fm = (f - target).pow(2).sum(-1).mean()
            l_opt = ((f - target) / loss_scale).pow(2).sum(-1).mean() if a.loss_norm else l_fm
            # the segmentation output is trained where RNODE trains it: at the data pose
            if a.gate:      # the mask acts at every pose: train it at the pose of the sample
                l_seg = (F.binary_cross_entropy_with_logits(seg, lab, reduction="none") * ok).sum() / ok.sum().clamp_min(1) / 4
            else:
                seg0 = rnode.field(net, table, P, cam, q, p, 0.0)[3] if step % 4 == 0 else None
                l_seg = (F.binary_cross_entropy_with_logits(seg0, lab, reduction="none") * ok).sum() / ok.sum().clamp_min(1) if seg0 is not None else l_fm * 0
            opt.zero_grad(set_to_none=True)
            (l_opt + 0.2 * l_seg).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            sched.step()
            ema_update(step)
            acc += [l_fm.item(), l_seg.item() * 4, 1]
            if step % a.log_every == 0:
                r = {"fm_loss": acc[0] / acc[2], "l_seg": acc[1] / acc[2]}
                acc[:] = 0
                emit(r, "fm", step, a.fm_steps)

    # ---- phase 2: RNODE (Algorithm 1 of the paper)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = schedule(opt, max(1, a.steps))
    acc = np.zeros(7)
    for step in range(1, a.steps + 1):
        # one update from a.accum batches (gradients summed): the gradient of the divergence term is close to
        # uncorrelated between samples, so the update needs hundreds of samples; they do not fit in memory at once
        opt.zero_grad(set_to_none=True)
        part = np.zeros(5)
        for _ in range(a.accum):
            idx = torch.randint(a.val, S, (a.batch,))
            P, cam, lab, ok, q, p, q1, p1 = make_batch(d, idx, dev)
            eps = rnode.EXACT if a.exact else torch.randn(a.batch, rnode.DIM, device=dev)     # once per batch
            nll, E, n, seg = rnode.rnode_terms(net, table, P, cam, q, p, q1, p1, eps, a.solver_steps)
            l_rnode = rnode.rnode_loss(nll, E, n, a.lambda_k, a.lambda_j)
            l_seg = (F.binary_cross_entropy_with_logits(seg, lab, reduction="none") * ok).sum() / ok.sum().clamp_min(1)
            ((l_rnode + 0.2 * l_seg) / a.accum).backward()
            part += np.array([nll.mean().item() / rnode.DIM, E.mean().item(), n.mean().item(), l_seg.item(), 0]) / a.accum
            part[4] = max(part[4], E.max().item())
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        if torch.isfinite(gn):
            opt.step()
        sched.step()
        acc += [part[0], part[1], part[2], part[3], float(torch.isfinite(gn)),
                min(gn.item(), 1e6) if torch.isfinite(gn) else 1e6, part[4]]
        if step % a.log_every == 0:
            k = a.log_every
            r = {"nll": acc[0] / k, "E": acc[1] / k, "n": acc[2] / k, "l_seg": acc[3] / k, "finite": acc[4] / k,
                 "grad_norm": acc[5] / k, "E_max": acc[6] / k}
            acc[:] = 0
            emit(r, "rnode", step, a.steps)


if __name__ == "__main__":
    main()
