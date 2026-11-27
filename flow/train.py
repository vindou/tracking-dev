"""Flow-matching training of the pose flow (flow/model.py) on the synthetic shards. Runs on a CUDA machine without
open3d / trimesh: it needs only the shards and the closest-point table (python -m flow.train --make-table, on the Mac).

    python -m flow.train --data flow/data/train --run flow/runs/flow [--steps 30000] [--batch 256]

Loss: squared error of the field against the velocity of the geodesic from the data pose to a source sample, at a
uniform time, plus 0.2 x the cross-entropy of the object / clutter output at the data pose (every fourth step)."""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import geom, model


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


def make_batch(d, idx, dev, g=None):
    """Data of the scenes idx: a seed near the true centroid, the points around the seed (P, cam, object label, mask
    of real points), the data pose (q, p) (the true pose with the small noise) and the true pose (q1, p1), all
    relative to the seed."""
    pts = d["pts"][idx].to(dev).float()                       # relative to the true centroid
    flag = d["flag"][idx].to(dev)
    valid = torch.arange(pts.shape[1], device=dev)[None] < d["n"][idx].to(dev)[:, None]
    q1 = d["q"][idx].to(dev)
    B = len(idx)
    u = torch.randn(B, 3, device=dev, generator=g)
    seed = u / u.norm(dim=-1, keepdim=True) * torch.rand(B, 1, device=dev, generator=g).pow(1 / 3) * model.SEED_MAX
    P, cam, lab, ok = model.select_points(pts - seed[:, None], valid, (flag & 1).long(), ((flag >> 1) & 1).float(), generator=g)
    p1 = -seed
    q = geom.qmul(q1, model.qexp(torch.randn(B, 3, device=dev, generator=g) * model.DEQ_ROT))
    p = p1 + torch.randn(B, 3, device=dev, generator=g) * model.DEQ_POS * model.POS_SCALE
    return P, cam, lab, ok, q, p, q1, p1


@torch.no_grad()
def validate(net, table, d, idx, dev, solver_steps=4, samples=16):
    """Held-out scenes: `samples` source samples per scene (around the true pose) integrated to the data end; share of
    samples, and of scenes with at least one sample, within 10 degrees and 10 mm of the truth, the same for the
    samples that start more than 45 degrees from the truth, and the median errors. This number did not predict the
    result through the tracker (flow/README.md): compare models with flow.evaluate and flow.compare."""
    net.eval()
    g = torch.Generator(device=dev).manual_seed(11)
    acc = {k: [] for k in ("hit", "any", "hit_far", "rot", "pos")}
    for i in range(0, len(idx), 64):
        P, cam, _, _, _, _, q1, p1 = make_batch(d, idx[i:i + 64], dev, g)
        B = len(q1)
        rep = torch.arange(B, device=dev).repeat_interleave(samples)
        qb, pb = model.sample_source(q1[rep], p1[rep], g)
        gen = [model.generate(net, table, P[r], cam[r], qb[j:j + 256], pb[j:j + 256], solver_steps)      # 256 at a time: memory
               for j, r in ((j, rep[j:j + 256]) for j in range(0, len(rep), 256))]
        qg, pg = torch.cat([x[0] for x in gen]), torch.cat([x[1] for x in gen])
        er = geom.qlog(geom.qmul(geom.qconj(qg), q1[rep])).norm(dim=-1).rad2deg()
        ep = (pg - p1[rep]).norm(dim=-1) * 1000
        far = geom.qlog(geom.qmul(geom.qconj(qb), q1[rep])).norm(dim=-1).rad2deg() > 45
        hit = ((er < 10) & (ep < 10)).float()
        acc["hit_far"].append(hit[far])
        acc["hit"].append(hit.reshape(B, samples).mean(1))
        acc["any"].append(hit.reshape(B, samples).max(1).values)
        acc["rot"].append(er)
        acc["pos"].append(ep)
    net.train()
    c = {k: torch.cat(v) for k, v in acc.items()}
    return {"gen_hit": c["hit"].mean().item(), "gen_any": c["any"].mean().item(), "gen_hit_from_over_45deg": c["hit_far"].mean().item(),
            "gen_rot_p50": c["rot"].median().item(), "gen_pos_p50": c["pos"].median().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="flow/data/train")
    ap.add_argument("--run", default="flow/runs/flow")
    ap.add_argument("--table", default="flow/data/table.npy")
    ap.add_argument("--make-table", action="store_true", help="write the closest-point table of the model and stop (needs open3d)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--act", default="gelu", choices=list(model.ACTIVATIONS), help="activation (relu: the fastest in TensorRT)")
    ap.add_argument("--val", type=int, default=512, help="held-out scenes (the first of the data)")
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.make_table:
        from . import scene
        np.save(a.table, model.build_table(scene.Rig()))
        print(a.table)
        return
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
    net = model.FlowNet(a.width, a.act).to(dev)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, device {dev}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    log = open(run / "log.jsonl", "a")
    t0 = time.time()
    acc = np.zeros(3)
    for step in range(1, a.steps + 1):
        idx = torch.randint(a.val, S, (a.batch,))
        P, cam, lab, ok, q, p, q1, p1 = make_batch(d, idx, dev)
        qb, pb = model.sample_source(q1, p1)
        t = torch.rand(a.batch, device=dev) * model.T_END
        qt, pt, target = model.fm_pair(q, p, qb, pb, t)
        f, _ = model.field(net, table, P, cam, qt, pt, t)
        l_fm = (f - target).pow(2).sum(-1).mean()
        if step % 4 == 0:       # the object / clutter output is trained at the data pose
            seg0 = model.field(net, table, P, cam, q, p, 0.0)[1]
            l_seg = (F.binary_cross_entropy_with_logits(seg0, lab, reduction="none") * ok).sum() / ok.sum().clamp_min(1)
        else:
            l_seg = l_fm * 0
        opt.zero_grad(set_to_none=True)
        (l_fm + 0.2 * l_seg).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        sched.step()
        acc += [l_fm.item(), l_seg.item() * 4, 1]
        if step % a.log_every == 0:
            r = {"step": step, "min": (time.time() - t0) / 60, "fm_loss": acc[0] / acc[2], "l_seg": acc[1] / acc[2]}
            acc[:] = 0
            if step % a.val_every == 0 or step == a.steps:
                r.update(validate(net, table, d, val_idx, dev))
                torch.save({"model": net.state_dict(), "width": a.width, "step": step, "config": vars(a)}, run / "model.pt")
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
