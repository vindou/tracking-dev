"""Training of the flow network on the synthetic shards (flow.make_data train).

    python -m flow.train --data flow/data/train --run flow/runs/a [--steps 40000] [--batch 128] [--device mps]"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import geom, sampler
from .model import FlowNet


def load(data, have=0):
    """All shards of the directory (None if there are not more than `have`). The generator may still be writing new
    ones; the training loop calls this again at every checkpoint."""
    files = sorted(glob.glob(f"{data}/shard_*.npz"))
    if len(files) <= have:
        return None
    z = [np.load(f) for f in files]
    d = {k: torch.from_numpy(np.concatenate([x[k] for x in z])) for k in ("pts", "flag", "n", "q")}
    print(f"{len(files)} shards, {len(d['n'])} scenes, {d['pts'].numel() * 2 / 1e9:.2f} GB", flush=True)
    d["files"] = len(files)
    return d


def batch(d, idx, dev):
    pts = d["pts"][idx].to(dev).float()            # relative to the true centroid: p_1 = 0
    flag = d["flag"][idx].to(dev)
    n = d["n"][idx].to(dev)
    valid = torch.arange(pts.shape[1], device=dev)[None] < n[:, None]
    return pts, valid, (flag & 1).long(), ((flag >> 1) & 1).float(), d["q"][idx].to(dev)


@torch.no_grad()
def validate(net, d, idx, dev):
    """Held-out scenes. Position stage: uniform rotation, 100 mm off, rotation held. Rotation flow: starts 20, 45 and
    65 degrees and 20 mm from the truth; share that ends within 10 degrees and 10 mm (inside the basin of ICP)."""
    net.eval()
    g = torch.Generator(device=dev).manual_seed(7)
    out = {}
    acc = {k: [] for k in ("pos", 20, 45, 65)}
    for i in range(0, len(idx), 64):
        pts, valid, cam, _, q1 = batch(d, idx[i:i + 64], dev)
        B = len(q1)
        ball = lambda rad: (lambda u: u / u.norm(dim=-1, keepdim=True) * rad)(torch.randn(B, 3, device=dev, generator=g))
        q, p, *_ = sampler.integrate(net, pts, valid, cam, geom.qrand(B, dev, g), ball(0.10), steps=4, refine=1, generator=g,
                                     position_only=True)
        acc["pos"].append(p.norm(dim=-1).cpu() * 1000)
        for ang in (20, 45, 65):
            q0 = geom.qmul(q1, geom.qexp(ball(np.radians(ang))))
            q, p, *_ = sampler.integrate(net, pts, valid, cam, q0, ball(0.02), generator=g)
            er = geom.qlog(geom.qmul(geom.qconj(q), q1)).norm(dim=-1).rad2deg().cpu()
            acc[ang].append(((er < 10) & (p.norm(dim=-1).cpu() < 0.010)).float())
    net.train()
    out["pos_stage_mm_p50"] = torch.cat(acc["pos"]).median().item()
    out["pos_stage_in_20mm"] = (torch.cat(acc["pos"]) < 20).float().mean().item()
    for ang in (20, 45, 65):
        out[f"from_{ang}deg_in_10_10"] = torch.cat(acc[ang]).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="flow/data/train")
    ap.add_argument("--run", default="flow/runs/a")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--val", type=int, default=512)
    ap.add_argument("--init", default="")
    a = ap.parse_args()
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(0)
    d = load(a.data)
    S = len(d["n"])
    val_idx = torch.arange(a.val)            # the first scenes of the first shard are held out
    net = FlowNet(a.width).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["model"])
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, device {dev}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.03)
    log = open(run / "log.jsonl", "a")
    t0 = time.time()
    acc = np.zeros(3)
    for step in range(1, a.steps + 1):
        idx = torch.randint(a.val, S, (a.batch,))
        pts, valid, cam, lab, q1 = batch(d, idx, dev)
        t = torch.rand(a.batch, device=dev)
        t = torch.where(torch.rand(a.batch, device=dev) < 0.15, torch.ones_like(t), t)   # extra weight on the end point
        qt, pt, target, rot_mask = sampler.training_pair(q1, torch.zeros(a.batch, 3, device=dev), t)
        x, pick, ok = sampler.crop_canonical(pts, valid, cam, qt, pt)
        xi, seg = net(x, t)
        l_rot, l_pos = sampler.flow_loss(xi, target, rot_mask)
        l_seg = (F.binary_cross_entropy_with_logits(seg, torch.gather(lab, 1, pick), reduction="none") * ok).sum() / ok.sum().clamp_min(1)
        loss = l_rot + l_pos + 0.2 * l_seg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        sched.step()
        acc += [l_rot.item(), l_pos.item(), l_seg.item()]
        if step % 250 == 0:
            r = {"step": step, "l_rot": acc[0] / 250, "l_pos": acc[1] / 250, "l_seg": acc[2] / 250, "min": (time.time() - t0) / 60}
            acc[:] = 0
            if step % 1000 == 0 or step == a.steps:
                r.update(validate(net, d, val_idx, dev))
                torch.save({"model": net.state_dict(), "width": a.width, "step": step}, run / "model.pt")
                new = load(a.data, d["files"])
                if new is not None:
                    d, S = new, len(new["n"])
                r["scenes"] = S
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
