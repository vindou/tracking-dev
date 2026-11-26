"""Diagnostics of a checkpoint of train_rnode, for the ablations of why RNODE does not improve the model.
   python -m flow.ablate_rnode --model flow/runs/abl/s1/model_fm.pt --out flow/runs/abl/s1_fm.json

  nll        likelihood loss of held-out data (nats per dimension) with the exact trace at 4 and 16 solver steps, its
             two parts, and the floor: the entropy of the dequantization noise.
  forward    where the data lands in the base: spread of the normalized end point (1 = fills the base) and its
             correlation with the normalized dequantization offset (1 = the map resolves the offset).
  generate   share of base samples within 10 mm / 10 degrees of the truth after 1, 2, 4, 8 solver steps; straightness of
             the forward paths (squared displacement / kinetic energy, 1 = straight at constant speed).
  field      error of the field against the geodesic velocity (the flow-matching target) by time: t near 0 is at the
             truth, t near 1 in the base.
  gradient   cosine between the gradients of independent batches, for each term of the RNODE loss."""
import argparse
import json
import math

import numpy as np
import torch

from . import geom, rnode, train_rnode as T
from .model import POS_SCALE


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    cfg = ck["config"]
    rnode.DEQ_ROT, rnode.DEQ_POS = cfg["deq_rot"], cfg["deq_pos"]
    rnode.SIGMA_ROT, rnode.SIGMA_POS = cfg.get("sigma_rot", rnode.SIGMA_ROT), cfg.get("sigma_pos", rnode.SIGMA_POS)
    return rnode.net_from_checkpoint(ck, dev), cfg


def batches(d, n, B, dev, g):
    for i in range(0, n, B):
        yield T.make_batch(d, torch.arange(i, min(n, i + B)), dev, g)


def nll_section(net, table, d, dev):
    out = {"floor": 0.5 * math.log(2 * math.pi * math.e) + 0.5 * (math.log(rnode.DEQ_ROT) + math.log(rnode.DEQ_POS))}
    for steps, B in ((4, 32), (16, 16)):
        g = torch.Generator(device=dev).manual_seed(11)
        base, div = [], []
        for P, cam, _, _, q, p, q1, p1 in batches(d, 256, B, dev, g):
            qT, pT, aug, _ = rnode.integrate(net, table, P, cam, q, p, 0.0, rnode.T_END, steps, rnode.EXACT)
            base.append(-rnode.log_base(qT, pT, q1, p1).detach() / 6)
            div.append(-aug[:, 0].detach() / 6)
        base, div = torch.cat(base), torch.cat(div)
        out[f"exact_{steps}"] = dict(nll=(base + div).mean().item(), base=base.mean().item(), div=div.mean().item())
    return out


@torch.no_grad()
def forward_section(net, table, d, dev):
    g = torch.Generator(device=dev).manual_seed(12)
    a, b = [], []
    for _ in range(2):
        for P, cam, _, _, q, p, q1, p1 in batches(d, 512, 64, dev, g):
            qT, pT, _, _ = rnode.integrate(net, table, P, cam, q, p, 0.0, rnode.T_END, 4)
            a.append(torch.cat([geom.qlog(geom.qmul(geom.qconj(q1), q)) / rnode.DEQ_ROT, (p - p1) / (rnode.DEQ_POS * POS_SCALE)], -1))
            b.append(torch.cat([geom.qlog(geom.qmul(geom.qconj(q1), qT)) / rnode.SIGMA_ROT, (pT - p1) / (rnode.SIGMA_POS * POS_SCALE)], -1))
    a, b = torch.cat(a), torch.cat(b)
    corr = [torch.corrcoef(torch.stack([a[:, i], b[:, i]]))[0, 1].item() for i in range(6)]
    return dict(end_std_rot=b[:, :3].std(0).mean().item(), end_std_pos=b[:, 3:].std(0).mean().item(),
                end_mean_norm=b.mean(0).norm().item(), corr_rot=float(np.mean(corr[:3])), corr_pos=float(np.mean(corr[3:])))


def generate_section(net, table, d, dev):
    out = {}
    for steps in (1, 2, 4, 8):
        g = torch.Generator(device=dev).manual_seed(13)
        er, ep = [], []
        with torch.no_grad():
            for P, cam, _, _, q, p, q1, p1 in batches(d, 512, 64, dev, g):
                rep = torch.arange(len(q), device=dev).repeat_interleave(8)
                qb, pb = rnode.sample_base(q1[rep], p1[rep], g)
                qg, pg = rnode.generate(net, table, P[rep], cam[rep], qb, pb, steps)
                er.append(geom.qlog(geom.qmul(geom.qconj(qg), q1[rep])).norm(dim=-1).rad2deg())
                ep.append((pg - p1[rep]).norm(dim=-1) * 1000)
        er, ep = torch.cat(er), torch.cat(ep)
        out[f"steps_{steps}"] = dict(hit=((er < 10) & (ep < 10)).float().mean().item(), rot_p50=er.median().item(), pos_p50=ep.median().item())
    g = torch.Generator(device=dev).manual_seed(14)
    E, D = [], []
    for P, cam, _, _, q, p, q1, p1 in batches(d, 256, 32, dev, g):
        eps = torch.randn(len(q), 6, device=dev, generator=g)
        qT, pT, aug, _ = rnode.integrate(net, table, P, cam, q, p, 0.0, rnode.T_END, 4, eps)
        E.append(aug[:, 1].detach())
        D.append((geom.qlog(geom.qmul(geom.qconj(q), qT)).pow(2).sum(-1) + ((pT - p) / POS_SCALE).pow(2).sum(-1)).detach())
    out["kinetic_energy"] = torch.cat(E).mean().item()
    out["straightness"] = (torch.cat(D).mean() / torch.cat(E).mean()).item()
    return out


@torch.no_grad()
def field_section(net, table, d, dev):
    g = torch.Generator(device=dev).manual_seed(15)
    t_all, e_all, n_all = [], [], []
    for _ in range(4):
        for P, cam, _, _, q, p, q1, p1 in batches(d, 512, 128, dev, g):
            qb, pb = rnode.sample_base(q1, p1, g)
            t = torch.rand(len(q), device=dev, generator=g) * rnode.T_END
            qt, pt, target = rnode.fm_pair(q, p, qb, pb, t)
            f = rnode.field(net, table, P, cam, qt, pt, t)[0]
            t_all.append(t), e_all.append((f - target).pow(2).sum(-1)), n_all.append(target.pow(2).sum(-1))
    t, e, n = torch.cat(t_all), torch.cat(e_all), torch.cat(n_all)
    out = {}
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        m = (t >= lo) & (t < hi)
        out[f"t_{lo:.2f}_{hi:.2f}"] = (e[m].mean() / n[m].mean()).item()
    out["all"] = (e.mean() / n.mean()).item()
    return out


def gradient_section(net, table, d, dev, K=10):
    params = list(net.parameters())
    S = len(d["n"])
    out = {}
    torch.manual_seed(0)
    for kind, B in (("hutchinson", 32), ("exact", 8)):
        gs = []
        for _ in range(K):
            P, cam, _, _, q, p, q1, p1 = T.make_batch(d, torch.randint(512, S, (B,)), dev)
            eps = rnode.EXACT if kind == "exact" else torch.randn(B, 6, device=dev)
            qT, pT, aug, _ = rnode.integrate(net, table, P, cam, q, p, 0.0, rnode.T_END, 4, eps, True)
            terms = {"base": -rnode.log_base(qT, pT, q1, p1).mean() / 6, "div": -aug[:, 0].mean() / 6}
            terms["total"] = terms["base"] + terms["div"] + 0.01 * (aug[:, 1].mean() + aug[:, 2].mean()) / 6
            names, r = list(terms), {}
            for i, k in enumerate(names):
                gr = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1, allow_unused=True)
                r[k] = torch.cat([(x if x is not None else torch.zeros_like(w)).flatten() for x, w in zip(gr, params)])
            gs.append(r)
        for k in gs[0]:
            G = torch.stack([g[k] for g in gs])
            C = torch.nn.functional.normalize(G, dim=1)
            cos = (C @ C.T)[~torch.eye(K, dtype=torch.bool, device=dev)].mean().item()
            out[f"{kind}_b{B}_{k}"] = dict(cosine=cos, norm=G.norm(dim=1).mean().item(),
                                           samples_for_cosine_half=B * (1 - cos) / max(cos, 1e-6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="flow/data/train")
    ap.add_argument("--table", default="flow/data/rnode_table.npy")
    ap.add_argument("--skip-gradient", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    net, cfg = load(a.model, dev)
    table = torch.from_numpy(np.load(a.table)).to(dev)
    d = T.load(a.data)
    r = dict(model=a.model, smooth=rnode.SMOOTH, deq_rot=rnode.DEQ_ROT, deq_pos=rnode.DEQ_POS)
    for name, fn in (("nll", nll_section), ("forward", forward_section), ("generate", generate_section), ("field", field_section)):
        r[name] = fn(net, table, d, dev)
        print(name, json.dumps(r[name]), flush=True)
    if not a.skip_gradient:
        r["gradient"] = gradient_section(net, table, d, dev)
        print("gradient", json.dumps(r["gradient"]), flush=True)
    json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
