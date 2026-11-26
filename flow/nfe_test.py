"""Accuracy against the number of network evaluations per trajectory, for the solvers of rnode.generate.
   python -m flow.nfe_test --model flow/runs/rnode/model_fm.pt [--weights ema_0.999]
Held-out scenes, 8 base samples each; share within 10 mm / 10 degrees and the median errors."""
import argparse

import numpy as np
import torch

from . import geom, rnode, train_rnode as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", default="model")
    ap.add_argument("--data", default="flow/data/train")
    ap.add_argument("--table", default="flow/data/rnode_table.npy")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.model, map_location=dev)
    cfg = ck["config"]
    rnode.DEQ_ROT, rnode.DEQ_POS = cfg["deq_rot"], cfg["deq_pos"]
    rnode.SIGMA_ROT, rnode.SIGMA_POS = cfg.get("sigma_rot", rnode.SIGMA_ROT), cfg.get("sigma_pos", rnode.SIGMA_POS)
    net = rnode.net_from_checkpoint(ck, dev, a.weights)
    table = torch.from_numpy(np.load(a.table)).to(dev)
    d = T.load(a.data.split(",")[0])
    print("solver   steps evals  hit    rot p50  pos p50")
    for solver, steps in (("euler", 1), ("euler", 2), ("midpoint", 1), ("heun", 1), ("euler", 4), ("midpoint", 2), ("heun", 2), ("rk4", 1),
                          ("midpoint", 3), ("euler", 8), ("midpoint", 4), ("heun", 4), ("rk4", 2), ("rk4", 4), ("rk4", 8)):
        g = torch.Generator(device=dev).manual_seed(13)
        er, ep = [], []
        for i in range(0, 512, 32):
            P, cam, _, _, q, p, q1, p1 = T.make_batch(d, torch.arange(i, i + 32), dev, g)
            rep = torch.arange(32, device=dev).repeat_interleave(8)
            qb, pb = rnode.sample_base(q1[rep], p1[rep], g)
            qg, pg = rnode.generate(net, table, P[rep], cam[rep], qb, pb, steps, solver=solver)
            er.append(geom.qlog(geom.qmul(geom.qconj(qg), q1[rep])).norm(dim=-1).rad2deg())
            ep.append((pg - p1[rep]).norm(dim=-1) * 1000)
        er, ep = torch.cat(er), torch.cat(ep)
        print(f"{solver:8s} {steps:5d} {steps * rnode.SOLVER_EVALS[solver]:5d}  {((er < 10) & (ep < 10)).float().mean().item():.3f}  {er.median().item():6.1f}  {ep.median().item():6.1f}", flush=True)


if __name__ == "__main__":
    main()
