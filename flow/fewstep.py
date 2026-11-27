"""Experiments: the pose stage in fewer than 4 network evaluations, starting from the frozen model. Nothing here is
used by the deployed path; flow.propose takes the result with --pose-run.

    python -m flow.fewstep deriv   --run flow/runs/few_deriv   [--steps 6000]
    python -m flow.fewstep distill --run flow/runs/few_distill2 --pose-steps 2 [--steps 8000]
    python -m flow.propose --eval ... --pose-run flow/runs/few_deriv/model.pt --solver-steps 2

deriv    A second head on the pooled features of the frozen network predicts the derivative of the field along the
         flow path, d/dt f(z(t), t). The target is a Jacobian-vector product of the frozen network in the direction
         (dz/dt, 1) (forward-mode automatic differentiation), at states of its own Euler x 8 paths. Only the new head
         is trained. A second-order step then costs one evaluation:
           rotation   q <- q exp(h w + h^2/2 w')          position   p <- p + R (h v + h^2/2 (w x v + v')) POS_SCALE
         (w, v: body-frame angular and linear velocity; w', v': the head's output).
distill  A student, initialised from the frozen weights, is trained so that its end pose after --pose-steps Euler
         steps equals the end pose of the frozen network after 8 (the teacher). Loss: squared geodesic distance on
         SO(3) plus the squared position difference in POS_SCALE units. Data and starts as flow/train.py.
Both are for the pose stage; the position stage keeps the frozen network."""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import geom, model, train

TEACHER_STEPS = 8


def pooled(net, x, t, seed_b):
    """The input of the network's head (FlowNet.forward up to it)."""
    N = x.shape[1]
    te = t[:, None] * net.freq[None]
    c = net.cond(torch.cat([torch.sin(te), torch.cos(te), seed_b], -1))
    f = net.local(x)
    g1 = f.max(dim=1).values
    h = net.mid(torch.cat([f, g1[:, None].expand(-1, N, -1), c[:, None].expand(-1, N, -1)], -1))
    return torch.cat([h.max(dim=1).values, h.mean(dim=1), c], -1)


def deriv_head(net):
    d = nn.Sequential(model.mlp([net.head[0][0].in_features, 512, 512]), nn.Linear(512, model.DIM))
    nn.init.zeros_(d[-1].weight)
    nn.init.zeros_(d[-1].bias)
    return d


def field2(net, dhead, table, pts, cam, q, p, t):
    """The field and the head's estimate of its derivative along the path, from one pass through the network."""
    R = geom.qmat(q)
    xm = (pts - p[:, None]) @ R
    seed_b = -(p[:, None] @ R).squeeze(1) / model.POS_SCALE
    if not torch.is_tensor(t):
        t = torch.full((q.shape[0],), float(t), device=q.device, dtype=q.dtype)
    z = pooled(net, model.features(table, xm, cam), t, seed_b)
    return net.head(z), dhead(z)


def rates(q, f):
    """Body-frame angular velocity and rig-frame linear velocity (m per unit time) of the field value f at q."""
    return f[:, :3], (geom.qmat(q) @ f[:, 3:, None]).squeeze(-1) * model.POS_SCALE


def field_dt(net, table, pts, cam, q, p, t):
    """f and d/dt f(z(t), t) along the flow path through (q, p) at time t [B], by a Jacobian-vector product."""
    with torch.no_grad():
        f0 = model.field(net, table, pts, cam, q, p, t)[0]
        w, v = rates(q, f0)

    def fn(u, dp, tt):
        return model.field(net, table, pts, cam, geom.qmul(q, model.qexp(u)), p + dp, tt)[0]
    zero = torch.zeros_like(w)
    f, df = torch.func.jvp(fn, (zero, zero, t), (w, v, torch.ones_like(t)))
    return f.detach(), df.detach()


def taylor_step(f, d, q, p, h):
    """The second-order step of size h (negative: towards the data) with the field f and its path derivative d."""
    w, v, dw, dv = f[:, :3], f[:, 3:], d[:, :3], d[:, 3:]
    q2 = geom.qmul(q, model.qexp(h * w + 0.5 * h * h * dw))
    lin = h * v + 0.5 * h * h * (torch.cross(w, v, dim=-1) + dv)
    return q2 / q2.norm(dim=-1, keepdim=True), p + (geom.qmat(q) @ lin[..., None]).squeeze(-1) * model.POS_SCALE


@torch.no_grad()
def solve_taylor(net, dhead, table, pts, cam, q, p, steps):
    h = -model.T_END / steps
    for i in range(steps):
        f, d = field2(net, dhead, table, pts, cam, q, p, model.T_END + i * h)
        q, p = taylor_step(f, d, q, p, h)
    return q, p


def pose_error(q, p, q1, p1):
    """Geodesic distance on SO(3) (rad) and position difference (POS_SCALE units), differentiable at zero."""
    d = geom.qmul(geom.qconj(q), q1)
    ang = 2.0 * torch.atan2(torch.sqrt(d[:, 1:].pow(2).sum(-1) + 1e-12), d[:, 0].abs())
    return ang, (p - p1).norm(dim=-1) / model.POS_SCALE


def load_pose(path, net, table, dev, steps):
    """The pose-stage solver of a run of this file: (pts, cam, q) -> q, p (starts at p = 0), and its name."""
    ck = torch.load(path, map_location=dev)
    if ck["kind"] == "deriv":
        dhead = deriv_head(net).to(dev)
        dhead.load_state_dict(ck["dhead"])
        dhead.eval()
        return (lambda X, c, q: solve_taylor(net, dhead, table, X, c, q, torch.zeros_like(q[:, :3]), steps)), f"second-order x {steps}"
    student = copy.deepcopy(net)
    student.load_state_dict(ck["model"])
    student.eval()
    assert steps == ck["pose_steps"], f"this student was trained for {ck['pose_steps']} pose steps"
    return (lambda X, c, q: model.generate(student, table, X, c, q, torch.zeros_like(q[:, :3]), steps, solver="euler")), f"distilled, Euler x {steps}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["deriv", "distill"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="flow/runs/flow/model.pt")
    ap.add_argument("--data", default="flow/data/train")
    ap.add_argument("--table", default="flow/data/table.npy")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--pose-steps", type=int, default=2, help="distill: Euler steps of the student")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.0, help="0: 1e-3 for the new head, 1e-4 for the student")
    ap.add_argument("--val", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    table = torch.from_numpy(np.load(a.table)).to(dev)
    d = train.load(a.data)
    S = len(d["n"])
    net = model.load(a.model, dev)
    for prm in net.parameters():
        prm.requires_grad_(False)
    if a.kind == "deriv":
        new = deriv_head(net).to(dev)
        params, lr = list(new.parameters()), a.lr or 1e-3
    else:
        new = copy.deepcopy(net)
        for prm in new.parameters():
            prm.requires_grad_(True)
        params, lr = list(new.parameters()), a.lr or 1e-4
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    log = open(run / "log.jsonl", "a")
    t0 = time.time()
    acc = np.zeros(4)
    h8 = -model.T_END / TEACHER_STEPS
    for step in range(1, a.steps + 1):
        idx = torch.randint(a.val, S, (a.batch,))
        P, cam, _, _, _, _, q1, p1 = train.make_batch(d, idx, dev)
        qb, pb = model.sample_source(q1, p1)
        if a.kind == "deriv":
            # a state of the frozen network's Euler x 8 path, at a random time of that path
            n = int(torch.randint(0, TEACHER_STEPS, (1,)))
            with torch.no_grad():
                q, p = (qb, pb) if n == 0 else model.integrate(net, table, P, cam, qb, pb, model.T_END, model.T_END + n * h8, n, solver="euler")
            t = torch.full((a.batch,), model.T_END + n * h8, device=dev)
            _, target = field_dt(net, table, P, cam, q, p, t)
            pred = field2(net, new, table, P, cam, q, p, t)[1]
            loss = nn.functional.smooth_l1_loss(pred, target)
            with torch.no_grad():
                acc += [loss.item(), (pred - target).pow(2).mean().item(), target.pow(2).mean().item(), 1]
        else:
            with torch.no_grad():
                qt, pt = model.integrate(net, table, P, cam, qb, pb, model.T_END, 0.0, TEACHER_STEPS, solver="euler")
            qs, ps = model.integrate(new, table, P, cam, qb, pb, model.T_END, 0.0, a.pose_steps, solver="euler")
            ang, pos = pose_error(qs, ps, qt, pt)
            loss = (ang.pow(2) + pos.pow(2)).mean()
            acc += [loss.item(), ang.mean().item() * 180 / math.pi, pos.mean().item() * model.POS_SCALE * 1e3, 1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == a.steps:
            m = acc / acc[3]
            r = {"step": step, "min": (time.time() - t0) / 60, "loss": m[0]}
            r.update({"unexplained": m[1] / max(m[2], 1e-12), "target_rms": math.sqrt(m[2])} if a.kind == "deriv" else {"rot_deg": m[1], "pos_mm": m[2]})
            acc[:] = 0
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
    ck = {"kind": a.kind, "config": vars(a), "minutes": (time.time() - t0) / 60}
    ck.update({"dhead": new.state_dict()} if a.kind == "deriv" else {"model": new.state_dict(), "pose_steps": a.pose_steps})
    torch.save(ck, run / "model.pt")


if __name__ == "__main__":
    main()
