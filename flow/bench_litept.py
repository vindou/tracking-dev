"""Cost of the encoders: training steps per second, memory, a profile of one training step, time of one solve.

    python -m flow.bench_litept caps  [--batches 60] [--seed 0]          cells per cloud at the five stages (for litept.CAPS)
    python -m flow.bench_litept profile --encoder litept_t [--impl sparse|fixed]
    python -m flow.bench_litept speed   --encoder E [--impl I] [--select] [--compile] [--amp] [--steps 200]
    python -m flow.bench_litept infer   --encoder E [--impl I] [--compile] [--amp]
Every mode appends one JSON line to --out. --mem-gb limits the GPU memory of this process (the GPU is shared).

The training step is that of the flow-matching phase of train_rnode (same batch, loss, every fourth step the second
pass for the object / clutter output, clipping, Adam, two weight averages)."""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import geom, litept, rnode
from .train_rnode import load, make_batch


def tick(name):
    litept._tick(name)


class Trainer:
    def __init__(self, a, dev):
        self.a, self.dev = a, dev
        self.table = torch.from_numpy(np.load(a.table)).to(dev)
        self.d = load(a.data)
        litept.IMPL = a.impl
        litept.AMP = a.amp
        litept.SELECT = a.select
        torch.manual_seed(0)
        self.net = rnode.make_net(1, a.encoder).to(dev)
        torch.nn.init.normal_(self.net.head[-1].weight, std=0.02)       # not the zero field of a new model
        if a.compile:
            self.net.core_fn = torch.compile(self.net._core)
        print(f"{a.encoder} {getattr(self.net, 'impl', '')} select {litept.SELECT} amp {litept.AMP} compile {a.compile}", flush=True)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.ema = [{k: v.detach().clone() for k, v in self.net.named_parameters()} for _ in range(2)]
        self.params = sum(p.numel() for p in self.net.parameters())

    def step(self, i):
        a, dev, net = self.a, self.dev, self.net
        B = a.batch
        idx = torch.randint(512, len(self.d["n"]), (B,))
        P, cam, lab, ok, q, p, q1, p1 = make_batch(self.d, idx, dev)
        qb, pb = rnode.sample_base(q1, p1)
        t = torch.rand(B, device=dev)
        qt, pt, target = rnode.fm_pair(q, p, qb, pb, t)
        tick("batch: crop of the scene, poses")
        f, _, _, _ = rnode.field(net, self.table, P, cam, qt, pt, t)
        loss = (f - target).pow(2).sum(-1).mean()
        if i % 4 == 0:
            seg0 = rnode.field(net, self.table, P, cam, q, p, 0.0)[3]
            loss = loss + 0.2 * (F.binary_cross_entropy_with_logits(seg0, lab, reduction="none") * ok).sum() / ok.sum().clamp_min(1)
        tick("head")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        tick("backward (all stages)")
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        self.opt.step()
        with torch.no_grad():
            for e in self.ema:
                for k, v in net.named_parameters():
                    e[k].lerp_(v, 0.001)
        tick("clipping, Adam, weight averages")
        return loss


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def mem(dev):
    if dev.type != "cuda":
        return {}
    return {"peak_allocated_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}


def speed(a, dev):
    tr = Trainer(a, dev)
    for i in range(a.warmup):
        tr.step(i)
    sync(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(a.steps):
        loss = tr.step(i)
    sync(dev)
    dt = time.perf_counter() - t0
    return {"steps_per_s": a.steps / dt, "ms_per_step": 1000 * dt / a.steps, "params_M": tr.params / 1e6, "loss": loss.item(), **mem(dev)}


def profile(a, dev):
    tr = Trainer(a, dev)
    for i in range(a.warmup):
        tr.step(i)
    n = 40
    litept.PROFILE = litept.Prof(dev)
    for i in range(n):
        tr.step(i)
    acc = litept.PROFILE.acc
    litept.PROFILE = None
    total = sum(acc.values())
    out = {"ms_per_step": 1000 * total / n, "sections_ms": {k: 1000 * v / n for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}}
    for k, v in out["sections_ms"].items():
        print(f"{v:9.1f} ms  {100 * v * n / 1000 / total:5.1f} %  {k}", flush=True)
    # operators (forward and backward) by GPU time
    from torch.profiler import ProfilerActivity, profile as tprof
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    with tprof(activities=acts) as prof:
        for i in range(8):
            tr.step(i)
        sync(dev)
    key = "self_device_time_total" if dev.type == "cuda" else "self_cpu_time_total"
    ev = sorted(prof.key_averages(), key=lambda e: -getattr(e, key))
    tot = sum(getattr(e, key) for e in ev) or 1
    out["ops"] = [{"op": e.key, "share": getattr(e, key) / tot, "ms_per_step": getattr(e, key) / 8e3, "calls_per_step": e.count / 8} for e in ev[:22]]
    for o in out["ops"]:
        print(f"{o['ms_per_step']:9.1f} ms  {100 * o['share']:5.1f} %  {o['calls_per_step']:7.1f} calls  {o['op']}", flush=True)
    out.update(mem(dev))
    return out


@torch.no_grad()
def infer(a, dev):
    """One solve as in stage 2 of infer_rnode: 128 hypotheses on the points of one crop, RK4 x 4 (16 evaluations)."""
    tr = Trainer(a, dev)
    tr.net.eval()
    K = 128
    P, cam, *_ = make_batch(tr.d, torch.arange(1), dev)
    times = []
    for r in range(a.warmup_solves + 10):
        g = torch.Generator(device=dev).manual_seed(r)
        rep = torch.zeros(K, dtype=torch.long, device=dev)
        qb = geom.qrand(K, dev, g)
        sync(dev)
        t0 = time.perf_counter()
        X, c = P[rep], cam[rep]                     # a new tensor per solve: the tables of the rig-frame variant are built once per solve
        q, p = rnode.generate(tr.net, tr.table, X, c, qb, torch.zeros(K, 3, device=dev), 4)
        sync(dev)
        times.append(time.perf_counter() - t0)
    t = np.array(times[a.warmup_solves:]) * 1000
    return {"solve_ms_p50": float(np.median(t)), "solve_ms_min": float(t.min()), "ms_per_evaluation": float(np.median(t)) / 16, **mem(dev)}


@torch.no_grad()
def caps(a, dev):
    """Cells per cloud at each stage over training batches, for the body-frame cells (at the pose of the sample on
    the flow-matching path, at the data pose, and at a uniformly random rotation at the seed, as the starts of the
    inference) and for the rig-frame cells. With --check: how often litept.CAPS would be exceeded."""
    torch.manual_seed(a.seed)
    d = load(a.data)
    B = 512
    cnt = {"body": [], "rig": []}

    def count(coord):
        keys, _, n = litept._rows(litept._key(coord), 512)
        out = [n]
        for _ in range(4):
            keys, _, n = litept._rows(torch.where(keys < litept.SENT, litept._key(litept._unkey(keys) // 2), keys), 512)
            out.append(n)
        return torch.stack(out, 1)

    for _ in range(a.batches):
        idx = torch.randint(0, len(d["n"]), (B,))
        P, cam, lab, ok, q, p, q1, p1 = make_batch(d, idx, dev)
        qb, pb = rnode.sample_base(q1, p1)
        qt, pt, _ = rnode.fm_pair(q, p, qb, pb, torch.rand(B, device=dev))
        for qq, pp in ((qt, pt), (q, p), (geom.qrand(B, dev), torch.zeros(B, 3, device=dev))):
            cnt["body"].append(count(litept.LitePT.cells((P - pp[:, None]) @ geom.qmat(qq))))
        cnt["rig"].append(count(litept.LitePT.cells(P)))
    out = {}
    for k, v in cnt.items():
        v = torch.cat(v).float()
        out[k] = {"clouds": len(v), "mean": v.mean(0).tolist(), "p99": v.quantile(0.99, dim=0).tolist(), "max": v.max(0).values.tolist(),
                  "caps": list(litept.CAPS[k]),
                  "share_over_caps_per_stage": (v > torch.tensor(litept.CAPS[k])).float().mean(0).tolist(),
                  "share_of_clouds_over_any_cap": (v > torch.tensor(litept.CAPS[k])).any(1).float().mean().item()}
        print(k, json.dumps(out[k]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("caps", "profile", "speed", "infer"))
    ap.add_argument("--encoder", default="litept_t")
    ap.add_argument("--impl", default="sparse", choices=("sparse", "fixed"))
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--select", action="store_true", help="litept.SELECT: index_select in place of f[idx]")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--warmup-solves", type=int, default=3)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=0.0)
    ap.add_argument("--data", default="flow/data/train")
    ap.add_argument("--table", default="flow/data/rnode_table.npy")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if a.mem_gb > 0:
            torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    r = {"mode": a.mode, "encoder": a.encoder, "impl": a.impl if a.encoder.startswith("litept") else "",
         "compile": a.compile, "amp": a.amp, "select": a.select, "batch": a.batch, "mem_gb": a.mem_gb}
    try:
        r.update({"caps": caps, "profile": profile, "speed": speed, "infer": infer}[a.mode](a, dev))
    except torch.OutOfMemoryError as e:
        r["error"] = "out of memory: " + str(e)[:160]
    print(json.dumps({k: v for k, v in r.items() if k not in ("ops", "sections_ms")}), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
