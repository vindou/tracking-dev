"""ONNX graphs of the frozen model for TensorRT on the Jetson, with inputs and reference outputs of real scenes.

    python -m flow.export --out flow/runs/export [--scenes 3,700,1500]

Fixed shapes (K1 = 32 seeds, K2 = 4 x 32 hypotheses, N = 512 points); the closest-point table is a constant of
each graph.
  stage1   pts [K1, N, 3], cam [K1, N], q [K1, 4]         -> q, p, seg [K1, N]   the solve from p = 0 (4 RK4 steps, 16
                                                                               evaluations) and the object logits
  stage2   pts [K2, N, 3], cam [K2, N], q [K2, 4]         -> q, p               the solve
  refine   pts [K2, N, 3], q, p                           -> q, p               verifier.icp (6 steps), float32
  score    pts [K2, N, 3], cam, q, p                      -> logit [K2]         the verifier
These four are the model as it is written, with 4 Runge-Kutta steps. The same call writes stage1e (4 Euler steps)
and stage2e (8 Euler steps) with the network rearranged for TensorRT (FastNet; same weights, same outputs; stage1f /
stage2f: the rearranged network with Runge-Kutta, stage2e1: one position); flow/trt_eval.py runs the evaluation set
through them on the Jetson.
The deployed graphs are those of --deploy (deploy(): the solver steps of flow.propose, the verifier rearranged, graphs
for one position), exported with --act tanh_gelu and run in float16 by flow/live/trk_flow.cpp.
For every scene of --scenes, <out>/s<scene>/ holds the inputs of each graph as raw float32 files and the outputs of
the PyTorch code (ref_*.bin); flow/jetson_time.sh runs the graphs on them and flow.export --check compares."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import geom, model, propose, scene, verifier as V
from .make_data import load_eval

K1, CENTERS, ORIENT = 32, 4, 32
DEPLOYED = "convfoldgridpool"       # the arrangement of FastNet of the deployed graphs (stage1e, stage2e), run in float16


class Stage(nn.Module):
    def __init__(self, net, table, seg, steps=4, solver="rk4"):
        super().__init__()
        self.net, self.seg, self.steps, self.solver = net, seg, steps, solver
        self.register_buffer("table", table)

    def forward(self, pts, cam, q):
        q, p = model.integrate(self.net, self.table, pts, cam, q, torch.zeros_like(q[:, :3]), model.T_END, 0.0, self.steps, solver=self.solver)
        return (q, p, model.field(self.net, self.table, pts, cam, q, p, 0.0)[1]) if self.seg else (q, p)


class Refine(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, pts, q, p):
        return V.icp(self.table, pts, q, p, export=True)


class Score(nn.Module):
    def __init__(self, ver, table):
        super().__init__()
        self.ver = ver
        self.register_buffer("table", table)

    def forward(self, pts, cam, q, p):
        return self.ver(self.table, pts, cam, q, p)


class FastNet(nn.Module):
    """model.FlowNet with the same weights and the same outputs, arranged for TensorRT.
      fold    the first layer of the second per-point stage acts on [f, g1, c], of which g1 and c are the same for all
              points of a hypothesis: W [f, g1, c] + b = W_f f + (W_g g1 + W_c c + b). The second term is computed once
              per hypothesis; the per-point product has 192 inputs instead of 448 and no concatenation is left.
      layout  "lin": [B, N, C] with linear layers (as the model); "conv": [B, C, H, W] (H W = N, grid) with 1 x 1
              convolutions, so the reductions over the points need no change of layout.
      pool    "reduce": maximum / mean over the point axes; "pool": pooling layers with the kernel (H, W)."""

    def __init__(self, net, fold=True, layout="conv", pool="reduce", grid=None):
        super().__init__()
        self.fold, self.layout, self.pool, self.grid = fold, layout, pool, grid
        self.cond, self.head, self.gseg, self.act = net.cond, net.head, net.gseg, copy.deepcopy(net.local[1])     # the model's activation
        self.register_buffer("freq", net.freq)
        b = net.local[4].out_features

        def per_point(W, bias):
            m = (nn.Conv2d if layout == "conv" else nn.Linear)(W.shape[1], W.shape[0], *((1,) if layout == "conv" else ()), bias=bias is not None)
            m.weight.data = W.clone()[..., None, None] if layout == "conv" else W.clone()
            if bias is not None:
                m.bias.data = bias.clone()
            return m
        self.local = nn.ModuleList(per_point(net.local[i].weight.data, net.local[i].bias.data) for i in (0, 2, 4))
        W, bias = net.mid[0].weight.data, net.mid[0].bias.data
        if fold:
            self.mid0 = per_point(W[:, :b], None)
            self.mid0g = nn.Linear(W.shape[1] - b, W.shape[0])
            self.mid0g.weight.data, self.mid0g.bias.data = W[:, b:].clone(), bias.clone()
        else:
            self.mid0 = per_point(W, bias)
        self.mid1 = per_point(net.mid[2].weight.data, net.mid[2].bias.data)
        self.seg0 = per_point(net.seg[0][0].weight.data, net.seg[0][0].bias.data)
        self.seg1 = per_point(net.seg[1].weight.data, net.seg[1].bias.data)

    def forward(self, x, t, seed_b):
        B, N = x.shape[0], x.shape[1]
        conv = self.layout == "conv"
        te = t[:, None] * self.freq[None]
        c = self.cond(torch.cat([torch.sin(te), torch.cos(te), seed_b], -1))
        if conv:
            H, W = self.grid or (N, 1)
            x = x.transpose(1, 2).reshape(B, -1, H, W)
            if self.pool == "pool":
                gmax_of = lambda a: nn.functional.max_pool2d(a, (H, W)).flatten(1)
                mean_of = lambda a: nn.functional.avg_pool2d(a, (H, W)).flatten(1)
            else:
                gmax_of, mean_of = (lambda a: a.amax((2, 3))), (lambda a: a.mean((2, 3)))
            per = lambda g: g[:, :, None, None]                 # a per-hypothesis vector for every point
            tile = lambda g: g[:, :, None, None].expand(-1, -1, H, W)
        else:
            gmax_of, mean_of = (lambda a: a.max(dim=1).values), (lambda a: a.mean(dim=1))
            per = lambda g: g[:, None]
            tile = lambda g: g[:, None].expand(-1, N, -1)
        cat = 1 if conv else -1
        f = x
        for m in self.local:
            f = self.act(m(f))
        g1 = gmax_of(f)
        if self.fold:
            h = self.mid0(f) + per(self.mid0g(torch.cat([g1, c], -1)))
        else:
            h = self.mid0(torch.cat([f, tile(g1), tile(c)], cat))
        h = self.act(self.mid1(self.act(h)))
        gmax = gmax_of(h)
        out = self.head(torch.cat([gmax, mean_of(h), c], -1))
        seg = self.seg1(self.act(self.seg0(torch.cat([f, tile(self.gseg(gmax))], cat)))).reshape(B, N)
        return out, seg


class FastVerifier(nn.Module):
    """verifier.Verifier with the same weights and outputs, arranged like FastNet: the per-point layers as 1 x 1
    convolutions on a grid with pooling layers, and the first layer of each second stage folded (the half of its
    input that is the maximum over the points is applied once per hypothesis). The nearest-point search between the
    model points and the cloud stays as it is (differences, not products: products lose the millimetres in float16)."""

    def __init__(self, ver, fold=True, grid_cloud=(16, 32), grid_model=(16, 16)):
        super().__init__()
        self.fold, self.gc, self.gm = fold, grid_cloud, grid_model
        self.register_buffer("model_pts", ver.model_pts)
        self.head, self.act = ver.head, nn.GELU()

        def conv(lin, cols=None, bias=True):
            W = lin.weight.data if cols is None else lin.weight.data[:, cols]
            m = nn.Conv2d(W.shape[1], W.shape[0], 1, bias=bias)
            m.weight.data = W.clone()[..., None, None]
            if bias:
                m.bias.data = lin.bias.data.clone()
            return m

        def second(seq, b):         # first layer of a second stage: [per-point part, per-hypothesis part]
            if not fold:
                return conv(seq[0]), None
            g = nn.Linear(seq[0].in_features - b, seq[0].out_features)
            g.weight.data, g.bias.data = seq[0].weight.data[:, b:].clone(), seq[0].bias.data.clone()
            return conv(seq[0], slice(0, b), bias=False), g
        self.cloud1 = nn.ModuleList(conv(ver.cloud1[i]) for i in (0, 2, 4))
        b = ver.cloud1[4].out_features
        self.cloud2a, self.cloud2g = second(ver.cloud2, b)
        self.cloud2b = conv(ver.cloud2[2])
        self.model1 = nn.ModuleList(conv(ver.model1[i]) for i in (0, 2))
        self.model2a, self.model2g = second(ver.model2, ver.model1[2].out_features)

    def stage(self, x, grid, first, a, g, b):
        """[B, N, C] features -> maximum and mean over the points of the second stage's output."""
        B = x.shape[0]
        H, W = grid
        pool_max = lambda t: nn.functional.max_pool2d(t, (H, W)).flatten(1)
        pool_mean = lambda t: nn.functional.avg_pool2d(t, (H, W)).flatten(1)
        f = x.transpose(1, 2).reshape(B, -1, H, W)
        for m in first:
            f = self.act(m(f))
        fmax = pool_max(f)
        h = a(f) + g(fmax)[:, :, None, None] if g is not None else a(torch.cat([f, fmax[:, :, None, None].expand(-1, -1, H, W)], 1))
        h = self.act(h)
        if b is not None:
            h = self.act(b(h))
        return pool_max(h), pool_mean(h)

    def forward(self, table, pts, cam, q, p):
        xm = (pts - p[:, None]) @ geom.qmat(q)
        hmax, hmean = self.stage(model.features(table, xm, cam), self.gc, self.cloud1, self.cloud2a, self.cloud2g, self.cloud2b)
        mp, mn = self.model_pts[None, :, :3], self.model_pts[None, :, 3:]
        d, j = torch.cdist(mp.expand(len(xm), -1, -1), xm).min(dim=-1)
        near = torch.gather(xm, 1, j[..., None].expand(-1, -1, 3))
        v = (near - mp) / model.RES_SCALE
        dn = d[..., None] / model.RES_SCALE
        x = torch.cat([mp.expand(len(xm), -1, -1) / model.SET_RADIUS, mn.expand(len(xm), -1, -1), v * torch.tanh(dn) / (dn + 1e-6)], -1)
        kmax, kmean = self.stage(x, self.gm, self.model1, self.model2a, self.model2g, None)
        return self.head(torch.cat([hmax, hmean, kmax, kmean], -1)).squeeze(-1)


@torch.no_grad()
def idea_parts(out, ver, table, t):
    """Graphs for timing the verifier and the refinement on the Jetson (flow/jetson_fast.sh start; they are named
    evalx_v_* so that its loop over parts/evalx_*.onnx takes them): the verifier as it is and rearranged
    (FastVerifier), for 128 hypotheses and for the 32 of one position, and the refinement for 32."""
    out.mkdir(parents=True, exist_ok=True)
    s = t["score"][0]
    ref = Score(ver, table)(*s.values())
    for b in (128, 32):
        inp = {k: v[:b].contiguous() for k, v in s.items()}
        for k, v in inp.items():
            v.numpy().astype(np.float32).tofile(out / f"in_v{b}_{k}.bin")
        for name, m in ((f"score{b}", Score(ver, table)), (f"scoref{b}", Score(FastVerifier(ver), table)), (f"scorenofold{b}", Score(FastVerifier(ver, fold=False), table))):
            d = (torch.sigmoid(m(*inp.values())) - torch.sigmoid(ref[:b])).abs().max().item()
            print(f"{name}: largest difference of the score to the verifier {d:.2e}")
            assert d < 1e-4
            (out / f"evalx_v_{name}.inputs").write_text(",".join(f"{k}_in:in_v{b}_{k}.bin" for k in inp))
            export(m, inp, ["logit"], out / f"evalx_v_{name}.onnx")
    r = {k: v[:32].contiguous() for k, v in t["refine"][0].items()}
    for k, v in r.items():
        v.numpy().astype(np.float32).tofile(out / f"in_vr32_{k}.bin")
    (out / "evalx_v_refine32.inputs").write_text(",".join(f"{k}_in:in_vr32_{k}.bin" for k in r))
    export(Refine(table), r, ["q", "p"], out / "evalx_v_refine32.onnx")


@torch.no_grad()
def deploy(out, net, ver, table, t):
    """The graphs of the inference process (flow/live/trk_flow.cpp), with the solver steps of flow.propose:
      stage1e          position stage, 32 seeds          stage2e / stage2e1    pose stage, 4 positions / 1 position
      refine / refine1  refinement, 128 / 32 hypotheses   score / score1        verifier (FastVerifier), 128 / 32
    Each is compared here with the model as it is written, on the tensors of one scene."""
    out.mkdir(parents=True, exist_ok=True)
    fast, fver = FastNet(net, **FAST[DEPLOYED]), FastVerifier(ver)
    s1, s2, rf, sc = (t[k][0] for k in ("stage1", "stage2", "refine", "score"))
    one = lambda d: {k: v[:ORIENT].contiguous() for k, v in d.items()}
    st = lambda n, seg, steps: Stage(n, table, seg, steps, "euler")
    n1, n2 = propose.STAGE1_STEPS, propose.SOLVER_STEPS
    graphs = {"stage1e": (st(fast, True, n1), st(net, True, n1), s1, ["q", "p", "seg"]),
              "stage2e": (st(fast, False, n2), st(net, False, n2), s2, ["q", "p"]),
              "stage2e1": (st(fast, False, n2), st(net, False, n2), one(s2), ["q", "p"]),
              "refine": (Refine(table), None, rf, ["q", "p"]), "refine1": (Refine(table), None, one(rf), ["q", "p"]),
              "score": (Score(fver, table), Score(ver, table), sc, ["logit"]), "score1": (Score(fver, table), Score(ver, table), one(sc), ["logit"])}
    for name, (m, ref, inp, outs) in graphs.items():
        if ref is not None:
            got, want = m(*inp.values()), ref(*inp.values())
            if outs == ["logit"]:
                print(f"{name}: largest difference of the score to the verifier {(torch.sigmoid(got) - torch.sigmoid(want)).abs().max().item():.2e}")
            else:
                er, ep = pose_diff(got[0], got[1], want[0], want[1])
                print(f"{name}: rearranged against the model, max {er.max():.4f} deg {ep.max():.4f} mm")
        export(m, inp, outs, out / f"{name}.onnx")
    (out / "deploy.json").write_text(json.dumps({"stage1_steps": n1, "pose_steps": n2, "activation": type(net.local[1]).__name__ + ("(tanh)" if getattr(net.local[1], "approximate", "none") == "tanh" else "")}))


FAST = {"lin": dict(fold=False, layout="lin"), "linfold": dict(fold=True, layout="lin"),
        "conv": dict(fold=False, layout="conv"), "convfold": dict(fold=True, layout="conv"),
        "convfoldpool": dict(fold=True, layout="conv", pool="pool"), "convfoldgrid": dict(fold=True, layout="conv", grid=(16, 32)),
        "convfoldgridpool": dict(fold=True, layout="conv", pool="pool", grid=(16, 32))}


@torch.no_grad()
def fast_parts(out, net, table, t):
    """One field evaluation (128 hypotheses, and 32) with every arrangement of FAST: <out>/evalx_<name>[_b32].onnx,
    on the inputs of eval_b128 / eval_b32; ref_evalx_f.bin is the output of the model itself."""
    out.mkdir(parents=True, exist_ok=True)
    s = t["stage2"][0]
    full = {"pts": s["pts"], "cam": s["cam"], "q": t["stage2"][1]["q"], "p": t["stage2"][1]["p"]}
    for b, tag in ((128, ""), (32, "_b32")):
        inp = {k: v[:b].contiguous() for k, v in full.items()}
        ref = Eval(net, table)(*inp.values())
        ref.numpy().astype(np.float32).tofile(out / f"ref_evalx{tag}_f.bin")
        for name, kw in FAST.items():
            m = Eval(FastNet(net, **kw), table)
            d = (m(*inp.values()) - ref).abs().max().item()
            print(f"{name}{tag}: max difference to the model {d:.2e} (output up to {ref.abs().max():.2f})")
            assert d < 1e-4
            (out / f"evalx_{name}{tag}.inputs").write_text(",".join(f"{k}_in:in_eval_b{b}_{k}.bin" for k in inp))
            export(m, inp, ["f"], out / f"evalx_{name}{tag}.onnx")


@torch.no_grad()
def fast_stages(out, net, table, scenes, kw):
    """Whole stages with FastNet(**kw): stage1f / stage2f (4 RK4 steps, as stage1 / stage2), stage1e (4 Euler steps),
    stage2e (8 Euler steps) and stage2e1 (8 Euler steps, the 32 hypotheses of one position). Inputs are those of
    stage1 / stage2 (stage2e1: in_stage2e1_*.bin); the references are the outputs of the model itself."""
    fast = FastNet(net, **kw)
    spec = {"stage1f": ("stage1", 4, "rk4", K1), "stage2f": ("stage2", 4, "rk4", CENTERS * ORIENT), "stage1e": ("stage1", 4, "euler", K1),
            "stage2e": ("stage2", 8, "euler", CENTERS * ORIENT), "stage2e1": ("stage2", 8, "euler", ORIENT)}
    for n, (i, t) in enumerate(scenes):
        sd = out / f"s{i}"
        for g, (src, steps, solver, b) in spec.items():
            inp = {k: v[:b].contiguous() for k, v in t[src][0].items()}
            seg = src == "stage1"
            ref = Stage(net, table, seg, steps, solver)(*inp.values())
            got = Stage(fast, table, seg, steps, solver)(*inp.values())
            er, ep = pose_diff(got[0], got[1], ref[0], ref[1])
            print(f"scene {i} {g}: rearranged against the model, max {er.max():.4f} deg {ep.max():.4f} mm", flush=True)
            for k, v in zip(("q", "p", "seg"), ref):
                v.numpy().astype(np.float32).tofile(sd / f"ref_{g}_{k}.bin")
            if g == "stage2e1":
                for k, v in inp.items():
                    v.numpy().astype(np.float32).tofile(sd / f"in_{g}_{k}.bin")
            if n == 0:
                export(Stage(fast, table, seg, steps, solver), inp, ["q", "p", "seg"][:len(ref)], out / f"{g}.onnx")


@torch.no_grad()
def scene_tensors(net, ver, table, rig, depth, seed):
    """The tensors of flow.propose.propose on one scene (CPU), per graph: inputs and outputs."""
    dev = torch.device("cpu")
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    P, C = rig.cloud(depth)
    S = propose.seeds_fps(P, K1, rng)
    assert len(S) == K1, "scene with fewer seeds than K1"
    X, c, ok = propose.point_sets(P, C, S, rng, dev, g)
    qb = geom.qrand(K1, dev, g)
    q1, p1 = model.generate(net, table, X, c, qb, torch.zeros(K1, 3), 4)
    seg = model.field(net, table, X, c, q1, p1, 0.0)[1]
    out = {"stage1": ({"pts": X, "cam": c, "q": qb}, {"q": q1, "p": p1, "seg": seg})}
    # stage 2 on the centres the position stage gives (as propose())
    of1 = (((torch.sigmoid(seg) > 0.5) & ok).float().sum(1) / ok.float().sum(1).clamp_min(1)).numpy()
    e = p1.numpy().astype(np.float64) + S
    clusters = []
    for i in np.argsort(-of1):
        for cl in clusters:
            if np.linalg.norm(e[i] - e[cl[0]]) < 0.03:
                cl.append(i)
                break
        else:
            clusters.append([i])
    clusters.sort(key=lambda cl: -sum(of1[j] for j in cl))
    cen = np.stack([np.average(e[cl], axis=0, weights=of1[cl] + 1e-3) for cl in clusters[:CENTERS]])
    cen = np.concatenate([cen, np.repeat(cen[:1], CENTERS - len(cen), 0)])              # fixed shape: repeat the first
    X, c, ok = propose.point_sets(P, C, cen, rng, dev, g)
    rep = np.repeat(np.arange(CENTERS), ORIENT)
    X, c = X[rep].contiguous(), c[rep].contiguous()
    qb = torch.from_numpy(np.tile(propose.orientations(ORIENT)[0], (CENTERS, 1)).astype(np.float32))
    q2, p2 = model.generate(net, table, X, c, qb, torch.zeros(len(rep), 3), 4)
    q3, p3 = V.icp(table, X, q2, p2)
    out["stage2"] = ({"pts": X, "cam": c, "q": qb}, {"q": q2, "p": p2})
    out["refine"] = ({"pts": X, "q": q2, "p": p2}, {"q": q3, "p": p3})
    out["score"] = ({"pts": X, "cam": c, "q": q3, "p": p3}, {"logit": ver(table, X, c, q3, p3)})
    return out


class HalfNet(nn.Module):
    """The network in float16 inside a float32 graph (for a strongly typed TensorRT build: lookup, features and the
    pose arithmetic stay float32)."""

    def __init__(self, net):
        super().__init__()
        self.net = copy.deepcopy(net).half()

    def forward(self, x, t, seed_b):
        f, seg = self.net(x.half(), t.half(), seed_b.half())
        return f.float(), seg.float()


class Eval(nn.Module):
    """One evaluation of the field (what the solve does 16 times)."""

    def __init__(self, net, table):
        super().__init__()
        self.net = net
        self.register_buffer("table", table)

    def forward(self, pts, cam, q, p):
        return model.field(self.net, self.table, pts, cam, q, p, 0.5)[0]


class NetOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, seed_b):
        return self.net(x, torch.full((x.shape[0],), 0.5), seed_b)[0]


class Feat(nn.Module):
    """Lookup and features without the network."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, pts, cam, q, p):
        return model.features(self.table, (pts - p[:, None]) @ geom.qmat(q), cam)


class Look(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, xm):
        return model.lookup(self.table, xm)


def export(m, inp, outs, path):
    torch.onnx.export(m.eval(), tuple(inp.values()), str(path), input_names=[k + "_in" for k in inp], output_names=outs, opset_version=17, dynamo=False)
    print(path, flush=True)


@torch.no_grad()
def parts(out, net, table, t):
    """Graphs for the timing split (flow/jetson_tune.sh): one field evaluation at several sizes, and its parts. Each
    graph <g>.onnx comes with <g>.inputs, the --loadInputs argument of trtexec."""
    out.mkdir(parents=True, exist_ok=True)
    s = t["stage2"][0]
    full = {"pts": s["pts"], "cam": s["cam"], "q": t["stage2"][1]["q"], "p": t["stage2"][1]["p"]}       # the state at the end of the flow
    xm = (full["pts"] - full["p"][:, None]) @ geom.qmat(full["q"])
    seed_b = -(full["p"][:, None] @ geom.qmat(full["q"])).squeeze(1) / model.POS_SCALE
    cut = lambda d, b, n=model.N_POINTS: {k: (v[:b, :n] if v.dim() > 1 and v.shape[1] == model.N_POINTS else v[:b]).contiguous() for k, v in d.items()}
    graphs = {f"eval_b{b}": (Eval(net, table), cut(full, b), ["f"]) for b in (32, 64, 128)}
    graphs["eval_b128_n256"] = (Eval(net, table), cut(full, 128, 256), ["f"])
    graphs["net_b128"] = (NetOnly(net), {"x": model.features(table, xm, full["cam"]), "seed_b": seed_b}, ["f"])
    graphs["feat_b128"] = (Feat(table), full, ["x"])
    graphs["look_b128"] = (Look(table), {"xm": xm}, ["e"])
    for name, (m, inp, outs) in graphs.items():
        for k, v in inp.items():
            v.numpy().astype(np.float32).tofile(out / f"in_{name}_{k}.bin")
        (out / f"{name}.inputs").write_text(",".join(f"{k}_in:in_{name}_{k}.bin" for k in inp))
        export(m, inp, outs, out / f"{name}.onnx")


def pose_diff(q, p, q0, p0):
    """Rotation (degrees) and position (mm) differences between two sets of poses."""
    er, ep = V.errors(torch.as_tensor(q0), torch.as_tensor(p0), torch.as_tensor(q), torch.as_tensor(p))
    return er.numpy(), ep.numpy()


def check(out, tags):
    """Compares the outputs trtexec exported (<out>/s*/trt_<graph>_<tag>.json) with the PyTorch references."""
    sig = lambda x: 1 / (1 + np.exp(-x))
    for tag in tags:
        for sd in sorted(p for p in Path(out).glob("s*") if p.is_dir()):
            row = [f"{tag:5s} {sd.name:6s}"]
            for name in ("stage1", "stage2", "refine", "score", "stage1f", "stage2f", "stage1e", "stage2e", "stage2e1"):
                f = sd / f"trt_{name}_{tag}.json"
                if not f.exists():
                    continue
                o = {x["name"]: np.array(x["values"], np.float32).reshape([int(n) for n in x["dimensions"].split("x")] if isinstance(x["dimensions"], str) else x["dimensions"]) for x in json.load(open(f))}
                ref = lambda k: np.fromfile(sd / f"ref_{name}_{k}.bin", np.float32).reshape(o[k].shape)
                if name == "score":
                    a, b = sig(o["logit"]), sig(ref("logit"))
                    row.append(f"score: max diff {np.abs(a - b).max():.3f}, decisions at {propose.MIN_SCORE} changed {int(((a >= propose.MIN_SCORE) != (b >= propose.MIN_SCORE)).sum())} of {len(a)}")
                else:
                    er, ep = pose_diff(o["q"], o["p"], ref("q"), ref("p"))
                    row.append(f"{name}: median {np.median(er):.3f} deg {np.median(ep):.3f} mm, max {er.max():.2f} deg {ep.max():.2f} mm")
                    if "seg" in o:
                        row.append(f"seg sign changed {float(((o['seg'] > 0) != (ref('seg') > 0)).mean()):.4f}")
            print("  ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="flow/runs/export")
    ap.add_argument("--eval", default="flow/data/eval2")
    ap.add_argument("--scenes", default="3,700,1500")
    ap.add_argument("--parts", action="store_true", help="export the graphs of the timing split and the mixed-precision solve instead")
    ap.add_argument("--fast", action="store_true", help="export one evaluation with every arrangement of FastNet instead")
    ap.add_argument("--fast-stages", default="", help="export whole stages with this arrangement of FastNet (a key of FAST) instead")
    ap.add_argument("--ideas", action="store_true", help="export the verifier rearranged and the one-position graphs for timing instead")
    ap.add_argument("--deploy", action="store_true", help="export the graphs of the inference process (deploy()) instead; see flow/jetson_deploy.sh")
    ap.add_argument("--model", default=propose.MODEL)
    ap.add_argument("--act", default="", choices=["", "tanh_gelu"], help="replace the activation of the flow network (model.set_activation)")
    ap.add_argument("--check", default="", help="comma-separated tags of trtexec runs to compare (no export)")
    a = ap.parse_args()
    out = propose.ROOT / a.out
    if a.check:
        return check(out, a.check.split(","))
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    net, ver, table = propose.load(dev, a.model)
    if a.act:
        model.set_activation(net, a.act)
    rig = scene.Rig()
    want = [int(s) for s in a.scenes.split(",")]
    first, kept = None, []
    for i, cat, T, depth, _ in load_eval(propose.ROOT / a.eval):
        if i not in want:
            continue
        t = scene_tensors(net, ver, table, rig, depth, i)
        first = first or t
        kept.append((i, t))
        if a.parts or a.fast or a.ideas or a.deploy:
            break
        sd = out / f"s{i}"
        sd.mkdir(exist_ok=True)
        for name, (inp, ref) in t.items():
            for k, v in inp.items():
                v.numpy().astype(np.float32).tofile(sd / f"in_{name}_{k}.bin")       # graph input <k>_in
            for k, v in ref.items():
                v.numpy().astype(np.float32).tofile(sd / f"ref_{name}_{k}.bin")
        # the export form of the refinement against the float64 one, in PyTorch
        r_in, r_ref = t["refine"]
        qx, px = V.icp(table, r_in["pts"], r_in["q"], r_in["p"], export=True)
        er, ep = pose_diff(qx, px, r_ref["q"], r_ref["p"])
        print(f"scene {i} ({cat}): float32 refinement against float64: max {er.max():.4f} deg, {ep.max():.4f} mm", flush=True)
    if a.deploy:
        return deploy(out, net, ver, table, first)
    if a.ideas:
        return idea_parts(out / "parts", ver, table, first)
    if a.fast_stages:
        return fast_stages(out, net, table, kept, FAST[a.fast_stages])
    if a.fast:
        return fast_parts(out / "parts", net, table, first)
    if a.parts:
        parts(out / "parts", net, table, first)
        inp = first["stage2"][0]
        return export(Stage(HalfNet(net), table, False), inp, ["q", "p"], out / "stage2mixed.onnx")
    fast_stages(out, net, table, kept, FAST[DEPLOYED])
    mods = {"stage1": Stage(net, table, True), "stage2": Stage(net, table, False), "refine": Refine(table), "score": Score(ver, table)}
    for name, m in mods.items():
        export(m, first[name][0], list(first[name][1]), out / f"{name}.onnx")


if __name__ == "__main__":
    main()
