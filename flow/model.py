"""The pose flow: a conditional flow on SO(3) x R^3, trained by flow matching (Lipman et al., arXiv 2210.02747; on
manifolds: Chen and Lipman, arXiv 2302.03660), that moves pose hypotheses towards the pose of the object.

  state      z = (q, p): rotation of the object (unit quaternion) and position of the model centroid, relative to a
             seed position (m). 6 dimensions.
  condition  N_POINTS measured points within SET_RADIUS of the seed (fixed along a trajectory) and the seed itself.
             The network sees the points in the body frame of z (features()) and the seed in that frame.
  field      f(z, t) in R^6: body-frame angular velocity (rad per unit time) and body-frame linear velocity
             (POS_SCALE m per unit time): q' = q (0, w) / 2, p' = R(q) v POS_SCALE.
  time       t = 0 is the data (the true pose with a small noise: DEQ_ROT, DEQ_POS), t = T_END the source.
  source     centred on the TRUE pose: rotation R_true exp(eta), eta ~ N(0, SIGMA_ROT^2), position
             N(p_true, (SIGMA_POS POS_SCALE)^2). It cannot be sampled at inference (the truth is not known), so the
             inference starts from a set of orientations at candidate positions (flow/propose.py). A source that does
             not use the truth (uniform rotation, Gaussian around the seed) did not train.
  training   regression of f on the constant velocity of the geodesic from a data pose to a source sample (fm_pair).
  solver     Runge-Kutta-Munthe-Kaas: classical RK4 in the rotation vector u of q_n exp(u) (with the terms of the
             inverse differential of exp up to second order) and in p; hypotheses are integrated from T_END to 0."""
import math
import os

import numpy as np
import torch
import torch.nn as nn

from . import geom

# A shared GPU: FLOW_GPU_GB limits the memory this process may take (training, inference).
if os.environ.get("FLOW_GPU_GB") and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ["FLOW_GPU_GB"]) * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)

T_END = 1.0
POS_SCALE = 0.1         # m per unit of the translation part of the field
SET_RADIUS = 0.16       # m: points within this distance of the seed are the condition
N_POINTS = 512
SEED_MAX = 0.08         # m: seeds of the training data are uniform in a ball of this radius around the true centroid
SIGMA_ROT = 0.6         # source: standard deviation of the rotation vector around the true rotation, rad (34 degrees)
SIGMA_POS = 0.4         # source: standard deviation of the position around the truth, in POS_SCALE units (40 mm)
DEQ_ROT = 0.05          # rad: noise on the true rotation (2.9 degrees)
DEQ_POS = 0.04          # POS_SCALE units: noise on the true position (4 mm)
TABLE_HALF = 0.20       # m: half-width of the closest-point table (object frame, around the model centroid)
TABLE_G = 128
RES_SCALE = 0.03        # m per unit of the residual features
N_FEAT = 19
DIM = 6


# ---------------- closest-surface table and per-point features ----------------
def build_table(rig):
    """[G, G, G, 6]: closest mesh point (relative to the model centroid) and its triangle normal at the cell centres.
    Needs open3d (made on the Mac: python -m flow.train --make-table)."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene(nthreads=4)
    sc.add_triangles(o3d.core.Tensor(rig.V.astype(np.float32)), o3d.core.Tensor(rig.F))
    a = (np.arange(TABLE_G) + 0.5) / TABLE_G * 2 * TABLE_HALF - TABLE_HALF
    cen = np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3) + rig.centroid
    ans = sc.compute_closest_points(o3d.core.Tensor(cen.astype(np.float32)))
    cp = ans["points"].numpy() - rig.centroid.astype(np.float32)
    return np.concatenate([cp, ans["primitive_normals"].numpy()], 1).astype(np.float32).reshape(TABLE_G, TABLE_G, TABLE_G, 6)


def lookup(table, xm):
    """Trilinear interpolation of the table at body-frame points xm [B, N, 3] (m); constant outside the table."""
    G = table.shape[0]
    flat = table.reshape(-1, 6)
    u = ((xm + TABLE_HALF) * (G / (2 * TABLE_HALF)) - 0.5).clamp(0.0, G - 1.0 - 1e-4)
    i0 = u.detach().floor().long()
    w = u - i0
    i1 = i0 + 1
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix, iy, iz = (i1 if dx else i0)[..., 0], (i1 if dy else i0)[..., 1], (i1 if dz else i0)[..., 2]
                wt = (w[..., 0] if dx else 1 - w[..., 0]) * (w[..., 1] if dy else 1 - w[..., 1]) * (w[..., 2] if dz else 1 - w[..., 2])
                out = out + wt[..., None] * flat[(ix * G + iy) * G + iz]
    return out


def features(table, xm, cam):
    """Per-point inputs [B, N, N_FEAT] from body-frame points xm [B, N, 3] (m, relative to the estimated centroid) and
    the camera of each point (-1 / +1): position, camera, vector to the closest surface point (bounded), its normal,
    the point-to-plane residual r, the bounded distance, r n, r (x cross n) (the terms of an ICP step), and a soft
    "near the estimate" weight."""
    e = lookup(table, xm)
    n = e[..., 3:]
    d = (e[..., :3] - xm) / RES_SCALE
    dist = torch.sqrt(d.pow(2).sum(-1, keepdim=True) + 1e-8)
    d = d * torch.tanh(dist) / dist
    r = (d * n).sum(-1, keepdim=True)
    x = xm / SET_RADIUS
    near = torch.sigmoid((0.13 - torch.sqrt(xm.pow(2).sum(-1, keepdim=True) + 1e-8)) / 0.02)
    return torch.cat([x, cam[..., None], d, n, r, torch.tanh(dist), r * n, r * torch.cross(x, n, dim=-1), near], -1)


# ---------------- network ----------------
class _PointLinearFn(torch.autograd.Function):
    """F.linear on [B, N, C] with the weight gradient as a batched product summed over B. Same result as nn.Linear;
    the MPS backend takes about 10 times longer for the single (C_in x B N) @ (B N x C_out) product of the default
    backward."""

    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)
        return torch.nn.functional.linear(x, w, b)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        return g @ w, torch.bmm(g.transpose(1, 2), x).sum(0), g.sum((0, 1))


class PointLinear(nn.Linear):
    def forward(self, x):
        if x.dim() == 3 and self.training and x.is_mps:
            return _PointLinearFn.apply(x, self.weight, self.bias)
        return super().forward(x)


ACTIVATIONS = {"gelu": nn.GELU, "tanh_gelu": lambda: nn.GELU(approximate="tanh"), "relu": nn.ReLU}


def mlp(dims, last_act=True, norm=True, act="gelu"):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(PointLinear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers += ([nn.LayerNorm(dims[i + 1])] if norm else []) + [ACTIVATIONS[act]()]
    return nn.Sequential(*layers)


def set_activation(net, act):
    """Replaces every activation of a network (in place): e.g. the tanh form of GELU, which TensorRT runs faster, in
    a network trained with GELU."""
    for m in net.modules():
        for name, child in list(m.named_children()):
            if isinstance(child, (nn.GELU, nn.ReLU)):
                setattr(m, name, ACTIVATIONS[act]())
    return net


class FlowNet(nn.Module):
    """The field f(z, t; cloud, seed), 1.18 M parameters. PointNet style: per-point features, a first global maximum
    fed back to every point with the time and the seed, a second per-point stage, then maximum and mean over the
    points. Outputs f [B, 6] and a per-point logit "this point is on the object" [B, N] (used to rank the positions
    of the first inference stage). The per-point layers have no normalization. The last layer starts at zero."""

    def __init__(self, w=1, act="gelu"):
        super().__init__()
        b, g = 192 * w, 512 * w
        self.local = mlp([N_FEAT, 64 * w, 128 * w, b], norm=False, act=act)
        self.cond = mlp([32 + 3, 64, 64], act=act)
        self.mid = mlp([2 * b + 64, 256 * w, g], norm=False, act=act)
        self.head = nn.Sequential(mlp([2 * g + 64, 512, 512], act=act), nn.Linear(512, DIM))
        self.gseg = mlp([g, 64], act=act)
        self.seg = nn.Sequential(mlp([b + 64, 128], norm=False, act=act), nn.Linear(128, 1))
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(16.0), 16)))    # slow in t: the solver takes few steps
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, t, seed_b):
        N = x.shape[1]
        te = t[:, None] * self.freq[None]
        c = self.cond(torch.cat([torch.sin(te), torch.cos(te), seed_b], -1))
        f = self.local(x)
        g1 = f.max(dim=1).values
        h = self.mid(torch.cat([f, g1[:, None].expand(-1, N, -1), c[:, None].expand(-1, N, -1)], -1))
        gmax = h.max(dim=1).values
        out = self.head(torch.cat([gmax, h.mean(dim=1), c], -1))
        seg = self.seg(torch.cat([f, self.gseg(gmax)[:, None].expand(-1, N, -1)], -1)).squeeze(-1)
        return out, seg


def load(path, dev):
    """The network of a checkpoint of flow.train."""
    ck = torch.load(path, map_location=dev)
    cfg = ck.get("config", {})
    assert cfg.get("encoder", "pointnet") == "pointnet" and not cfg.get("gate") and not cfg.get("smooth"), \
        "checkpoint of an experimental variant that is no longer in the code"
    net = FlowNet(ck.get("width", 1), cfg.get("act", "gelu")).to(dev)
    net.load_state_dict(ck["model"])
    return net.eval()


# ---------------- field and solver ----------------
def qexp(w):
    """Rotation vector -> unit quaternion, differentiable at w = 0."""
    a = torch.sqrt(w.pow(2).sum(-1, keepdim=True) + 1e-12)
    return torch.cat([torch.cos(0.5 * a), torch.sin(0.5 * a) / a * w], -1)


def field(net, table, pts, cam, q, p, t):
    """f(z, t) [B, 6] and the per-point object / clutter logits [B, N] at the state (q, p)."""
    R = geom.qmat(q)
    xm = (pts - p[:, None]) @ R                                 # R^T (point - p)
    seed_b = -(p[:, None] @ R).squeeze(1) / POS_SCALE           # the seed (origin) in the body frame
    if not torch.is_tensor(t):
        t = torch.full((q.shape[0],), float(t), device=q.device, dtype=q.dtype)
    return net(features(table, xm, cam), t, seed_b)


def _step(net, table, pts, cam, q, p, t, h, solver):
    """One step of size h (negative: towards the data) from (q, p) at time t."""
    cross = lambda a, b: torch.cross(a, b, dim=-1)

    def stage(u, dp, ts):
        qs = q if u is None else geom.qmul(q, qexp(u))
        f = field(net, table, pts, cam, qs, p if dp is None else p + dp, ts)[0]
        w = f[:, :3]
        du = w if u is None else w + 0.5 * cross(u, w) + cross(u, cross(u, w)) / 12.0     # d/dt of the rotation vector
        return du, (geom.qmat(qs) @ f[:, 3:, None]).squeeze(-1) * POS_SCALE

    k1 = stage(None, None, t)
    if solver == "euler":
        du, dv = h * k1[0], h * k1[1]
    else:
        k2 = stage(0.5 * h * k1[0], 0.5 * h * k1[1], t + 0.5 * h)
        k3 = stage(0.5 * h * k2[0], 0.5 * h * k2[1], t + 0.5 * h)
        k4 = stage(h * k3[0], h * k3[1], t + h)
        du, dv = ((h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in (0, 1))
    q2 = geom.qmul(q, qexp(du))
    return q2 / q2.norm(dim=-1, keepdim=True), p + dv


def integrate(net, table, pts, cam, q, p, t0, t1, steps, trajectory=False, solver="rk4", power=1.0):
    """Fixed steps from t0 to t1 (solver: "rk4", 4 network evaluations per step; "euler", 1; "ab2", 1: the
    two-step Adams-Bashforth method, which extrapolates with the velocity of the previous step, 1.5 f_n - 0.5 f_(n-1);
    its first step is an Euler step). power: the step sizes; 1: equal steps; p > 1: the times are
    t1 + (t0 - t1) (1 - i / steps)^p, larger steps at the start and smaller ones towards t1 (not for "ab2").
    Returns q, p and, with trajectory = True, the list of states."""
    assert power == 1.0 or solver != "ab2", "the two-step method here is for equal steps"
    times = [t1 + (t0 - t1) * (1 - i / steps) ** power for i in range(steps + 1)]
    traj = [(q, p)]
    prev = None
    for i in range(steps):
        h = times[i + 1] - times[i]
        if solver == "ab2":
            f = field(net, table, pts, cam, q, p, times[i])[0]
            k = (f[:, :3], (geom.qmat(q) @ f[:, 3:, None]).squeeze(-1) * POS_SCALE)     # body-frame angular, rig-frame linear velocity
            w, v = k if prev is None else (1.5 * k[0] - 0.5 * prev[0], 1.5 * k[1] - 0.5 * prev[1])
            q = geom.qmul(q, qexp(h * w))
            q, p, prev = q / q.norm(dim=-1, keepdim=True), p + h * v, k
        else:
            q, p = _step(net, table, pts, cam, q, p, times[i], h, solver)
        traj.append((q, p))
    return (q, p, traj) if trajectory else (q, p)


@torch.no_grad()
def generate(net, table, pts, cam, q, p, steps, trajectory=False, solver="rk4", power=1.0):
    """Integrates starts (q, p) from T_END to the data end (t = 0)."""
    return integrate(net, table, pts, cam, q, p, T_END, 0.0, steps, trajectory, solver, power)


# ---------------- training pairs ----------------
def sample_source(q_ref, p_ref, generator=None):
    """A sample of the source around the reference pose(s) q_ref [B, 4], p_ref [B, 3]."""
    B, dev = q_ref.shape[0], q_ref.device
    q = geom.qmul(q_ref, qexp(torch.randn(B, 3, device=dev, generator=generator) * SIGMA_ROT))
    return q, p_ref + torch.randn(B, 3, device=dev, generator=generator) * SIGMA_POS * POS_SCALE


def fm_pair(q, p, qb, pb, t):
    """The point at time t [B] of the geodesic from the data (q, p) at t = 0 to a source sample (qb, pb) at t = T_END,
    and its constant velocity in the frame of that point: the regression target for f."""
    eta = geom.qlog(geom.qmul(geom.qconj(q), qb))
    s = (t / T_END)[:, None]
    qt = geom.qmul(q, qexp(s * eta))
    pt = (1 - s) * p + s * pb
    v = ((pb - p)[:, None] @ geom.qmat(qt)).squeeze(1) / POS_SCALE
    return qt, pt, torch.cat([eta, v], -1) / T_END


def select_points(pts, valid, cam, lab=None, n=N_POINTS, generator=None):
    """pts [B, M, 3] relative to the seed, valid [B, M], cam [B, M] (0 / 1): n random valid points within SET_RADIUS
    of the seed (fewer: the first chosen point is repeated; none: zeros). Returns points [B, n, 3], cam (-1 / +1),
    the labels of the chosen points if lab is given, and the mask of real points."""
    inside = valid & (pts.pow(2).sum(-1) < SET_RADIUS ** 2)
    score = torch.rand(inside.shape, device=pts.device, generator=generator) + inside.float() * 2.0
    top, idx = score.topk(n, dim=1)
    ok = top >= 2.0
    idx = torch.where(ok, idx, idx[:, :1])
    keep = ok[:, :1].float()
    out = torch.gather(pts, 1, idx[..., None].expand(-1, -1, 3)) * keep[..., None]
    c = (torch.gather(cam, 1, idx).float() * 2 - 1) * keep
    return out, c, (torch.gather(lab, 1, idx) if lab is not None else None), ok
