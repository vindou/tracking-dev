"""Vector-field network of the flow: a point cloud in the frame of the current pose estimate -> the remaining motion.

Input  x  [B, N, 18]: per point, in the body frame of the estimate x_t (centred on the estimated centroid): position,
                     camera (-1 / +1), and the residual to the closest point of the model surface with the terms of a
                     point-to-plane step (modelgrid.features);
       t  [B]:       flow time in [0, 1].
Output xi [B, 6]:    estimate of log(x_t^-1 x_1) on SO(3) x R^3 in the body frame: rotation vector (rad) and
                     translation (units of POS_SCALE). The flow velocity is xi / (1 - t).
       seg [B, N]:   logit "this point is on the object" (auxiliary; training signal against clutter)."""
import math

import torch
import torch.nn as nn

CROP_RADIUS = 0.15   # m, points farther from the estimated centroid are not shown to the network
POS_SCALE = 0.1      # m per unit of the translation output
N_POINTS = 512
N_FEAT = 18       # per-point inputs, see modelgrid.features


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


def mlp(dims, last_act=True, norm=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(PointLinear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers += ([nn.LayerNorm(dims[i + 1])] if norm else []) + [nn.GELU()]
    return nn.Sequential(*layers)


class FlowNet(nn.Module):
    """PointNet-style: per-point features, a first global maximum fed back to every point, a second per-point stage,
    then maximum and mean over the points. The per-point layers have no normalization (cost on the training machine)."""

    def __init__(self, w=1):
        super().__init__()
        b, g = 192 * w, 512 * w
        self.local = mlp([N_FEAT, 64 * w, 128 * w, b], norm=False)
        self.temb = mlp([32, 64])
        self.mid = mlp([2 * b + 64, 256 * w, g], norm=False)
        self.head = nn.Sequential(mlp([2 * g + 64, 512, 512]), nn.Linear(512, 6))
        self.gseg = mlp([g, 64])
        self.seg = nn.Sequential(mlp([b + 64, 128], norm=False), PointLinear(128, 1))
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(200.0), 16)))

    def forward(self, x, t):
        N = x.shape[1]
        te = t[:, None] * self.freq[None]
        te = self.temb(torch.cat([torch.sin(te), torch.cos(te)], -1))                 # [B, 64]
        f = self.local(x)                                                             # [B, N, b]
        g1 = f.max(dim=1).values
        h = self.mid(torch.cat([f, g1[:, None].expand(-1, N, -1), te[:, None].expand(-1, N, -1)], -1))
        gmax = h.max(dim=1).values
        xi = self.head(torch.cat([gmax, h.mean(dim=1), te], -1))
        seg = self.seg(torch.cat([f, self.gseg(gmax)[:, None].expand(-1, N, -1)], -1)).squeeze(-1)
        return xi, seg
