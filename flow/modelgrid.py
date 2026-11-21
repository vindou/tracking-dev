"""Closest-surface lookup of the object model on a grid, and the per-point input features of the flow network.

The network sees the cloud in the body frame of the pose estimate. If the estimate is right the points lie on the
model surface, so the vector from each point to its closest surface point is the alignment residual; with the surface
normal it gives the point-to-plane residual r and the terms r n and r (x cross n), whose sums are the right-hand side
of one point-to-plane ICP step. They are computed here (a table lookup) and given to the network as inputs, so the
network does not have to learn the shape of the object from scratch.

Grid: G^3 cells over the cube of half-width CROP_RADIUS around the model centroid (object frame); per cell the closest
point of the mesh to the cell centre and the normal of its triangle."""
import numpy as np
import open3d as o3d
import torch

from .model import CROP_RADIUS

G = 96
RES_SCALE = 0.03    # m per unit of the residual features


def build(rig):
    sc = o3d.t.geometry.RaycastingScene(nthreads=4)
    sc.add_triangles(o3d.core.Tensor(rig.V.astype(np.float32)), o3d.core.Tensor(rig.F))
    a = (np.arange(G) + 0.5) / G * 2 * CROP_RADIUS - CROP_RADIUS
    cen = np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3) + rig.centroid
    ans = sc.compute_closest_points(o3d.core.Tensor(cen.astype(np.float32)))
    cp = ans["points"].numpy() - rig.centroid.astype(np.float32)      # relative to the centroid, like the points
    n = ans["primitive_normals"].numpy()
    return torch.from_numpy(np.concatenate([cp, n], 1).astype(np.float32))    # [G^3, 6]


_cache = {}


def table(dev):
    key = str(dev)
    if key not in _cache:
        from . import scene
        if "cpu" not in _cache:
            _cache["cpu"] = build(scene.Rig())
        _cache[key] = _cache["cpu"].to(dev)
    return _cache[key]


def features(xm, cam):
    """xm [B, N, 3]: points in the body frame of the estimate, metres, relative to the estimated centroid (inside the
    crop). cam [B, N] in {-1, +1}. Returns [B, N, N_FEAT]."""
    T = table(xm.device)
    i = ((xm + CROP_RADIUS) * (G / (2 * CROP_RADIUS))).long().clamp_(0, G - 1)
    e = T[(i[..., 0] * G + i[..., 1]) * G + i[..., 2]]            # [B, N, 6]
    n = e[..., 3:]
    d = (e[..., :3] - xm) / RES_SCALE                              # to the closest surface point
    dist = d.norm(dim=-1, keepdim=True)
    d = d * torch.tanh(dist) / dist.clamp_min(1e-6)               # bounded: far points are clutter or a wrong estimate
    r = (d * n).sum(-1, keepdim=True)                              # point-to-plane residual
    x = xm / CROP_RADIUS
    return torch.cat([x, cam[..., None], d, n, r, torch.tanh(dist), r * n, r * torch.cross(x, n, dim=-1)], -1)
