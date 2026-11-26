"""The encoder of LitePT (Yue et al., "LitePT: Lighter Yet Stronger Point Transformer", arXiv 2512.13689; code
github.com/prs-eth/LitePT, litept/model.py) as the point encoder of the dynamics.

As in the paper: five stages with grid pooling of stride 2 between them; the first three stages are convolution
blocks, x + LayerNorm(Linear(SubMConv3d 3x3x3 (x))), without attention; the last two are attention blocks (pre-norm,
x + Attn(LN x), x + MLP(LN x), MLP ratio 4, GELU) without convolution; the only positional encoding is PointROPE, a
rotary encoding of the integer grid coordinates applied to queries and keys (head dimension 18, 6 per axis, base
100); the stem is a 5x5x5 submanifold convolution; pooling is Linear, maximum over the cell, normalization, GELU.
Widths, depths and heads of LitePT-S: (36, 72, 144, 252, 504), (2, 2, 2, 6, 2), (2, 4, 8, 14, 28).

Differences, because the input here is 512 points of a 0.3 m region evaluated inside an ODE:
- No spconv: the submanifold convolution is written with a sorted-key neighbour search (same values).
- The first grid is 8 mm (the paper: 20 mm for rooms), in the body frame of the pose estimate; the features of the
  points of a cell are averaged.
- All tokens of a cloud are one attention group (the paper groups 1024 serialized points; a cloud here has fewer), so
  the serialization orders have no effect and are not implemented.
- LayerNorm where the code uses BatchNorm (stem, pooling); no drop path; no half-precision attention.
- No decoder: the pose head reads the maximum and the mean of the last tokens. The time and seed condition is added
  to the tokens after the stem and before each attention stage, and given to the head.

Three ways to run it (LitePT(size, grid, impl); python -m flow.tests_litept checks them against each other):
- impl "sparse", grid "body": the first version. One list of occupied cells for the whole batch (torch.unique), so
  every tensor has a different size at every pass.
- impl "fixed", grid "body": the same function with the same weights on tensors of fixed shape. Every stage has
  CAPS cells per cloud (padding is masked); the convolution reads a fixed [cells, 27] neighbour table (125 in the
  stem) whose empty entries point at a zero row; pooling is a gather of the 8 children and a maximum. Cells beyond
  the capacity (the highest keys) are dropped.
- grid "rig" (encoders litept_t_rig, litept_s_rig; either implementation): the cells are those of the points relative to
  the crop seed in the rig axes, not in the body frame of the pose estimate. Cells, neighbour tables and pooling
  maps then depend on the cloud only: they are built once per cloud (LitePT.cached, like LPNet.graph) and reused by
  every evaluation of an integration. The per-point features are still those of the body frame. This is a different
  model: the field is continuous in the pose (no point changes its cell), but the convolution and PointROPE see
  rig-frame offsets."""
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rnode
from .model import mlp

GRID0 = 0.008           # m, cell of the first stage
RADIX = 128             # cells per axis in the key; coordinates are kept in [2, RADIX - 3] (offsets of +-2 do not wrap)
HEAD_DIM = 18
CONFIGS = {             # channels, depths (heads = channels / 18)
    "s": ((36, 72, 144, 252, 504), (2, 2, 2, 6, 2)),          # LitePT-S
    "t": ((36, 72, 144, 216, 288), (1, 1, 1, 3, 1)),          # reduced
}
SENT = RADIX ** 3       # key of "no cell" (larger than every key)
# Cells per cloud at the five stages. Stage 0 cannot exceed the 512 points. The others: the largest count measured
# with `python -m flow.bench_litept caps` (body frame 344 / 162 / 56 / 21 over 61,440 clouds and poses, rig frame
# 342 / 154 / 59 / 19 over 20,480 clouds; the means are 137 / 51 / 18 / 9) plus a margin of 10 to 50 %.
CAPS = {"body": (512, 384, 192, 72, 32), "rig": (512, 384, 192, 72, 32)}
IMPL = os.environ.get("LITEPT_IMPL", "sparse")      # implementation: "sparse" or "fixed" (same function)
AMP = os.environ.get("LITEPT_AMP", "0") == "1"         # fixed implementation: the network in bfloat16 (features and outputs stay float32)
COMPILE = os.environ.get("LITEPT_COMPILE", "0") == "1" # fixed implementation: torch.compile of the network on the tables
SELECT = os.environ.get("LITEPT_SELECT", "1") == "1"   # rows are read with index_select (see _take): 19 times faster training
COUNT = None            # set to [0, 0] to count clouds and clouds with more cells than a capacity (costs a sync)


class Prof:
    """Wall time per section of a pass, with the GPU synchronized at every mark (python -m flow.bench_litept profile)."""

    def __init__(self, dev):
        self.acc, self.cuda = {}, dev.type == "cuda"
        self.mark()

    def mark(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t = time.perf_counter()

    def tick(self, name):
        t = self.t
        self.mark()
        self.acc[name] = self.acc.get(name, 0.0) + self.t - t


PROFILE = None


def _tick(name):
    if PROFILE is not None:
        PROFILE.tick(name)


def _take(f, idx):
    """f [T, C], idx [M, K] -> [M, K, C]. The same values either way; the backward pass differs: that of f[idx] is
    index_put with accumulation (on the GPU it sorts all M K indices), that of index_select is index_add."""
    if SELECT:
        return f.index_select(0, idx.reshape(-1)).reshape(*idx.shape, f.shape[1])
    return f[idx]


def _offsets(k):
    r = torch.arange(k) - k // 2
    o = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    return (o[:, 0] * RADIX + o[:, 1]) * RADIX + o[:, 2]


def neighbours(keys, delta):
    """keys [T] sorted unique cell keys, delta [K] key offsets -> [T, K] index of the neighbour cell, T where it is
    empty."""
    T = keys.shape[0]
    want = keys[:, None] + delta[None]
    idx = torch.searchsorted(keys, want).clamp(max=T - 1)
    return torch.where(keys[idx] == want, idx, torch.full_like(idx, T))


class SubMConv(nn.Module):
    """Submanifold sparse convolution: output at the occupied cells only, from the occupied cells of the window."""

    def __init__(self, c_in, c_out, k, bias=True):
        super().__init__()
        self.register_buffer("delta", _offsets(k), persistent=False)
        self.weight = nn.Parameter(torch.empty(k ** 3 * c_in, c_out))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(2.0 / (k ** 3 * c_in) / 0.3))    # most cells of a window are empty
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None

    def forward(self, f, nb):
        g = _take(torch.cat([f, f.new_zeros(1, f.shape[1])]), nb).flatten(1) @ self.weight
        return g if self.bias is None else g + self.bias


class ConvBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv, self.lin, self.norm = SubMConv(c, c, 3), nn.Linear(c, c), nn.LayerNorm(c)

    def forward(self, f, nb):
        return f + self.norm(self.lin(self.conv(f, nb)))


def point_rope(x, pos, inv_freq):
    """x [B, H, L, 18], pos [B, L, 3] integer grid coordinates: a 1-D rotary encoding per axis on a third of the
    channels each (rotate-half pairing, as libs/pointrope/pointrope_torch.py)."""
    out = []
    for a, v in enumerate(x.chunk(3, dim=-1)):
        ang = pos[:, None, :, a, None].to(x.dtype) * inv_freq
        ang = torch.cat([ang, ang], -1)
        h = v.shape[-1] // 2
        out.append(v * ang.cos() + torch.cat([-v[..., h:], v[..., :h]], -1) * ang.sin())
    return torch.cat(out, -1)


class AttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c // HEAD_DIM
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.qkv, self.proj = nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        d = HEAD_DIM // 3
        self.register_buffer("inv_freq", 1.0 / 100.0 ** (torch.arange(0, d, 2).float() / d), persistent=False)

    def forward(self, x, pos, mask):
        B, L, C = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, L, 3, self.h, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k = point_rope(q, pos, self.inv_freq), point_rope(k, pos, self.inv_freq)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, C))
        return x + self.mlp(self.n2(x))


class Pool(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.lin, self.norm = nn.Linear(c_in, c_out), nn.LayerNorm(c_out)

    def forward(self, f, inv, n):
        h = self.lin(f)
        out = h.new_zeros(n, h.shape[1]).scatter_reduce(0, inv[:, None].expand(-1, h.shape[1]), h, "amax", include_self=False)
        return F.gelu(self.norm(out))


def _cells(batch, coord):
    """Sorted unique cells of (batch [M], coord [M, 3]) and the cell of each row."""
    key = ((batch * RADIX + coord[:, 0]) * RADIX + coord[:, 1]) * RADIX + coord[:, 2]
    keys, inv = torch.unique(key, return_inverse=True)
    c = torch.stack([keys // RADIX ** 2 % RADIX, keys // RADIX % RADIX, keys % RADIX], -1)
    return keys, inv, keys // RADIX ** 3, c


# ---------------- fixed-shape tables ----------------
def _key(c):
    return (c[..., 0] * RADIX + c[..., 1]) * RADIX + c[..., 2]


def _unkey(k):
    return torch.stack([k // RADIX ** 2, k // RADIX % RADIX, k % RADIX], -1)


def _rows(key, cap):
    """key [B, M] (SENT: no cell) -> the distinct keys of each row in ascending order, padded with SENT [B, cap]; the
    position of each entry's key in that row [B, M] (cap: none, or dropped); the number of distinct keys [B]."""
    B = key.shape[0]
    sk, order = key.sort(dim=1)
    new = torch.ones_like(sk, dtype=torch.bool)
    new[:, 1:] = sk[:, 1:] != sk[:, :-1]
    real = sk < SENT
    cid = new.long().cumsum(1) - 1
    cid = torch.where((cid < cap) & real, cid, torch.full_like(cid, cap))
    keys = sk.new_full((B, cap + 1), SENT).scatter_(1, cid, sk)[:, :cap].contiguous()
    return keys, torch.empty_like(cid).scatter_(1, order, cid), (new & real).sum(1)


def _find(keys, want):
    """keys [B, L] ascending with SENT padding, want [B, M, K] -> flat index b L + position of the key [B M, K];
    B L where the cell does not exist."""
    B, L = keys.shape
    w = want.reshape(B, -1)
    idx = torch.searchsorted(keys, w).clamp(max=L - 1)
    hit = (keys.gather(1, idx) == w) & (w < SENT)
    idx = torch.where(hit, idx + torch.arange(B, device=keys.device)[:, None] * L, torch.full_like(idx, B * L))
    return idx.reshape(B * want.shape[1], want.shape[2])


@torch.no_grad()
def build_grid(coord, caps, delta5, delta3, off8):
    """coord [B, N, 3] integer cell of each point -> the tables of the five stages, all of fixed shape:
      inv0   [B N] flat cell of each point at stage 0 (B L0: dropped);  cnt0 [B L0, 1] points of the cell
      nb5    [B L0, 125], nb[s] [B Ls, 27] (s = 0, 1, 2): flat index of the neighbour cells (B Ls: empty)
      child[s] [B Ls, 8] (s = 1..4): flat index of the cells of stage s - 1 pooled into the cell (B L(s-1): none)
      mask[s] [B, Ls] real cells;  coord[s] [B, Ls, 3] (s = 3, 4) for PointROPE."""
    B, N, _ = coord.shape
    keys, inv, n = _rows(_key(coord), caps[0])
    off = torch.arange(B, device=coord.device)[:, None]
    g = {"inv0": torch.where(inv < caps[0], inv + off * caps[0], torch.full_like(inv, B * caps[0])).reshape(-1),
         "nb": [], "child": [None], "mask": [], "coord": [None] * 5, "n": [n]}
    cnt = torch.zeros(B * caps[0] + 1, device=coord.device).index_add_(0, g["inv0"], torch.ones(B * N, device=coord.device))
    g["cnt0"] = cnt[:-1, None].clamp_min(1.0)
    g["nb5"] = _find(keys, keys[:, :, None] + delta5)
    for s in range(5):
        if s > 0:
            fine = keys
            keys, _, n = _rows(torch.where(fine < SENT, _key(_unkey(fine) // 2), fine), caps[s])
            g["n"].append(n)
            g["child"].append(_find(fine, _key(2 * _unkey(keys)[:, :, None, :] + off8)))
        g["mask"].append(keys < SENT)
        if s < 3:
            g["nb"].append(_find(keys, keys[:, :, None] + delta3))
        else:
            g["coord"][s] = _unkey(keys) * g["mask"][s][..., None]
    if COUNT is not None:
        COUNT[0] += B
        COUNT[1] += int(torch.stack([m > c for m, c in zip(g["n"], caps)]).any(0).sum())
    return g


class LitePT(nn.Module):
    def __init__(self, size="s", grid="body", impl=None):
        super().__init__()
        ch, depth = CONFIGS[size]
        self.grid, self.impl = grid, impl or IMPL
        self.caps = CAPS[grid]
        self.register_buffer("off8", torch.tensor([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]), persistent=False)
        self._cache = None
        self.core_fn = torch.compile(self._core) if COMPILE and self.impl == "fixed" else self._core
        self.cond = mlp([32 + 3, 64, 64])
        self.stem = SubMConv(rnode.N_FEAT, ch[0], 5, bias=False)
        self.stem_norm = nn.LayerNorm(ch[0])
        self.cond_in = nn.ModuleList([nn.Linear(64, ch[0]), nn.Linear(64, ch[3]), nn.Linear(64, ch[4])])
        self.pool = nn.ModuleList([Pool(ch[i], ch[i + 1]) for i in range(4)])
        self.blocks = nn.ModuleList([nn.ModuleList([(ConvBlock if s < 3 else AttnBlock)(ch[s]) for _ in range(depth[s])]) for s in range(5)])
        self.register_buffer("delta3", _offsets(3), persistent=False)
        self.out_norm = nn.LayerNorm(ch[4])
        self.head = nn.Sequential(mlp([2 * ch[4] + 64, 512, 512]), nn.Linear(512, rnode.DIM))
        self.gseg = mlp([2 * ch[4], 64])
        self.seg = nn.Sequential(mlp([ch[0] + 64, 128], norm=False), nn.Linear(128, 1))
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(16.0), 16)))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def cells(v):
        """Positions [B, N, 3] (m) -> integer cells."""
        return (torch.floor(v.detach() / GRID0).long() + RADIX // 2).clamp(2, RADIX - 3)

    def tables(self, coord):
        return build_grid(coord, self.caps, self.stem.delta, self.delta3, self.off8)

    def cached(self, pts):
        """The tables of a cloud (grid "rig"), kept while the same tensor is passed again."""
        if self._cache is None or self._cache[0] is not pts:
            self._cache = (pts, (self.tables_sparse if self.impl == "sparse" else self.tables)(self.cells(pts)))
        return self._cache[1]

    def forward(self, x, t, seed_b, xm=None, pts=None):
        _tick("features (table lookup), pose")
        if self.grid == "rig":
            g = self.cached(pts)
        else:
            g = (self.tables_sparse if self.impl == "sparse" else self.tables)(self.cells(xm))
        _tick("grid: cells, neighbour and pooling tables")
        if self.impl == "sparse":
            return self._forward_sparse(x, t, seed_b, g)
        if AMP:
            with torch.autocast(x.device.type, dtype=torch.bfloat16):
                out, seg = self.core_fn(x, t, seed_b, g)
            return out.float(), seg.float()
        return self.core_fn(x, t, seed_b, g)

    def _core(self, x, t, seed_b, g):
        """The network on the tables of build_grid. Cell features are flat, [B L, C], for the convolutions."""
        B, N, _ = x.shape
        L = self.caps
        te = t[:, None] * self.freq[None]
        c = self.cond(torch.cat([torch.sin(te), torch.cos(te), seed_b], -1))
        f = x.new_zeros(B * L[0] + 1, x.shape[-1]).index_add_(0, g["inv0"], x.reshape(B * N, -1))[:-1] / g["cnt0"].to(x.dtype)
        _tick("cell features (mean of points), condition")
        f = F.gelu(self.stem_norm(self.stem(f, g["nb5"]))) + self.cond_in[0](c).repeat_interleave(L[0], 0)
        _tick("stem convolution 5x5x5")
        f0 = None
        for s in range(5):
            if s > 0:
                h = self.pool[s - 1].lin(f)
                h = _take(torch.cat([h, h.new_full((1, h.shape[1]), float("-inf"))]), g["child"][s]).amax(1)
                f = F.gelu(self.pool[s - 1].norm(torch.where(g["mask"][s].reshape(-1, 1), h, torch.zeros_like(h))))
                _tick(f"pooling into stage {s}")
            if s < 3:
                for blk in self.blocks[s]:
                    f = blk(f, g["nb"][s])
                if s == 0:
                    f0 = f
                _tick(f"stage {s} convolution blocks")
            else:
                f = f + self.cond_in[s - 2](c).repeat_interleave(L[s], 0)
                xp = f.reshape(B, L[s], -1)
                for blk in self.blocks[s]:
                    xp = blk(xp, g["coord"][s], g["mask"][s])
                f = xp.reshape(B * L[s], -1)
                _tick(f"stage {s} attention blocks")
        m = g["mask"][4][..., None]
        f = self.out_norm(f).reshape(B, L[4], -1)
        mx = f.masked_fill(~m, float("-inf")).amax(1)
        mean = (f * m).sum(1) / m.sum(1)
        glob = torch.cat([mx, mean], -1)
        out = self.head(torch.cat([glob, c], -1))
        f0 = _take(torch.cat([f0, f0.new_zeros(1, f0.shape[1])]), g["inv0"][:, None]).reshape(B, N, -1)
        seg = self.seg(torch.cat([f0, self.gseg(glob)[:, None].expand(-1, N, -1)], -1)).squeeze(-1)
        _tick("head")
        return out, seg

    @torch.no_grad()
    def tables_sparse(self, coord):
        """coord [B, N, 3] -> the index tensors of the sparse implementation: one list of occupied cells for the
        whole batch at every stage (sizes depend on the clouds)."""
        B, N, _ = coord.shape
        dev = coord.device
        keys, inv0, tb, tc = _cells(torch.arange(B, device=dev).repeat_interleave(N), coord.reshape(-1, 3))
        g = {"inv0": inv0, "cnt0": torch.bincount(inv0, minlength=len(keys))[:, None], "nb5": neighbours(keys, self.stem.delta),
             "tb": [], "inv": [None], "nb": [], "attn": [None] * 5}
        for s in range(5):
            if s > 0:
                keys, inv, tb, tc = _cells(tb, tc // 2)
                g["inv"].append(inv)
            g["tb"].append(tb)
            if s < 3:
                g["nb"].append(neighbours(keys, self.delta3))
            else:       # the cells of each cloud side by side for the attention
                cnt = torch.bincount(tb, minlength=B)
                L = int(cnt.max())
                pos = torch.arange(len(keys), device=dev) - (torch.cumsum(cnt, 0) - cnt)[tb]
                cp = tc.new_zeros(B, L, 3)
                cp[tb, pos] = tc
                g["attn"][s] = (L, pos, cp, torch.arange(L, device=dev)[None] < cnt[:, None])
        g["cnt4"] = torch.bincount(tb, minlength=B)[:, None]
        return g

    def _forward_sparse(self, x, t, seed_b, g):
        B, N, _ = x.shape
        te = t[:, None] * self.freq[None]
        c = self.cond(torch.cat([torch.sin(te), torch.cos(te), seed_b], -1))
        inv0, tb = g["inv0"], g["tb"][0]
        f = x.new_zeros(len(tb), x.shape[-1]).index_add_(0, inv0, x.reshape(B * N, -1))
        f = f / g["cnt0"].to(f.dtype)
        _tick("cell features (mean of points), condition")
        f = F.gelu(self.stem_norm(self.stem(f, g["nb5"]))) + _take(self.cond_in[0](c), tb[:, None]).squeeze(1)
        _tick("stem convolution 5x5x5")
        f0 = None
        for s in range(5):
            tb = g["tb"][s]
            if s > 0:
                f = self.pool[s - 1](f, g["inv"][s], len(tb))
                _tick(f"pooling into stage {s}")
            if s < 3:
                for blk in self.blocks[s]:
                    f = blk(f, g["nb"][s])
                if s == 0:
                    f0 = f
                _tick(f"stage {s} convolution blocks")
            else:
                f = f + _take(self.cond_in[s - 2](c), tb[:, None]).squeeze(1)
                L, pos, cp, mask = g["attn"][s]
                xp = f.new_zeros(B, L, f.shape[1])
                xp[tb, pos] = f
                for blk in self.blocks[s]:
                    xp = blk(xp, cp, mask)
                f = xp[tb, pos]
                _tick(f"stage {s} attention blocks")
        f = self.out_norm(f)
        mx = f.new_full((B, f.shape[1]), -1e4).scatter_reduce(0, tb[:, None].expand(-1, f.shape[1]), f, "amax", include_self=False)
        mean = f.new_zeros(B, f.shape[1]).index_add_(0, tb, f) / g["cnt4"].to(f.dtype)
        glob = torch.cat([mx, mean], -1)
        out = self.head(torch.cat([glob, c], -1))
        seg = self.seg(torch.cat([_take(f0, inv0[:, None]).reshape(B, N, -1), self.gseg(glob)[:, None].expand(-1, N, -1)], -1)).squeeze(-1)
        _tick("head")
        return out, seg
