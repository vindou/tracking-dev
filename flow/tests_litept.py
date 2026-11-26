"""Checks of flow/litept.py.   python -m flow.tests_litept

1. The sparse convolution (list form and fixed-table form) equals a dense convolution at the occupied cells.
2. PointROPE: attention does not change when all cells are shifted by the same offset.
3. Every variant: the output does not depend on the order of the points or on the other clouds of the batch.
4. The fixed-shape implementation gives the outputs of the first (sparse) implementation with the same weights,
   also for the derivative with respect to the pose input.
5. Fixed shapes with capacities that are too small: runs, finite.
6. Rig-frame cells: the cached tables are reused and give the same result as fresh ones."""
import torch
import torch.nn.functional as F

from . import litept, rnode

torch.manual_seed(0)


def cloud(B, N=rnode.N_POINTS, g=None):
    """Points on bent sheets within SET_RADIUS (m), like a depth crop, and random features."""
    u = (torch.rand(B, N, 2, generator=g) - 0.5) * 0.22
    z = 0.4 * u[..., :1] ** 2 * 10 + 0.02 * torch.sin(40 * u[..., 1:]) + 0.05 * (torch.rand(B, N, 1, generator=g) > 0.7)
    pts = torch.cat([u, z], -1)
    q = torch.randn(B, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    from . import geom
    return pts @ geom.qmat(q).float(), torch.randn(B, N, rnode.N_FEAT, generator=g)


def net(size, grid, impl):
    torch.manual_seed(1)
    n = litept.LitePT(size, grid, impl).double()
    n.caps = (512, 512, 256, 128, 64)                      # no cell of the test clouds is dropped
    torch.nn.init.normal_(n.head[-1].weight, std=0.05)     # the last layer starts at zero
    return n.eval()


def run(n, pts, x, t=None, seed=None):
    B = len(pts)
    t = torch.full((B,), 0.4, dtype=torch.float64) if t is None else t
    seed = torch.zeros(B, 3, dtype=torch.float64) if seed is None else seed
    return n(x.double(), t, seed, xm=pts.double(), pts=pts)


def test_conv_dense():
    for k in (3, 5):
        conv = litept.SubMConv(4, 6, k).double()
        G = 12
        occ = torch.rand(G, G, G) < 0.3
        c = occ.nonzero() + 20
        keys, order = litept._key(c).sort()
        c = c[order]
        f = torch.randn(len(c), 4, dtype=torch.float64)
        dense = torch.zeros(1, 4, G, G, G, dtype=torch.float64)
        dense[0, :, c[:, 0] - 20, c[:, 1] - 20, c[:, 2] - 20] = f.T
        w = conv.weight.reshape(k, k, k, 4, 6).permute(4, 3, 0, 1, 2)
        ref = F.conv3d(dense, w, conv.bias, padding=k // 2)[0][:, c[:, 0] - 20, c[:, 1] - 20, c[:, 2] - 20].T
        a = conv(f, litept.neighbours(keys, conv.delta))
        cap = len(c) + 7                                        # fixed table with padding
        kp = torch.cat([keys, keys.new_full((cap - len(c),), litept.SENT)])[None]
        fp = torch.cat([f, torch.randn(cap - len(c), 4, dtype=torch.float64)])      # padding rows hold anything
        b = conv(fp, litept._find(kp, kp[:, :, None] + conv.delta))[:len(c)]
        assert (a - ref).abs().max() < 1e-10 and (b - ref).abs().max() < 1e-10, (k, (a - ref).abs().max(), (b - ref).abs().max())
    print("1 sparse convolution = dense convolution (3 and 5 wide; list and fixed table)")


def test_rope():
    blk = litept.AttnBlock(72).double()
    x = torch.randn(2, 9, 72, dtype=torch.float64)
    pos = torch.randint(0, 20, (2, 9, 3))
    mask = torch.ones(2, 9, dtype=torch.bool)
    a, b = blk(x, pos, mask), blk(x, pos + torch.tensor([5, -3, 11]), mask)
    c = blk(x, pos * 2, mask)
    assert (a - b).abs().max() < 1e-10 and (a - c).abs().max() > 1e-4
    print("2 PointROPE: shift of all cells changes nothing (and a change of the offsets does)")


def test_order_batch():
    pts, x = cloud(5)
    for size, grid, impl in (("t", "body", "sparse"), ("t", "body", "fixed"), ("t", "rig", "sparse"), ("t", "rig", "fixed"),
                             ("s", "body", "fixed"), ("s", "rig", "sparse")):
        n = net(size, grid, impl)
        with torch.no_grad():
            o, s = run(n, pts, x)
            perm = torch.randperm(pts.shape[1])
            o2, s2 = run(n, pts[:, perm], x[:, perm])
            o3, s3 = run(n, pts[2:3].clone(), x[2:3])
        e = max((o - o2).abs().max(), (s[:, perm] - s2).abs().max(), (o[2:3] - o3).abs().max(), (s[2:3] - s3).abs().max())
        assert e < 1e-9, (size, grid, impl, e)
        print(f"3 order and batch independence, {size} {grid} {impl}: largest difference {e:.1e}")


def test_fixed_equals_sparse():
    pts, x = cloud(6)
    pts[5] = 0                                                  # a cloud without points (select_points gives zeros)
    for size, grid in (("t", "body"), ("s", "body"), ("t", "rig"), ("s", "rig")):
        a, b = net(size, grid, "sparse"), net(size, grid, "fixed")
        b.load_state_dict(a.state_dict())
        t, seed = torch.rand(6, dtype=torch.float64), torch.randn(6, 3, dtype=torch.float64)
        xa = x.double().requires_grad_(True)
        xb = x.double().requires_grad_(True)
        oa, sa = a(xa, t, seed, xm=pts.double(), pts=pts)
        ob, sb = b(xb, t, seed, xm=pts.double(), pts=pts)
        w = torch.randn_like(oa)
        ga, = torch.autograd.grad((oa * w).sum() + sa.sum(), xa)
        gb, = torch.autograd.grad((ob * w).sum() + sb.sum(), xb)
        e = [(oa - ob).abs().max().item(), (sa - sb).abs().max().item(), (ga - gb).abs().max().item()]
        assert max(e) < 1e-9, (size, e)
        print(f"4 fixed = sparse, LitePT-{size}, {grid}-frame cells (float64): output {e[0]:.1e}, per-point logit {e[1]:.1e}, input gradient {e[2]:.1e} "
              f"(output scale {oa.abs().mean():.2f})")


def test_overflow():
    pts, x = cloud(4)
    n = net("t", "body", "fixed")
    n.caps = (200, 60, 20, 8, 4)
    litept.COUNT = [0, 0]
    with torch.no_grad():
        o, s = run(n, pts, x)
    assert torch.isfinite(o).all() and torch.isfinite(s).all() and litept.COUNT == [4, 4], litept.COUNT
    litept.COUNT = None
    print("5 capacities too small: finite output, all 4 clouds counted as overflowing")


def test_rig_cache():
    pts, x = cloud(3)
    n = net("t", "rig", "sparse")
    calls = []
    orig = n.tables_sparse
    n.tables_sparse = lambda c: calls.append(1) or orig(c)
    with torch.no_grad():
        o1, _ = run(n, pts, x)
        o2, _ = run(n, pts, x)                  # same tensor: cached
        o3, _ = run(n, pts.clone(), x)          # another tensor: rebuilt
    assert len(calls) == 2 and (o1 - o2).abs().max() == 0 and (o1 - o3).abs().max() < 1e-12
    # the cells do not depend on the body-frame positions
    with torch.no_grad():
        o4 = n(x.double(), torch.full((3,), 0.4, dtype=torch.float64), torch.zeros(3, 3, dtype=torch.float64), xm=pts.double() * 0.3 + 0.1, pts=pts)[0]
    assert (o1 - o4).abs().max() == 0
    print("6 rig-frame cells: tables built once per cloud tensor; independent of the body-frame positions")


if __name__ == "__main__":
    test_conv_dense()
    test_rope()
    test_order_batch()
    test_fixed_equals_sparse()
    test_overflow()
    test_rig_cache()
    print("ok")
