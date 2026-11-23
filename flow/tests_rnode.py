"""Checks of the RNODE implementation (CPU, double precision).   python -m flow.tests_rnode

A small random network stands in for the trained one; the table is that of a sphere. The table is piecewise
trilinear, so the field has kinks at the cell faces and the solver tests use loose tolerances."""
import numpy as np
import torch

from . import geom, rnode


def _setup(B=3, N=40, seed=0):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    net = rnode.RNet()
    for p in net.head[-1].parameters():          # the last layer starts at zero: give it weights for the tests
        torch.nn.init.normal_(p, std=0.05)
    net = net.double()
    G = 40                                       # closest-point table of a sphere of radius 50 mm
    a = (torch.arange(G) + 0.5) / G * 2 * rnode.TABLE_HALF - rnode.TABLE_HALF
    c = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), -1)
    nrm = c / c.norm(dim=-1, keepdim=True)
    table = torch.cat([0.05 * nrm, nrm], -1)
    pts = torch.randn(B, N, 3) * 0.06
    cam = torch.sign(torch.randn(B, N))
    q = geom.qrand(B, "cpu").double()
    p = torch.randn(B, 3) * 0.02
    return net, table, pts, cam, q, p


def _perturb(q, p, d):
    q2 = geom.qmul(q, rnode.qexp(d[:, :3]))
    return q2, p + (geom.qmat(q) @ d[:, 3:, None]).squeeze(-1) * rnode.POS_SCALE


def _diff(qa, pa, qb, pb):
    """Body-frame coordinates (at a) of the difference between two states."""
    w = geom.qlog(geom.qmul(geom.qconj(qa), qb))
    v = ((pb - pa)[:, None] @ geom.qmat(qa)).squeeze(1) / rnode.POS_SCALE
    return torch.cat([w, v], -1)


def _fd_jacobian(net, table, pts, cam, q, p, t, h=1e-6):
    J = torch.zeros(q.shape[0], 6, 6)
    for i in range(6):
        d = torch.zeros(q.shape[0], 6)
        d[:, i] = h
        fp = rnode.field(net, table, pts, cam, *_perturb(q, p, d), t)[0]
        fm = rnode.field(net, table, pts, cam, *_perturb(q, p, -d), t)[0]
        J[:, :, i] = (fp - fm) / (2 * h)
    return J


def test_jacobian_terms_match_finite_differences():
    """Hutchinson: div = eps^T J eps and frob = |eps^T J|^2; exact: trace(J) and |J|_F^2."""
    net, table, pts, cam, q, p = _setup()
    J = _fd_jacobian(net, table, pts, cam, q, p, 0.3)
    eps = torch.randn(q.shape[0], 6)
    _, div, frob, _ = rnode.field(net, table, pts, cam, q, p, 0.3, eps)
    e = torch.einsum("bi,bij->bj", eps, J)
    assert (div - (e * eps).sum(-1)).abs().max() < 1e-6 and (frob - e.pow(2).sum(-1)).abs().max() < 1e-6
    _, div, frob, _ = rnode.field(net, table, pts, cam, q, p, 0.3, rnode.EXACT)
    assert (div - torch.diagonal(J, dim1=1, dim2=2).sum(-1)).abs().max() < 1e-6 and (frob - J.pow(2).sum((1, 2))).abs().max() < 1e-6


def test_forward_then_backward_returns():
    net, table, pts, cam, q, p = _setup()
    q1, p1, _, _ = rnode.integrate(net, table, pts, cam, q, p, 0.0, 1.0, 16)
    q0, p0, _, _ = rnode.integrate(net, table, pts, cam, q1, p1, 1.0, 0.0, 16)
    moved = _diff(q, p, q1, p1).norm(dim=-1).min()
    back = _diff(q, p, q0, p0).norm(dim=-1).max()
    assert moved > 1e-2 and back < 2e-3 * moved, (moved, back)


def test_solver_order():
    """Halving the step reduces the error of the end state by clearly more than 2^3 (4th order; the table has kinks)."""
    net, table, pts, cam, q, p = _setup()
    ref = rnode.integrate(net, table, pts, cam, q, p, 0.0, 1.0, 64)
    err = [_diff(ref[0], ref[1], *rnode.integrate(net, table, pts, cam, q, p, 0.0, 1.0, s)[:2]).norm(dim=-1).max().item() for s in (4, 8, 16)]
    assert err[0] / err[1] > 8 and err[1] / err[2] > 8, err


def test_log_det_is_the_integral_of_the_divergence():
    """The exact trace integrated along the trajectory equals the log-determinant of the flow map in body-frame
    coordinates (finite differences), up to the solver error: the change-of-variables term of the likelihood."""
    net, table, pts, cam, q, p = _setup(B=2)
    steps = 32
    l = rnode.integrate(net, table, pts, cam, q, p, 0.0, 1.0, steps, rnode.EXACT)[2][:, 0]
    q1, p1 = rnode.integrate(net, table, pts, cam, q, p, 0.0, 1.0, steps)[:2]
    h = 1e-5
    J = torch.zeros(q.shape[0], 6, 6)
    for i in range(6):
        d = torch.zeros(q.shape[0], 6)
        d[:, i] = h
        a = rnode.integrate(net, table, pts, cam, *_perturb(q, p, d), 0.0, 1.0, steps)[:2]
        b = rnode.integrate(net, table, pts, cam, *_perturb(q, p, -d), 0.0, 1.0, steps)[:2]
        J[:, :, i] = (_diff(q1, p1, *a) - _diff(q1, p1, *b)) / (2 * h)
    logdet = torch.linalg.slogdet(J)[1]
    print('   integral of the trace', l.tolist(), 'log det of the flow map', logdet.tolist())
    assert l.abs().min() > 1e-3 and (logdet - l).abs().max() < 1e-2 * max(1.0, l.abs().max().item()), (l, logdet)


def test_loss_backward_runs():
    for eps_kind in ("hutchinson", rnode.EXACT):
        net, table, pts, cam, q, p = _setup()
        eps = torch.randn(q.shape[0], 6) if eps_kind == "hutchinson" else rnode.EXACT
        nll, E, n, _ = rnode.rnode_terms(net, table, pts, cam, q, p, q, p, eps, 2)
        rnode.rnode_loss(nll, E, n, 0.01, 0.01).backward()
        g = torch.cat([x.grad.flatten() for x in net.parameters() if x.grad is not None])
        assert torch.isfinite(g).all() and g.abs().max() > 0


def test_fm_pair_is_the_geodesic_velocity():
    """Moving the point at time t by (1 - t) times the target reaches the base sample."""
    torch.manual_seed(5)
    torch.set_default_dtype(torch.float64)
    B = 50
    q, qb = geom.qrand(B, "cpu").double(), geom.qrand(B, "cpu").double()
    p, pb = torch.randn(B, 3) * 0.05, torch.randn(B, 3) * 0.05
    t = torch.rand(B)
    qt, pt, v = rnode.fm_pair(q, p, qb, pb, t)
    q2 = geom.qmul(qt, rnode.qexp((1 - t)[:, None] * v[:, :3]))
    p2 = pt + (1 - t)[:, None] * (geom.qmat(qt) @ v[:, 3:, None]).squeeze(-1) * rnode.POS_SCALE
    assert _diff(qb, pb, q2, p2).norm(dim=-1).max() < 1e-6


def test_zero_init_is_identity_with_finite_gradients():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    net = rnode.RNet().double()
    _, table, pts, cam, q, p = _setup()
    eps = torch.randn(q.shape[0], 6)
    nll, E, n, _ = rnode.rnode_terms(net, table, pts, cam, q, p, q, p, eps, 2)
    rnode.rnode_loss(nll, E, n, 0.01, 0.01).backward()
    assert E.abs().max() == 0 and all(torch.isfinite(x.grad).all() for x in net.parameters() if x.grad is not None)


def test_base_density_integrates_to_one():
    """Monte Carlo over rotations drawn uniformly (Haar) and positions drawn from a wide Gaussian."""
    torch.manual_seed(3)
    torch.set_default_dtype(torch.float64)
    n = 400000
    q = geom.qrand(n, "cpu").double()
    sp = 3 * rnode.SIGMA_POS * rnode.POS_SCALE
    p = torch.randn(n, 3) * sp
    q0, p0 = torch.tensor([[1.0, 0, 0, 0]]).expand(n, -1), torch.zeros(n, 3)
    import math
    log_prop = -math.log(8 * math.pi ** 2) - 0.5 * (p / sp).pow(2).sum(-1) - 3 * math.log(sp / rnode.POS_SCALE * math.sqrt(2 * math.pi))
    est = torch.exp(rnode.log_base(q, p, q0, p0) - log_prop).mean().item()
    assert abs(est - 1) < 0.03, est


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("PASS", name)
    torch.set_default_dtype(torch.float32)
