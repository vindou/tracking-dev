"""Checks of the flow mathematics (CPU, double precision).   python -m flow.tests

A small random network stands in for the trained one; the table is that of a sphere. The table is piecewise
trilinear, so the field has kinks at the cell faces and the solver tests use loose tolerances."""
import numpy as np
import torch

from . import geom, model, verifier


def _setup(B=3, N=40, seed=0):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    net = model.FlowNet()
    for p in net.head[-1].parameters():          # the last layer starts at zero: give it weights for the tests
        torch.nn.init.normal_(p, std=0.05)
    net = net.double()
    G = 40                                       # closest-point table of a sphere of radius 50 mm
    a = (torch.arange(G) + 0.5) / G * 2 * model.TABLE_HALF - model.TABLE_HALF
    c = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), -1)
    nrm = c / c.norm(dim=-1, keepdim=True)
    table = torch.cat([0.05 * nrm, nrm], -1)
    pts = torch.randn(B, N, 3) * 0.06
    cam = torch.sign(torch.randn(B, N))
    q = geom.qrand(B, "cpu").double()
    p = torch.randn(B, 3) * 0.02
    return net, table, pts, cam, q, p


def _diff(qa, pa, qb, pb):
    """Body-frame coordinates (at a) of the difference between two states."""
    w = geom.qlog(geom.qmul(geom.qconj(qa), qb))
    v = ((pb - pa)[:, None] @ geom.qmat(qa)).squeeze(1) / model.POS_SCALE
    return torch.cat([w, v], -1)


def test_exp_log_round_trip():
    torch.manual_seed(0)
    w = torch.randn(2000, 3, dtype=torch.double)
    w = w / w.norm(dim=-1, keepdim=True) * torch.rand(2000, 1, dtype=torch.double) * 3.1
    assert (geom.qlog(geom.qexp(w)) - w).abs().max() < 1e-9
    assert (geom.qlog(model.qexp(w)) - w).abs().max() < 1e-6
    small = torch.randn(100, 3, dtype=torch.double) * 1e-9
    assert (geom.qlog(geom.qexp(small)) - small).abs().max() < 1e-15


def test_matrix_agrees_with_numpy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = geom.random_R(rng)
        q = geom.quat_from_R(R)
        assert np.abs(geom.qmat(torch.from_numpy(q)).numpy() - R).max() < 1e-12
        assert np.abs(geom.R_from_quat(q) - R).max() < 1e-12


def test_fm_pair_is_the_geodesic_velocity():
    """Moving the point at time t by (1 - t) times the target reaches the source sample."""
    torch.manual_seed(5)
    torch.set_default_dtype(torch.float64)
    B = 50
    q, qb = geom.qrand(B, "cpu").double(), geom.qrand(B, "cpu").double()
    p, pb = torch.randn(B, 3) * 0.05, torch.randn(B, 3) * 0.05
    t = torch.rand(B)
    qt, pt, v = model.fm_pair(q, p, qb, pb, t)
    q2 = geom.qmul(qt, model.qexp((1 - t)[:, None] * v[:, :3]))
    p2 = pt + (1 - t)[:, None] * (geom.qmat(qt) @ v[:, 3:, None]).squeeze(-1) * model.POS_SCALE
    assert _diff(qb, pb, q2, p2).norm(dim=-1).max() < 1e-6


def test_zero_init_is_identity():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    net = model.FlowNet().double()
    _, table, pts, cam, q, p = _setup()
    q1, p1 = model.integrate(net, table, pts, cam, q, p, 1.0, 0.0, 4)
    assert _diff(q, p, q1, p1).abs().max() < 1e-12


def test_forward_then_backward_returns():
    net, table, pts, cam, q, p = _setup()
    q1, p1 = model.integrate(net, table, pts, cam, q, p, 0.0, 1.0, 16)
    q0, p0 = model.integrate(net, table, pts, cam, q1, p1, 1.0, 0.0, 16)
    moved = _diff(q, p, q1, p1).norm(dim=-1).min()
    back = _diff(q, p, q0, p0).norm(dim=-1).max()
    assert moved > 1e-2 and back < 2e-3 * moved, (moved, back)


def test_solver_order():
    """Halving the step reduces the error of the end state by clearly more than 2^3 for RK4 (4th order; the table has
    kinks) and by about 2 for Euler."""
    net, table, pts, cam, q, p = _setup()
    ref = model.integrate(net, table, pts, cam, q, p, 0.0, 1.0, 64)
    err = lambda s, solver: _diff(*ref, *model.integrate(net, table, pts, cam, q, p, 0.0, 1.0, s, solver=solver)).norm(dim=-1).max().item()
    e = [err(s, "rk4") for s in (4, 8, 16)]
    assert e[0] / e[1] > 8 and e[1] / e[2] > 8, e
    e = [err(s, "euler") for s in (8, 16)]
    assert 1.5 < e[0] / e[1] < 2.6, e


def test_trajectory_is_the_solver_states():
    net, table, pts, cam, q, p = _setup()
    q1, p1, traj = model.generate(net, table, pts, cam, q, p, 4, trajectory=True)
    assert len(traj) == 5 and (traj[0][0] - q).abs().max() == 0 and (traj[-1][1] - p1).abs().max() == 0


def test_icp_recovers_a_small_offset():
    """Points on the sphere of the table, pose moved by a few millimetres: the refinement brings the centre back
    (the rotation of a sphere is free)."""
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)
    _, table, _, _, _, _ = _setup()
    d = torch.randn(4, 200, 3)
    pts = 0.05 * d / d.norm(dim=-1, keepdim=True)
    q = geom.qrand(4, "cpu").double()
    p0 = torch.randn(4, 3) * 0.003
    _, p = verifier.icp(table, pts, q, p0)
    assert p.norm(dim=-1).max() < 0.1 * p0.norm(dim=-1).min(), (p.norm(dim=-1), p0.norm(dim=-1))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("PASS", name)
    torch.set_default_dtype(torch.float32)
