"""Checks of the flow mathematics (CPU, double precision where possible).   python -m flow.tests"""
import numpy as np
import torch

from . import geom, sampler
from .model import POS_SCALE


def test_exp_log_round_trip():
    torch.manual_seed(0)
    w = torch.randn(2000, 3, dtype=torch.double)
    w = w / w.norm(dim=-1, keepdim=True) * torch.rand(2000, 1, dtype=torch.double) * 3.1
    assert (geom.qlog(geom.qexp(w)) - w).abs().max() < 1e-9
    small = torch.randn(100, 3, dtype=torch.double) * 1e-9
    assert (geom.qlog(geom.qexp(small)) - small).abs().max() < 1e-15


def test_matrix_agrees_with_numpy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = geom.random_R(rng)
        q = geom.quat_from_R(R)
        assert np.abs(geom.qmat(torch.from_numpy(q)).numpy() - R).max() < 1e-12
        assert np.abs(geom.R_from_quat(q) - R).max() < 1e-12


def test_target_reaches_x1():
    """Applying the whole target xi to the perturbed x_t gives x_1, for every t."""
    torch.manual_seed(1)
    B = 500
    q1 = geom.qrand(B, "cpu").double()
    p1 = torch.randn(B, 3, dtype=torch.double) * 0.2
    t = torch.rand(B, dtype=torch.double)
    qt, pt, xi, _ = sampler.training_pair(q1, p1, t)
    q, p = sampler.apply(qt, pt, xi, 1.0)
    assert geom.qlog(geom.qmul(geom.qconj(q), q1)).norm(dim=-1).max() < 1e-6
    assert (p - p1).norm(dim=-1).max() < 1e-7


def test_euler_with_exact_field_follows_geodesic():
    """With the exact remaining motion as the prediction, Euler steps of dt / (1 - t) stay on the geodesic and end at x_1."""
    torch.manual_seed(2)
    B, steps = 200, 8
    q0, q1 = geom.qrand(B, "cpu").double(), geom.qrand(B, "cpu").double()
    p0, p1 = torch.randn(B, 3, dtype=torch.double) * 0.1, torch.zeros(B, 3, dtype=torch.double)
    w = geom.qlog(geom.qmul(geom.qconj(q0), q1))
    q, p = q0, p0
    for i in range(steps):
        t = i / steps
        xi = torch.cat([geom.qlog(geom.qmul(geom.qconj(q), q1)), ((p1 - p)[:, None] @ geom.qmat(q)).squeeze(1) / POS_SCALE], -1)
        q, p = sampler.apply(q, p, xi, 1.0 if i == steps - 1 else (1 / steps) / (1 - t))
        on_path = geom.qmul(q0, geom.qexp((i + 1) / steps * w))
        assert geom.qlog(geom.qmul(geom.qconj(on_path), q)).norm(dim=-1).max() < 1e-6
    assert (p - p1).norm(dim=-1).max() < 1e-9


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("PASS", name)
