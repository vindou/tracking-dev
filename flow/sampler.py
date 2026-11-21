"""Flow on SO(3) x R^3 around padded clouds: the conditional path and its target (training) and the Euler integration
(inference). A pose is (q, p): unit quaternion of R_rig_from_object and the position of the model centroid.

Path from x_0 to x_1:  R_t = R_0 exp(t log(R_0^T R_1)),  p_t = (1 - t) p_0 + t p_1.
The network sees the cloud in the frame of x_t and predicts xi = (log(R_t^T R_1), R_t^T (p_1 - p_t)), the remaining
motion; the vector field of the paper is xi / (1 - t). One Euler step of size dt from t applies the fraction
dt / (1 - t) of the predicted remaining motion (the last step applies all of it)."""
import torch

from . import geom, modelgrid
from .model import CROP_RADIUS, N_POINTS, POS_SCALE


def crop_canonical(pts, valid, cam, q, p, n=N_POINTS, generator=None):
    """pts [B, M, 3] (any common frame), valid [B, M] bool, cam [B, M] (0 / 1), pose estimate q [B, 4], p [B, 3].
    Random n of the valid points within CROP_RADIUS of p, in the body frame of the estimate. Returns the network input x [B, n, N_FEAT] (modelgrid.features),
    the indices [B, n] and their mask [B, n] (False where the crop had fewer than n points; such entries repeat the
    first chosen point, or are zero if the crop is empty)."""
    d = pts - p[:, None]
    inside = valid & (d.pow(2).sum(-1) < CROP_RADIUS ** 2)
    score = torch.rand(inside.shape, device=pts.device, generator=generator) + inside.float() * 2.0
    top, idx = score.topk(n, dim=1)
    ok = top >= 2.0
    idx = torch.where(ok, idx, idx[:, :1])
    any_ok = ok[:, :1]
    xm = torch.gather(d, 1, idx[..., None].expand(-1, -1, 3)) @ geom.qmat(q)    # R^T (pt - p), metres
    c = torch.gather(cam, 1, idx).float() * 2 - 1
    x = modelgrid.features(xm, c) * any_ok[..., None].float()
    return x, idx, ok


MAX_ANGLE = 1.22      # rad (70 deg): largest rotation offset of the source distribution of the rotation flow
P_GLOBAL = 0.3        # share of training samples with a uniform rotation (position only)


def training_pair(q1, p1, t, sigma_rot=0.07, sigma_pos=0.005, generator=None):
    """Source sample x_0, the point x_t of the path, a small perturbation of it (so that the network also learns to
    correct an estimate that left the path), and the target xi. Two kinds of source:
      local  (70 %): rotation within MAX_ANGLE of R_1 (angle uniform, random axis), position within 40 mm;
      global (30 %): uniform rotation, position within 100 mm. For these only the translation is a target
                     (rot_mask 0): the remaining rotation from an arbitrary start is not predictable enough to
                     regress, and at inference the rotation flow is started from a set of orientations instead.
    Returns q_t, p_t, xi, rot_mask."""
    B, dev = q1.shape[0], q1.device
    r = lambda *shape: torch.rand(*shape, device=dev, generator=generator)
    n = lambda *shape: torch.randn(*shape, device=dev, generator=generator)
    glob = r(B) < P_GLOBAL
    ax = n(B, 3)
    ax = ax / ax.norm(dim=-1, keepdim=True)
    q0 = torch.where(glob[:, None], geom.qrand(B, dev, generator).to(q1.dtype), geom.qmul(q1, geom.qexp(ax * r(B, 1) * MAX_ANGLE)))
    u = n(B, 3)
    u = u / u.norm(dim=-1, keepdim=True) * r(B, 1).pow(1 / 3) * torch.where(glob[:, None], 0.10, 0.04)
    p0 = p1 + u
    w = geom.qlog(geom.qmul(geom.qconj(q0), q1))
    qt = geom.qmul(q0, geom.qexp(t[:, None] * w))
    pt = (1 - t[:, None]) * p0 + t[:, None] * p1
    qt = geom.qmul(qt, geom.qexp(n(B, 3) * sigma_rot))
    pt = pt + n(B, 3) * sigma_pos
    xi_rot = geom.qlog(geom.qmul(geom.qconj(qt), q1))
    xi_pos = ((p1 - pt)[:, None] @ geom.qmat(qt)).squeeze(1) / POS_SCALE
    return qt, pt, torch.cat([xi_rot, xi_pos], -1), (~glob).to(q1.dtype)


def flow_loss(xi, target, rot_mask):
    """Squared error of the predicted remaining motion relative to its size (so that the last degrees and millimetres
    weigh as much as the first 70 degrees). Returns the rotation and the translation term."""
    e_rot = (xi[:, :3] - target[:, :3]).pow(2).sum(-1) / (target[:, :3].pow(2).sum(-1) + 0.1 ** 2)
    e_pos = (xi[:, 3:] - target[:, 3:]).pow(2).sum(-1) / (target[:, 3:].pow(2).sum(-1) + 0.1 ** 2)
    return (e_rot * rot_mask).sum() / rot_mask.sum().clamp_min(1), e_pos.mean()


def apply(q, p, xi, alpha):
    """x <- x exp(alpha xi) on SO(3) x R^3 (xi in the body frame, translation in POS_SCALE units)."""
    q2 = geom.qmul(q, geom.qexp(alpha * xi[:, :3]))
    p2 = p + alpha * POS_SCALE * (geom.qmat(q) @ xi[:, 3:, None]).squeeze(-1)
    return q2 / q2.norm(dim=-1, keepdim=True), p2


@torch.no_grad()
def integrate(net, pts, valid, cam, q, p, steps=8, refine=2, trajectory=False, generator=None, position_only=False):
    """Euler integration of the learned field from (q, p) at t = 0 to t = 1, then `refine` more evaluations at t = 1
    (each applies the whole predicted remaining motion). Returns q, p, the norm of the last predicted remaining motion
    (rad, m) and the object fraction of the last crop (segmentation head); with trajectory = True also every state."""
    traj = [(q, p)]
    ts = [i / steps for i in range(steps)] + [1.0] * refine
    xi = seg = ok = None
    for i, t in enumerate(ts):
        x, _, ok = crop_canonical(pts, valid, cam, q, p, generator=generator)
        xi, seg = net(x, torch.full((q.shape[0],), t, device=q.device))
        if position_only:
            xi = torch.cat([torch.zeros_like(xi[:, :3]), xi[:, 3:]], -1)
        alpha = 1.0 if i >= steps - 1 else (1.0 / steps) / (1.0 - t)
        q, p = apply(q, p, xi, alpha)
        traj.append((q, p))
    res_rot, res_pos = xi[:, :3].norm(dim=-1), xi[:, 3:].norm(dim=-1) * POS_SCALE
    objfrac = ((torch.sigmoid(seg) > 0.5) & ok).float().sum(1) / ok.float().sum(1).clamp_min(1)
    out = (q, p, res_rot, res_pos, objfrac)
    return out + (traj,) if trajectory else out
