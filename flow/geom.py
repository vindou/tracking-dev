"""Rotation helpers. Quaternions are (w, x, y, z); poses are (q, p) with p the position of the model centroid in the
rig frame, or 4x4 T_rig_from_object. numpy versions for data, torch versions (batched, last dim) for the flow."""
import numpy as np
import torch


# ---------- numpy ----------
def quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def R_from_quat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def random_R(rng):
    q = rng.normal(size=4)
    return R_from_quat(q / np.linalg.norm(q))


def rotvec_R(v):
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_angle_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))


# ---------- torch ----------
def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def qexp(w):
    """Rotation vector (..., 3) -> unit quaternion."""
    a = w.norm(dim=-1, keepdim=True)
    h = 0.5 * a
    k = torch.where(a > 1e-6, torch.sin(h) / a.clamp_min(1e-6), 0.5 - a * a / 48.0)
    return torch.cat([torch.cos(h), k * w], -1)


def qlog(q):
    """Unit quaternion -> rotation vector with angle in [0, pi]."""
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(n, q[..., :1])
    k = torch.where(n > 1e-6, ang / n.clamp_min(1e-6), 2.0 + 0 * n)
    return k * v


def qmat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def qrand(n, device, generator=None):
    q = torch.randn(n, 4, device=device, generator=generator)
    return q / q.norm(dim=-1, keepdim=True)
