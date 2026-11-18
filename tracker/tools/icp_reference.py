#!/usr/bin/env python3
"""ICP reference (Mac): runs the Open3D library (pip open3d 0.20.0) point-to-plane ICP on test clouds and writes the
inputs and the results for the C++ port's reference test (tests/test_icp.cpp).

    icp_reference.py [--model models/test_object.model] [--stl models/test_object.stl] [--out tracker/tests/data/icp_ref.bin]

Cases: 20 random object poses; for each pose: without and with depth noise (1 mm, Gaussian), L2 and Tukey kernel
(k = 5 mm): 80 cases. Source = 1000 points sampled on the mesh surface, moved to the rig frame by the pose (and noise
added); target = the model points with normals. Init = inverse of the pose perturbed by up to 10 degrees and 15 mm.
max_correspondence_distance 25 mm, criteria (1e-6, 1e-6, 30).

Open3D does not return the iteration count; it is found as the smallest max_iteration that gives the same
transformation as the run with 30 iterations (difference < 1e-12).

File format (little-endian): "ICPREF01", u32 version 1, u32 case count, u32 target count M, then M x 6 f64
(target points and normals). Per case: u32 kernel (0 L2, 1 Tukey), u32 noise flag, f64 kernel k, f64 max distance,
u32 max iteration, u32 reserved, f64 relative fitness, f64 relative rmse, f64 init[16] row-major, f64 truth[16]
(T_object_from_rig) row-major, u32 source count N, u32 reserved, N x 3 f64 source points, then the results:
f64 transformation[16] row-major, f64 fitness, f64 inlier rmse, u32 iterations, u32 correspondences.
"""
import argparse
import os
import struct
import sys

import numpy as np
import open3d as o3d
import trimesh

MAX_DIST = 0.025
TUKEY_K = 0.005
CRIT = (1e-6, 1e-6, 30)


def read_model(path):
    with open(path, "rb") as f:
        b = f.read()
    assert b[:8] == b"TRKMODEL"
    ver, n = struct.unpack_from("<II", b, 8)
    assert ver == 1
    a = np.frombuffer(b, dtype="<f8", offset=64, count=n * 6).reshape(n, 6)
    return a[:, :3].copy(), a[:, 3:].copy()


def rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_angle(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def run_icp(src, tgt, init, kernel, max_iter):
    loss = o3d.pipelines.registration.L2Loss() if kernel == 0 else o3d.pipelines.registration.TukeyLoss(k=TUKEY_K)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(CRIT[0], CRIT[1], max_iter)
    return o3d.pipelines.registration.registration_icp(src, tgt, MAX_DIST, init, est, crit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/test_object.model")
    ap.add_argument("--stl", default="models/test_object.stl")
    ap.add_argument("--out", default="tracker/tests/data/icp_ref.bin")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    tp, tn = read_model(a.model)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tp)
    tgt.normals = o3d.utility.Vector3dVector(tn)
    mesh = trimesh.load(a.stl)
    cases = []
    for k in range(20):
        R = rand_rot(rng)
        t = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.3, 0.5)])
        T_rig_obj = np.eye(4)
        T_rig_obj[:3, :3], T_rig_obj[:3, 3] = R, t
        truth = np.linalg.inv(T_rig_obj)                       # T_object_from_rig
        pts, _ = trimesh.sample.sample_surface(mesh, 1000, seed=rng)
        src_clean = (R @ np.asarray(pts).T).T + t
        dR = axis_angle(rng.normal(size=3), np.deg2rad(rng.uniform(0, 10)))
        dt = rng.normal(size=3)
        dt *= rng.uniform(0, 0.015) / np.linalg.norm(dt)
        P = np.eye(4)
        P[:3, :3], P[:3, 3] = dR, dt
        init = P @ truth
        for noise in (0, 1):
            src_pts = src_clean + (rng.normal(scale=0.001, size=src_clean.shape) if noise else 0)
            for kernel in (0, 1):
                src = o3d.geometry.PointCloud()
                src.points = o3d.utility.Vector3dVector(src_pts)
                full = run_icp(src, tgt, init, kernel, CRIT[2])
                T = np.asarray(full.transformation)
                iters = CRIT[2]
                for m in range(1, CRIT[2] + 1):
                    r = run_icp(src, tgt, init, kernel, m)
                    if np.max(np.abs(np.asarray(r.transformation) - T)) < 1e-12:
                        iters = m
                        break
                cases.append(dict(kernel=kernel, noise=noise, init=init, truth=truth, src=src_pts, T=T,
                                  fitness=full.fitness, rmse=full.inlier_rmse, iters=iters,
                                  ncorr=len(full.correspondence_set)))
                err_t = np.linalg.norm(T[:3, 3] - truth[:3, 3]) * 1000
                print(f"pose {k:2d} noise {noise} kernel {'L2   ' if kernel == 0 else 'Tukey'}: fitness {full.fitness:.4f}"
                      f" rmse {full.inlier_rmse * 1000:.3f} mm iterations {iters:2d} correspondences "
                      f"{len(full.correspondence_set)} |t - truth| {err_t:.3f} mm")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(b"ICPREF01")
        f.write(struct.pack("<III", 1, len(cases), len(tp)))
        f.write(np.hstack([tp, tn]).astype("<f8").tobytes())
        for c in cases:
            f.write(struct.pack("<IIddII", c["kernel"], c["noise"], TUKEY_K, MAX_DIST, CRIT[2], 0))
            f.write(struct.pack("<dd", CRIT[0], CRIT[1]))
            f.write(np.asarray(c["init"], dtype="<f8").tobytes())
            f.write(np.asarray(c["truth"], dtype="<f8").tobytes())
            f.write(struct.pack("<II", len(c["src"]), 0))
            f.write(np.asarray(c["src"], dtype="<f8").tobytes())
            f.write(np.asarray(c["T"], dtype="<f8").tobytes())
            f.write(struct.pack("<ddII", c["fitness"], c["rmse"], c["iters"], c["ncorr"]))
    print(f"open3d {o3d.__version__}: {len(cases)} cases written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
