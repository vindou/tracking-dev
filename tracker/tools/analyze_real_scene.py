#!/usr/bin/env python3
"""Analysis of a real pair (Mac; numpy, open3d, Pillow from the venv): depth images as pictures, points per camera in
the work volume, and the agreement of the two cameras' scenes under other camera-frame assumptions.
    analyze_real_scene.py CLOUD_CSV OUTDIR      (CLOUD_CSV from trk_clouddump --stride 1)"""
import os, sys
import numpy as np
import open3d as o3d
from PIL import Image

INTR = {0: (431.372, 424.663, 234.116), 1: (430.382, 424.649, 235.569)}
T1 = np.array([[-1, 0, 0, 17.5], [0, -0.906307787, 0.422618262, -240.664], [0, 0.422618262, 0.906307787, 56.1495], [0, 0, 0, 1.]])
T2 = np.array([[1, 0, 0, -17.5], [0, 0.906307787, -0.422618262, 240.664], [0, 0.422618262, 0.906307787, 56.1495], [0, 0, 0, 1.]])
T1[:3, 3] /= 1000; T2[:3, 3] /= 1000
FLIPS = {"nominal": np.eye(3), "rz180": np.diag([-1, -1, 1.]), "rx180": np.diag([1, -1, -1.]), "ry180": np.diag([-1, 1, -1.])}


def main():
    d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "analysis.txt"), "w")
    def p(*a):
        print(*a); print(*a, file=log)
    imgs, near_imgs = [], []
    for c in (0, 1):
        m = d["camera"] == c
        D = np.zeros((480, 848)); D[d["v"][m].astype(int), d["u"][m].astype(int)] = d["depth_m"][m]
        with np.errstate(divide="ignore"):
            v = np.where(D > 0, 1 - np.clip((np.log(np.where(D > 0, D, 1)) - np.log(0.2)) / (np.log(3.5) - np.log(0.2)), 0, 1), 0)
        imgs.append((np.stack([v, v ** 0.5, np.where(D > 0, 0.3, 0)], -1) * 255).astype(np.uint8))
        near = (D > 0) & (D < 0.9); t = np.clip((D - 0.2) / 0.7, 0, 1)
        near_imgs.append((np.stack([np.where(near, 1 - t, 0), np.where(near, np.sin(np.pi * t), 0), np.where(near, t, np.where(D > 0, 0.15, 0))], -1) * 255).astype(np.uint8))
        z = d["depth_m"][m]
        p(f"camera {'AB'[c]}: {m.sum()} points; depth histogram (0-0.3-0.5-0.7-1-1.5-2-3-5 m): {np.histogram(z, bins=[0, .3, .5, .7, 1, 1.5, 2, 3, 5])[0].tolist()}")
    gap = np.full((480, 16, 3), 255, np.uint8)
    Image.fromarray(np.hstack([imgs[0], gap, imgs[1]])).save(os.path.join(out, "depth_A_B.png"))
    Image.fromarray(np.hstack([near_imgs[0], gap, near_imgs[1]])).save(os.path.join(out, "depth_A_B_near_0.2-0.9m.png"))
    P = np.c_[d["x"], d["y"], d["z"]]; cam = d["camera"]
    lo, hi = np.array([-0.765, -0.24, 0.2]), np.array([0.765, 0.24, 0.85])
    inb = np.all((P >= lo) & (P <= hi), axis=1)
    p(f"points in the work volume: camera A {int((inb & (cam == 0)).sum())}, camera B {int((inb & (cam == 1)).sum())}")
    def campts(c, maxd, step):
        m = (d["camera"] == c) & (d["depth_m"] < maxd) & (d["u"] % step == 0) & (d["v"] % step == 0)
        f, cx, cy = INTR[c]; z = d["depth_m"][m]
        return np.c_[(d["u"][m] - cx) / f * z, (d["v"][m] - cy) / f * z, z]
    A, B = campts(0, 3.5, 7), campts(1, 3.5, 7)
    p("scene agreement under camera-frame assumptions (camera A points within 3 cm of camera B points):")
    for fn, F in FLIPS.items():
        PA = (T1[:3, :3] @ F @ A.T).T + T1[:3, 3]; PB = (T2[:3, :3] @ F @ B.T).T + T2[:3, 3]
        da = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(PA)).compute_point_cloud_distance(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(PB))))
        p(f"   {fn:8}: {100 * (da < 0.03).mean():5.1f} %, median distance {np.median(da) * 1000:6.0f} mm")
    p("(a name swap rotates both clouds together and does not change the agreement)")
    log.close()


if __name__ == "__main__":
    main()
