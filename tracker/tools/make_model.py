#!/usr/bin/env python3
"""T1: convert a STEP solid into the tracker's model files (Mac; trimesh + cascadio, numpy).

    make_model.py [--step assets/test_object.step] [--out models] [--points 3000] [--seed 1]
    make_model.py --rig assets/test_rig_main_assem.step [--out models]

Object mode writes
  <out>/<name>.stl         binary STL in meters, object frame of the STEP file, vertices joined
  <out>/<name>.model       surface points with the normal of the face of each point (format below)
  <out>/<name>.model.json  the check values of the conversion
Rig mode writes <out>/rig.stl (display only).

.model format (little-endian, fixed size header, version field):
  offset 0   char[8]  magic "TRKMODEL"
  offset 8   u32      version = 1
  offset 12  u32      point count N
  offset 16  f64[3]   bounding box min (m)      offset 40  f64[3] bounding box max (m)
  offset 64  N x { f64 px, py, pz, nx, ny, nz }  (48 bytes per point; meters, unit normal pointing out of the solid)
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import trimesh


def load_mesh(path):
    s = trimesh.load(path)   # STEP via cascadio; the loader gives meters
    m = s.to_geometry() if isinstance(s, trimesh.Scene) else s
    m.merge_vertices()
    return m


def even_samples(m, n, seed):
    """About n points spread evenly over the surface: 10 n area-weighted random samples, then farthest-point
    selection (numpy only; scipy is not a permitted package)."""
    rng = np.random.default_rng(seed)
    cand, fid = trimesh.sample.sample_surface(m, 10 * n, seed=rng)
    sel = np.empty(n, dtype=np.int64)
    sel[0] = 0
    d = np.linalg.norm(cand - cand[0], axis=1)
    for i in range(1, n):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(cand - cand[sel[i]], axis=1))
    return cand[sel], fid[sel]


def write_model(path, pts, nrm, bmin, bmax):
    with open(path, "wb") as f:
        f.write(b"TRKMODEL")
        f.write(struct.pack("<II", 1, len(pts)))
        f.write(struct.pack("<3d", *bmin))
        f.write(struct.pack("<3d", *bmax))
        f.write(np.hstack([pts, nrm]).astype("<f8").tobytes())


def object_mode(a):
    m = load_mesh(a.step)
    name = os.path.splitext(os.path.basename(a.step))[0]
    os.makedirs(a.out, exist_ok=True)
    if not m.is_winding_consistent or m.volume < 0:
        m.fix_normals()
    checks = {
        "source": os.path.basename(a.step),
        "triangles": int(len(m.faces)),
        "vertices": int(len(m.vertices)),
        "closed_surface": bool(m.is_watertight),
        "winding_consistent": bool(m.is_winding_consistent),
        "size_mm": [round(float(v) * 1000, 3) for v in m.extents],
        "bounds_m": m.bounds.tolist(),
        "volume_m3": float(m.volume),
        "area_m2": float(m.area),
    }
    pts, fid = even_samples(m, a.points, a.seed)
    nrm = m.face_normals[fid]
    checks["model_points"] = int(len(pts))
    m.export(os.path.join(a.out, name + ".stl"))
    write_model(os.path.join(a.out, name + ".model"), pts, nrm, m.bounds[0], m.bounds[1])
    # check: normals point out of the solid (closed, consistent winding and positive volume)
    checks["normals_point_out"] = bool(m.is_watertight and m.is_winding_consistent and m.volume > 0)
    with open(os.path.join(a.out, name + ".model.json"), "w") as f:
        json.dump(checks, f, indent=1)
    print(json.dumps(checks, indent=1))
    ok = (checks["triangles"] == 74 and checks["closed_surface"] and
          np.allclose(sorted(checks["size_mm"]), sorted([102.5, 92.5, 67.84]), atol=0.01))
    print("check values (74 triangles, closed, 102.5 x 92.5 x 67.84 mm):", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


def rig_mode(a):
    s = trimesh.load(a.rig)
    m = s.to_geometry() if isinstance(s, trimesh.Scene) else s
    m.merge_vertices()
    os.makedirs(a.out, exist_ok=True)
    m.export(os.path.join(a.out, "rig.stl"))
    info = {"source": os.path.basename(a.rig), "triangles": int(len(m.faces)), "bounds_m": m.bounds.tolist(),
            "parts": len(s.geometry) if isinstance(s, trimesh.Scene) else 1}
    with open(os.path.join(a.out, "rig.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="assets/test_object.step")
    ap.add_argument("--rig")
    ap.add_argument("--out", default="models")
    ap.add_argument("--points", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    return rig_mode(a) if a.rig else object_mode(a)


if __name__ == "__main__":
    sys.exit(main())
