#!/usr/bin/env python3
"""T6 test evaluation (standard library): the background map made from the empty synthetic scene removes the support
plane points and keeps the object points; the tracker acquires the object with the map.
    check_bgmap.py MAP CLOUD_CSV OFFLINE_WITHOUT_CSV OFFLINE_WITH_CSV
Pass: >= 99 % of the plane points (z = 0.55 m, behind the object) in the work volume are in occupied cells; <= 5 % of
the object points are; with the map the tracker acquires the object and its valid poses are within 5 mm / 5 degrees
of the truth."""
import csv, math, struct, sys


def load_map(p):
    b = open(p, "rb").read()
    assert b[:8] == b"TRKBGMAP"
    o = struct.unpack_from("<3d", b, 16)
    cell, = struct.unpack_from("<d", b, 40)
    nx, ny, nz, _ = struct.unpack_from("<4I", b, 48)
    occ = b[64:64 + nx * ny * nz]
    def occupied(x, y, z):
        ix, iy, iz = int((x - o[0]) / cell), int((y - o[1]) / cell), int((z - o[2]) / cell)
        if min(ix, iy, iz) < 0 or ix >= nx or iy >= ny or iz >= nz:
            return None
        return occ[(ix * ny + iy) * nz + iz] != 0
    return occupied, sum(occ)


def stats(path):
    rows = list(csv.DictReader(open(path)))
    valid = [r for r in rows if r["valid"] == "1"]
    first = next((int(r["pair_seq"]) for r in rows if r["state"] == "1"), None)
    err = []
    for r in valid:
        dp = 1000 * math.dist([float(r[c]) for c in ("icp_px", "icp_py", "icp_pz")], [float(r[c]) for c in ("true_px", "true_py", "true_pz")])
        q1 = [float(r[c]) for c in ("icp_qw", "icp_qx", "icp_qy", "icp_qz")]; q2 = [float(r[c]) for c in ("true_qw", "true_qx", "true_qy", "true_qz")]
        d = abs(sum(a * b for a, b in zip(q1, q2)))
        err.append((dp, 2 * math.degrees(math.acos(min(1, d)))))
    bad = sum(1 for e in err if e[0] > 5 or e[1] > 5)
    return len(rows), len(valid), first, bad, max((e[0] for e in err), default=float("nan"))


WV_MIN, WV_MAX = (-0.765, -0.290, 0.235), (0.765, 0.290, 0.825)   # work volume of config/tracker.json


def main():
    occupied, ncells = load_map(sys.argv[1])
    plane = obj = plane_in = obj_in = 0
    for r in csv.DictReader(open(sys.argv[2])):
        x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
        if not all(lo <= v <= hi for v, lo, hi in zip((x, y, z), WV_MIN, WV_MAX)):
            continue   # outside the work volume: removed by the work-volume filter, not by the map
        o = occupied(x, y, z)
        if o is None:
            continue
        if abs(z - 0.55) < 0.005:
            plane += 1; plane_in += o
        elif z < 0.53:
            obj += 1; obj_in += o
    pf = 100.0 * plane_in / max(1, plane); of = 100.0 * obj_in / max(1, obj)
    print(f"map: {ncells} occupied cells; plane points in occupied cells {plane_in}/{plane} ({pf:.1f} %), object points {obj_in}/{obj} ({of:.1f} %)")
    w = stats(sys.argv[3]); m = stats(sys.argv[4])
    print(f"without map: {w[1]} valid of {w[0]}, first track at pair {w[2]}, valid poses off by > 5 mm/5 deg: {w[3]}, max position error {w[4]:.2f} mm")
    print(f"with map:    {m[1]} valid of {m[0]}, first track at pair {m[2]}, valid poses off by > 5 mm/5 deg: {m[3]}, max position error {m[4]:.2f} mm")
    ok = pf >= 99.0 and of <= 5.0 and m[2] is not None and m[3] == 0
    print("t6:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
