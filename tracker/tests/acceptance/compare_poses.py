#!/usr/bin/env python3
"""Compares live poses (trk_posedump CSV) with offline poses (trk_offline CSV) by pair index (standard library).
    compare_poses.py OFFLINE_CSV LIVE_CSV [--tol-mm 0.1]
Pairs compared: those where both runs are in track state with a valid pose. Prints the largest ICP and filtered
position differences and the largest orientation differences; exit code 0 if the ICP poses agree within the tolerance."""
import csv, math, sys

def load(p):
    return {int(r["pair_seq"]): r for r in csv.DictReader(open(p))}

def qang(a, b):
    # angle of the relative rotation from the vector part of conj(a) * b (well conditioned for small angles;
    # acos of the dot product would turn the 8-decimal CSV rounding into ~0.01 degree)
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    rw = aw * bw + ax * bx + ay * by + az * bz
    rx = aw * bx - ax * bw - ay * bz + az * by
    ry = aw * by + ax * bz - ay * bw - az * bx
    rz = aw * bz - ax * by + ay * bx - az * bw
    return 2 * math.degrees(math.atan2(math.sqrt(rx * rx + ry * ry + rz * rz), abs(rw)))

def main():
    off, live = load(sys.argv[1]), load(sys.argv[2])
    tol = float(sys.argv[sys.argv.index("--tol-mm") + 1]) if "--tol-mm" in sys.argv else 0.1
    both = [k for k in sorted(off) if k in live and off[k]["state"] == "1" and live[k]["state"] == "1"
            and off[k]["valid"] == "1" and live[k]["valid"] == "1"]
    mi = mf = ri = rf = 0.0
    for k in both:
        o, l = off[k], live[k]
        mi = max(mi, 1000 * math.dist([float(o[c]) for c in ("icp_px", "icp_py", "icp_pz")], [float(l[c]) for c in ("icp_px", "icp_py", "icp_pz")]))
        mf = max(mf, 1000 * math.dist([float(o[c]) for c in ("px", "py", "pz")], [float(l[c]) for c in ("px", "py", "pz")]))
        ri = max(ri, qang([o[c] for c in ("icp_qw", "icp_qx", "icp_qy", "icp_qz")], [l[c] for c in ("icp_qw", "icp_qx", "icp_qy", "icp_qz")]))
        rf = max(rf, qang([o[c] for c in ("qw", "qx", "qy", "qz")], [l[c] for c in ("qw", "qx", "qy", "qz")]))
    first_off = min((k for k in off if off[k]["state"] == "1"), default=None)
    first_live = min((k for k in live if live[k]["state"] == "1"), default=None)
    print(f"offline pairs {len(off)}, live poses {len(live)} (pair indexes {min(live) if live else '-'}..{max(live) if live else '-'})")
    print(f"first track state: offline pair {first_off}, live pair {first_live}")
    print(f"pairs compared (both in track, both valid): {len(both)}")
    print(f"max difference ICP pose: {mi:.4f} mm, {ri:.4f} deg; filtered pose: {mf:.4f} mm, {rf:.4f} deg")
    ok = len(both) > 0 and mi <= tol
    print("RESULT", "PASS" if ok else "FAIL", f"(ICP position difference <= {tol} mm)")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
