"""Synthetic data for the flow module.

    python -m flow.make_data train --n 40000 [--out flow/data/train] [--seed 1] [--procs 8]
    python -m flow.make_data eval  --per-cat 80 [--out flow/data/eval] [--seed 990001]

train: shards with, per scene, the cloud points within STORE_RADIUS of the object centroid (at most MAX_PTS, relative to
the centroid, float16), camera and object/clutter label per point, and the true pose.
eval:  scenes.bin (depth pairs for tracker/build-mac/trk_hint_eval, format in trk_hint_eval.cpp) and meta.npz
(category, true pose, label images)."""
import argparse
import os
import struct
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import geom, scene

STORE_RADIUS = 0.26
MAX_PTS = 6144
SHARD = 2000
_rig = None


def _train_scene(seed):
    global _rig
    if _rig is None:
        _rig = scene.Rig()
    rng = np.random.default_rng(seed)
    T, depth, labels, _ = scene.make_scene(_rig, rng)
    P, C, L = _rig.cloud(depth, labels=labels)
    c = T[:3, :3] @ _rig.centroid + T[:3, 3]
    d = P - c.astype(np.float32)
    keep = np.flatnonzero(np.einsum("ij,ij->i", d, d) < STORE_RADIUS ** 2)
    if len(keep) > MAX_PTS:
        keep = rng.choice(keep, MAX_PTS, replace=False)
    pts = np.zeros((MAX_PTS, 3), np.float16)
    flag = np.zeros(MAX_PTS, np.uint8)          # bit 0 camera, bit 1 object
    pts[:len(keep)] = d[keep]
    flag[:len(keep)] = C[keep] | (L[keep] << 1)
    return pts, flag, len(keep), geom.quat_from_R(T[:3, :3]).astype(np.float32), c.astype(np.float32)


def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool:
        for s in range(0, a.n, SHARD):
            f = out / f"shard_{a.seed}_{s // SHARD:04d}.npz"
            if f.exists():
                continue
            seeds = [a.seed * 10_000_000 + s + i for i in range(min(SHARD, a.n - s))]
            r = pool.map(_train_scene, seeds, chunksize=16)
            np.savez(f, pts=np.stack([x[0] for x in r]), flag=np.stack([x[1] for x in r]),
                     n=np.array([x[2] for x in r], np.int32), q=np.stack([x[3] for x in r]), c=np.stack([x[4] for x in r]))
            print(f"{f.name}: {len(r)} scenes, median points {int(np.median([x[2] for x in r]))}", flush=True)


def _eval_scene(arg):
    global _rig
    if _rig is None:
        _rig = scene.Rig()
    seed, cat = arg
    rng = np.random.default_rng(seed)
    T, depth, labels, _ = scene.make_scene(_rig, rng, scene.CATEGORIES[cat])
    return T, depth, [np.packbits(l) for l in labels]


def evalset(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cats = [c for c in scene.CATEGORIES for _ in range(a.per_cat)]
    with Pool(a.procs) as pool:
        r = pool.map(_eval_scene, [(a.seed + i, c) for i, c in enumerate(cats)], chunksize=4)
    with open(out / "scenes.bin", "wb") as f:
        f.write(b"TRKSCENE" + struct.pack("<IIII", 1, len(r), scene.W, scene.H))
        for T, depth, _ in r:
            f.write(T.astype("<f8").tobytes())
            f.write(depth[0].astype("<u2").tobytes())
            f.write(depth[1].astype("<u2").tobytes())
    np.savez_compressed(out / "meta.npz", cat=np.array(cats), T=np.stack([x[0] for x in r]),
                        labels=np.stack([np.stack(x[2]) for x in r]))
    print(f"{out}/scenes.bin: {len(r)} scenes ({os.path.getsize(out / 'scenes.bin') / 1e6:.0f} MB)")


def load_eval(path):
    """Yields (index, category, T_rig_obj, depth pair, label pair) of an eval set."""
    path = Path(path)
    meta = np.load(path / "meta.npz")
    n = scene.W * scene.H
    with open(path / "scenes.bin", "rb") as f:
        assert f.read(8) == b"TRKSCENE"
        _, count, _, _ = struct.unpack("<IIII", f.read(16))
        for i in range(count):
            T = np.frombuffer(f.read(128), "<f8").reshape(4, 4)
            depth = [np.frombuffer(f.read(2 * n), "<u2").reshape(scene.H, scene.W) for _ in range(2)]
            labels = [np.unpackbits(meta["labels"][i][c])[:n].reshape(scene.H, scene.W) for c in range(2)]
            yield i, str(meta["cat"][i]), T, depth, labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--n", type=int, default=40000)
    t.add_argument("--out", default="flow/data/train")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--procs", type=int, default=8)
    e = sub.add_parser("eval")
    e.add_argument("--per-cat", type=int, default=80)
    e.add_argument("--out", default="flow/data/eval")
    e.add_argument("--seed", type=int, default=990001)
    e.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evalset(a)
