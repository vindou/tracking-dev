"""The inference process on the Jetson: depth pairs -> pose hypotheses for the tracker in search state.

    python3 live.py --tracker ~/tracker [--engines .] [--tag fp16] [--always]            live
    python3 live.py --tracker ~/tracker --scenes scenes.bin --out hints.bin              a scenes file (no buffers)

Needs only numpy and the tensorrt module of JetPack (no torch): the networks are the TensorRT engines of
flow/export.py (stage1e, stage2e, refine, score; trt_run.Engine), and everything from the tracker side comes through
the C interface tracker/src/flowio (libtrk_flowio.so, loaded with ctypes): configuration, calibration, the newest
pair of the capture pair buffer and its cloud, the state of the tracker, and the writer of the hint buffer.

Live: while the tracker is in search state (newest pose record; --always: regardless of the tracker), take the newest
pair, compute the hypotheses (numpy version of flow/propose.py: seeds, point sets, position stage, grouping of the
end positions, pose stage with refinement and verifier, grouping) and publish up to 16 of them, best first, with the
capture time of the pair (HintRecord, tracker/DESIGN.md section 13). The tracker tries them in its next search cycle
if they are not too old. One line of statistics every 10 s.

Scenes file: the same computation for every scene (format: tracker/src/apps/trk_hint_eval.cpp), written as a hints
file for trk_hint_eval; this checks the numpy version against flow/propose.py and times the preparation."""
import argparse
import ctypes
import signal
import struct
import time

import numpy as np

from trt_run import Engine

K1, CENTERS, ORIENT, N_POINTS, N_HINTS = 32, 4, 32, 512, 16
SET_RADIUS = 0.16
MIN_SCORE = 0.85
STRIDE = 2
MAX_CLOUD = 2 * (848 // STRIDE) * (480 // STRIDE)


class FioInfo(ctypes.Structure):
    _fields_ = [("centroid", ctypes.c_double * 3), ("radius", ctypes.c_double), ("wv_min", ctypes.c_double * 3), ("wv_max", ctypes.c_double * 3),
                ("pair_buffer", ctypes.c_char * 64), ("pose_buffer", ctypes.c_char * 64), ("hint_buffer", ctypes.c_char * 64),
                ("width", ctypes.c_int), ("height", ctypes.c_int)]


class FioPair(ctypes.Structure):
    _fields_ = [("pair_seq", ctypes.c_uint64), ("capture_instance_id", ctypes.c_uint64), ("meas_time_mono_ns", ctypes.c_int64),
                ("read_done_mono_ns", ctypes.c_int64), ("points", ctypes.c_int32), ("reserved", ctypes.c_int32)]


class Tracker:
    """The C interface of the tracker side (tracker/src/flowio/flowio.cpp)."""

    def __init__(self, lib, config):
        L = self.L = ctypes.CDLL(lib)
        L.fio_open.restype = ctypes.c_void_p
        L.fio_mono_ns.restype = ctypes.c_int64
        self.err = ctypes.create_string_buffer(512)
        self.h = ctypes.c_void_p(L.fio_open(config.encode(), self.err, 512))
        if not self.h:
            raise SystemExit(f"error: {self.err.value.decode()}")
        self.info = FioInfo()
        L.fio_info(self.h, ctypes.byref(self.info))
        self.centroid = np.array(self.info.centroid)
        self.P = np.zeros((MAX_CLOUD, 3), np.float32)
        self.C = np.zeros(MAX_CLOUD, np.uint8)
        self.pair = FioPair()
        self._p = self.P.ctypes.data_as(ctypes.c_void_p)
        self._c = self.C.ctypes.data_as(ctypes.c_void_p)

    def msg(self):
        return self.err.value.decode()

    def default_cameras(self):
        assert self.L.fio_default_cameras(self.h, self.err, 512) == 0, self.msg()

    def attach(self):
        return self.L.fio_attach_pairs(self.h, self.err, 512)

    def cloud(self, a, b):
        """The cloud of two full images (uint16 arrays)."""
        n = self.L.fio_cloud(self.h, a.ctypes.data_as(ctypes.c_void_p), b.ctypes.data_as(ctypes.c_void_p), STRIDE, self._p, self._c, MAX_CLOUD)
        return self.P[:n], self.C[:n]

    def read_cloud(self, timeout_ms=200):
        """Newest pair: (status, points, cameras); status 0 ok, 1 timeout, 2 writer gone. self.pair describes the pair."""
        rc = self.L.fio_read_cloud(self.h, timeout_ms, STRIDE, self._p, self._c, MAX_CLOUD, ctypes.byref(self.pair))
        n = self.pair.points if rc == 0 else 0
        return rc, self.P[:n], self.C[:n]

    def state(self):
        return self.L.fio_tracker_state(self.h)

    def hints_open(self):
        if self.L.fio_hints_open(self.h, self.err, 512) != 0:
            raise SystemExit(f"error: hint buffer: {self.msg()}")

    def publish(self, T, score, prep_ms, net_ms, post_ms, hypotheses):
        T = np.ascontiguousarray(T, np.float64).reshape(-1, 16)
        s = np.ascontiguousarray(score, np.float32)
        self.L.fio_hints_publish(self.h, ctypes.byref(self.pair), len(T), T.ctypes.data_as(ctypes.c_void_p), s.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_float(prep_ms), ctypes.c_float(net_ms), ctypes.c_float(post_ms), int(hypotheses))

    def close(self):
        self.L.fio_close(self.h)


# ---------------- the proposal (numpy version of flow/propose.py) ----------------
def qmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def orientations(n):
    """n rotations spread evenly over SO(3), as flow.propose.orientations (same random numbers, same set)."""
    rng = np.random.default_rng(5)
    q = rng.normal(size=(4000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sel = [0]
    d = 1 - np.abs(q @ q[0])
    while len(sel) < n:
        sel.append(int(np.argmax(d)))
        d = np.minimum(d, 1 - np.abs(q @ q[sel[-1]]))
    return q[sel].astype(np.float32)


def seeds_fps(P, k, rng, voxel=0.03, min_pts=4):
    """k start positions spread over the cloud: farthest-point selection on the centres of the occupied 3 cm cells."""
    key = np.floor(P / voxel).astype(np.int64) + 512
    key = (key[:, 0] << 20 | key[:, 1]) << 20 | key[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    cen = np.stack([np.bincount(inv, P[:, i]) for i in range(3)], 1) / cnt[:, None]
    cen = cen[cnt >= min_pts]
    if len(cen) == 0:
        return np.zeros((0, 3))
    sel = [int(rng.integers(len(cen)))]
    d = np.linalg.norm(cen - cen[sel[0]], axis=1)
    while len(sel) < min(k, len(cen)):
        sel.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(cen - cen[sel[-1]], axis=1))
    return cen[sel]


class Slabs:
    """The cloud sorted along x, so that the points near a seed are looked for in a slab only."""

    def __init__(self, P, C):
        o = np.argsort(P[:, 0], kind="stable")
        self.P, self.C = P[o], C[o]
        self.x = self.P[:, 0].copy()

    def near(self, s, r):
        """Points within r of s, relative to s, and their cameras."""
        a, b = np.searchsorted(self.x, [s[0] - r, s[0] + r])
        d = self.P[a:b] - s.astype(np.float32)
        m = np.einsum("ij,ij->i", d, d) < r * r
        return d[m], self.C[a:b][m]


def point_sets(cloud, seeds, rng, rows):
    """Per seed: N_POINTS random points of the cloud within SET_RADIUS, relative to the seed, their camera (-1 / +1)
    and the mask of real points (fewer points: the first chosen one is repeated; none: zeros). Arrays of `rows`
    seeds: the rows after the last seed are copies of the first."""
    X = np.zeros((rows, N_POINTS, 3), np.float32)
    c = np.zeros((rows, N_POINTS), np.float32)
    ok = np.zeros((rows, N_POINTS), bool)
    for i, s in enumerate(seeds):
        d, cam = cloud.near(s, SET_RADIUS)
        if len(d) == 0:
            continue
        m = min(len(d), N_POINTS)
        idx = rng.choice(len(d), m, replace=False) if len(d) > N_POINTS else rng.permutation(len(d))
        X[i, :m], X[i, m:] = d[idx], d[idx[0]]
        c[i, :m], c[i, m:] = cam[idx] * 2.0 - 1.0, cam[idx[0]] * 2.0 - 1.0
        ok[i, :m] = True
    X[len(seeds):], c[len(seeds):] = X[:1], c[:1]
    return X, c, ok


def group(q, p, res, tol_m=0.015, tol_deg=15.0):
    """Groups hypotheses that agree (within tol of the group's best member); see flow.propose.group."""
    cosmin = np.cos(np.radians(tol_deg) / 2)
    groups, lead = [], []
    for i in np.argsort(res):
        if lead:
            L = np.array(lead)
            hit = np.flatnonzero((np.linalg.norm(p[L] - p[i], axis=1) < tol_m) & (np.abs(q[L] @ q[i]) > cosmin))
            if len(hit):
                groups[hit[0]].append(int(i))
                continue
        groups.append([int(i)])
        lead.append(int(i))
    groups.sort(key=lambda g: (-len(g), res[g[0]]))
    return groups


class Proposer:
    def __init__(self, engines, tag, centroid):
        self.stage1, self.stage2, self.refine, self.score = (Engine(f"{engines}/{g}.{tag}.engine") for g in ("stage1e", "stage2e", "refine", "score"))
        self.centroid = centroid
        self.Q0 = orientations(ORIENT)
        self.q0 = np.tile(self.Q0, (CENTERS, 1))
        self.rep = np.repeat(np.arange(CENTERS), ORIENT)

    def __call__(self, P, C, seed):
        """Cloud -> (T [n, 4, 4] of the hints, best first, their scores, timings in ms (prep, net, post), number of
        hypotheses, all hypotheses (q, p, score))."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        none = (np.zeros((0, 4, 4)), np.zeros(0, np.float32), (0.0, 0.0, 0.0), 0, None)
        S = seeds_fps(P, K1, rng)
        if len(S) == 0:
            return none
        cloud = Slabs(P, C)
        X, c, ok = point_sets(cloud, S, rng, K1)
        qb = rng.normal(size=(K1, 4)).astype(np.float32)
        qb /= np.linalg.norm(qb, axis=1, keepdims=True)
        t1 = time.perf_counter()
        o = self.stage1(pts=X, cam=c, q=qb)
        t2 = time.perf_counter()
        k = len(S)
        of1 = ((o["seg"][:k] > 0) & ok[:k]).sum(1) / np.maximum(ok[:k].sum(1), 1)
        e = o["p"][:k].astype(np.float64) + S
        clusters = []
        for i in np.argsort(-of1):
            for cl in clusters:
                if np.linalg.norm(e[i] - e[cl[0]]) < 0.03:
                    cl.append(i)
                    break
            else:
                clusters.append([i])
        clusters.sort(key=lambda cl: -sum(of1[j] for j in cl))
        cen = np.stack([np.average(e[cl], axis=0, weights=of1[cl] + 1e-3) for cl in clusters[:CENTERS]])
        X, c, ok = point_sets(cloud, cen, rng, CENTERS)
        X, c = X[self.rep], c[self.rep]
        t3 = time.perf_counter()
        a = self.stage2(pts=X, cam=c, q=self.q0)
        b = self.refine(pts=X, q=a["q"], p=a["p"])
        s = self.score(pts=X, cam=c, q=b["q"], p=b["p"])
        t4 = time.perf_counter()
        K = len(cen) * ORIENT
        q = b["q"][:K].astype(np.float64)
        p = b["p"][:K].astype(np.float64) + cen[self.rep[:K]]
        score = 1 / (1 + np.exp(-s["logit"][:K].astype(np.float64)))
        # the best member of each group of agreeing hypotheses, highest score first. Only hypotheses above the minimum
        # score are grouped: one below it cannot be the best of a group that has one above it.
        good = np.flatnonzero(score >= MIN_SCORE)
        sel = sorted((good[g[0]] for g in group(q[good], p[good], -score[good])), key=lambda j: -score[j])[:N_HINTS]
        R = qmat(q[sel])
        T = np.tile(np.eye(4), (len(sel), 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = p[sel] - R @ self.centroid
        t5 = time.perf_counter()
        ms = lambda x: 1e3 * x
        return T, score[sel].astype(np.float32), (ms(t1 - t0 + t3 - t2), ms(t2 - t1 + t4 - t3), ms(t5 - t4)), K, (q, p, score)


# ---------------- modes ----------------
def run_scenes(trk, prop, a):
    """Every scene of a scenes file -> hints file for trk_hint_eval."""
    trk.default_cameras()
    f = open(a.scenes, "rb")
    assert f.read(8) == b"TRKSCENE"
    _, count, w, h = struct.unpack("<IIII", f.read(16))
    count = min(count, a.limit) if a.limit else count
    times = []
    with open(a.out, "wb") as o:
        o.write(b"TRKHINTS" + struct.pack("<III", 1, count, N_HINTS))
        for i in range(count):
            f.read(128)
            img = [np.frombuffer(f.read(2 * w * h), "<u2") for _ in range(2)]
            t0 = time.perf_counter()
            P, C = trk.cloud(img[0], img[1])
            t_cloud = 1e3 * (time.perf_counter() - t0)
            T, score, t, K, _ = prop(P, C, i)
            times.append((t_cloud,) + t + (len(P), len(T)))
            o.write(struct.pack("<I", len(T)) + np.concatenate([T.reshape(-1), np.zeros(16 * (N_HINTS - len(T)))]).astype("<f8").tobytes())
    t = np.array(times)
    np.save(a.out + ".times.npy", t)
    print(f"{count} scenes: ms per scene median (p90): cloud {np.median(t[:, 0]):.1f} ({np.percentile(t[:, 0], 90):.1f}), preparation "
          f"{np.median(t[:, 1]):.1f} ({np.percentile(t[:, 1], 90):.1f}), networks {np.median(t[:, 2]):.1f} ({np.percentile(t[:, 2], 90):.1f}), "
          f"grouping {np.median(t[:, 3]):.1f} ({np.percentile(t[:, 3], 90):.1f}); points {np.median(t[:, 4]):.0f}, hints {t[:, 5].mean():.1f}")


def run_live(trk, prop, a):
    stop = []
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.append(1))
    trk.hints_open()
    print(f"flow: hint buffer {trk.info.hint_buffer.decode()}, pair buffer {trk.info.pair_buffer.decode()}, "
          f"{'always' if a.always else 'while the tracker is in search state'}", flush=True)
    attached, n_prop, n_hint, n_empty, rows = False, 0, 0, 0, []
    t_status = t_start = time.monotonic()
    while not stop:
        now = time.monotonic()
        if now - t_status >= a.status:
            t_status = now
            if rows:
                r = np.array(rows)
                print(f"[flow +{now - t_start:.0f}s] proposals {n_prop} (with hints {n_hint}, empty cloud {n_empty}) | ms p50: cloud+prep "
                      f"{np.median(r[:, 0]):.0f}, networks {np.median(r[:, 1]):.0f}, grouping {np.median(r[:, 2]):.0f}, capture->publish "
                      f"{np.median(r[:, 3]):.0f} (p90 {np.percentile(r[:, 3], 90):.0f}) | points {np.median(r[:, 4]):.0f}", flush=True)
                rows = []
            else:
                print(f"[flow +{now - t_start:.0f}s] proposals {n_prop} (idle: tracker state {trk.state()}, pair buffer {'attached' if attached else 'none'})", flush=True)
        if not attached:
            if trk.attach() != 0:
                time.sleep(0.2)
                continue
            attached = True
            print(f"[flow] pair buffer attached: {trk.msg()}", flush=True)
        if not a.always and trk.state() != 0:
            time.sleep(0.005)
            continue
        rc, P, C = trk.read_cloud()
        if rc == 2:
            attached = False
            print("[flow] pair buffer writer gone", flush=True)
            continue
        if rc == 1:
            continue
        t_cloud = (trk.L.fio_mono_ns() - trk.pair.read_done_mono_ns) / 1e6      # building the cloud, after the copy of the images
        if len(P) < 50:
            n_empty += 1
            continue
        T, score, (prep, net, post), K, _ = prop(P, C, trk.pair.pair_seq)
        n_prop += 1
        n_hint += len(T) > 0
        trk.publish(T, score, prep + t_cloud, net, post, K)
        rows.append((prep + t_cloud, net, post, (trk.L.fio_mono_ns() - trk.pair.meas_time_mono_ns) / 1e6, len(P)))
    trk.close()
    print(f"flow: stopped after {n_prop} proposals", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="/home/gchang/tracker", help="tracker directory (config/tracker.json, build/libtrk_flowio.so)")
    ap.add_argument("--config", default="")
    ap.add_argument("--lib", default="")
    ap.add_argument("--engines", default=".")
    ap.add_argument("--tag", default="fp16")
    ap.add_argument("--always", action="store_true", help="propose for every pair it can, whatever the tracker does")
    ap.add_argument("--status", type=float, default=10.0)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default="hints.bin")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    trk = Tracker(a.lib or f"{a.tracker}/build/libtrk_flowio.so", a.config or f"{a.tracker}/config/tracker.json")
    prop = Proposer(a.engines, a.tag, trk.centroid)
    if a.scenes:
        run_scenes(trk, prop, a)
    else:
        run_live(trk, prop, a)


if __name__ == "__main__":
    main()
