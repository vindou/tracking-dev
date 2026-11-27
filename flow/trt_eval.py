"""The evaluation set through the TensorRT engines on the Jetson (flow/trt_run.py there) and the tracker here.

    python -m flow.trt_eval --tags fp32,fp16 [--src flow/data/eval_demo]

The Jetson runs only the networks; clouds, seeds, point sets, grouping and the tracker run on the Mac, so every
scene is prepared in passes: (1) inputs of the position stage -> Jetson; (2) with its outputs, the inputs of the pose
stage -> Jetson (flow, refinement, verifier); (3) with those, the hints, trk_hint_eval and the table. The random
numbers of a scene are the same in every pass and for every tag (CPU generator), so two tags differ only in the
engines. Results: flow/data/eval_trt_<tag>/; files exchanged: flow/runs/trt_eval/, ~/flow_probe/frozen/trt_eval/."""
import argparse
import json
import subprocess

import numpy as np
import torch

from . import evaluate, export, propose, scene
from .make_data import load_eval

ROOT = propose.ROOT
N = 512
REMOTE = "flow_probe/frozen"


class Pending(Exception):
    """The outputs of this stage are not there yet."""


class Replay:
    """The engine of propose(): records the inputs of a stage, or returns the outputs the Jetson computed. Shapes are
    fixed, so sets with fewer seeds or positions are filled with copies of the first and cut afterwards."""

    def __init__(self, d, tag, n):
        self.d, self.tag, self.i = d, tag, 0
        f1, f2 = d / f"s1_{tag}_q.npy", d / f"s2_{tag}_q.npy"
        self.out1 = {k: np.load(d / f"s1_{tag}_{k}.npy") for k in ("q", "p", "seg")} if f1.exists() else None
        self.out2 = {k: np.load(d / f"s2_{tag}_{k}.npy") for k in ("q", "p", "logit")} if f2.exists() else None
        self.in1 = None if self.out1 else {"pts": np.zeros((n, export.K1, N, 3), np.float32), "cam": np.zeros((n, export.K1, N), np.float32),
                                           "q": np.tile(np.float32([1, 0, 0, 0]), (n, export.K1, 1))}
        self.in2 = None if self.out2 else {"pts": np.zeros((n, export.CENTERS, N, 3), np.float32), "cam": np.zeros((n, export.CENTERS, N), np.float32)}

    @staticmethod
    def fill(a, k):
        a = a.numpy()
        return np.concatenate([a, np.repeat(a[:1], k - len(a), 0)])

    def stage1(self, X, c, qb):
        if self.out1 is None:
            for k, v in (("pts", X), ("cam", c), ("q", qb)):
                self.in1[k][self.i] = self.fill(v, export.K1)
            raise Pending
        k = len(X)
        return tuple(torch.from_numpy(self.out1[name][self.i][:k]) for name in ("q", "p", "seg"))

    def stage2(self, X, c, rep, qb):
        if self.out2 is None:
            self.in2["pts"][self.i], self.in2["cam"][self.i] = self.fill(X, export.CENTERS), self.fill(c, export.CENTERS)
            raise Pending
        k = len(rep)
        return (torch.from_numpy(self.out2["q"][self.i][:k]), torch.from_numpy(self.out2["p"][self.i][:k]),
                1 / (1 + np.exp(-self.out2["logit"][self.i][:k].astype(np.float64))))

    def save(self):
        """Writes the recorded inputs of the next stage; False if nothing is pending."""
        if self.out1 is None:
            for k, v in self.in1.items():
                np.save(self.d / f"s1_{k}.npy", v)
        elif self.out2 is None:
            for k, v in self.in2.items():
                np.save(self.d / f"s2_{self.tag}_{k}.npy", v)
        return self.out2 is None


def run_pass(ev, d, tag, rig, n):
    """One pass over the scenes; with all outputs there, writes the hints. Returns the Replay."""
    eng = Replay(d, tag, n)
    dev = torch.device("cpu")

    def proposer(i, depth):
        eng.i = i
        try:
            return propose.propose(None, None, rig, depth, dev, seed=i, engine=eng)
        except Pending:
            return None
    if eng.out2 is None:
        for i, cat, T, depth, _ in load_eval(ev):
            proposer(i, depth)
    else:
        propose.write_hints(ev, rig, proposer, 16)
    return eng


def remote(what, tag):
    out = subprocess.run(["ssh", "jetson", f"cd ~/{REMOTE} && python3 trt_run.py {what} trt_eval {tag}"], check=True, capture_output=True, text=True).stdout
    print(out.strip().splitlines()[-1], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="fp32,fp16")
    ap.add_argument("--src", default="flow/data/eval_demo", help="an evaluation folder (scenes.bin, meta.npz, baseline.csv)")
    a = ap.parse_args()
    src = (ROOT / a.src).resolve()
    d = ROOT / "flow/runs/trt_eval"
    d.mkdir(parents=True, exist_ok=True)
    rig = scene.Rig()
    n = len(np.load(src / "meta.npz")["cat"])
    np.save(d / "orient.npy", propose.orientations(export.ORIENT)[0].astype(np.float32))
    sync = lambda *files: subprocess.run(["rsync", "-a", *[str(d / f) for f in files], f"jetson:{REMOTE}/trt_eval/"], check=True)
    fetch = lambda pat: subprocess.run(["rsync", "-a", f"jetson:{REMOTE}/trt_eval/{pat}", str(d) + "/"], check=True)
    subprocess.run(["ssh", "jetson", f"mkdir -p ~/{REMOTE}/trt_eval"], check=True)
    subprocess.run(["scp", "-q", str(ROOT / "flow/trt_run.py"), f"jetson:{REMOTE}/"], check=True)
    for tag in a.tags.split(","):
        ev = src.parent / f"eval_trt_{tag}"
        ev.mkdir(exist_ok=True)
        for f in ("scenes.bin", "meta.npz", "baseline.csv"):
            if not (ev / f).exists():
                (ev / f).symlink_to((src / f).resolve())
        if not (d / "s1_pts.npy").exists():
            run_pass(ev, d, tag, rig, n).save()
        if not (d / f"s1_{tag}_q.npy").exists():
            sync("s1_pts.npy", "s1_cam.npy", "s1_q.npy", "orient.npy")
            remote("stage1", tag)
            fetch(f"s1_{tag}_*.npy")
        if not (d / f"s2_{tag}_q.npy").exists():
            run_pass(ev, d, tag, rig, n).save()
            sync(f"s2_{tag}_pts.npy", f"s2_{tag}_cam.npy")
            remote("stage2", tag)
            fetch(f"s2_{tag}_*.npy")
        run_pass(ev, d, tag, rig, n)
        subprocess.run([ROOT / "tracker/build-mac/trk_hint_eval", "--scenes", ev / "scenes.bin", "--hints", ev / "hints.bin", "--baseline", "0",
                        "--out", ev / "hints.csv"], check=True, cwd=ROOT / "tracker", stdout=subprocess.DEVNULL)
        evaluate.summarize(ev)
        v = [s["flow"] for s in json.load(open(ev / "results.json"))["scenes"]]
        print(f"{tag}: " + " / ".join(str(sum(x == k for x in v)) for k in ("correct", "wrong", "none")), flush=True)


if __name__ == "__main__":
    main()
