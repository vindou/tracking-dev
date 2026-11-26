"""What two ways of spending fewer evaluations in the pose stage would do, from the hypotheses of a finished run.

    python -m flow.propose --eval flow/data/eval_X --dump ...      # writes hyp.pkl
    python -m flow.simulate --eval flow/data/eval_X

Both keep a subset of the hypotheses of the run and change nothing else, so no network is evaluated here; every
variant is run through the tracker (trk_hint_eval) like a normal evaluation.
  cascade   the positions one at a time, the most likely first; the tracker gets the hints of one position and the
            next position is processed only if it accepts none. Simulated from four runs, each with the hypotheses of
            one position alone.
  merge     during the flow, hypotheses of one position that have come within a tolerance of each other are replaced
            by one of them (the flow is deterministic: equal states stay equal). Simulated by keeping, of every such
            group, the member with the lowest index; the cost is the number of hypothesis evaluations left."""
import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from . import evaluate, propose, scene

ROOT = propose.ROOT
N_ORIENT = 32


def subset(h, keep):
    """The hypotheses dictionary write_hints needs, restricted to the indices keep."""
    keep = np.asarray(keep, int)
    q, p = h["q"][keep], h["p"][keep]
    score = None if h.get("score") is None else h["score"][keep]
    return dict(q=q, p=p, T=h["T"][keep], score=score, min_score=propose.MIN_SCORE, centers=h["centers"], seeds=h["centers"],
                groups=propose.group(q, p, -score) if len(keep) else [], t_prep=0.0, t_flow=0.0)


def run_variant(src, name, hyp, keep_of, rig):
    """Writes the hints of the variant (keep_of(i, h) -> indices), runs the tracker, returns the per-scene results."""
    ev = src.parent / f"{src.name}_sim_{name}"
    ev.mkdir(exist_ok=True)
    for f in ("scenes.bin", "meta.npz", "baseline.csv"):
        if not (ev / f).exists():
            (ev / f).symlink_to((src / f).resolve())
    if not (ev / "results.json").exists():
        propose.write_hints(ev, rig, lambda i, depth: subset(hyp[i], keep_of(i, hyp[i])), 16)
        subprocess.run([ROOT / "tracker/build-mac/trk_hint_eval", "--scenes", ev / "scenes.bin", "--hints", ev / "hints.bin", "--baseline", "0",
                        "--out", ev / "hints.csv"], check=True, cwd=ROOT / "tracker", stdout=subprocess.DEVNULL)
        evaluate.summarize(ev)
    return json.load(open(ev / "results.json"))["scenes"]


def counts(verdicts):
    return " / ".join(str(sum(v == k for v in verdicts)) for k in ("correct", "wrong", "none"))


def merge_keep(h, steps, tol_mm, tol_deg):
    """Indices that survive merging after the given flow steps, and the hypothesis evaluations spent (exact, and with
    the batch rounded up to 32, 64 or 128)."""
    tq, tp = h["traj_q"], h["traj_p"]
    S, K = len(tq) - 1, tq.shape[1]
    alive = np.arange(K)
    cost = cost_pad = 0
    pad = lambda n: 0 if n == 0 else 32 if n <= 32 else 64 if n <= 64 else 128
    cosmin = np.cos(np.radians(tol_deg) / 2)
    for s in range(S):
        if s in steps:
            new = []
            for c in range((K + N_ORIENT - 1) // N_ORIENT):          # only within a position: the points differ between positions
                reps = []
                for i in alive[alive // N_ORIENT == c]:
                    for j in reps:
                        if np.linalg.norm(tp[s, i] - tp[s, j]) < tol_mm * 1e-3 and abs(float(tq[s, i] @ tq[s, j])) > cosmin:
                            break
                    else:
                        reps.append(i)
                new += reps
            alive = np.array(new, int)
        cost += len(alive)
        cost_pad += pad(len(alive))
    return alive, cost, cost_pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="flow/data/eval_euler8")
    a = ap.parse_args()
    src = (ROOT / a.eval).resolve()
    hyp = pickle.load(open(src / "hyp.pkl", "rb"))
    for h in hyp:
        if h.get("score") is None:
            h["score"] = np.zeros(len(h["q"]))
    rig = scene.Rig()
    cats = np.array([str(c) for c in np.load(src / "meta.npz")["cat"]])
    full = [s["flow"] for s in json.load(open(src / "results.json"))["scenes"]]
    print(f"the run itself                 {counts(full)}")
    same = [s["flow"] for s in run_variant(src, "all", hyp, lambda i, h: np.arange(len(h["q"])), rig)]
    print(f"all hypotheses (must agree)    {counts(same)}   scenes that differ: {sum(x != y for x, y in zip(full, same))}")

    # ---- cascade ----
    per = [[s["flow"] for s in run_variant(src, f"c{c}", hyp, lambda i, h, c=c: np.flatnonzero(np.arange(len(h["q"])) // N_ORIENT == c), rig)]
           for c in range(4)]
    ncen = np.array([(len(h["q"]) + N_ORIENT - 1) // N_ORIENT for h in hyp])
    casc, used = [], []
    for i in range(len(hyp)):
        v, u = "none", max(1, ncen[i])
        for c in range(ncen[i]):
            if per[c][i] != "none":
                v, u = per[c][i], c + 1
                break
        casc.append(v), used.append(u)
    used = np.array(used)
    print(f"\ncascade (tracker decides)      {counts(casc)}   positions processed: mean {used.mean():.2f} of {ncen.mean():.2f}")
    for c in range(4):
        print(f"  position {c + 1} alone             {counts(per[c])}")
    print("  positions processed 1 / 2 / 3 / 4: " + " / ".join(str(int((used == k).sum())) for k in (1, 2, 3, 4)))
    for cat in scene.CATEGORIES:
        m = cats == cat
        print(f"  {cat:11s} all {counts([v for v, k in zip(full, m) if k])}   cascade {counts([v for v, k in zip(casc, m) if k])}   positions {used[m].mean():.2f}")
    ch = [(x, y) for x, y in zip(full, casc) if x != y]
    print("  changes against all positions: " + ", ".join(f"{x}->{y} {ch.count((x, y))}" for x, y in sorted(set(ch))))
    # a rule without the tracker: stop at the first position with a hypothesis above the minimum score
    vr = np.array([next((c + 1 for c in range(n) if (h["score"][c * N_ORIENT:(c + 1) * N_ORIENT] >= propose.MIN_SCORE).any()), max(1, n))
                   for h, n in zip(hyp, ncen)])
    print(f"  stopping on the verifier alone would process {vr.mean():.2f} positions; same count as with the tracker in {np.mean(vr == used):.3f} of the scenes")

    # ---- merging ----
    S = len(hyp[0]["traj_q"]) - 1 if len(hyp[0]["q"]) else 8
    print(f"\nmerging ({S} flow steps; evaluations relative to all hypotheses in all steps)")
    # how many distinct states are left at each step
    for tol in (2.0, 5.0, 10.0):
        left = np.array([[len(merge_keep(h, [s], tol, tol)[0]) if len(h["q"]) else 0 for s in range(1, S)] for h in hyp[::8]])
        print(f"  tolerance {tol:.0f} mm / {tol:.0f} deg: distinct states after step 1..{S - 1} (mean over 200 scenes): " + " ".join(f"{x:.0f}" for x in left.mean(0)))
    for name, steps, tol in (("m2", (2, 4, 6), 2.0), ("m5", (2, 4, 6), 5.0), ("m10", (2, 4, 6), 10.0), ("m5late", (4, 6), 5.0)):
        info = {}

        def keep(i, h):
            if not len(h["q"]):
                info[i] = (0, 0, 0)
                return np.zeros(0, int)
            al, c, cp = merge_keep(h, steps, tol, tol)
            info[i] = (c, cp, S * len(h["q"]))
            return al
        v = [s["flow"] for s in run_variant(src, name, hyp, keep, rig)]
        if not info:                # results were cached: recompute the cost
            for i, h in enumerate(hyp):
                keep(i, h)
        c = np.array([info[i] for i in range(len(hyp))], float)
        ch = [(x, y) for x, y in zip(full, v) if x != y]
        print(f"  merge within {tol:.0f} mm / {tol:.0f} deg after steps {steps}: {counts(v)}   evaluations {c[:, 0].sum() / c[:, 2].sum():.2f} "
              f"(batches of 32/64/128: {c[:, 1].sum() / c[:, 2].sum():.2f})   changes: " + ", ".join(f"{x}->{y} {ch.count((x, y))}" for x, y in sorted(set(ch))))


if __name__ == "__main__":
    main()
