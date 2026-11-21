"""Precomputes the flow-field slices of the demo for every evaluation scene (about 8 s per scene on an M2).

    python -m flow.precompute_fields [--model flow/runs/a/model.pt] [--eval flow/data/eval]
Scenes where the classical search fails and the flow succeeds (the demo's first selection) are done first. Scenes that
are already in <eval>/fields/ are skipped; the demo computes a missing scene itself when it is first shown."""
import argparse
import json
from pathlib import Path

import torch

from . import infer, scene
from .make_data import load_eval

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flow/runs/a/model.pt")
    ap.add_argument("--eval", default="flow/data/eval")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    ev = ROOT / a.eval
    dev = torch.device(a.device)
    net = infer.load_model(ROOT / a.model, dev)
    rig = scene.Rig()
    res = json.load(open(ev / "results.json"))["scenes"]
    first = {s["scene"] for s in res if s["classical"] != "correct" and s["flow"] == "correct"}
    for want in (True, False):
        for i, cat, T, depth, _ in load_eval(ev):
            if (i in first) != want or infer.field_cache_path(ev, i).exists():
                continue
            P, C = rig.cloud(depth)
            infer.cached_field_frames(net, rig, ev, i, P, C, T, dev)
            print(f"scene {i} ({cat})", flush=True)


if __name__ == "__main__":
    main()
