# flow: learned pose proposals for the tracker

A network trained only on synthetic depth data proposes where the object is, a second small network rejects the
proposals it does not believe, and the tracker's own ICP and fit check decide. This folder holds the one model that
is meant to be deployed, its training, its evaluation through the tracker core, and a viser demo.

Status: evaluated on synthetic scenes through the tracker core on the Mac. Not connected to the live tracker, not
tested on real data, not timed on the Jetson (see "Not done").

## Result (1,600 synthetic scenes, 320 per kind; `flow/data/eval2`)

One depth pair per scene, tracker in search state. Correct = a pose was accepted and is within 10 mm and 10 degrees
of the truth; wrong = accepted and farther; none = nothing accepted. Both columns use the same ICP and fit check
(`tracker/build-mac/trk_hint_eval`).

| scenes | classical search: correct / wrong / none | flow + verifier + ICP: correct / wrong / none |
|---|---|---|
| clean (object only) | 313 / 4 / 3 | 317 / 0 / 3 |
| hand (held from above: palm, fingers, forearm) | 237 / 16 / 67 | 263 / 3 / 54 |
| rod (object on a thin rod) | 303 / 6 / 11 | 320 / 0 / 0 |
| structures (boxes, cylinders, a plate behind) | 45 / 64 / 211 | 295 / 5 / 20 |
| everything (hand + structures + plate) | 60 / 44 / 216 | 209 / 34 / 77 |
| all | 958 / 134 / 508 | 1404 / 42 / 154 |

- These are the proposals made on the Mac (`flow/data/eval_demo`, what the demo shows). The same model run on the
  GPU machine gives 1407 / 42 / 151; the two runs differ only in their random numbers.
- 47 % more correct poses than the classical search and a third of the wrong ones. Of the accepted poses, 97 % are
  correct (classical: 88 %).
- The weak kind is heavy clutter: 34 of the 42 wrong poses.
- The classical numbers are without a background map (`trk_bgmap`), the tracker's intended defence against static
  structure; a hand or a rod cannot be mapped.
- If the learned path hands nothing over, the tracker continues with the classical search. Counting that: about 1420 /
  60 / 120 (the classical part varies by a few scenes between runs).
- The minimum score of the verifier (0.85) was chosen on this same set; the figure is not confirmed on fresh scenes.

## How it works

Pose z = (q, p): rotation of the object in the rig frame and position of the model centroid.

- **Flow** (`model.py`). A vector field on SO(3) x R^3, trained by flow matching: the network regresses the
  velocity of the geodesic from the true pose (t = 0) to a source sample (t = 1). The source is centred on the true
  pose (34 degrees, 40 mm), so it cannot be sampled at inference; inference starts from a set of orientations at
  candidate positions. Solver: 4 Runge-Kutta steps from t = 1 to t = 0 (16 network evaluations).
- **Input.** 512 measured points within 0.16 m of a seed position, in the body frame of the current pose. Per
  point, 19 numbers: position, camera, the vector to the closest point of the CAD surface with its normal, the
  point-to-plane residual r, and r n, r (x cross n) (the terms of an ICP step). The closest-point table is a 128^3
  grid (`flow/data/table.npy`).
- **Network** (`model.FlowNet`). PointNet style, 1.18 M parameters. Outputs the velocity (6 numbers) and a per-point
  object / clutter logit.
- **Proposals** (`propose.py`). (1) Position: 32 seeds spread over the cloud, each integrated from a random rotation;
  the end positions are grouped and the 4 groups with the most object points are kept. (2) Pose: at each position
  the flow starts from 32 orientations spread over SO(3). (3) Verifier, then grouping (15 mm, 15 degrees); the best of
  each group goes to the tracker, highest score first, at most 16 (`TrackerCore::set_hints`, tracker/DESIGN.md
  section 13).
- **Verifier** (`verifier.py`, 0.53 M parameters). Each end pose is refined by 6 point-to-plane steps against the
  table on its own 512 points, then scored from the residual features of the cloud and from how much of 256 model
  points the cloud explains. Poses below 0.85 are not handed over. It is trained on outputs of the flow itself,
  labelled by their error after the refinement. Rejecting is what helps; reordering alone does not.

## Synthetic data (`scene.py`, `make_data.py`)

Ray casting of the object mesh and clutter from both cameras (rig, intrinsics and work volume of the tracker), then a
sensor model: disparity noise (0.05 to 0.22 px, half of it correlated over 16 px; sigma_z grows with z^2), flying
pixels and holes at depth edges, holes at grazing angles, dropout patches, missing pixels, 1 mm quantization. Each
camera is rendered from a pose 0.15 degrees / 1 mm (sigma) off its nominal pose. Clutter is made of boxes, capsules
and cylinders. None of the noise parameters was fitted to recordings of the real cameras.

## Files

| file | what |
|---|---|
| `model.py` | constants, closest-point table and per-point features, `FlowNet`, the field, the solver, training pairs |
| `verifier.py` | refinement (`icp`), `Verifier`, its training |
| `train.py` | flow-matching training of `FlowNet` |
| `propose.py` | depth pair -> hypotheses for the tracker; writes the hint file of an evaluation set |
| `evaluate.py`, `compare.py` | tracker runs and the table; paired comparison of two result folders (exact sign test) |
| `demo.py` | viser demo |
| `scene.py`, `make_data.py`, `geom.py` | synthetic scenes, data sets, rotations |
| `tests.py` | checks of the mathematics |
| `remote_eval.sh` | proposals on the GPU machine, tracker runs on the Mac |
| `anim/` | animations that explain the method (their own README) |

Models and data are not in git: `flow/runs/flow/model.pt` (flow), `flow/runs/verifier/model.pt` (verifier),
`flow/data/table.npy`, `flow/data/train` (24,000 scenes), `flow/data/eval2` (1,600 scenes).

## Run

```
.venv/bin/python -m flow.tests
.venv/bin/python -m flow.make_data train --n 24000                    # about 40 min on 8 cores, 1.0 GB
.venv/bin/python -m flow.make_data eval --per-cat 320 --out flow/data/eval2 --seed 770001
.venv/bin/python -m flow.train --make-table                           # closest-point table (needs open3d)
# on the GPU machine (data and table copied there; see tracker/MACHINE_CHANGES.md); FLOW_GPU_GB=7 limits the memory
python -m flow.train --run flow/runs/flow                             # 30,000 steps of 256; 47 steps/s on an RTX 4090
python -m flow.verifier --flow flow/runs/flow/model.pt --run flow/runs/verifier      # 15,000 steps, about 35 min
# on the Mac
.venv/bin/python -m flow.evaluate --eval flow/data/eval_demo          # proposals (about 50 min on an M2), tracker runs, table
.venv/bin/python -m flow.compare flow/data/eval_A flow/data/eval_B    # paired comparison, first folder = reference
.venv/bin/python -m flow.demo                                         # http://localhost:8090
```

An evaluation folder holds links to `scenes.bin`, `meta.npz` and `baseline.csv` of the scene set
(`flow/remote_eval.sh` makes them). Compare models through the tracker, not on the validation numbers of the
trainer: three times a model that looked better on validation was not better through the tracker.

## What was tried and left out

The code of everything below is in git at the commit "Checkpoint: verifier, encoder and training experiments"
(c8f5bc9) and was removed after it. Figures are correct / wrong / none on the 1,600 scenes; the frozen flow without
the verifier is 1359 / 142 / 99, and two seeds of it differ by +31 -34 scenes (the noise floor).

- **First flow model** (network predicts the remaining motion, Euler steps with end-point corrections): 318 / 46 /
  36 on an earlier 400-scene set, against 327 / 44 / 29 for the frozen flow there.
- **RNODE** (Finlay et al., arXiv 2002.02798: likelihood training with kinetic and Jacobian regularizers). Did not
  train from scratch; as fine-tuning it was at best neutral. The frozen flow is its network trained with the
  flow-matching warm start alone.
- **Verifier threshold.** Reorder only 1344 / 163 / 93; 0.5: 1403 / 62 / 135; 0.7: 1407 / 51 / 142; 0.85:
  1407 / 42 / 151; 0.95: 1389 / 32 / 179.
- **Solver.** Euler x 8 equals RK4 x 4 at half the evaluations (1364 / 142 / 94); Euler x 4 is 1346 / 151 / 103.
  `model.generate(..., solver="euler")` is kept because the verifier is trained on Euler x 4 outputs.
- **Encoders.** Lean Point Networks (Le et al., arXiv 1907.00960): 1345 / 177 / 78. LitePT (Yue et al., arXiv
  2512.13689), small: 1355 / 167 / 78 with cells in the body frame, 1351 / 184 / 65 with cells in the rig frame.
  Both better on validation, no more correct poses through the tracker, significantly more wrong ones; with the
  verifier they equal the frozen model (1402 / 78 / 120, 1415 / 51 / 134, 1413 / 50 / 137).
- **Training recipe** (all within the noise floor unless noted): logit-normal or beta time sampling, 30 % uniform
  source, loss normalization, wider source, weight decay, weight averaging; mask-gated pooling was worse (1274 / 182 /
  144).
- **Data.** 2 and 3 times the scenes, and added hand and heavy-clutter scenes: no change.

## Not done

- No real data. The noise model is a guess; the result says the approach is worth real recordings, not that it works
  on them.
- No live path: there is no hint buffer and no inference process on the Jetson.
- No export and no timing of this model on the Jetson. On the RTX 4090 one stage-2 solve (128 hypotheses, 16
  evaluations) takes 53 ms without the verifier; on the M2 a whole proposal takes 1 to 3.6 s.
- No uncertainty output for the filter.
- The remaining wrong poses pass both the verifier and the tracker's fit check; most are in heavy clutter.
- One training run of each network, one object.
