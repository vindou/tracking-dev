# flow: learned pose proposals for the tracker

A network trained only on synthetic depth data proposes where the object is, a second small network rejects the
proposals it does not believe, and the tracker's own ICP and fit check decide. This folder holds the one model that
is meant to be deployed, its training, its evaluation through the tracker core, and a viser demo.

Status: evaluated on synthetic scenes through the tracker core on the Mac, with the networks run either in PyTorch
on the Mac or as TensorRT engines on the Jetson (about 170 ms per depth pair there). Not connected to the live
tracker, not tested on real data (see "Not done").

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
- The table is the model with its first solver (4 Runge-Kutta steps per stage). The deployed configuration uses
  Euler steps and gives the same result within the noise; see "Deployed configuration".
- If the learned path hands nothing over, the tracker continues with the classical search. Counting that: about 1420 /
  60 / 120 (the classical part varies by a few scenes between runs).
- The minimum score of the verifier (0.85) was chosen on this same set; the figure is not confirmed on fresh scenes.

## Deployed configuration (Jetson Orin Nano, TensorRT 10.3, 25 W mode)

Same weights as the table above; what changed is how they are run.

- **Solver.** Euler: 4 steps in the position stage, 8 in the pose stage (13 network evaluations with the object
  logits, instead of 33). These are the defaults of `propose.py`.
- **Graph.** The network is rearranged for TensorRT (`export.FastNet`): the part of the 448 -> 256 layer that is the
  same for all points of a hypothesis is computed once per hypothesis, and the per-point layers are 1 x 1
  convolutions on a 16 x 32 grid with pooling layers. Same outputs as the model (below 0.01 degrees in PyTorch).
- **Precision.** float16 engines.

| run (1,600 scenes) | correct / wrong / none |
|---|---|
| Mac, PyTorch, 4 Runge-Kutta steps in both stages (the table above) | 1404 / 42 / 154 |
| Mac, PyTorch, Euler 4 + 8 | 1401 / 44 / 155 |
| Jetson, TensorRT float32, Euler 4 + 8 | 1394 / 41 / 165 |
| Jetson, TensorRT float16, Euler 4 + 8 (deployed) | 1395 / 42 / 163 |

The two Jetson runs have identical inputs and differ in 25 scenes (+13 -12 correct). The Jetson and Mac runs use
different random numbers; their differences are within the noise floor (p = 0.35 to 0.57, paired sign test).

Time per depth pair on the Jetson, networks only (median over the 1,600 scenes, including the copies between the
engines; preparing the cloud, the seeds and the point sets is not included and was not timed there):

| | position stage | pose stage, refinement, verifier | sum |
|---|---|---|---|
| float16 | 23 ms | 150 ms | 173 ms |
| float32 | 34 ms | 228 ms | 261 ms |

The first export (model as written, 4 Runge-Kutta steps, float16) took about 415 ms. Tried and not adopted: handing
over one position at a time and stopping at the first the tracker accepts (about 60 ms typical; simulated: 1393 / 46
/ 161, but 12 more wrong poses among structures); merging hypotheses that have converged (they do not converge before
the refinement); builder optimization levels 4 and 5, CUDA graphs and mixed precision (8 % at best).

## How it works

Pose z = (q, p): rotation of the object in the rig frame and position of the model centroid.

- **Flow** (`model.py`). A vector field on SO(3) x R^3, trained by flow matching: the network regresses the
  velocity of the geodesic from the true pose (t = 0) to a source sample (t = 1). The source is centred on the true
  pose (34 degrees, 40 mm), so it cannot be sampled at inference; inference starts from a set of orientations at
  candidate positions. Solver: fixed steps from t = 1 to t = 0 (Euler by default, see above; `solver="rk4"` for
  Runge-Kutta).
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
| `export.py` | ONNX graphs for TensorRT (`FastNet`: the rearranged network), inputs and reference outputs of scenes |
| `trt_run.py`, `trt_eval.py` | runs the engines on the Jetson (numpy and `tensorrt` only); the evaluation set through them and the tracker |
| `jetson_time.sh`, `jetson_tune*.sh`, `jetson_fast*.sh` | engine builds and timing experiments on the Jetson (`trtexec`) |
| `simulate.py` | stopping early and merging, simulated from the hypotheses of a finished run |
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
# Jetson (ssh jetson): graphs, engines, then the evaluation set through the engines
.venv/bin/python -m flow.export                                       # flow/runs/export/*.onnx
flow/jetson_time.sh && flow/jetson_fast.sh start stage1e:stage1 stage2e:stage2    # builds refine, score, stage1e, stage2e (detached; fast.done)
.venv/bin/python -m flow.trt_eval --tags fp16                         # about 25 min; flow/data/eval_trt_fp16
```

The demo and the animations show the stored Runge-Kutta run (`flow/data/eval_demo`) and call the solver explicitly.

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
- **Solver** (without the verifier). Euler x 8 equals RK4 x 4 at half the evaluations (1364 / 142 / 94); Euler x 4
  in both stages is 1346 / 151 / 103. The verifier is trained on Euler x 4 outputs.
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
- No live path: there is no hint buffer and no inference process on the Jetson. `trt_run.py` runs the engines on
  files of prepared inputs; the cloud, the seeds, the point sets and the grouping still run in Python on the Mac.
- On the M2 a whole proposal takes 1 to 3.6 s with the Runge-Kutta solver.
- No uncertainty output for the filter.
- The remaining wrong poses pass both the verifier and the tracker's fit check; most are in heavy clutter.
- One training run of each network, one object.
