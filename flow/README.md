# flow: learned pose proposals for the tracker

A network trained only on synthetic depth data proposes where the object is, a second small network rejects the
proposals it does not believe, and the tracker's own ICP and fit check decide. This folder holds the one model that
is meant to be deployed, its training, its evaluation through the tracker core, and a viser demo.

Status: evaluated on synthetic scenes through the tracker core on the Mac, and live on the Jetson: an inference
process (`live/trk_flow`, C++ with TensorRT) feeds the tracker's hint buffer while the tracker searches, tested with
synthetic scenes played into the pair buffer (about 175 ms per depth pair). Not tested with the cameras or on real
data (see "Not done").

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

## Deployed configuration (Jetson Orin Nano, TensorRT 10.3)

Same weights as the table above; what changed is how they are run.

- **Solver.** Euler: 4 steps in the position stage, 4 in the pose stage (9 network evaluations with the object
  logits, instead of 33). These are the defaults of `propose.py`. Until 2026-11-27 the pose stage took 8 steps; the
  tables of this section up to "Current engines" are from that configuration.
- **Activation.** The engines use the tanh form of GELU (`export.py --act tanh_gelu`), which TensorRT runs about
  30 % faster; the weights are unchanged.
- **Verifier and single positions.** The verifier is rearranged like the network (`export.FastVerifier`, same
  output). For a scene that gives one position, `trk_flow` runs engines for 32 hypotheses instead of 128.
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

**Current engines** (`export.py --deploy`, `jetson_deploy.sh`, folder `~/flow_probe/tanh`; MAXN_SUPER mode, GPU at
1020 MHz, where the earlier engines take 153 ms instead of 173). The 1,600 scenes through `trk_flow` on the Jetson,
float16, judged by the tracker; all three runs have the same inputs:

| engines | correct / wrong / none | ms per pair, median (p90) | networks, 4 positions / 1 position |
|---|---|---|---|
| GELU, Euler 4 + 8 (`~/flow_probe/frozen`) | 1402 / 43 / 155 | 163 (179) | 159 ms / - |
| tanh-GELU, Euler 4 + 4 (deployed) | 1401 / 52 / 147 | 88 (104) | 84 ms / 32 ms |
| ReLU retrain, Euler 4 + 4 (`flow/runs/flow_relu`, not adopted) | 1366 / 47 / 187 | 75 (91) | 72 ms / 29 ms |

Deployed against the earlier engines: correct +30 -31; wrong +17 -8 (p = 0.11, not significant, but worth watching
on fresh scenes). The ReLU model (same training, `train.py --act relu`) equals the frozen model in PyTorch with 6
pose steps (1399 / 43 / 158) but loses 36 correct scenes through these engines (p < 0.001); whether the 4 steps or
float16 cause that was not separated. Engine times: position stage 13.5 ms, pose stage 41.6 (one position 10.7),
refinement 8.1 (2.6), verifier 15.9 (4.1).

Solver and other runs behind these choices (PyTorch on linbox01, tracker on the Mac, paired against Euler 4 + 6 =
1396 / 44 / 160): pose stage with Euler x 8, 5, 4, 3: 1395 / 43 / 162, 1396 / 42 / 162, 1395 / 38 / 167, 1392 / 44 /
164; two-step Adams-Bashforth x 4, 3: 1396 / 44 / 160, 1398 / 42 / 160; unequal step sizes: no difference; dropping
hypotheses during the solve (racing): 36 to 77 fewer correct; scoring one of each set of agreeing hypotheses: 4 to 11
fewer correct; INT8: no gain with GELU. A position from the tracker's last pose instead of the position stage (true
position moved by 10 to 40 mm, 32 hypotheses): 1433 to 1437 correct and 4 to 8 wrong; not built into the live path.

The first export (model as written, 4 Runge-Kutta steps, float16) took about 415 ms. Tried and not adopted: handing
over one position at a time and stopping at the first the tracker accepts (about 60 ms typical; simulated: 1393 / 46
/ 161, but 12 more wrong poses among structures); merging hypotheses that have converged (they do not converge before
the refinement); builder optimization levels 4 and 5, CUDA graphs and mixed precision (8 % at best).

## Live path on the Jetson

`live/trk_flow` reads the newest depth pair from the capture buffer while the tracker is in search state, computes
the hypotheses and publishes up to 16 in the hint buffer `/trk_hints`; the tracker tries them in its next search
cycle if they are at most 400 ms old (tracker/DESIGN.md section 13). It is the pipeline of `propose.py` in C++ with
the four engines; `live.py` is the same in Python (numpy) and was the reference for it.

The same 1,600 scenes through the whole process on the Jetson, hints judged by `trk_hint_eval` on the Mac:

| process | correct / wrong / none | ms per pair, median (p90) | of which outside the networks (median by kind) |
|---|---|---|---|
| `trt_run.py` engines, preparation in PyTorch on the Mac | 1395 / 42 / 163 | 173 (networks only) | not timed |
| `live.py` (numpy) | 1388 / 50 / 162 | 211 (314) | 24 to 116 |
| `live/trk_flow` (C++) | 1402 / 43 / 155 | 175 (191) | 2 to 15 |

The three use different random numbers; the differences are within the noise floor.

Live test (`live_test.sh`): 200 scenes (every 8th, 40 per kind), each shown for 2 s after 0.7 s of empty images, in
the pair buffer at 90 Hz, with the tracker and the inference process running together on the Jetson. First pose the
tracker accepts while the scene is shown:

| | correct / wrong / none | scene appears -> pose published, median (p90) | accepted from a hypothesis |
|---|---|---|---|
| tracker alone | 112 / 17 / 71 | 835 ms (860) | - |
| with `live.py` | 181 / 7 / 12 | 305 ms (891) | 183 |
| with `live/trk_flow` | 179 / 10 / 11 | 269 ms (310) | 186 |

- By kind, with `trk_flow` against the tracker alone: structures 35 / 4 / 1 against 1 / 9 / 30, everything 29 / 5 / 6
  against 7 / 4 / 29, hand 35 / 1 / 4 against 28 / 2 / 10.
- A proposal reaches the tracker 253 ms (median) after its pair was captured; running next to the tracker the
  networks take about 195 ms instead of 172.
- The tracker's search cycle is unchanged (p50 6.7 ms against 7.5); the cycle that tries the hypotheses takes 6.8 ms
  (median), at most 49 ms.
- About a third of the hint records are not used because they are older than 400 ms when the tracker next searches
  (a proposal finished after the tracker had already found the object).
- One run of each; the scenes are static and synthetic.

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
| `live/` | `trk_flow`: the inference process on the Jetson (C++, TensorRT); `build_jetson.sh` |
| `live.py` | the same process in Python (numpy, no torch); reference for `trk_flow` |
| `live_test.sh`, `live_eval.py` | live test on the Jetson without cameras, and its table |
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
.venv/bin/python -m flow.export --deploy --act tanh_gelu --out flow/runs/export_tanh    # the graphs of trk_flow
flow/jetson_deploy.sh start tanh                                      # ~/flow_probe/tanh: engines and deploy.tsv (detached, 20 min)
# the earlier engines (Euler 4 + 8, ~/flow_probe/frozen) and the evaluation that prepares the scenes on the Mac:
#   python -m flow.export; flow/jetson_time.sh; flow/jetson_fast.sh start stage1e:stage1 stage2e:stage2; python -m flow.trt_eval --tags fp16
# live path: tracker and inference process on the Jetson
tracker/tools/sync_to_jetson.sh && flow/live/build_jetson.sh          # ~/tracker/build, ~/flow_live/build/trk_flow
flow/live_test.sh 8 0 "hints nohints" tanh && .venv/bin/python -m flow.live_eval    # 200 scenes with and without hints, about 20 min
# by hand on the Jetson, next to d435_capture and trk_tracker:
#   ~/flow_live/build/trk_flow --config ~/tracker/config/tracker.json --engines ~/flow_probe/tanh
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
- The live path was run only with synthetic static scenes played into the pair buffer, not with the cameras, a
  moving object, a background map or a calibration file. `trk_flow` is not in `viewer/demo.sh`.
- On the M2 a whole proposal takes 1 to 3.6 s with the Runge-Kutta solver.
- No uncertainty output for the filter.
- The remaining wrong poses pass both the verifier and the tracker's fit check; most are in heavy clutter.
- One training run of each network, one object.
