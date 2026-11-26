# flow: pose hypotheses for the tracker from a learned flow on SE(3)

A network trained only on synthetic depth data proposes where the object is; the tracker's own ICP and fit check
decide. Method: Riemannian flow matching (Chen and Lipman, arXiv 2302.03660) on SO(3) x R^3, with the cloud shown to
the network in the frame of the current pose estimate.

Status: prototype on the Mac, evaluated on synthetic scenes through the tracker core. Not connected to the live
tracker, not tested on real data (see "Not done").

## Result (400 synthetic scenes, 80 per kind; model `runs/a/model.pt`)

One depth pair per scene, tracker in search state. Correct = a pose was accepted and is within 10 mm and 10 degrees of
the truth; wrong = accepted and farther; none = nothing accepted in 180 cycles (2 s of pairs). Both columns use the
same ICP and fit check (`tracker/build-mac/trk_hint_eval`).

| scenes | classical search: correct / wrong / none | flow + ICP: correct / wrong / none |
|---|---|---|
| clean (object only) | 79 / 1 / 0 | 78 / 2 / 0 |
| hand (held from above: palm, fingers, forearm) | 58 / 3 / 19 | 56 / 11 / 13 |
| rod (object on a thin rod) | 72 / 4 / 4 | 79 / 1 / 0 |
| structures (boxes, cylinders, a plate behind) | 7 / 12 / 61 | 73 / 3 / 4 |
| everything (hand + structures + plate) | 15 / 16 / 49 | 32 / 29 / 19 |
| all | 231 / 36 / 133 | 318 / 46 / 36 |

- The flow finds the object in 318 of 400 scenes, the classical search in 231. The gain is in clutter that is not
  attached to the object (structures: 73 against 7).
- It does not reduce wrong acceptances: 46 against 36. Of the accepted poses, 87 % are correct with either method.
  With a hand on the object the flow is not better (56 against 58 correct, 11 against 3 wrong). The wrong poses pass
  the tracker's fit check; the flow only chooses where ICP starts.
- The classical numbers are without a background map. For static structure the tracker's intended defence is the map
  (`trk_bgmap`), which this comparison does not use; a hand or a rod cannot be mapped.
- Time. Classical: 30 to 63 pairs until the pose (p50 per kind, 0.33 to 0.70 s; computed on the Mac, the Jetson is
  slower). Flow: the network takes 5.2 ms per evaluation of 32 hypotheses on the Jetson Orin Nano (TensorRT 10.3,
  fp16, measured with `trtexec`; 2.8 ms for 16). One proposal is 5 evaluations of 32 and 10 evaluations of 128
  hypotheses: about 234 ms by that measure (not measured end to end; cloud preparation not included). The ICP and fit
  check of the 16 hypotheses took 44 ms (p50) in one tracker cycle on the Mac. The Mac inference in this repository
  is slow (about 1.6 s per scene) and is not the number to compare.
- Trained for 64 minutes on an M2 (9000 steps of 96, after a 27-minute start with an earlier source distribution)
  on 24000 synthetic scenes. Held-out starts 20 / 45 / 65 degrees and 20 mm from the truth end within 10 degrees and
  10 mm in 99 / 95 / 66 % of the cases.

## How it works

Pose x = (R, p): rotation of the object in the rig frame and position of the model centroid.

- **Path and target** (`sampler.py`). R_t = R_0 exp(t log(R_0^T R_1)), p_t = (1 - t) p_0 + t p_1. The network predicts
  the remaining motion xi = (log(R_t^T R_1), R_t^T (p_1 - p_t)) in the body frame; the vector field of the paper is
  xi / (1 - t). Loss: squared error relative to the size of the target. x_t is perturbed by 4 degrees / 5 mm so that
  the network also corrects estimates that left the path.
- **Input** (`modelgrid.py`). The measured points within 0.15 m of p_t, in the body frame of x_t (512 random points).
  Per point: position, camera, the vector to the closest point of the CAD surface with its normal, the point-to-plane
  residual r, and r n, r (x cross n) (the terms of an ICP step). The closest-point table is a 96^3 grid.
- **Network** (`model.py`). PointNet style, 1.2 M parameters; outputs xi and a per-point object / clutter logit.
- **Source distribution.** Rotation within 70 degrees of the truth (angle uniform), position within 40 mm; 30 % of the
  samples have a uniform rotation and 100 mm, with the translation as the only target.
- **Inference** (`infer.py`). (1) Position: 32 seeds spread over the cloud, 5 evaluations with the rotation held; the
  end points are grouped, the 4 groups with the most object points are kept. (2) Rotation: at each position the flow
  starts from 32 orientations spread over SO(3) (covering radius 63 degrees), 8 Euler steps and 2 end-point
  corrections. (3) Hypotheses that agree (15 mm, 15 degrees) are grouped; the best member of each of the 16 largest
  groups goes to the tracker (`TrackerCore::set_hints`, tracker/DESIGN.md section 13), which accepts the best one
  that passes its fit check.

Two things differ from the first plan, both found by training:
1. A uniform rotation source did not train: after 3500 steps the network did not reduce the rotation error even from
   10 degrees. Most of the regression target is then not predictable from the input. The bounded source with
   orientation starts at inference replaced it.
2. Without the closest-surface inputs the rotation loss did not move in the first 750 steps; they were added before
   the run above (not compared after the change of the source, so their separate effect is not measured).

## Synthetic data (`scene.py`, `make_data.py`)

Ray casting of the object mesh and clutter from both cameras (rig, intrinsics and work volume of the tracker), then a
sensor model: disparity noise (0.05 to 0.22 px, half of it correlated over 16 px; sigma_z grows with z^2), flying
pixels and holes at depth edges, holes at grazing angles, dropout patches, missing pixels, 1 mm quantization. Each
camera is rendered from a pose 0.15 degrees / 1 mm (sigma) off its nominal pose. Clutter is made of boxes, capsules
and cylinders. None of the noise parameters was fitted to recordings of the real cameras.

## Run

```
.venv/bin/python -m flow.tests                               # flow mathematics
.venv/bin/python -m flow.make_data train --n 24000           # about 40 min on 8 cores, 1.0 GB
.venv/bin/python -m flow.make_data eval --per-cat 80         # 400 scenes, 650 MB
.venv/bin/python -m flow.train --steps 9000 --batch 96       # about 1 h on an M2
.venv/bin/python -m flow.evaluate                            # hypotheses, tracker runs, table; needs tracker/build-mac
.venv/bin/python -m flow.precompute_fields                   # flow-field slices of the demo, about 1 h (optional: the demo computes a missing scene on first view)
.venv/bin/python -m flow.demo                                # http://localhost:8090
.venv/bin/python -m flow.export --model flow/runs/a/model.pt # ONNX for TensorRT
```

Packages added to `.venv`: torch, onnx, onnxscript. `flow/data/` and `flow/runs/` are not in git.

## RNODE (`rnode.py`, `train_rnode.py`, `infer_rnode.py`, `ablate_rnode.py`, `tests_rnode.py`)

An implementation of RNODE (Finlay et al. 2020, arXiv 2002.02798) on SO(3) x R^3, tried as the training objective.
**Outcome: it does not improve the tracker result. The network built for it, trained with flow matching alone, is the
best model so far (327 correct against 318 for `runs/a`).** `runs/a` is still what the demo uses.

| model (+ ICP), 400 scenes | correct / wrong / none |
|---|---|
| classical | 231 / 36 / 133 |
| `runs/a/model.pt` (flow matching, `model.py`) | 318 / 46 / 36 |
| `runs/rnode/model_fm.pt` (RNODE network, flow matching only, 30,000 steps) | 327 / 44 / 29 |
| `runs/rnode/model_smooth_fm.pt` (smooth network, flow matching only) | 327 / 46 / 27 |
| `runs/rnode/model_smooth_rnode.pt` (smooth network + 300 RNODE updates) | 325 / 46 / 29 |

One run and one seed each; differences of a few scenes are not significant.

What is implemented: the augmented system (state, log-determinant, kinetic energy, Jacobian norm), the loss of eq. 17,
Hutchinson's estimate with one noise vector per batch, fixed-grid RK4 (4 steps). `tests_rnode.py` checks the Jacobian
terms and the log-determinant against finite differences.

Differences from the paper:
- The state is a pose. The divergence is the trace of the Jacobian with respect to a body-frame perturbation; the
  solver is RK4 on the rotation vector of each step.
- Gradients go through the solver (no adjoint).
- The base distribution is centred on the true pose (0.6 rad, 40 mm). The likelihood is a training objective only:
  it cannot be evaluated at inference, where the truth is unknown.
- The data is the true pose with dequantization noise (0.05 rad, 4 mm).
- An exact trace (6 vector-Jacobian products, `--exact`) is available in place of Hutchinson's estimate.
- A flow-matching warm start (`--fm-steps`) and an auxiliary object / clutter loss.

What happened, in order:
1. From scratch (four attempts, including the paper's settings): no learning. The network learned a drift that did
   not depend on the state.
2. Fine-tuning the warm start at batch 32: the model was destroyed in 500 updates (hit rate 68 % to 2 %).
3. Fine-tuning at 512 samples per update (`--accum 16`): stable, slightly worse.
4. The same on a network with a continuous Jacobian (`--smooth`: softmax pooling, smoothstep table weights): the
   likelihood with the 4 training steps improved (-1.559 to -1.611 nats per dimension) but with 16 steps it did not
   (-1.515 to -1.518), and the hit rate at 1 and 2 solver steps fell (19 % to 5 %, 60 % to 45 %). The model fitted
   the error of the 4-step solver.

Ablations (`runs/abl/`, `ablate_rnode.py`; 10,000-step warm starts, then 75 updates of 512 samples):
- The likelihood is close to its floor, the entropy of the dequantization noise (-1.69): the gap is 0.10 to 0.33 at
  dequantization widths of 1, 3 and 6 times, and 0.20 after 30,000 steps (exact trace, 16 solver steps). There is little to gain.
- Widening the dequantization noise did not help: RNODE did not improve the likelihood at any width, and the cosine
  between the gradients of independent batches stayed low (0.03 to 0.09 for the divergence term).
- The regularizers changed nothing useful: path straightness was the same before and after in every variant (0.9).
- The smooth network is the only variant RNODE did not degrade. Likely reason (not measured directly): with maximum
  pooling and trilinear interpolation the divergence jumps where the selected point or cell changes, and the
  gradient does not contain those jumps.
- 4 solver steps match 8 in every variant.

The smooth network costs about 7 times more per scene on the Mac CPU (10 s against 1.5 s) for the same tracker
result. Not timed on the Jetson.

```
# on the GPU machine (data copied there; see tracker/MACHINE_CHANGES.md)
python -m flow.train_rnode --run flow/runs/rnode --fm-steps 30000 --steps 0                    # warm start only
python -m flow.train_rnode --run flow/runs/x --init flow/runs/rnode/model_fm.pt --steps 300 --batch 32 --accum 16 --lr 5e-5 --clip 1e9
python -m flow.ablate_rnode --model flow/runs/rnode/model_fm.pt --out out.json
# on the Mac (use --device cpu: the MPS path stalled)
python -m flow.evaluate --method rnode --model flow/runs/rnode/model_fm.pt --eval flow/data/eval_rnode_fm --device cpu
python -m flow.tests_rnode
```

## Verifier, encoders and training variants (2026-11-24; `verifier.py`, `litept.py`, `compare.py`, `remote_eval.sh`)

**Outcome: a learned verifier that drops hypotheses before the tracker sees them is the only change that improved
the tracker result. No encoder, training recipe or data variant did.**

Evaluation set: `flow/data/eval2`, 1,600 scenes (320 per kind, seed 770001). Models are compared on the same scenes
with an exact sign test on the scenes that change (`python -m flow.compare REF OTHER...`). Two seeds of the same
recipe differ by 3 scenes in total while 65 scenes flip (+31 -34), so that is the noise floor.

| configuration (+ ICP), 1,600 scenes | correct / wrong / none | correct, paired | wrong, paired |
|---|---|---|---|
| classical | 958 / 134 / 508 | | |
| baseline: `runs/rnode/model_fm.pt`, RK4 x 4 | 1359 / 142 / 99 | reference | reference |
| same recipe, seed 1 | 1356 / 143 / 101 | +31 -34 | -30 +31 |
| verifier, reorder only (threshold 0) | 1344 / 163 / 93 | +35 -50 | -29 +50 |
| verifier, threshold 0.5 | 1403 / 62 / 135 | +71 -27 | -89 +9 |
| verifier, threshold 0.7 | 1407 / 51 / 142 | +76 -28 | -98 +7 |
| verifier, threshold 0.85 | 1407 / 42 / 151 | +82 -34 | -107 +7 |
| verifier, threshold 0.95 | 1389 / 32 / 179 | +82 -52 | -116 +6 |
| verifier 0.7, Euler x 8 | 1408 / 54 / 138 | +82 -33 | -95 +7 |

The thresholds were compared on this same set; the figure at the chosen threshold is not confirmed on fresh scenes.

- **Verifier** (`verifier.py`, 0.53 M parameters). Each hypothesis is refined by 6 point-to-plane steps against the
  closest-point table and scored from the residual features of the cloud and the coverage of 256 model points.
  Trained on hypotheses of the baseline flow from the training scenes (starts near the truth, far starts, base
  samples), label: within 10 mm and 10 degrees after refinement. Hypotheses below the threshold are not sent
  (`infer_rnode --verifier MODEL --min-score X`). Reordering alone does not help; rejecting does.
- **Solver.** Euler x 8 equals RK4 x 4 at half the network evaluations (1364 / 142 / 94 without verifier);
  Euler x 4 is 1346 / 151 / 103 and Euler x 2 is worse (p = 0.04). On validation Euler x 4 looked much better than
  RK4 x 4 (78 % against 68 %); that did not carry through the tracker.
- **Lean Point Networks encoder** (Le et al., arXiv 1907.00960; `--encoder lpn`, 2.65 M parameters, 4 times slower
  to train). Validation 80 % against 68 %, but through the tracker 1345 / 177 / 78: more wrong acceptances
  (-29 +64, p < 0.001). With the verifier at 0.5: 1402 / 78 / 120, not better than the baseline with the verifier.
- **LitePT encoder** (Yue et al., arXiv 2512.13689; `--encoder litept_t`, `litept_s`; differences from the paper at
  the top of `litept.py`). Through the tracker on eval2 (correct / wrong / none; baseline 1359 / 142 / 99, with the
  verifier at 0.85 1407 / 42 / 151): `litept_t` 1355 / 167 / 78, with verifier 1415 / 51 / 134; `litept_t_rig`
  1351 / 184 / 65 (wrong: -37 +79, p < 0.001), with verifier 1413 / 50 / 137. No more correct poses than the
  baseline, more wrong ones without the verifier; not adopted.
- **LitePT speed** (`bench_litept.py`, `tests_litept.py`; batch 256, RTX 4090, 7 GB limit). The first code ran at
  0.75 steps/s; 96.5 % of a step was the backward pass of the `f[idx]` gathers (`index_put_` with accumulation).
  `index_select` (`LITEPT_SELECT=1`, the default) gives the same values and gradients: 16.0 steps/s (baseline 46.7,
  `lpn` 10.5; 5.45 GB). Cells in the rig frame (`litept_t_rig`, `litept_s_rig`; tables built once per cloud and
  cached) do not change the training speed (15.6) but one solve of 128 hypotheses takes 148 ms in place of 190 ms
  (baseline 53, `lpn` 179), and the validation likelihood is normal again (-1.48; body-frame cells +0.08).
  Fixed-shape tensors (`LITEPT_IMPL=fixed`; same outputs to 1e-15 in float64) are slower and need twice the memory:
  out of memory at batch 256, 9.1 steps/s with `LITEPT_COMPILE=1 LITEPT_AMP=1`. `litept_s` does not fit at batch
  256 under 7 GB (13.6 steps/s at batch 128). `FLOW_GPU_GB` limits the GPU memory of any flow process.
- **Training variants** (30,000 steps each, all within the noise floor unless noted): logit-normal time sampling
  1343 / 149 / 108; beta time sampling 1358 / 140 / 102; 30 % uniform source 1355 / 147 / 98; loss normalization
  1352 / 155 / 93; wider rotation source (0.9 rad) 1346 / 157 / 97; weight decay 0.01 1358 / 144 / 98; mask-gated
  pooling 1274 / 182 / 144 (worse, p < 0.001). Weight averaging (EMA 0.999, 0.9999) changed nothing on validation.
- **Data** (same number of steps): 2 times the scenes 1344 / 149 / 107; 3 times 1349 / 150 / 101; added hand and
  heavy-clutter scenes 1348 / 145 / 107. With the verifier at 0.7 they equal the baseline with the verifier.

```
flow/remote_eval.sh NAME REMOTE_MODEL [STEPS] [EVAL_SET] [WEIGHTS] [SOLVER] ["EXTRA"]   # proposals on linbox01, tracker here
flow/remote_eval.sh e2_ver70 flow/runs/rnode/model_fm.pt 4 eval2 model rk4 "--verifier flow/runs/verifier/model.pt --min-score 0.7"
python -m flow.compare flow/data/eval_e2_base flow/data/eval_e2_ver70
python -m flow.verifier --flow flow/runs/rnode/model_fm.pt --run flow/runs/verifier        # on the GPU machine, 35 min
```

## Not done

- No real data. The noise model is a guess; the result says the approach is worth real recordings, not that it works
  on them.
- No live path: there is no hint buffer and no inference process on the Jetson. Timing on the Jetson is the network
  alone.
- No uncertainty output for the filter. The agreement between hypotheses did not separate right from wrong poses
  well enough to filter with (tried: group size and predicted remaining motion; each wrong pose removed cost about
  one correct pose).
- The wrong acceptances with a hand on the object are a property of the fit check (inlier ratio and free-space test
  on points that include the hand). The per-point object / clutter output of the network is not used by the tracker.
- One training run, one seed; 80 scenes per kind (a difference of a few scenes is not significant).
