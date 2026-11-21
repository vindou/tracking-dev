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
.venv/bin/python -m flow.demo                                # http://localhost:8090
.venv/bin/python -m flow.export --model flow/runs/a/model.pt # ONNX for TensorRT
```

Packages added to `.venv`: torch, onnx, onnxscript. `flow/data/` and `flow/runs/` are not in git.

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
