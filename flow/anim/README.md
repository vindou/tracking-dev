# Animations of the flow-based pose tracker

Eight Manim (Community Edition 0.22) scenes that explain `flow/`: how a depth cloud becomes an accepted object pose and
why each part exists. Silent, with captions. Everything shown is evaluated on synthetic scenes only.

| file | scene class | content |
|---|---|---|
| `s1_pipeline.py` | `PipelineOverview` | cloud -> flow proposals -> verifier -> ICP + fit check -> pose; the classical search for contrast |
| `s2_state_flow.py` | `StateAndFlow` | a pose as a frame; SO(3) as the ball of rotation vectors; the geodesic of `model.fm_pair` and its velocity target |
| `s3_inputs.py` | `NetworkInputs` | the cloud in the body frame of the estimate, closest-surface vectors, the 19 features |
| `s4_network.py` | `Network` | `RNet`, conditioning, the two heads; `LPNet` and `LitePT` in the encoder slot |
| `s5_ode.py` | `SolvingODE` | 32 starts flowing to a few modes; RK4 x 4 against Euler x 8 in network evaluations |
| `s6_two_stage.py` | `TwoStageInference` | seeds -> positions -> 32 orientations each -> grouping |
| `s7_verifier.py` | `VerifierScene` | refine, score, threshold; what the tracker accepts without and with it |
| `s8_results.py` | `Results` | the 1,600-scene figures of `flow/README.md` with its caveats |

`style.py` holds the look: white background, text in New Computer Modern (the LaTeX text face, taken from the TeX
installation and registered with Pango), formulas set by LaTeX itself, numbered section headings with a rule, and one
colour per concept (true pose green, estimate orange, object points blue, clutter grey, accepted indigo, rejected red,
residual / velocity vectors ochre, seeds magenta). It also holds the 3-D view (projection in numpy, drawn with 2-D
objects), the object mesh and pose glyphs, and sets the output folder and 30 fps.

## Render

From this folder, with the repository's virtual environment:

```
../../.venv/bin/manim -ql s1_pipeline.py PipelineOverview     # 480p, while iterating
../../.venv/bin/manim -qh s1_pipeline.py PipelineOverview     # 1080p, 30 fps
./render_all.sh                                               # all eight at 1080p
```

Videos go to `media/videos/<file>/1080p30/<Scene>.mp4` (`media/` is ignored by git). Needs ffmpeg, cairo, pango and a
LaTeX installation with `dvisvgm` and the New Computer Modern fonts (TeX Live has both; without the fonts the text
falls back to Times New Roman, the formulas still need LaTeX). Render one scene at a time: parallel manim processes
share a text cache and corrupt it.

## Where the pictures come from

Scenes 1, 3, 5, 6 and 7 draw stored outputs of the real pipeline for one synthetic scene (number 1451 of
`flow/data/eval2`, kind "everything"), in `assets/scene.npz` (1.2 MB):

```
.venv/bin/python -m flow.anim.precompute            # from the repository root, CPU, a few minutes
```

It runs the trained flow (`flow/runs/flow/model.pt`, RK4 x 4, and Euler x 8 from the same starts) and the verifier
(`flow/runs/verifier/model.pt`) with the solver states kept, checks the result against `propose.propose`, and
copies the tracker's accepted poses for that scene from the result files of the evaluation (`eval2/baseline.csv`,
`eval_e2_base/hints.csv`, `eval_e2_ver85/hints.csv`). The scene was chosen because it shows every step: several seeds
land on the object, the tracker accepts a wrong pose without the verifier and the true one with it. It is one scene,
not a typical one; the totals are in scene 8.

Scene 2 is an illustration (a made-up pair of poses, the offset drawn 3 times larger than a typical sample).
Scene 4 is a block diagram; widths and parameter counts were read from `flow/rnode.py` and `flow/litept.py`
(the two other encoders were removed from the code after git commit c8f5bc9; the frozen network is `model.FlowNet`). Scene 8 uses the figures of `flow/README.md`.

The cloud on screen is a sample (900 object and 3,600 clutter points); the networks used all points.
