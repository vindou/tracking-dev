# Changes made on the Jetson and on the Mac

Nothing was installed or removed as a system package, no `sudo` was used, and no system setting, firmware, kernel or
clock was changed on either machine. Everything below is in the user's home directory or the repository.

## Jetson (`ssh jetson`, home `/home/gchang`)

| Path | What | Size | Remove with |
|---|---|---|---|
| `~/tracker/` | copy of `tracker/` (by `tools/sync_to_jetson.sh`) with three builds: `build/`, `build-tsan/`, `build-asan/` (ASan + UBSan) | 1.5 GB | `rm -rf ~/tracker` (stops the demo) |
| `~/tracker/third_party/` | Eigen and nanoflann headers, copied from the Mac (part of the sync) | | with `~/tracker` |
| `~/models/` | object and rig models (`test_object.model`, `.stl`, `rig.stl`) | 5.3 MB | `rm -rf ~/models` (stops the demo) |
| `~/tracker_logs/` | logs of all Jetson-side test runs (copied to `tracker/logs/` on the Mac) | 20 MB | `rm -rf ~/tracker_logs` |
| `~/tracker_data/` | recorded and synthetic pair files (`*.trk`) and cloud dumps (`*.csv`): T4/T5/T6 recordings, real-scene recordings (`real_static_1`, `held_top`, `look`, `now`), synthetic replay file | 1.8 GB | `rm -rf ~/tracker_data` |
| `~/tracker/config/bgmap_real.bin` | **deleted** by the sync script (rsync `--delete`, D45); it was made by the T6 real-scene run and not used by any test. Make a new map before the demo (`DEMO.md`) | | |
| `~/flow_probe/` (2026-11-20) | two ONNX files of the flow network (untrained weights) and the `trtexec` logs of the timing run (`trt_k16.log`, `trt_k32.log`) | 10 MB | `rm -rf ~/flow_probe` |

Unchanged: `~/capture/` (capture program from the earlier task; only its existing build is run).

Shared-memory objects created while programs run (removed when they stop): `/d435_pairs` (capture), `/trk_pose`,
`/trk_cloud` (tracker).

## linbox01 (`chang940@linbox01`, shared RTX 4090; 2026-11-22 and 23, RNODE training)

| Path | What | Size | Remove with |
|---|---|---|---|
| `~/tracking-flow/` | copy of the `flow` sources needed for training, a Python 3.12 virtual environment with `torch` 2.15.0+cu132 (`.venv/`), the 12 training shards (`flow/data/train`, copied with `scp -r`), the closest-point table, all training runs (`flow/runs/`), run scripts and logs | 6.5 GB | `rm -rf ~/tracking-flow` |

Added on 2026-11-23 and 24 (encoder, verifier and training experiments), all inside `~/tracking-flow/`, which is now
14 GB: packages `open3d` 0.20.0 and `trimesh` 5.2.2 in `.venv/`; `models/` and `tracker/config/tracker.json` (copies,
for scene generation); generated data `flow/data/train2` (48,000 scenes), `train2a` (links into `train2`),
`train_hard` (24,000), `eval`, `eval2` and `eval_*` result folders; runs `flow/runs/{lpn, b_seed1, verifier, t_*,
litept_t, litept_s}`; `queue_runner2.sh`, `queue.txt`, `queue.log` (one training job at a time).

Added on 2026-11-24 (LitePT speed work), also inside `~/tracking-flow/`: `bench_all.sh`, `bench2.sh`, `bench3.sh`,
`wait_gpu.sh` (waits for free GPU memory), `bench*.jsonl`, `bench*.log`, `wait.log`, `eval_*.log`, `queue.txt.bak`,
run `flow/runs/litept_t_rig`, result folders `flow/data/eval_e2_litept_t*`. No package was installed.

Not yet updated to the cleaned code of 2026-11-25: `~/tracking-flow/flow/` still holds the old module names
(`rnode.py`, `infer_rnode.py`, `train_rnode.py`, ...). Before the next run there, copy `flow/*.py` again and copy
`flow/runs/rnode/model_fm.pt` to `flow/runs/flow/model.pt` and `flow/data/rnode_table.npy` to `flow/data/table.npy`.

Nothing outside `~/tracking-flow/` was changed. The GPU is shared: two early runs at batch 128 ran out of memory
while another user's job was on it.

## Mac (repository `tracking-dev/`)

| Path | What | In git |
|---|---|---|
| `tracker/`, `viewer/`, `models/`, `assets/` | the deliverables | yes |
| `.venv/` | Python virtual environment: `viser`, `numpy`, `trimesh`, `cascadio`, `open3d` and their dependencies (500 MB) | ignored |
| `third_party_src/` | downloaded Eigen and nanoflann sources (164 MB) | ignored |
| `tracker/build-mac/` | Mac build of the tracker library, tools and unit tests | ignored |
| `capture/build-mac/`, `capture/build-mac-tsan/` | Mac builds of the capture library, needed to link the pair reader (no source change in `capture/`) | ignored |
| `viewer/demo_logs/` | logs of the demo runs (50 MB) | ignored |
| `.venv/` (2026-11-20, flow module) | added `torch` 2.15.0, `onnx`, `onnxscript` and their dependencies (`sympy`, `networkx`, `mpmath`, `filelock`, `fsspec`, `protobuf`, `ml_dtypes`, `onnx_ir`) | ignored |
| `flow/` | flow module (source in git); `flow/data/` (synthetic training and evaluation scenes, about 1.7 GB) and `flow/runs/` (trained model, ONNX files) | source yes, data and runs ignored |
| `flow/data/rnode_table.npy`, `flow/data/eval_rnode*`, `flow/data/eval_smooth_*`, `flow/runs/rnode/`, `flow/runs/abl/` (2026-11-23, RNODE) | closest-point table (48 MB), evaluation folders (links to `flow/data/eval` plus hint files), models copied back from linbox01, ablation results | ignored |
| `.venv/` (2026-11-23) | added `pypdf` (to read a paper) | ignored |
| `flow/data/eval2/`, `flow/data/eval_e2_*`, `flow/data/eval_fm_euler*`, `flow/runs/verifier/` (2026-11-24) | 1,600-scene evaluation set (2.6 GB), one result folder per evaluated configuration (links plus hint files), verifier model copied back from linbox01; `flow/data/` is now 4.2 GB | ignored |
| `.venv/` (2026-11-24, animations in `flow/anim/`; they also read fonts and run `latex` from the existing TeX Live, nothing installed there) | added `manim` 0.22.1 (Manim Community) and its dependencies: `av`, `beautifulsoup4`, `cloup`, `Cython`, `decorator`, `glcontext`, `isosurfaces`, `ManimPango`, `mapbox_earcut`, `moderngl`, `moderngl-window`, `pycairo`, `pydub`, `pyglet`, `pyglm`, `pyobjc-core`, `pyobjc-framework-Cocoa`, `scipy`, `screeninfo`, `skia-pathops`, `soupsieve`, `srt`, `svgelements`, `watchdog`. No installed package changed version. Nothing system-wide: ffmpeg, cairo, pango and LaTeX were already present. `.venv/` is now 1.5 GB | ignored |
| `flow/anim/media/` (2026-11-24) | rendered animation videos and manim's intermediate files | ignored |
| `flow/data/eval_demo/` (2026-11-25) | proposals of the frozen model (baseline network + verifier at 0.85) made on the Mac for the 1,600 scenes, tracker results, and the demo's cache of field plots (`fields/`, grows as scenes are viewed) | ignored |
| `flow/runs/flow/model.pt`, `flow/data/table.npy` (2026-11-25, cleanup of `flow/`) | copies of `flow/runs/rnode/model_fm.pt` and `flow/data/rnode_table.npy` under the names the cleaned code uses (the frozen model). The old runs (`flow/runs/a`, `abl`, `rnode`) and the result folders of the experiments (`flow/data/eval_e2_*`, `eval_fm_*`, `eval_rnode*`, `eval_smooth_*`, `eval_c4`) are no longer used by the flow code and can be deleted, except `eval_e2_base` and `eval_e2_ver85`, which `flow/anim/precompute.py` reads | ignored |
| `.gitignore` | entries for `.venv/`, `third_party_src/`, `viewer/poses.csv`, `viewer/demo_logs/` | yes |

Note: `.gitignore` also ignores `*.log` and `*.out`, so most files in `tracker/logs/` would not be committed as they
are. Add `!tracker/logs/**` to `.gitignore` if the logs should go into the repository.
