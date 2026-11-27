# Changes made on the Jetson and on the Mac

Nothing was installed or removed as a system package, and no firmware or kernel was changed on either machine. One
system setting was changed on the Jetson, with `sudo`, at the user's request (2026-11-26): the power mode, from 25W
(GPU up to 918 MHz) to MAXN_SUPER (`sudo nvpmodel -m 2`, GPU up to 1020 MHz; no reboot was needed; the mode persists
across reboots; back with `sudo nvpmodel -m 1`). All timings recorded before that were taken in the 25W mode.
A second setting was changed with `sudo` at the user's request (2026-11-27): the lowest GPU clock, from 306 MHz to
1020 MHz (`echo 1020000000 | sudo tee /sys/class/devfreq/17000000.gpu/min_freq`), so that the clock does not drop
while the inference process waits (flow/jetson_contention.sh: a proposal after 0.7 s of idle took 148 ms of network
time, 96 ms with the floor). Idle board power rose from 3.1 W to 4.4 W. The setting is lost at a reboot (nothing was
installed to restore it); back with `echo 306000000 | sudo tee /sys/class/devfreq/17000000.gpu/min_freq`.
Everything else below is in the user's home directory or the repository.

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
| `~/tracker_bench/` (2026-11-26) | a second copy of `tracker/` with its own builds (`build-generic`, `build-tuned`, `build-lto`, `build-tsan`) and the results of the CPU benchmark (`tracker/tools/bench_cpu_jetson.sh`); the demo copy `~/tracker` is not touched by it | about 1 GB | `rm -rf ~/tracker_bench` |
| `~/flow_live/` (2026-11-26) | copy of `flow/live/` and its build (`build/trk_flow`, with its own build of the tracker libraries); links TensorRT and CUDA of JetPack, nothing installed | 5 MB | `rm -rf ~/flow_live` |
| `~/flow_probe/frozen/` (2026-11-25) | ONNX graphs of the frozen flow and verifier (`flow/export.py`), the TensorRT engines built from them (float32 and float16), inputs and outputs of three evaluation scenes, `trtexec` logs (`flow/jetson_time.sh`). Engines of the timing experiments (`flow/jetson_tune.sh`: parts of one evaluation, builder levels 3 to 5, mixed precision) are in the same folder. Graphs and engines of the rearranged network (2026-11-26, `flow/jetson_fast.sh`: `parts/evalx_*`, `stage1f`, `stage2f`, `stage1e`, `stage2e`, `stage2e1`) are there too. `trt_run.py` and `trt_eval/` (2026-11-26, `flow/trt_eval.py`: inputs and outputs of the 1,600 evaluation scenes, 0.5 GB) are there too. Only `trtexec` and `python3 trt_run.py` (the `tensorrt` module and `libcudart` of JetPack) were run; nothing installed. Added for the live path: `live.py`, `trt_eval/scenes.bin` (the 1,600 evaluation scenes, 2.6 GB), the hint files of the offline checks, and `live/` (logs and pose files of the live test) | 7.8 GB | `rm -rf ~/flow_probe/frozen` |
| `~/flow_probe/tanh/`, `~/flow_probe/relu/` (2026-11-27) | graphs and float16 engines of the inference process (`flow/jetson_deploy.sh`): the frozen weights with tanh-GELU (deployed) and the ReLU retrain (not adopted), each 7 engines, `deploy.tsv` and the hints of the 1,600 scenes; `~/flow_live/build/trk_flow` rebuilt | 0.8 GB each | `rm -rf ~/flow_probe/tanh ~/flow_probe/relu` |
| `~/flow_probe/contention/` (2026-11-27) | per-scene timings, clock readings and `tegrastats` power logs of the idle-clock experiment (`flow/jetson_contention.sh`); float32 engines of the ReLU model in `~/flow_probe/relu/` | 0.4 GB with the engines | `rm -rf ~/flow_probe/contention ~/flow_probe/relu/*.fp32.engine` |
| `~/flow_probe/tanh2/` (2026-11-27) | the deployed engines: tanh-GELU, Euler 4 + 2 (two pose-stage engines built there, the other five copied from `~/flow_probe/tanh`), and the hints of the 1,600 scenes | 0.7 GB with the graphs | `rm -rf ~/flow_probe/tanh2` |

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

Updated on 2026-11-26: `flow/{__init__,model,geom,propose,scene,make_data,verifier,train}.py` copied again (the cleaned code; the other old modules are still there), and `flow/runs/flow/model.pt` and `flow/data/table.npy` made as copies under the new names. (A field computation for portfolio figures ran there with a 2 GB memory limit; its script and output were removed again.)

Added on 2026-11-26 and 27 (speed runs), inside `~/tracking-flow/`: `queue_speed.sh` and `queue2_speed.sh` to
`queue6_speed.sh` with their `.out` and `.log` files (one job at a time, GPU memory capped at 4 to 4.5 GB), the
proposal folders `flow/data/eval_lb_*` (38, one per solver, activation or idea run), and the run `flow/runs/flow_relu`
(the one training of these two days: the frozen recipe with ReLU, 30,000 steps); `flow/*.py` copied again. No
package was installed.

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
| `flow/runs/export/` (2026-11-25) | ONNX graphs of the frozen model, scene inputs and PyTorch reference outputs, and the outputs and logs fetched from the Jetson (1.5 GB with the graphs of the timing experiments and of the rearranged network) | ignored |
| `flow/runs/trt_eval/`, `flow/data/eval_trt_fp32`, `flow/data/eval_trt_fp16`, `flow/data/eval_euler8*` (2026-11-26) | inputs and outputs exchanged with the Jetson for the evaluation through TensorRT (568 MB), its result folders, and the result folders of the solver and simulation runs | ignored |
| `flow/runs/live/`, `flow/data/eval_live_np`, `flow/data/eval_live_cpp` (2026-11-26) | logs and pose files of the live test fetched from the Jetson (53 MB), result folders of the offline checks of the inference process | ignored |
| `flow/runs/export_tanh/`, `flow/runs/export_relu/`, `flow/runs/flow_relu/`, `flow/data/eval_lb_*`, `flow/data/eval_jet_*`, `flow/runs/cpuopt/`, `tracker/build-opt*`, `tracker/build-ref`, `tracker/build-prof` (2026-11-26 and 27) | graphs of the inference process (0.7 GB), the ReLU model copied back from linbox01, result folders of the solver, activation and idea runs (proposals from linbox01, links plus hint files) and of the runs through the Jetson engines, results and separate builds of the tracker CPU work | ignored |
| `flow/runs/export_tanh2/`, `flow/runs/contention/`, `flow/runs/live_{frozen,tanh_nofloor,tanh_floor,tanh2_floor}/` (2026-11-27) | graphs of the deployed engines (Euler 4 + 2), results of the idle-clock experiment, copies of the live-test results per engine set | ignored |
| `.gitignore` | entries for `.venv/`, `third_party_src/`, `viewer/poses.csv`, `viewer/demo_logs/` | yes |

Note: `.gitignore` also ignores `*.log` and `*.out`, so most files in `tracker/logs/` would not be committed as they
are. Add `!tracker/logs/**` to `.gitignore` if the logs should go into the repository.
