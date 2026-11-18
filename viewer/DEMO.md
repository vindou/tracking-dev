# Running the tracking demo

The demo tracks the test object live with the two D435i cameras on the Jetson and shows it in a browser on the Mac.
One script starts and stops everything.

```
Jetson: d435_capture -> trk_tracker -> trk_bridge  ==USB network==>  Mac: viewer.py (browser, http://localhost:8080)
```

## Before you start (once)

1. **Hardware.** Both cameras on the Jetson's USB 3 ports, the Jetson connected to the Mac with the USB-C network cable.
   `ssh jetson` must work from the Mac without a password prompt (the script uses SSH).
2. **Python on the Mac.** In the repository root:
   ```
   python3 -m venv .venv
   .venv/bin/pip install -r viewer/requirements.txt
   ```
3. **Programs on the Jetson.** If `~/capture/build/d435_capture`, `~/tracker/build/trk_tracker` and
   `~/tracker/build/trk_bridge` exist, skip this. Otherwise, from the Mac:
   ```
   tracker/tools/sync_to_jetson.sh            # copies tracker/ and models/ to the Jetson and builds
   ```
4. **Models.** `models/test_object.stl`, `models/test_object.model` and `models/rig.stl` must exist. To make them again:
   ```
   .venv/bin/python tracker/tools/make_model.py --step assets/test_object.step --out models
   .venv/bin/python tracker/tools/make_model.py --rig assets/test_rig_main_assem.step --out models
   ```

## Run

From the repository root on the Mac:

```
viewer/demo.sh
```

The script starts the capture program, the tracker and the bridge on the Jetson, starts the viewer, and prints:

```
  capture running (pid ...)
  tracker running (pid ...)
  bridge running (pid ...)

viewer: http://localhost:8080   (Ctrl-C stops the demo)
```

Open **http://localhost:8080** in a browser.

## Showing the object

- **Where:** in the region both cameras see. The cameras sit in the rig and look upward, angled toward each other.
  Hold the object roughly centred over the rig, about **0.4 to 0.6 m above the rig plate**, inside the overlap of the
  two green frustums. Closer than 0.3 m to a camera (red part of the frustums) gives no depth.
- **How to hold it:** from **above** (fingertips on the top edge or a corner). A hand under the object is between the
  cameras and the object and hides it.
- **Start:** hold it still for about a second. The status panel turns from orange (ACQUIRING) to green (TRACKING) and
  the orange object model appears on the object.
- **Moving:** move slowly (tested up to 0.2 m/s and 30 degrees/s). The model and its 10 s path trace follow.
- **If the panel turns red (LOST):** hold still near the centre of the view; after about half a second the tracker
  searches again and normally finds the object within a second.

## What the viewer shows

| Item | Meaning |
|---|---|
| Status panel | **ACQUIRING** (orange, with progress bar): searching for the object. **TRACKING** (green): pose updated this cycle. **TRACKING** (light green): this cycle's fit was rejected, the pose is predicted; the reason is shown. **LOST** (red): only predicting; search starts again after 45 rejected cycles. **NO LINK** (gray): the bridge is not reachable (the viewer retries every second). |
| Orange object model | Solid: fit accepted. Transparent: predicted pose. Small axes: the object frame. |
| Blue line | Path of the object over the last 10 s. |
| Point cloud | What the tracker sees around the object (whole work volume while searching). Green: matched to the model; red: not matched; light blue: not used by the solver. |
| Gray mesh, frames | The rig, the rig frame and the two camera frames. |
| Frustums | Field of view of each camera: red = 0 to 0.3 m (too close), green = 0.3 to 0.8 m. |
| Text | Object position (mm) and orientation (roll, pitch, yaw in degrees, rig frame); RMSE, inlier ratio and point count of the fit; pose rate; calculation time per cycle; delay from capture to pose publish (measured on the Jetson); link state and messages dropped by the bridge. |

Controls (folder "Display"): show / hide the point cloud, show / hide the cameras and frustums, show the filtered or the
unfiltered (raw ICP) pose, clear the path trace.

## Stop

Press **Ctrl-C** in the terminal that runs `demo.sh`. It stops the viewer, then the bridge, the tracker and the capture
program, and checks that nothing is left:

```
  processes left on the Jetson: 0, shared-memory objects left: 0
logs: viewer/demo_logs/<date_time>
```

The log folder holds the logs of all four programs and `poses.csv` (every pose the viewer received).

## Troubleshooting

| Symptom | Check |
|---|---|
| A program is "NOT running" at the start | Its log is printed below the line. A camera unplugged or a capture program still running from an earlier session: `ssh jetson pgrep -fa d435_capture`; stop it with `kill <pid>`. |
| Panel stays "NO LINK" | The bridge listens on 192.168.55.1:9870. `ping 192.168.55.1` from the Mac; if the address differs, run `JETSON_IP=<address> viewer/demo.sh` and set `bridge.listen_address` in `tracker/config/tracker.json`. |
| Port 8080 in use | `HTTP_PORT=8081 viewer/demo.sh` |
| Stays ACQUIRING with the object in view | Hand under the object, object too close (red frustum) or outside the green overlap, or too much of it hidden. Hold it still from above near the centre. |
| Model sits beside the object, or the two cameras' points do not overlap | The camera transforms do not match the rig. See "Calibration" below. |
| Ctrl-C did not stop the Jetson programs | `ssh jetson "pkill -TERM -f 'build/(trk_|d435_)'"` |

## Calibration and set-up steps (optional, improve accuracy)

The demo uses the camera transforms in `tracker/config/tracker.json`. They come from the corrected rig assembly
(`assets/test_rig_main_assem.step` of 2026-11-17: cameras at y = ±240.664 mm, the same values as
`assets/d435i_depth_transforms.json`). They are not yet calibrated. To calibrate (run these on the Jetson in `~/tracker`, with the
capture program running: `~/capture/build/d435_capture --config ~/capture/capture.conf &`, and stop it afterwards):

1. **Which camera is which** (only if the cameras were swapped or replaced):
   `build/trk_camid --config config/tracker.json --write`, then cover the camera on the -Y side of the rig when asked.
2. **Background map** (scene without the object): `build/trk_bgmap --out config/bgmap.bin --config config/tracker.json`,
   then set `"background_map": "bgmap.bin"` in the configuration.
3. **Camera calibration** (object held completely still in view of both cameras for about 4 s):
   `build/trk_calib --out config/calib.json --config config/tracker.json`. If it reports a difference within 20 mm and
   5 degrees, set `"calibration_file": "calib.json"` under `"cameras"`.

Then copy the configuration and the calibration file back to the Mac
(`scp jetson:tracker/config/tracker.json jetson:tracker/config/calib.json tracker/config/`) so that the viewer draws
the cameras where the tracker has them.
