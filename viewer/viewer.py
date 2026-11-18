#!/usr/bin/env python3
"""Live viewer (Mac) for the tracker: receives pose and cloud messages from trk_bridge (TCP) and shows them with viser.

    viewer.py [--host 192.168.55.1] [--port 9870] [--http-port 8080] [--csv FILE] [--config ../tracker/config/tracker.json]
              [--duration S] [--stats-json FILE]

Receiver thread: connects to the bridge (again every 1 s after a failure or a disconnect), reads messages, keeps the
newest pose and cloud, and writes every pose message to the CSV file (full rate). Display loop (about 30 Hz): rig frame,
camera frames, rig mesh, object mesh at the filtered (or unfiltered) pose, cloud with inlier colours, path trace of the
last 10 s, text values. Controls: clear trace, show cloud, filtered / unfiltered pose.
"""
import argparse
import csv
import signal
import json
import os
import socket
import sys
import threading
import time
from collections import deque

import numpy as np
import trimesh
import viser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Receiver(threading.Thread):
    def __init__(self, host, port, csv_path):
        super().__init__(daemon=True, name="receiver")
        self.host, self.port = host, port
        self.lock = threading.Lock()
        self.pose = None
        self.cloud = None
        self.status = None
        self.connected = False
        self.connects = 0
        self.stop = False
        self.poses = 0
        self.clouds = 0
        self.rate_times = deque(maxlen=200)
        self.per_instance = {}          # tracker instance -> {"first", "last", "received", "gaps", "out_of_order"}
        self.csv_file = open(csv_path, "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["recv_time_s"] + P.POSE_FIELDS)

    def _recv_exact(self, s, n):
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = s.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError("closed")
            got += k
        return bytes(buf)

    def run(self):
        while not self.stop:
            try:
                s = socket.create_connection((self.host, self.port), timeout=2.0)
            except OSError:
                time.sleep(1.0)        # bridge not there yet: try again every second
                continue
            s.settimeout(2.0)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connected = True
            self.connects += 1
            print(f"[viewer] connected to {self.host}:{self.port}", flush=True)
            try:
                while not self.stop:
                    try:
                        hdr = self._recv_exact(s, P.HEADER.size)
                    except socket.timeout:
                        continue
                    magic, typ, ver, length, _ = P.HEADER.unpack(hdr)
                    if magic != P.NET_MAGIC or ver != P.NET_VERSION:
                        raise ConnectionError(f"bad header magic {magic:#x} version {ver}")
                    payload = self._recv_exact(s, length)
                    now = time.time()
                    if typ == P.TYPE_POSE:
                        p = P.decode_pose(payload)
                        self.csv.writerow([f"{now:.6f}"] + [p[f] for f in P.POSE_FIELDS])
                        inst = self.per_instance.setdefault(p["writer_instance_id"], {"first": p["pose_seq"], "last": None, "received": 0, "gaps": 0, "out_of_order": 0})
                        if inst["last"] is not None:
                            if p["pose_seq"] > inst["last"] + 1:
                                inst["gaps"] += p["pose_seq"] - inst["last"] - 1
                            elif p["pose_seq"] <= inst["last"]:
                                inst["out_of_order"] += 1
                        inst["last"] = p["pose_seq"]
                        inst["received"] += 1
                        with self.lock:
                            self.pose = p
                            self.poses += 1
                            self.rate_times.append(now)
                    elif typ == P.TYPE_CLOUD:
                        c = P.decode_cloud(payload)
                        with self.lock:
                            self.cloud = c
                            self.clouds += 1
                    elif typ == P.TYPE_STATUS:
                        with self.lock:
                            self.status = P.decode_status(payload)
            except (OSError, ConnectionError) as e:
                print(f"[viewer] disconnected: {e}", flush=True)
            finally:
                self.connected = False
                s.close()
                self.csv_file.flush()
            time.sleep(1.0)

    def pose_rate(self):
        with self.lock:
            t = list(self.rate_times)
        if len(t) < 2 or time.time() - t[-1] > 1.0:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0])


def load_cameras(config_path):
    """Camera transforms from the tracker configuration (and its calibration file if one is named and present)."""
    cams = {}
    try:
        cfg = json.load(open(config_path))
        for name in ("D435i_1", "D435i_2"):
            T = np.array(cfg["cameras"][name]["T_rig_from_camera_mm"], dtype=float)
            T[:3, 3] /= 1000.0
            cams[name] = T
        cal = cfg["cameras"].get("calibration_file", "")
        if cal:
            p = cal if os.path.isabs(cal) else os.path.join(os.path.dirname(config_path), cal)
            if os.path.exists(p):
                c = json.load(open(p))
                for name in cams:
                    cams[name] = np.array(c["cameras"][name]["T_rig_from_camera"], dtype=float)
    except (OSError, KeyError, ValueError) as e:
        print(f"[viewer] camera transforms not read from {config_path}: {e}", flush=True)
    return cams


# Depth intrinsics of the two cameras (848 x 480; values of the pair-buffer header), for drawing the frustums.
INTRINSICS = {"D435i_1": (431.372, 431.372, 424.663, 234.116), "D435i_2": (430.382, 430.382, 424.649, 235.569)}
REJECT_NAMES = [(1, "too few points"), (2, "RMSE too high"), (4, "inlier ratio too low"), (8, "degenerate view"),
                (16, "jump from prediction"), (32, "searching"), (64, "free-space violation")]


def wxyz_from_R(R):
    """Rotation matrix -> quaternion (w, x, y, z), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rpy_deg(q):
    """Quaternion (w, x, y, z) -> roll, pitch, yaw (deg; rotations about rig x, y, z)."""
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


def frustum_section(T, intr, z0, z1):
    """Vertices and faces of the frustum between depths z0 and z1 (camera frame), in the rig frame."""
    fx, fy, cx, cy = intr
    corners = [(0, 0), (847, 0), (847, 479), (0, 479)]
    v = []
    for z in (z0, z1):
        for u, w in corners:
            v.append([(u - cx) / fx * z, (w - cy) / fy * z, z])
    v = (T[:3, :3] @ np.array(v).T).T + T[:3, 3]
    f = []
    for i in range(4):   # four sides, and both end caps
        j = (i + 1) % 4
        f += [[i, j, 4 + j], [i, 4 + j, 4 + i]]
    f += [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]
    return v.astype(np.float32), np.array(f, dtype=np.uint32)


PHASE_STYLE = ("height:3.6em;overflow:hidden;padding:6px 8px;border-radius:6px;color:white;"
               "font-family:monospace;line-height:1.5em;white-space:nowrap;text-overflow:ellipsis")


def _box(col, title, text):
    # fixed height, one title line and one detail line, so the panel does not change size between phases
    return (f'<div style="{PHASE_STYLE};background:{col}"><b>{title:<10}</b><br>'
            f'<span style="font-size:0.85em">{text}</span></div>').replace("  ", "&nbsp;&nbsp;")


def phase_html(pose, connected, status):
    if not connected:
        return _box("#888", "NO LINK", "waiting for the bridge")
    if pose is None:
        return _box("#888", "CONNECTED", "waiting for poses")
    st, valid = pose["track_state"], pose["valid"]
    why = ", ".join(n for b, n in REJECT_NAMES if pose["reject_reasons"] & b and b != 32) or "rejected"
    if st == 0:
        return _box("#d08a00", "ACQUIRING", f"search {pose['search_done']}/{pose['search_total']}, {pose['point_count']} points")
    if st == 1 and valid:
        return _box("#1f9d3a", "TRACKING", "pose updated (fit accepted)")
    if st == 1:
        return _box("#5b9d1f", "TRACKING", f"predicting: {why}")
    return _box("#c0392b", "LOST", f"predicting, {pose['rejected_in_row']} rejected: {why}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.55.1")
    ap.add_argument("--port", type=int, default=9870)
    ap.add_argument("--http-port", type=int, default=8080)
    ap.add_argument("--csv", default=os.path.join(HERE, "poses.csv"))
    ap.add_argument("--config", default=os.path.join(ROOT, "tracker", "config", "tracker.json"))
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--stats-json")
    ap.add_argument("--deadzone", type=float, default=0.3, help="near limit of the cameras drawn in red (m)")
    ap.add_argument("--max-range", type=float, default=0.8, help="far end of the drawn frustums (m)")
    a = ap.parse_args()

    def on_term(*_):
        raise KeyboardInterrupt()   # SIGTERM: the same clean stop as Ctrl-C
    signal.signal(signal.SIGTERM, on_term)

    rx = Receiver(a.host, a.port, a.csv)
    rx.start()
    server = viser.ViserServer(host="0.0.0.0", port=a.http_port, verbose=False)
    print(f"[viewer] open http://localhost:{a.http_port} in a browser", flush=True)
    sc = server.scene
    sc.add_frame("/rig", axes_length=0.1, axes_radius=0.002)
    # rig mesh, all gray (add_mesh_simple: add_mesh_trimesh needs scipy, which is not a permitted package)
    rig_path = os.path.join(ROOT, "models", "rig.stl")
    if os.path.exists(rig_path):
        rig = trimesh.load(rig_path)
        sc.add_mesh_simple("/rig_mesh", rig.vertices.astype(np.float32), rig.faces.astype(np.uint32),
                           color=(150, 150, 150), opacity=0.7)
    # cameras: frame, label, frustum in two sections (dead zone red, usable range green), can be hidden
    cam_handles = []
    for name, T in load_cameras(a.config).items():
        cam_handles.append(sc.add_frame(f"/cams/{name}", axes_length=0.04, axes_radius=0.001,
                                        wxyz=wxyz_from_R(T[:3, :3]), position=T[:3, 3]))
        cam_handles.append(sc.add_label(f"/cams/{name}/label", name))
        intr = INTRINSICS.get(name, (431.0, 431.0, 424.0, 240.0))
        v, f = frustum_section(T, intr, 0.0, a.deadzone)
        cam_handles.append(sc.add_mesh_simple(f"/frustum/{name}/dead", v, f, color=(220, 40, 40), opacity=0.12, side="double"))
        v, f = frustum_section(T, intr, a.deadzone, a.max_range)
        cam_handles.append(sc.add_mesh_simple(f"/frustum/{name}/usable", v, f, color=(40, 200, 70), opacity=0.08, side="double"))
    obj = trimesh.load(os.path.join(ROOT, "models", "test_object.stl"))
    ov, of = obj.vertices.astype(np.float32), obj.faces.astype(np.uint32)
    obj_solid = sc.add_mesh_simple("/object/solid", ov, of, color=(240, 140, 30), visible=False)
    obj_ghost = sc.add_mesh_simple("/object/predicted", ov, of, color=(240, 140, 30), opacity=0.35, visible=False)
    obj_frame = sc.add_frame("/object/frame", axes_length=0.05, axes_radius=0.0012, visible=False)

    gui = server.gui
    phase = gui.add_html(phase_html(None, False, None))
    prog = gui.add_progress_bar(0.0, visible=False, color="orange")
    txt = gui.add_markdown("")
    with gui.add_folder("Display"):
        chk_cloud = gui.add_checkbox("Show point cloud", initial_value=True)
        chk_cams = gui.add_checkbox("Show cameras and frustums", initial_value=True)
        dd_pose = gui.add_dropdown("Pose shown", options=("filtered", "unfiltered (ICP)"), initial_value="filtered")
        btn_clear = gui.add_button("Clear path trace")
    gui.add_markdown("Cloud: green = inlier, red = outlier, light blue = not used by the solver. Object: solid = fit "
                     "accepted, transparent = predicted.")

    @chk_cams.on_update
    def _(_):
        for h in cam_handles:
            h.visible = chk_cams.value

    trace = deque()
    btn_clear.on_click(lambda _: trace.clear())
    trace_h = None
    last_cloud_seq = None
    cloud_h = None
    flag_colors = np.array([[220, 50, 50], [40, 200, 60], [120, 170, 230]], dtype=np.uint8)
    t_start = time.time()
    try:
        while a.duration <= 0 or time.time() - t_start < a.duration:
            time.sleep(1.0 / 30.0)
            with rx.lock:
                pose, cloud, status = rx.pose, rx.cloud, rx.status
            now = time.time()
            phase.content = phase_html(pose, rx.connected, status)
            if pose is not None:
                searching = pose["track_state"] == 0
                prog.visible = searching
                if searching and pose["search_total"]:
                    prog.value = 100.0 * pose["search_done"] / pose["search_total"]
                filt = dd_pose.value == "filtered"
                pos = np.array([pose["px"], pose["py"], pose["pz"]] if filt else [pose["icp_px"], pose["icp_py"], pose["icp_pz"]])
                q = np.array([pose["qw"], pose["qx"], pose["qy"], pose["qz"]] if filt else [pose["icp_qw"], pose["icp_qx"], pose["icp_qy"], pose["icp_qz"]])
                have = not searching and np.linalg.norm(q) > 0.5
                solid = have and (pose["valid"] == 1)
                obj_solid.visible = bool(solid)
                obj_ghost.visible = bool(have and not solid)
                obj_frame.visible = bool(have)
                if have:
                    q = q / np.linalg.norm(q)
                    for h in (obj_solid, obj_ghost, obj_frame):
                        h.position = pos
                        h.wxyz = q
                    if not trace or np.linalg.norm(trace[-1][1] - pos) > 2e-4:
                        trace.append((now, pos))
                while trace and now - trace[0][0] > 10.0:
                    trace.popleft()
                if len(trace) >= 2:
                    pts = np.array([p for _, p in trace])
                    trace_h = sc.add_line_segments("/trace", np.stack([pts[:-1], pts[1:]], axis=1), colors=(40, 120, 230), thickness=0.002)
                elif trace_h is not None:
                    trace_h.visible = False
                delay_ms = (pose["publish_time_mono_ns"] - pose["meas_time_mono_ns"]) / 1e6
                r, pch, y = rpy_deg(q) if have else (0, 0, 0)
                txt.content = (
                    (f"**Object position** x {pos[0]*1000:7.1f}  y {pos[1]*1000:7.1f}  z {pos[2]*1000:7.1f} mm  \n"
                     f"**Orientation** roll {r:6.1f}  pitch {pch:6.1f}  yaw {y:6.1f} deg  \n" if have else
                     "**Object position** -  \n**Orientation** -  \n")
                    + f"**RMSE** {pose['fit_rmse']*1000:.2f} mm, **inliers** {pose['inlier_ratio']*100:.0f} %, "
                    f"**points** {pose['point_count']}  \n"
                    f"**Pose rate** {rx.pose_rate():.0f} Hz, **calculation** {pose['t_total_us']/1000:.2f} ms, "
                    f"**capture -> publish** {delay_ms:.1f} ms  \n"
                    f"**Link** {'connected' if rx.connected else 'reconnecting'}; dropped by the bridge: "
                    f"{status['poses_dropped'] if status else 0} poses, {status['clouds_dropped'] if status else 0} clouds")
            if cloud is not None and cloud["pose_seq"] != last_cloud_seq:
                last_cloud_seq = cloud["pose_seq"]
                if len(cloud["points"]):
                    col = flag_colors[np.minimum(cloud["inlier"], 2)]
                    cloud_h = sc.add_point_cloud("/cloud", cloud["points"].astype(np.float32), col, point_size=0.0025,
                                                 visible=chk_cloud.value)
            if cloud_h is not None:
                cloud_h.visible = chk_cloud.value
    except KeyboardInterrupt:
        pass
    rx.stop = True
    time.sleep(0.3)
    rx.csv_file.flush()
    stats = {"poses": rx.poses, "clouds": rx.clouds, "connects": rx.connects,
             "instances": {f"{k:016x}": v for k, v in rx.per_instance.items()}}
    print(f"[viewer] stats: {json.dumps(stats)}", flush=True)
    if a.stats_json:
        json.dump(stats, open(a.stats_json, "w"), indent=1)
    server.stop()


if __name__ == "__main__":
    main()
