#!/usr/bin/env python3
"""Visual check of the nominal camera transforms (Mac, viser): the rig assembly from the STEP file (one colour per
part), the camera frames and viewing frustums from the nominal T_rig_from_camera, and the real depth points of both
cameras (from a trk_clouddump CSV) placed with those transforms.

    inspect_transforms.py [--step assets/test_rig_main_assem.step] [--cloud CSV] [--port 8080]
"""
import argparse
import time

import numpy as np
import trimesh
import viser

T = {
    "D435i_1 (camera A)": np.array([[-1, 0, 0, 17.5], [0, -0.906307787, 0.422618262, -240.664], [0, 0.422618262, 0.906307787, 56.1495], [0, 0, 0, 1.]]),
    "D435i_2 (camera B)": np.array([[1, 0, 0, -17.5], [0, 0.906307787, -0.422618262, 240.664], [0, 0.422618262, 0.906307787, 56.1495], [0, 0, 0, 1.]]),
}
for k in T:
    T[k][:3, 3] /= 1000.0
INTR = [(431.372, 424.663, 234.116), (430.382, 424.649, 235.569)]
CAM_COLORS = [(230, 60, 60), (60, 120, 230)]
PART_COLORS = [(170, 170, 180), (120, 200, 120), (220, 180, 80), (180, 120, 220), (90, 200, 210), (200, 140, 140), (150, 150, 100)]


def wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4); q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="assets/test_rig_main_assem.step")
    ap.add_argument("--cloud")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    server = viser.ViserServer(host="0.0.0.0", port=a.port, verbose=False)
    sc = server.scene
    sc.add_frame("/rig_frame", axes_length=0.1, axes_radius=0.002)
    sc.add_label("/rig_frame/label", "rig frame (assembly)")
    # rig parts from the STEP file, each with the transform of its scene node
    scene = trimesh.load(a.step)
    parts = []
    for i, node in enumerate(scene.graph.nodes_geometry):
        Tn, gname = scene.graph[node]
        g = scene.geometry[gname]
        v = trimesh.transform_points(g.vertices, Tn)
        parts.append(node)
        h = sc.add_mesh_simple(f"/rig/{i}_{node}", v.astype(np.float32), g.faces.astype(np.uint32),
                               color=(150, 150, 150), opacity=0.6, side="double")
    print("rig parts:", parts, flush=True)
    frames = []
    for c, (name, Tc) in enumerate(T.items()):
        q, p = wxyz(Tc[:3, :3]), Tc[:3, 3]
        fr = sc.add_frame(f"/cam{c}", axes_length=0.04, axes_radius=0.0008, wxyz=q, position=p)
        frames.append(fr)
        sc.add_label(f"/cam{c}/label", name + "  (x red, y green, z blue = viewing direction)")
        f = INTR[c][0]
        sc.add_camera_frustum(f"/cam{c}/frustum", fov=2 * np.arctan(240 / f), aspect=848 / 480, scale=0.15,
                              color=CAM_COLORS[c], thickness=1.0, thickness_units="screen")
    gui = server.gui
    dd = gui.add_dropdown("Camera transforms", options=("nominal", "x offset reflected (x -> -x)"), initial_value="nominal")
    gui.add_markdown("Red = camera A (D435i_1), blue = camera B (D435i_2). The frustum is the depth field of view; its "
                     "axis is the camera z axis.")
    if a.cloud:
        d = np.genfromtxt(a.cloud, delimiter=",", names=True)
        handles = []
        base = []
        for c in (0, 1):
            m = (d["camera"] == c) & (d["depth_m"] < 1.0)
            idx = np.flatnonzero(m)[::3]
            pts = np.c_[d["x"][idx], d["y"][idx], d["z"][idx]].astype(np.float32)
            base.append(pts)
            handles.append(sc.add_point_cloud(f"/cloud{c}", pts, CAM_COLORS[c], point_size=0.0015))
        chk = gui.add_checkbox("Show depth points (< 1 m)", initial_value=True)

        @chk.on_update
        def _(_):
            for h in handles:
                h.visible = chk.value
    else:
        handles, base = [], []

    @dd.on_update
    def _(_):
        refl = dd.value != "nominal"
        for c, Tc in enumerate(T.values()):
            p = Tc[:3, 3].copy()
            if refl:
                p[0] = -p[0]
            frames[c].position = p
            if handles:
                pts = base[c].copy()
                if refl:
                    pts[:, 0] += 2 * p[0]   # points move with the camera origin (rotation unchanged)
                handles[c].points = pts
    print(f"open http://localhost:{a.port}", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
