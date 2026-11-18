"""Network protocol of trk_bridge (tracker/README.md "Network protocol"); layouts as in tracker/src/records/records.hpp.
All little-endian, fixed size, version fields."""
import struct

import numpy as np

NET_MAGIC = 0x4E4B5254          # "TRKN"
NET_VERSION = 1
TYPE_POSE, TYPE_CLOUD, TYPE_STATUS = 1, 2, 3
HEADER = struct.Struct("<IHHII")   # magic, type, version, payload_len, reserved (16 bytes)

# Pose record, version 1, 640 bytes
POSE = struct.Struct("<II4Q2qd2I3d4d3d3d3d4d36d3d2I2I8f2qQIf16x")
POSE_FIELDS = (["version", "track_state", "pose_seq", "pair_seq", "writer_instance_id", "capture_instance_id",
                "meas_time_mono_ns", "publish_time_mono_ns", "pair_dt_ms", "valid", "reject_reasons",
                "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz",
                "icp_px", "icp_py", "icp_pz", "icp_qw", "icp_qx", "icp_qy", "icp_qz"]
               + [f"cov{i}" for i in range(36)]
               + ["fit_rmse", "inlier_ratio", "min_eigenvalue", "point_count", "icp_iterations", "search_done",
                  "search_total"]
               + [f"t_{s}_us" for s in ("read", "build", "filter", "solve", "check", "track", "publish", "total")]
               + ["capture_time_a_ns", "capture_time_b_ns", "pairs_skipped", "rejected_in_row", "free_space_violation"])
assert POSE.size == 640 and len(POSE_FIELDS) == len(POSE.unpack(bytes(640)))
STATE_NAMES = {0: "search", 1: "track", 2: "lost"}

CLOUD_HEAD = struct.Struct("<IIQQ3d")       # version, count, pose_seq, pair_seq, center (48 bytes)
CLOUD_POINT = np.dtype([("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("inlier", "u1"), ("camera", "u1")])
CLOUD_MAX = 8192
CLOUD_SIZE = CLOUD_HEAD.size + CLOUD_MAX * CLOUD_POINT.itemsize
STATUS = struct.Struct("<IIQQQQQqQ")        # 64 bytes
STATUS_FIELDS = ["version", "tracker_attached", "poses_read", "clouds_read", "poses_dropped", "clouds_dropped",
                 "tracker_instance_id", "bridge_time_mono_ns", "reserved"]


def decode_pose(payload):
    return dict(zip(POSE_FIELDS, POSE.unpack(payload)))


def decode_cloud(payload):
    version, count, pose_seq, pair_seq, cx, cy, cz = CLOUD_HEAD.unpack_from(payload, 0)
    pts = np.frombuffer(payload, dtype=CLOUD_POINT, count=count, offset=CLOUD_HEAD.size)
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64) * 1e-4 + np.array([cx, cy, cz])
    return {"version": version, "pose_seq": pose_seq, "pair_seq": pair_seq, "points": xyz,
            "inlier": pts["inlier"].copy(), "camera": pts["camera"].copy()}


def decode_status(payload):
    return dict(zip(STATUS_FIELDS, STATUS.unpack(payload)))
