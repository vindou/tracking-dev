"""Runs the TensorRT engines of the model on the Jetson (needs only numpy and the tensorrt module of JetPack; device
memory through libcudart). Used by flow/trt_eval.py; this file is copied to ~/flow_probe/frozen and run there.

    python3 trt_run.py stage1 <dir> <tag>      <dir>/s1_{pts,cam,q}.npy -> <dir>/s1_<tag>_{q,p,seg}.npy
    python3 trt_run.py stage2 <dir> <tag>      <dir>/s2_<tag>_{pts,cam,n}.npy, orient.npy -> <dir>/s2_<tag>_{q,p,logit}.npy

tag = fp32 or fp16: the engines <graph>.<tag>.engine of the current directory (graphs stage1e, stage2e, refine,
score; fixed shapes: 32 seeds, 4 x 32 hypotheses, 512 points). Prints the time per scene."""
import ctypes
import sys
import time

import numpy as np
import tensorrt as trt

try:
    cuda = ctypes.CDLL("libcudart.so")
except OSError:
    cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
LOG = trt.Logger(trt.Logger.WARNING)


class Engine:
    def __init__(self, path):
        self.engine = trt.Runtime(LOG).deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.io = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            assert self.engine.get_tensor_dtype(name) == trt.DataType.FLOAT, name
            ptr = ctypes.c_void_p()
            assert cuda.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(int(np.prod(shape)) * 4)) == 0
            self.io[name] = (ptr, shape, self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
            self.ctx.set_tensor_address(name, ptr.value)

    def __call__(self, **inputs):
        """inputs by name without the "_in" suffix; returns the outputs by name."""
        for name, (ptr, shape, is_in) in self.io.items():
            if is_in:
                a = np.ascontiguousarray(inputs[name[:-3]], np.float32)
                assert a.shape == shape, (name, a.shape, shape)
                assert cuda.cudaMemcpy(ptr, a.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.nbytes), 1) == 0
        assert self.ctx.execute_v2([p.value for p, _, _ in self.io.values()])
        out = {}
        for name, (ptr, shape, is_in) in self.io.items():
            if not is_in:
                a = np.empty(shape, np.float32)
                assert cuda.cudaMemcpy(a.ctypes.data_as(ctypes.c_void_p), ptr, ctypes.c_size_t(a.nbytes), 2) == 0
                out[name] = a
        return out


def main():
    what, d, tag = sys.argv[1:4]
    times = []
    if what == "stage1":
        e = Engine(f"stage1e.{tag}.engine")
        pts, cam, q = (np.load(f"{d}/s1_{k}.npy", mmap_mode="r") for k in ("pts", "cam", "q"))
        res = {k: [] for k in ("q", "p", "seg")}
        for i in range(len(pts)):
            t0 = time.perf_counter()
            o = e(pts=pts[i], cam=cam[i], q=q[i])
            times.append(time.perf_counter() - t0)
            res["q"].append(o["q"]), res["p"].append(o["p"]), res["seg"].append(o["seg"] > 0)
        for k, v in res.items():
            np.save(f"{d}/s1_{tag}_{k}.npy", np.stack(v))
    else:
        flow, refine, score = (Engine(f"{g}.{tag}.engine") for g in ("stage2e", "refine", "score"))
        pts, cam = (np.load(f"{d}/s2_{tag}_{k}.npy", mmap_mode="r") for k in ("pts", "cam"))
        orient = np.load(f"{d}/orient.npy").astype(np.float32)
        K = flow.io["q_in"][1][0]
        rep = np.repeat(np.arange(K // len(orient)), len(orient))
        q0 = np.tile(orient, (K // len(orient), 1))
        res = {k: [] for k in ("q", "p", "logit")}
        for i in range(len(pts)):
            t0 = time.perf_counter()
            X, c = pts[i][rep], cam[i][rep]
            a = flow(pts=X, cam=c, q=q0)
            b = refine(pts=X, q=a["q"], p=a["p"])
            s = score(pts=X, cam=c, q=b["q"], p=b["p"])
            times.append(time.perf_counter() - t0)
            res["q"].append(b["q"]), res["p"].append(b["p"]), res["logit"].append(s["logit"])
        for k, v in res.items():
            np.save(f"{d}/s2_{tag}_{k}.npy", np.stack(v))
    t = np.array(times[5:]) * 1e3
    print(f"{what} {tag}: {len(times)} scenes, ms per scene median {np.median(t):.1f}, p90 {np.percentile(t, 90):.1f}, max {t.max():.1f}", flush=True)


if __name__ == "__main__":
    main()
