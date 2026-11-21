"""ONNX export of the flow network (one evaluation of the vector field for K hypotheses) for TensorRT on the Jetson.

    python -m flow.export --model flow/runs/a/model.pt --out flow/runs/a/flow_k32.onnx [--k 32]
On the Jetson: /usr/src/tensorrt/bin/trtexec --onnx=flow_k32.onnx --fp16"""
import argparse

import torch

from .model import N_FEAT, N_POINTS, FlowNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="flow/runs/a/flow_k32.onnx")
    ap.add_argument("--k", type=int, default=32)
    a = ap.parse_args()
    net = FlowNet()
    if a.model:
        ck = torch.load(a.model, map_location="cpu")
        net = FlowNet(ck.get("width", 1))
        net.load_state_dict(ck["model"])
    net.eval()
    x, t = torch.randn(a.k, N_POINTS, N_FEAT), torch.rand(a.k)
    torch.onnx.export(net, (x, t), a.out, input_names=["x", "t"], output_names=["xi", "seg"], opset_version=17, dynamo=False)
    print(a.out)


if __name__ == "__main__":
    main()
