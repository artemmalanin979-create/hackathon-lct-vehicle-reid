#!/usr/bin/env python3
"""Export the small learned head from NPZ without importing training runtime.

The graph directly encodes the frozen residual-MLP contract. Both head-only
and combined_v1+head image models are checked with onnx.checker. Inputs/outputs
are compared to saved Torch/NumPy vectors, never to the exported graph itself.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def nodes(weights, input_name, output_name):
    initializers = [numpy_helper.from_array(weights[k].astype(np.float32), "distill_"+k.replace(".", "_"))
                    for k in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]]
    graph = [
        helper.make_node("ReduceL2", [input_name], ["distill_input_norm"], axes=[1], keepdims=1),
        helper.make_node("Div", [input_name, "distill_input_norm"], ["distill_x"]),
        helper.make_node("Gemm", ["distill_x", "distill_fc1_weight", "distill_fc1_bias"], ["distill_hidden"], transB=1),
        helper.make_node("Relu", ["distill_hidden"], ["distill_relu"]),
        helper.make_node("Gemm", ["distill_relu", "distill_fc2_weight", "distill_fc2_bias"], ["distill_residual"], transB=1),
        helper.make_node("Add", ["distill_x", "distill_residual"], ["distill_unnormalized"]),
        helper.make_node("ReduceL2", ["distill_unnormalized"], ["distill_output_norm"], axes=[1], keepdims=1),
        helper.make_node("Div", ["distill_unnormalized", "distill_output_norm"], [output_name]),
    ]
    return graph, initializers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backbone", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w = dict(np.load(a.checkpoint, allow_pickle=False))
    ns, initializers = nodes(w, "embedding", "student")
    graph = helper.make_graph(ns, "distilled_residual_head", [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None,512])],
                              [helper.make_tensor_value_info("student", TensorProto.FLOAT, [None,512])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, a.out / "head.onnx")
    opts = ort.SessionOptions(); opts.intra_op_num_threads=2; opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(a.out / "head.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    with np.load(a.validation, allow_pickle=False) as z:
        x=z["x"] if "x" in z else z["input"]
        expected=z["torch_output"] if "torch_output" in z else z["output"]
    actual=session.run(None,{"embedding":x})[0]
    delta=float(np.max(np.abs(actual-expected)))
    assert delta <= 1e-5, delta
    report={"status":"PASS","rows":len(x),"max_abs":delta,"onnx":onnx.__version__,"onnxruntime":ort.__version__}
    np.save(a.out / "onnx_validation.npy",actual)
    if a.backbone:
        full=onnx.load(a.backbone)
        original=full.graph.output[0].name
        ns, initializers=nodes(w,original,"student_embedding")
        full.graph.node.extend(ns); full.graph.initializer.extend(initializers)
        del full.graph.output[:]
        full.graph.output.append(helper.make_tensor_value_info("student_embedding",TensorProto.FLOAT,[None,512]))
        onnx.checker.check_model(full)
        onnx.save(full,a.out / "student_combined_v1.onnx")
    report["artifacts"]={f.name:{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"bytes":f.stat().st_size} for f in a.out.glob("*.onnx")}
    (a.out / "export.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
