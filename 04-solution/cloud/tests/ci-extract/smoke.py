# Самопроверка Re-ID стека на GPU. Запуск: /opt/reid/venv/bin/python /opt/lct/smoke.py
import sys, subprocess
import numpy, PIL, torch, timm, transformers, open_clip
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda-runtime", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
ok = torch.cuda.is_available()
print("torch.cuda.is_available", ok, "-", torch.cuda.get_device_name(0) if ok else "НЕТ GPU")
if ok:
    x = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    print("matmul на GPU ok, сумма", float((x @ x).sum()))
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
print("onnxruntime", ort.__version__, "providers", ort.get_available_providers())
import onnx
from onnx import helper, TensorProto
node = helper.make_node("Add", ["a", "b"], ["y"])
g = helper.make_graph([node], "add", [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2]),
                                      helper.make_tensor_value_info("b", TensorProto.FLOAT, [2])],
                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ort session providers", s.get_providers())
print("ort add ->", s.run(None, {"a": numpy.array([1, 2], numpy.float32), "b": numpy.array([3, 4], numpy.float32)})[0])
print("timm", timm.__version__, "transformers", transformers.__version__, "open_clip", open_clip.__version__,
      "numpy", numpy.__version__, "Pillow", PIL.__version__)
print("SMOKE_REID", "OK" if ok and "CUDAExecutionProvider" in s.get_providers() else "FAIL")
