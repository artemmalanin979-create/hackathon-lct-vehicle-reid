#!/usr/bin/env python3
"""New job_73 adapter: reconstruct initial NPZ from shipped OMZ initializers.

The historical one-off ONNX-to-NPZ script was not found. This adapter is explicitly
new; the initializer tensors were checked against the preserved original NPZ/PT.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--onnx", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
digest = hashlib.sha256(args.onnx.read_bytes()).hexdigest()
if digest != "4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f":
    raise SystemExit("Expected the original OMZ ONNX, not the fine-tuned combined_v1")
if args.out.exists():
    raise SystemExit("Output already exists; choose a new path")
arrays = {entry.name: numpy_helper.to_array(entry) for entry in onnx.load(args.onnx).graph.initializer}
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, **dict(sorted(arrays.items())))
print(f"{len(arrays)} initializers; sha256={hashlib.sha256(args.out.read_bytes()).hexdigest()}")
