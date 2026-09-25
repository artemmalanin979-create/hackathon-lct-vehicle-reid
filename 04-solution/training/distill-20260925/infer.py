#!/usr/bin/env python3
"""Run the experimental 512-d student on one real image/bbox, without a server."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE=Path(__file__).resolve().parent
EXPECTED_MODEL="188284ff7a56ff915ea6143cca62dca0a381e3ca7fc784b996b876b641ad3d06"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,default=HERE/"artifacts/run/export/student_combined_v1.onnx")
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--bbox",nargs=4,type=int,metavar=("X","Y","W","H"),required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit("refusing to overwrite existing output")
    if hashlib.sha256(args.model.read_bytes()).hexdigest()!=EXPECTED_MODEL:raise SystemExit("model SHA-256 mismatch")
    sys.path.insert(0,str(HERE.parents[2]/"04-solution/service"))
    from app.core.preprocess import validate_bbox
    with Image.open(args.image) as im:
        im.load();x,y,w,h=args.bbox;validate_bbox(im,x,y,w,h)
        crop=im.convert("RGB").crop((x,y,x+w,y+h)).resize((208,208),Image.Resampling.BILINEAR)
        tensor=np.asarray(crop,dtype=np.float32).transpose(2,0,1)[None]
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.model),sess_options=opts,providers=["CPUExecutionProvider"])
    vector=session.run(None,{session.get_inputs()[0].name:tensor})[0]
    if vector.shape!=(1,512) or not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector))-1)>1e-6:
        raise SystemExit("invalid student output")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as f:np.save(f,vector)
    print(json.dumps({"status":"PASS","shape":list(vector.shape),"model_sha256":EXPECTED_MODEL,"output":str(args.out),"model_version":"experimental-distill-20260925-not-release"}))


if __name__=="__main__":main()
