#!/usr/bin/env python3
"""Same-node CPU benchmark, real service baseline, cached raw crops as input.

Fusion reuses the second backbone output rather than evaluating it twice.
No image decode, gallery search or KR is included in descriptor timings.
"""
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--crops",type=Path,default=Path("/home/fedora/lct-reid/data/crops_208.npy"))
    p.add_argument("--cold",choices=["baseline","student","fusion"])
    args=p.parse_args()
    sys.path.insert(0,str(args.repo / "04-solution/service"))
    from app.core.model import Embedder,l2norm
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    crops=np.load(args.crops,mmap_mode="r",allow_pickle=False)
    weight=json.loads((args.out / "selection.json").read_text())["fusion_student_weight"]
    tensors=[np.ascontiguousarray(crops[i:i+1].transpose(0,3,1,2),dtype=np.float32) for i in range(45)]

    def build(name):
        if name == "student":
            s=ort.InferenceSession(str(args.out / "export/student_combined_v1.onnx"),sess_options=opts,providers=["CPUExecutionProvider"])
            return lambda x:s.run(None,{s.get_inputs()[0].name:x})[0]
        base=Embedder(threads=2)
        if name == "baseline":
            return base.embed_tensors
        head=ort.InferenceSession(str(args.out / "export/head.onnx"),sess_options=opts,providers=["CPUExecutionProvider"])
        def fusion(x):
            a=base.session.run(None,{base.input_name:x})[0]
            b=base.session2.run(None,{base.input_name2:x})[0]
            ens=l2norm(l2norm(a)+l2norm(b))
            teacher=l2norm((ens-base.m)@base.P.T)
            student=head.run(None,{"embedding":l2norm(b)})[0]
            return np.concatenate([np.sqrt(1-weight)*teacher,np.sqrt(weight)*student],axis=1)
        return fusion

    if args.cold:
        t=time.perf_counter();runner=build(args.cold);output=runner(tensors[0]); elapsed=time.perf_counter()-t
        print(json.dumps({"seconds":elapsed,"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"dimensions":output.shape[1]}))
        return
    modes=["baseline","student","fusion"]
    cold={}
    for mode in modes:
        r=subprocess.run([sys.executable,__file__,"--repo",str(args.repo),"--out",str(args.out),"--crops",str(args.crops),"--cold",mode],capture_output=True,text=True,check=True,timeout=60)
        cold[mode]=json.loads(r.stdout)
    runners={m:build(m) for m in modes}
    for tensor in tensors[:5]:
        for m in modes:runners[m](tensor)
    durations={m:[] for m in modes}
    outputs={m:[] for m in modes}
    for i,tensor in enumerate(tensors[5:]):
        order=modes if i%2==0 else list(reversed(modes))
        for m in order:
            t=time.perf_counter();y=runners[m](tensor);elapsed=time.perf_counter()-t
            durations[m].append(elapsed);outputs[m].append(y[0])
    # Independent cached raw combined_v1 features -> learned NumPy head,
    # compared with the single exported image model on exactly these crops.
    from experiment import head_numpy,l2
    raw=np.load("/home/fedora/lct-reid/jobs/job_48/out/train_fit_j48.npy",allow_pickle=False)
    expected=head_numpy(raw[5:45],dict(np.load(args.out / "main.npz",allow_pickle=False)))
    delta=float(np.max(np.abs(expected-np.array(outputs["student"]))))
    if delta>1e-5:raise AssertionError(f"image ONNX mismatch: {delta}")
    modeldir=args.repo / "04-solution/service/model"
    basebytes=sum((modeldir/f).stat().st_size for f in ["osnet_ain_x1_0_vehicle_reid.onnx","osnet_ain_combined_v1.onnx","lw_ens_j48_rho0.5.npz"])
    sizes={"baseline":basebytes,"student":(args.out / "export/student_combined_v1.onnx").stat().st_size,
           "fusion":basebytes+(args.out / "export/head.onnx").stat().st_size}
    measurements={m:{"batch1_p50_ms":float(np.quantile(durations[m],.5)*1000),"batch1_p95_ms":float(np.quantile(durations[m],.95)*1000),
          "FPS_batch1":float(1/np.mean(durations[m])),"cold_start_seconds":cold[m]["seconds"],"cold_process_peak_rss_mib":cold[m]["peak_rss_mib"],
          "weights_bytes":sizes[m],"embedding_dimensions":512 if m!="fusion" else 1024,"gallery750_raw_index_bytes":750*(512 if m!="fusion" else 1024)*4,
          "VRAM":"NOT APPLICABLE: CPUExecutionProvider","samples_seconds":durations[m]} for m in modes}
    report={"measurements":measurements,"threads":2,"batch":1,"warmup":5,"samples":40,"input":"train crops208 indices5..44, uint8->float32 conversion outside timer",
          "boundary":"preprocessed tensor to normalized descriptor; excludes decode/API/Qdrant/KR","ORT":ort.__version__,"numpy":np.__version__,
          "image_export_max_abs_vs_cached_torch_head":delta,"concurrent_sessions_peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          "scope":"shared VM under other workloads; alternating paired measurements mitigate but do not remove contention",
          "source_hashes":{f:hashlib.sha256((args.repo / "04-solution/service/app/core" / f).read_bytes()).hexdigest() for f in ["model.py","config.py","preprocess.py","validation.py"]}}
    (args.out / "benchmark.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({m:{k:v for k,v in d.items() if k!="samples_seconds"} for m,d in measurements.items()}))


if __name__ == "__main__":main()
