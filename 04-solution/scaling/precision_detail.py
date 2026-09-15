"""Inspect stored neighbors to separate rounding from candidate-search errors."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
r=root/"results"
f32=np.load(r/"truth_1000000_0.08.npz")
f64=np.load(r/"truth64_1000000_0.08.npz")
ann=np.load(r/"neighbors_1000000_0.08_p16.npz")
cases=[]
for i in range(len(f32["ids"])):
    a=set(f32["ids"][i,:10].tolist())
    b=set(f64["ids"][i,:10].tolist())
    if a!=b:
        ids=sorted(a.symmetric_difference(b))
        entry={"sample_row":i,"original_query_row":int(f32["query_indices"][i]),
               "float32_only":sorted(a-b),"float64_only":sorted(b-a),
               "ann_matches_float64":set(ann["ids"][i,:10].tolist())==b,"scores":{}}
        for name,data in [("flat_float32",f32),("cosine_float64",f64),("ivf_float32",ann)]:
            values=dict(zip(data["ids"][i].tolist(),data["scores"][i].tolist()))
            entry["scores"][name]={str(idx):values.get(idx) for idx in ids}
        cases.append(entry)
out={"n":1000000,"queries":256,"top_k":10,"nprobe":16,
     "disagreeing_queries":len(cases),"details":cases}
(root/"precision_detail.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
