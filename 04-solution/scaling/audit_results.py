"""Validate completed evidence without recomputing or replacing Re-ID metrics."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
r=root/"results"
expected=[]
for n in [750,7500,75000,1000000]:
    expected += [(mode,n,.08) for mode in ["flat","current","ivf"]]
expected += [("rerank-scale",n,.08) for n in [750,1500,3000,6000,7500,10000,1000000]]
expected += [("quality",750,.08),("env",750,.08)]
expected += [("candidate-cost",k,.08) for k in [30,100,300]]
expected += [("truth64",n,.08) for n in [75000,1000000]]
expected += [(mode,75000,.25) for mode in ["flat","ivf"]]
for mode,n,sigma in expected:
    data=json.loads((r/f"{mode}_{n}_{sigma}.json").read_text())
    skip=(mode=="current" and n==1000000) or (mode=="rerank-scale" and n>=7500)
    assert data["status"]==("skipped_budget" if skip else "ok"),(mode,n,data["status"])
    if "resident" in data:
        m=data["resident"]
        assert m["maxrss_mib"]<4096
        assert m["memory.peak"]<=4*1024**3
        assert m["memory.swap.current"]==0
    if mode=="ivf":
        truth=np.load(r/f"truth_{n}_{sigma}.npz")
        for setting in data["settings"]:
            ann=np.load(r/f"neighbors_{n}_{sigma}_p{setting['nprobe']}.npz")
            assert np.array_equal(ann["query_indices"],truth["query_indices"])
            assert abs(float(ann["recall10"].mean())-setting["recall10"])<1e-14
            assert np.all((ann["ids"]>=-1)&(ann["ids"]<n))
            assert ann["ids"].shape==(256,100)
q=json.loads((r/"quality_750_0.08.json").read_text())
assert abs(q["baseline"]["mAP"]-.6565692724907637)<1e-12
assert abs(q["full_rerank"]["mAP"]-.6936584724586873)<1e-12
assert q["baseline"]["valid_queries"]==832
for c in q["candidates"]:
    order=np.load(r/f"order_{c['source']}_K{c['K']}.npy")
    assert order.shape==(1110,750)
    assert np.all(np.sort(order,axis=1)==np.arange(750))
assert json.loads((root/"service_integrity.json").read_text())["unchanged"]
print(f"OK: {len(expected)} result records, bounded memory, zero cgroup swap, aligned ANN queries, reproduced mAP, complete candidate permutations, unchanged service")
