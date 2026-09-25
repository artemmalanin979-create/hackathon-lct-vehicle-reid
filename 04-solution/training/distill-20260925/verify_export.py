#!/usr/bin/env python3
"""Compare actual ONNX outputs with independent training-runtime predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import experiment as e


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    protocol=json.loads((e.HERE/"protocol.json").read_text())
    files=e.setup(a.repo,protocol)
    qm=e.read_csv(files["query_csv"]);gm=e.read_csv(files["gallery_csv"])
    x=np.load(a.out/"export/onnx_validation.npy",allow_pickle=False)
    y=np.load(a.out/"export_validation.npz",allow_pickle=False)["output"]
    np.testing.assert_allclose(x,y,atol=1e-5,rtol=0)
    thresholds=json.loads((a.out/"selection.json").read_text())["thresholds"]["student"]
    result={"max_abs":float(np.max(np.abs(x-y))),"rows":len(x),"modes":{}}
    for mode in ["cosine","KR"]:
        q=len(qm)
        rs=[e.evaluate(e.scores(z[:q],z[q:],mode),qm,gm,thresholds[mode]) for z in [x,y]]
        top_diff=sum(u["top_gallery_index"]!=v["top_gallery_index"] for u,v in zip(rs[0]["per_query"],rs[1]["per_query"]))
        refusal_diff=sum(u["accepted"]!=v["accepted"] for u,v in zip(rs[0]["per_query"],rs[1]["per_query"]))
        delta=abs(rs[0]["ranking_full_gallery"]["mAP"]-rs[1]["ranking_full_gallery"]["mAP"])
        assert top_diff==0 and refusal_diff==0 and delta<=1e-6,(mode,top_diff,refusal_diff,delta)
        result["modes"][mode]={"top1_mismatches":top_diff,"refusal_mismatches":refusal_diff,"mAP_abs_delta":delta,"queries":len(qm)}
    result["status"]="PASS"
    e.dump(a.out/"export_ranking_check.json",result);print(json.dumps(result))


if __name__ == "__main__":main()
