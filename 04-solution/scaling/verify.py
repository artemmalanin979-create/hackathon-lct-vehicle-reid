"""Checks the experimental machinery and frozen source integrity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import faiss
import bench

root=Path(__file__).resolve().parent
meta=json.loads((root/"provenance.json").read_text())
for name, item in meta["files"].items():
    assert hashlib.sha256((root/"inputs"/name).read_bytes()).hexdigest()==item["sha256"]
q,g=bench.arrays()
base=bench.normalized(g)
short=bench.dense(base,7500,.08)
long=bench.dense(base,10000,.08)
assert np.array_equal(short,long[:7500])
assert np.array_equal(short[:750],base)
assert np.max(np.abs(np.linalg.norm(long,axis=1)-1))<1e-6
assert not np.array_equal(short[750:1500],base)
ordinary=faiss.IndexFlatIP(512)
ordinary.add(short)
preallocated=faiss.IndexFlatIP(512)
preallocated.codes.resize(short.size*4)
preallocated.ntotal=len(short)
view=faiss.rev_swig_ptr(preallocated.get_xb(),short.size).reshape(short.shape)
view[:]=short
for nqueries in (1,32):
    d0,i0=ordinary.search(bench.normalized(q[:nqueries]),10)
    d1,i1=preallocated.search(bench.normalized(q[:nqueries]),10)
    assert np.array_equal(i0,i1) and np.array_equal(d0,d1)
order=np.argsort(-bench.cosine_scores(q[:7],base),axis=1,kind="stable")
recovered=np.argsort(-bench.scores_from_order(order,len(base)),axis=1,kind="stable")
assert np.array_equal(order,recovered)
assert np.all(bench.overlap(i0,i0,10)==1)
np.testing.assert_array_equal(bench.overlap(np.array([[0,1,2]]),np.array([[2,3,4]]),3),[1/3])
print("OK: source hashes, deterministic prefixes, real anchors, normalization, preallocated FAISS exact index, rank-score mapping, neighbor agreement")
