"""Bounded, foreground benchmark. Re-ID metrics are imported, never reimplemented.

Synthetic extension: first 750 vectors are real. Remaining rows are normalized
real gallery anchors plus deterministic isotropic noise with L2 RMS sigma.
The block seed is independent of the requested gallery size: prefixes match.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "inputs"))
from ranking import cosine_scores, ranked_indices
from rerank import rerank_scores
from reid_metrics import evaluate

faiss.omp_set_num_threads(1)
SEED = 20260917
BLOCK = 4096
MIB = 1024**2
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def jsonwrite(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def memory():
    mem = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in Path("/proc/meminfo").read_text().splitlines()}
    status = dict(l.split(":", 1) for l in Path("/proc/self/status").read_text().splitlines() if ":" in l)
    cg = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split(":")[-1].lstrip("/")
    out = {"available_mib": mem["MemAvailable"]/MIB, "swap_used_mib": (mem["SwapTotal"]-mem["SwapFree"])/MIB,
           "rss_mib": int(status["VmRSS"].split()[0])/1024,
           "maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.max", "memory.swap.max"):
        p = cg / name
        if p.exists():
            val = p.read_text().strip()
            out[name] = int(val) if val.isdigit() else val
    return out


def guard(label, predicted_extra_mib=0):
    subprocess.run(["free", "-m"], check=True)
    m = memory()
    print(json.dumps({"guard": label, **m}), flush=True)
    if m["available_mib"] < max(2500, predicted_extra_mib + 2500):
        raise MemoryError(f"Insufficient available memory before {label}: {m}")
    if m["rss_mib"] + predicted_extra_mib > 3500:
        raise MemoryError(f"Conservative 3500 MiB allocation budget before {label}")
    return m


def arrays():
    return (np.load(ROOT/"inputs/val_query.npy"), np.load(ROOT/"inputs/val_gallery.npy"))


def normalized(a):
    a = np.array(a, dtype=np.float32, copy=True)
    faiss.normalize_L2(a)
    return a


def block(base, block_id, sigma):
    start = block_id * BLOCK
    ids = np.arange(start, start+BLOCK) % len(base)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, block_id]))
    noise = rng.standard_normal((BLOCK, base.shape[1]), dtype=np.float32)
    noise *= np.float32(sigma / np.sqrt(base.shape[1]))
    noise += base[ids]
    faiss.normalize_L2(noise)
    if start < len(base):
        k = min(BLOCK, len(base)-start)
        noise[:k] = base[start:start+k]
    return noise


def blocks(base, n, sigma):
    for start in range(0, n, BLOCK):
        yield start, block(base, start//BLOCK, sigma)[:min(BLOCK, n-start)]


def dense(base, n, sigma):
    out = np.empty((n, base.shape[1]), np.float32)
    for start, b in blocks(base, n, sigma):
        out[start:start+len(b)] = b
    return out


def query_sample(q):
    idx = np.random.default_rng(SEED).permutation(len(q))[:256]
    return normalized(q[idx]), idx


def timed(fn, q, repeats=3, single_n=64):
    # Warm query, then independent single-query requests and batch-32 throughput.
    fn(q[:1])
    lat = []
    for _ in range(repeats):
        for row in q[:single_n]:
            t = time.perf_counter_ns()
            fn(row[None])
            lat.append((time.perf_counter_ns()-t)/1e6)
    batch_seconds = []
    for _ in range(repeats):
        t = time.perf_counter()
        for start in range(0, len(q), 32):
            fn(q[start:start+32])
        batch_seconds.append(time.perf_counter()-t)
    return {"single_median_ms": float(np.median(lat)), "single_p95_ms": float(np.percentile(lat,95)),
            "single_qps": 1000/float(np.mean(lat)), "batch32_qps": len(q)/float(np.median(batch_seconds)),
            "single_raw_ms": lat, "batch_raw_s": batch_seconds, "queries": len(q), "repeats": repeats}


def overlap(found, reference, k):
    # Neighbor-set agreement, not a replacement for the project's Re-ID metrics.
    return np.array([len(set(a[:k].tolist()) & set(b[:k].tolist()))/k
                     for a,b in zip(found, reference)], np.float64)


def evaluate_scores(scores, label):
    qm = list(csv.DictReader((ROOT/"inputs/val_query.csv").open()))
    gm = list(csv.DictReader((ROOT/"inputs/val_gallery.csv").open()))
    res = evaluate(scores,
        query_ids=[r["vehicle_id"] for r in qm], gallery_ids=[r["vehicle_id"] for r in gm],
        query_cameras=[r["camera_id"] for r in qm], gallery_cameras=[r["camera_id"] for r in gm],
        known_absent=np.array([r["has_mate"]=="0" for r in qm]),
        threshold=0., camera_policy="market", refusal_mode="presence")
    # Scores assembled from ranks have no calibrated refusal interpretation.
    keep = {k: res[k] for k in ("protocol", "ranking_full_gallery", "ranking_top_k", "ranking_top_k_by_filter_order")}
    keep["per_query"] = [{k:r[k] for k in ("query_index", "ap", "rank1", "rank5", "num_relevant") if k in r}
                         for r in res["per_query"]]
    jsonwrite(RESULTS/f"eval_{label}.json", keep)
    return {"mAP": res["ranking_full_gallery"]["mAP"],
            "Rank-1": res["ranking_full_gallery"]["Rank-1"],
            "Rank-5": res["ranking_full_gallery"]["Rank-5"],
            "mAP@10": res["ranking_top_k"]["mAP"],
            "mAP@10_filter_first": res["ranking_top_k_by_filter_order"]["filter_then_top_k"]["mAP"],
            "valid_queries": res["ranking_full_gallery"]["num_valid_queries"]}


def scores_from_order(order, g):
    out = np.empty(order.shape, np.float64)
    np.put_along_axis(out, order, np.broadcast_to(g-np.arange(g), order.shape), axis=1)
    return out


def fingerprint_env():
    cmds = {"free": ["free", "-m"], "cpu": ["lscpu"], "processes": ["ps", "-eo", "pid,comm,%cpu,%mem,rss", "--sort=-rss"],
            "uptime": ["uptime"]}
    return {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version,
            "affinity": sorted(os.sched_getaffinity(0)), "threads": faiss.omp_get_max_threads(),
            "packages": {n:md.version(n) for n in ("faiss-cpu", "numpy", "packaging")},
            "numpy_config": str(np.show_config(mode="dicts")), "memory": memory(),
            **{k:subprocess.check_output(cmd,text=True).splitlines()[:28] for k,cmd in cmds.items()}}


def serialized_size(index):
    count = [0]
    def consume(data):
        count[0] += len(data)
        return len(data)
    writer = faiss.PyCallbackIOWriter(consume)
    faiss.write_index(index, writer)
    return count[0]


def run_flat(args, q, base):
    n = args.n
    qs, qi = query_sample(q)
    guard("flat_allocate", n*512*4/MIB+100)
    index = faiss.IndexFlatIP(512)
    # Allocate once; no 2x temporary gallery and no geometric vector growth.
    t = time.perf_counter()
    index.codes.resize(n*512*4)
    index.ntotal = n
    view = faiss.rev_swig_ptr(index.get_xb(), n*512).reshape(n,512)
    generation_s = 0.
    for start in range(0,n,BLOCK):
        tg = time.perf_counter()
        b = block(base, start//BLOCK, args.sigma)[:min(BLOCK,n-start)]
        generation_s += time.perf_counter()-tg
        view[start:start+len(b)] = b
        if start % (BLOCK*32) == 0:
            if memory()["available_mib"] < 2500: raise MemoryError("Available RAM fell below threshold")
    build_s = time.perf_counter()-t
    del view
    k = min(100,n)
    guard("flat_search")
    ds, ids = index.search(qs,k)
    np.savez(RESULTS/f"truth_{n}_{args.sigma}.npz", ids=ids, scores=ds, query_indices=qi)
    performance = timed(lambda x:index.search(x,10), qs)
    return {"build_including_generation_s":build_s, "generation_s":generation_s,
            "index_payload_bytes":index.codes.size(), "serialized_index_bytes":serialized_size(index),
            "resident":memory(), "performance":performance,
            "reference":"exhaustive normalized float32 inner product, FAISS IndexFlatIP"}


def run_current(args,q,base):
    n = args.n
    # Live arrays: input float32; cast float64; scaled gallery; norm temporary;
    # normalized output. Conservative guard, never deliberately allocate past it.
    predicted = n*512*36/MIB + 150
    if predicted>3500:
        return {"status":"skipped_budget", "predicted_peak_mib":predicted,
                "reason":"Unmodified service converts and normalizes the entire gallery in float64"}
    guard("service_exact", predicted)
    g = dense(base,n,args.sigma)
    qs,_ = query_sample(q)
    def search(x):
        scores = cosine_scores(x,g)
        return np.array([ranked_indices(row)[:10] for row in scores])
    perf = timed(search,qs[:32],repeats=3,single_n=16)
    # Record float32 vs canonical float64 arithmetic separately from ANN loss.
    exact64 = search(qs)
    ref = np.load(RESULTS/f"truth_{n}_{args.sigma}.npz")["ids"]
    agreement = overlap(exact64,ref,10)
    return {"performance":perf,"resident":memory(), "predicted_peak_mib":predicted,
            "float32_vs_float64_recall10":float(agreement.mean()),
            "float32_vs_float64_top1":float((exact64[:,0]==ref[:,0]).mean())}


def run_truth64(args,q,base):
    """Canonical service arithmetic, blocked, as an independent ANN oracle."""
    guard("canonical_float64_blocked_oracle",300)
    qs,qi=query_sample(q)
    top_scores=np.empty((len(qs),0),np.float64)
    top_ids=np.empty((len(qs),0),np.int64)
    kernel_s=0.
    total=time.perf_counter()
    for start,b in blocks(base,args.n,args.sigma):
        t=time.perf_counter()
        scores=cosine_scores(qs,b)
        ids=np.broadcast_to(np.arange(start,start+len(b)),scores.shape)
        merged_scores=np.concatenate([top_scores,scores],axis=1)
        merged_ids=np.concatenate([top_ids,ids],axis=1)
        # Stable service sort plus IDs resolves ties identically across blocks.
        keep=np.empty((len(qs),min(100,merged_scores.shape[1])),np.int64)
        for i in range(len(qs)):
            ties=np.argsort(merged_ids[i],kind="stable")
            keep[i]=ties[ranked_indices(merged_scores[i,ties])[:keep.shape[1]]]
        top_scores=np.take_along_axis(merged_scores,keep,axis=1)
        top_ids=np.take_along_axis(merged_ids,keep,axis=1)
        kernel_s += time.perf_counter()-t
        if start%(BLOCK*32)==0:
            m=memory()
            print(json.dumps({"oracle_rows":start+len(b),"rss_mib":m["rss_mib"]}),flush=True)
            if m["available_mib"]<2500: raise MemoryError("Available RAM fell below threshold")
    np.savez(RESULTS/f"truth64_{args.n}_{args.sigma}.npz",ids=top_ids,scores=top_scores,query_indices=qi)
    ref=np.load(RESULTS/f"truth_{args.n}_{args.sigma}.npz")
    return {"kernel_and_top100_s":kernel_s,"including_generation_s":time.perf_counter()-total,
            "float32_vs_float64_recall10":float(overlap(ref["ids"],top_ids,10).mean()),
            "float32_vs_float64_recall100":float(overlap(ref["ids"],top_ids,100).mean()),
            "float32_vs_float64_top1":float(np.mean(ref["ids"][:,0]==top_ids[:,0])),
            "queries":len(qs),"block_rows":BLOCK,"resident":memory()}


def run_candidate_cost(args,q,base):
    guard("isolated_candidate_rerank",100)
    import tracemalloc
    k=args.n
    qs,_=query_sample(q)
    cand=np.argsort(-cosine_scores(qs[:1],base),axis=1,kind="stable")[0,:k]
    # Time without allocation tracing; then one independent allocation trace.
    perf=timed(lambda x:rerank_scores(x,base[cand],6,3,.3),qs[:32],repeats=1,single_n=32)
    tracemalloc.start()
    rerank_scores(qs[:1],base[cand],6,3,.3)
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"K":k,"single_query_traced_peak_bytes":peak,"performance":perf,"resident":memory(),
            "note":"one fixed candidate set for resource microbenchmark; batch32 field is not an online endpoint"}


def train_ivf(base,n,nlist,sigma):
    nt = min(n, max(10000,nlist*64))
    train = dense(base,nt,sigma)
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(512),512,nlist,faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = SEED
    index.cp.niter = 15
    index.cp.min_points_per_centroid = 1
    t = time.perf_counter()
    index.train(train)
    dt = time.perf_counter()-t
    del train
    return index,dt,nt


def run_ivf(args,q,base):
    n = args.n
    nlist = 16 if n==750 else (1024 if n>=1000000 else 128)
    guard("ivf_build", n*512*4*1.55/MIB+200)
    qs,qi = query_sample(q)
    index, train_s, nt = train_ivf(base,n,nlist,args.sigma)
    add_s = gen_s = 0.
    for start in range(0,n,BLOCK):
        t = time.perf_counter()
        b = block(base,start//BLOCK,args.sigma)[:min(BLOCK,n-start)]
        gen_s += time.perf_counter()-t
        t = time.perf_counter()
        index.add(b)
        add_s += time.perf_counter()-t
        if start % (BLOCK*32)==0:
            m=memory()
            print(json.dumps({"added":index.ntotal,"rss_mib":m["rss_mib"]}),flush=True)
            if m["available_mib"]<2500: raise MemoryError("Available RAM fell below threshold")
    assert index.ntotal==n
    truth=np.load(RESULTS/f"truth_{n}_{args.sigma}.npz")
    assert np.array_equal(truth["query_indices"],qi)
    rows=[]
    settings=[1,2,4,8,16] if nlist==16 else [1,2,4,8,16,32,64,128]
    for probe in settings:
        guard(f"ivf_search_{probe}")
        index.nprobe=probe
        ds,ids=index.search(qs,100)
        rec10=overlap(ids,truth["ids"],10)
        rec100=overlap(ids,truth["ids"],100)
        top1=ids[:,0]==truth["ids"][:,0]
        perf=timed(lambda x:index.search(x,10),qs)
        row={"nprobe":probe,"recall10":float(rec10.mean()),"recall100":float(rec100.mean()),
             "top1_agreement":float(top1.mean()),"tune_recall10":float(rec10[:128].mean()),
             "holdout_recall10":float(rec10[128:].mean()),
             "complete_top10_fraction":float(np.mean(rec10==1)), "performance":perf}
        np.savez(RESULTS/f"neighbors_{n}_{args.sigma}_p{probe}.npz",ids=ids,scores=ds,
                 recall10=rec10,recall100=rec100,query_indices=qi)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!="performance"}),flush=True)
    return {"nlist":nlist,"train_rows":nt,"kmeans_iterations":15,"train_s":train_s,"add_s":add_s,
            "generation_s":gen_s,"build_s":train_s+add_s,"index_payload_bytes":n*(512*4+8)+nlist*512*4,
            "serialized_index_bytes":serialized_size(index),"resident":memory(),"settings":rows}


def run_rerank_scale(args,q,base):
    n=args.n
    total=len(q)+n
    predicted=48*total*total/MIB + 3*total*512*8/MIB+120
    if predicted>3500:
        return {"status":"skipped_budget","total_nodes":total,"predicted_peak_mib":predicted,
                "reason":"dense k-reciprocal arrays would exceed conservative allocation budget"}
    guard("rerank_scale",predicted)
    g=dense(base,n,args.sigma)
    t=time.perf_counter()
    scores=rerank_scores(q,g,6,3,.3)
    dt=time.perf_counter()-t
    assert np.isfinite(scores).all()
    return {"total_nodes":total,"queries":len(q),"seconds":dt,"predicted_peak_mib":predicted,"resident":memory()}


def run_quality(args,q,base):
    guard("quality",500)
    raw=cosine_scores(q,base)
    order=np.argsort(-raw,axis=1,kind="stable")
    baseline=evaluate_scores(raw,"baseline")
    t=time.perf_counter()
    full=rerank_scores(q,base,6,3,.3)
    full_time=time.perf_counter()-t
    full_metrics=evaluate_scores(full,"full_rerank")
    np.savez(RESULTS/"quality_reference.npz",baseline_scores=raw,full_scores=full)
    del full
    index,_,_=train_ivf(base,len(base),16,args.sigma)
    index.add(base)
    index.nprobe=8
    rows=[]
    for k in [10,30,100,300,750]:
        guard(f"candidate_rerank_K{k}",250)
        for candidate_source in (["exact","ivf"] if k in [30,100] else ["exact"]):
            out_order=order.copy()
            lat=[]
            if candidate_source=="ivf":
                candidates=index.search(normalized(q),k)[1]
                assert np.all(candidates>=0)
            else:
                candidates=order[:,:k]
            for i in range(len(q)):
                cand=candidates[i]
                t=time.perf_counter()
                scores=rerank_scores(q[i:i+1],base[cand],6,3,.3)[0]
                local_order=cand[ranked_indices(scores)]
                # The candidate prefix is reranked; the exact tail is retained
                # only for the diagnostic full-gallery mAP on the small set.
                tail=order[i][~np.isin(order[i],cand)]
                out_order[i]=np.concatenate([local_order,tail])
                lat.append(time.perf_counter()-t)
            label=f"{candidate_source}_K{k}"
            metrics=evaluate_scores(scores_from_order(out_order,len(base)),label)
            np.save(RESULTS/f"order_{label}.npy",out_order)
            rows.append({"source":candidate_source,"K":k,**metrics,
                         "rerank_total_s":sum(lat),"rerank_single_median_ms":1000*float(np.median(lat)),
                         "rerank_single_p95_ms":1000*float(np.percentile(lat,95)),
                         "gain_retained_fraction":(metrics["mAP"]-baseline["mAP"])/(full_metrics["mAP"]-baseline["mAP"]),
                         "gain_at10_retained_fraction":(metrics["mAP@10"]-baseline["mAP@10"])/(full_metrics["mAP@10"]-baseline["mAP@10"]),
                         "resident":memory()})
            print(json.dumps(rows[-1]),flush=True)
    return {"baseline":baseline,"full_rerank":full_metrics,"full_rerank_s":full_time,"candidates":rows,
            "ivf_nlist":16,"ivf_nprobe":8,"graph":"one query plus K candidates; no other queries"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["flat","current","ivf","rerank-scale","quality","env","truth64","candidate-cost"])
    parser.add_argument("--n",type=int,default=750)
    parser.add_argument("--sigma",type=float,default=.08)
    args=parser.parse_args()
    out={"mode":args.mode,"n":args.n,"sigma":args.sigma,"seed":SEED,"start":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "command":" ".join(sys.argv),"environment":fingerprint_env()}
    q,g=arrays()
    base=normalized(g)
    try:
        if args.mode=="env":
            # Real-anchor perturbation statistics and deterministic prefix check.
            b=block(base,1,args.sigma)
            ids=np.arange(BLOCK,2*BLOCK)%len(base)
            anchor_cos=np.sum(base[ids]*b,axis=1)
            assert np.array_equal(dense(base,7500,args.sigma)[:750],base)
            out["data"]={"query_shape":list(q.shape),"gallery_shape":list(g.shape),
                         "gallery_identities":len(set(r["vehicle_id"] for r in csv.DictReader((ROOT/"inputs/val_gallery.csv").open()))),
                         "anchor_cos_mean":float(anchor_cos.mean()),"anchor_cos_min":float(anchor_cos.min()),
                         "anchor_cos_max":float(anchor_cos.max()),"prefix_sha256":hashlib.sha256(b.tobytes()).hexdigest()}
        else:
            out.update(globals()["run_"+args.mode.replace("-","_")](args,q,g if args.mode=="quality" else base))
        out.setdefault("status","ok")
    except MemoryError as exc:
        out.update(status="stopped_memory_guard",error=str(exc),resident=memory())
    out["end"]=time.strftime("%Y-%m-%dT%H:%M:%S%z")
    jsonwrite(RESULTS/f"{args.mode}_{args.n}_{args.sigma}.json",out)
    print(json.dumps({k:v for k,v in out.items() if k not in ("environment","settings","performance","candidates")}),flush=True)


if __name__=="__main__":
    main()
