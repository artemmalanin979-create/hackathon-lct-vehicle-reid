#!/usr/bin/env python3
"""Bounded cached-feature distillation. No network access, no release writes.

All outputs are confined to --out. Existing checkpoints are never overwritten
by a second run: use a fresh output directory. Evaluation uses the frozen
repository evaluator and service KR from --repo, verified before import.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from contracts import checked_features, checked_hash, check_disjoint, grouped_split

HERE = Path(__file__).resolve().parent


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def l2(x):
    x = np.asarray(x, np.float64)
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) == 0):
        raise ValueError("nonfinite/zero embedding")
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def setup(repo, protocol):
    files = {}
    for record in protocol["inputs"]:
        p = repo / record["repo_path"] if "repo_path" in record else Path(record["path"])
        checked_hash(p, record["sha256"])
        if "ids_path" in record:
            checked_hash(record["ids_path"], record["ids_sha256"])
        files[record["name"]] = p
    # Import only the hash-checked canonical evaluator/service; old worker repo
    # may contain older code and is intentionally not on this path.
    sys.path[:0] = [str(repo / "04-solution/eval"), str(repo / "04-solution/service")]
    from reid_metrics import evaluate, scores_from_embeddings
    from app.core.rerank import rerank_scores
    global EVALUATE, COSINE, RERANK
    EVALUATE, COSINE, RERANK = evaluate, scores_from_embeddings, rerank_scores
    # Execute the exact archived learn_lw definition without its hard-coded
    # historical imports, paths or output side effects. Its source hash is above.
    tree = ast.parse(files["learn_lw"].read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "learn_lw")
    ns = {"np": np}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(files["learn_lw"]), "exec"), ns)
    global LEARN_LW
    LEARN_LW = ns["learn_lw"]
    return files


def load_data(files, protocol):
    meta = {s: read_csv(files[k]) for s, k in (("train_fit", "train_csv"), ("val_query", "query_csv"), ("val_gallery", "gallery_csv"))}
    vectors = {}
    for record in protocol["inputs"]:
        if "ids_path" not in record:
            continue
        name = record["name"]
        split = name[2:]
        x = np.load(files[name], allow_pickle=False)
        if list(x.shape) != record["shape"]:
            raise ValueError("wrong feature shape")
        ids = Path(record["ids_path"]).read_text().splitlines()
        vectors[name] = l2(checked_features(x, ids, [r["image_id"] for r in meta[split]]))
    vids = np.array([int(r["vehicle_id"]) for r in meta["train_fit"]])
    cams = np.array([int(r["camera_id"]) for r in meta["train_fit"]])
    fit, dev = grouped_split(vids)
    external_ids = [int(r["vehicle_id"]) for s in ["val_query", "val_gallery"] for r in meta[s]]
    check_disjoint(vids[fit], vids[dev], external_ids)
    ids_text = "".join(str(v) + "\n" for v in sorted(set(vids[dev])))
    if hashlib.sha256(ids_text.encode()).hexdigest() != protocol["data"]["historical_dev_ids_sha256"]:
        raise ValueError("historic dev selection changed")
    q, g = [], []
    absent_ids = set(sorted(set(vids[dev]))[::4])
    for v in sorted(set(vids[dev])):
        rows = np.flatnonzero(vids == v)
        if v in absent_ids:
            q.extend(rows.tolist())
            continue
        qv, gv = [], []
        for c in np.unique(cams[rows]):
            group = rows[cams[rows] == c]
            gv.append(int(group[0]))
            qv.extend(group[1:].tolist())
        if not qv and len(gv) >= 2:
            qv.append(gv.pop())
        q.extend(qv)
        g.extend(gv)
    q, g = np.array(sorted(q)), np.array(sorted(g))
    qm = [dict(meta["train_fit"][i], has_mate="0" if vids[i] in absent_ids else "1") for i in q]
    gm = [meta["train_fit"][i] for i in g]
    return vectors, meta, vids, cams, fit, dev, q, g, qm, gm


def apply_whitening(x, lw):
    return l2((l2(x).astype(np.float64) - lw["m"]) @ lw["P"].T)


def evaluate(scores, qm, gm, threshold=0.0, fake_cameras=False):
    return EVALUATE(scores, query_ids=[r["vehicle_id"] for r in qm], gallery_ids=[r["vehicle_id"] for r in gm],
                    query_cameras=[f"q:{i}" if fake_cameras else r["camera_id"] for i, r in enumerate(qm)],
                    gallery_cameras=[f"g:{i}" if fake_cameras else r["camera_id"] for i, r in enumerate(gm)],
                    known_absent=[r.get("has_mate", "1") == "0" for r in qm], threshold=float(threshold),
                    camera_policy="market", refusal_mode="presence")


def calibrate(scores, qm, gm):
    curve = evaluate(scores, qm, gm)["refusal"]["pr_curve"]
    choices = []
    for t, p, r in zip(curve["thresholds"], curve["precision"], curve["recall"]):
        if t is not None and p is not None and r is not None:
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            choices.append((f1, float(t)))
    return max(choices)[1]


def fuse(base, student, w):
    return np.concatenate([math.sqrt(1-w) * l2(base), math.sqrt(w) * l2(student)], axis=1)


def head_numpy(x, weights):
    x = l2(x)
    hidden = np.maximum(x @ weights["fc1.weight"].T + weights["fc1.bias"], 0)
    return l2(x + hidden @ weights["fc2.weight"].T + weights["fc2.bias"])


def train(args, protocol, files, data):
    import torch
    from torch import nn
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    vectors, meta, vids, cams, fit, dev, q, g, qm, gm = data
    out = args.out
    if (out / "training.json").exists() or (out / "main.npz").exists():
        raise ValueError("output already has a training run; refuse overwrite")
    ensemble = l2(vectors["A_train_fit"] + vectors["B_train_fit"])
    lw = LEARN_LW(ensemble[fit], vids[fit], cams[fit], 0.5)
    target = apply_whitening(ensemble, lw)
    np.savez(out / "teacher_fit_only.npz", P=lw["P"], m=lw["m"])
    split = {"fit_indices": fit.tolist(), "dev_indices": dev.tolist(), "dev_query_indices": q.tolist(),
             "dev_gallery_indices": g.tolist(), "fit_ids": sorted(set(vids[fit].tolist())),
             "dev_ids": sorted(set(vids[dev].tolist())), "teacher_fit_pair_count": lw["n_pairs"]}
    dump(out / "split.json", split)
    x = torch.from_numpy(vectors["B_train_fit"])
    y = torch.from_numpy(target)
    ids_tensor, cams_tensor = torch.from_numpy(vids), torch.from_numpy(cams)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 512)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, v):
            return F.normalize(v + self.fc2(F.relu(self.fc1(v))), dim=1)

    def losses(pred, truth, batch):
        emb = (1 - (pred * truth).sum(1)).mean()
        mask = cams_tensor[batch, None] != cams_tensor[batch][None, :]
        pos = mask & (ids_tensor[batch, None] == ids_tensor[batch][None, :])
        neg = mask & ~pos
        delta = ((pred @ pred.T) - (truth @ truth.T)).square()
        groups = [delta[m].mean() for m in [pos, neg] if m.any()]
        relation = torch.stack(groups).mean() if groups else pred.sum() * 0
        return emb, relation

    byid = {v: np.flatnonzero((vids == v) & np.isin(np.arange(len(vids)), fit)) for v in sorted(set(vids[fit]))}
    unique = np.array(list(byid))
    def batch_indices(rng):
        selected = rng.choice(unique, 32, replace=False)
        batch = []
        for v in selected:
            rows = byid[v]
            cameras = rng.permutation(np.unique(cams[rows]))
            batch.extend(int(rng.choice(rows[cams[rows] == cameras[k % len(cameras)]])) for k in range(4))
        return np.array(batch)

    seed = protocol["training"]["seed"]
    torch.manual_seed(seed)
    smoke = Head()
    opt = torch.optim.AdamW(smoke.parameters(), lr=0.001, weight_decay=0.0001)
    b = batch_indices(np.random.default_rng(seed))
    before = smoke.fc2.weight.detach().clone()
    smoke_rows = []
    for _ in range(2):
        opt.zero_grad()
        emb, rel = losses(smoke(x[b]), y[b], b)
        loss = emb + 0.5 * rel
        loss.backward()
        grads = [p.grad for p in smoke.parameters() if p.grad is not None]
        assert all(torch.isfinite(v).all() for v in grads) and sum(float(v.abs().sum()) for v in grads) > 0
        opt.step()
        smoke_rows.append({"embedding_loss": float(emb.detach()), "relational_loss": float(rel.detach())})
    assert not torch.equal(before, smoke.fc2.weight)
    with torch.no_grad():
        expected = smoke(x[b]).numpy()
    sw = {k: v.detach().numpy() for k, v in smoke.state_dict().items()}
    np.testing.assert_allclose(head_numpy(vectors["B_train_fit"][b], sw), expected, atol=1e-6)
    np.savez(out / "smoke.npz", **sw)
    np.savez(out / "smoke_vectors.npz", x=vectors["B_train_fit"][b], torch_output=expected)
    dump(out / "smoke.json", {"status": "PASS", "batches": 2, "losses": smoke_rows, "numpy_max_abs": float(np.max(np.abs(head_numpy(vectors["B_train_fit"][b], sw)-expected)))})
    if args.smoke_only:
        return

    start_all = time.monotonic()
    runs = {}
    for name, coefficient in [("main", 0.5), ("ablation", 0.0)]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = Head()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        with torch.no_grad():
            init_dev = float((1 - (model(x[dev]) * y[dev]).sum(1)).mean())
            init_fit = float((1 - (model(x[fit]) * y[fit]).sum(1)).mean())
        best, best_epoch, history = float("inf"), None, []
        for epoch in range(1, 31):
            t0 = time.monotonic()
            summed = []
            model.train()
            for _ in range(math.ceil(len(fit) / 128)):
                b = batch_indices(rng)
                optimizer.zero_grad()
                emb, rel = losses(model(x[b]), y[b], b)
                loss = emb + coefficient * rel
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                optimizer.step()
                summed.append([float(emb.detach()), float(rel.detach())])
            model.eval()
            with torch.no_grad():
                dev_loss = float((1 - (model(x[dev]) * y[dev]).sum(1)).mean())
            row = {"epoch": epoch, "train_embedding": float(np.mean(summed, 0)[0]), "train_relation": float(np.mean(summed, 0)[1]),
                   "dev_embedding": dev_loss, "seconds": time.monotonic()-t0, "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
            history.append(row)
            print(json.dumps({"run": name, **row}), flush=True)
            if dev_loss < best:
                best, best_epoch = dev_loss, epoch
                np.savez(out / f"{name}.npz", **{k: v.detach().numpy() for k, v in model.state_dict().items()})
            if row["max_rss_mib"] > 2048 or time.monotonic()-start_all > 3600:
                raise RuntimeError("frozen resource budget exceeded")
            if epoch == 3:
                passed = dev_loss < init_dev and row["train_embedding"] < init_fit and sum(r["seconds"] for r in history)*10 <= 3600
                dump(out / f"{name}_pilot.json", {"status": "PASS" if passed else "FAIL", "initial_dev": init_dev, "initial_fit": init_fit, "epochs": history})
                if not passed:
                    runs[name] = {"status": "STOPPED_PILOT", "history": history}
                    dump(out / "training.json", runs)
                    raise RuntimeError("pilot failed; no new hypothesis allowed")
        runs[name] = {"status": "PASS", "initial_dev": init_dev, "initial_fit": init_fit, "best_epoch": best_epoch, "best_dev_loss": best, "history": history, "checkpoint_sha256": sha(out / f"{name}.npz")}
        dump(out / "training.json", runs)
    dump(out / "environment.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
         "threads": torch.get_num_threads(), "protocol_sha256": sha(HERE / "protocol.json"), "experiment_sha256": sha(__file__),
         "total_seconds": time.monotonic()-start_all, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


def scores(q, g, mode):
    return COSINE(q, g) if mode == "cosine" else RERANK(q, g, 6, 3, 0.3)


def summary(result):
    r, f = result["ranking_full_gallery"], result["refusal"]
    return {"mAP": r["mAP"], "Rank-1": r["Rank-1"], "Rank-5": r["Rank-5"], "mINP": r["mINP"],
            "F1": f["f1"], "TNR": f["tnr"], "threshold": f["threshold"], "counts": result["counts"]}


def paired_bootstrap(a, b, ids):
    unique = sorted(set(ids))
    delta = np.asarray(a)-np.asarray(b)
    sums = np.array([delta[np.array(ids) == v].sum() for v in unique])
    counts = np.array([np.count_nonzero(np.array(ids) == v) for v in unique])
    rng = np.random.default_rng(20260925)
    values = []
    for _ in range(4000):
        pick = rng.integers(len(unique), size=len(unique))
        values.append(float(sums[pick].sum()/counts[pick].sum()))
    values = np.array(values)
    return {"delta": float(delta.mean()), "ci95": np.quantile(values, [.025, .975]).tolist(),
            "p": min(1.0, 2*min((np.count_nonzero(values <= 0)+1)/4001, (np.count_nonzero(values >= 0)+1)/4001)),
            "clusters": len(unique), "queries": len(ids), "replicates": 4000, "seed": 20260925}


def evaluate_run(args, protocol, files, data):
    v, meta, vids, cams, fit, dev, dq, dg, dqm, dgm = data
    out = args.out
    lw = dict(np.load(out / "teacher_fit_only.npz", allow_pickle=False))
    dev_base = apply_whitening(l2(v["A_train_fit"] + v["B_train_fit"]), lw)
    main = dict(np.load(out / "main.npz", allow_pickle=False))
    ablation = dict(np.load(out / "ablation.npz", allow_pickle=False))
    ds = head_numpy(v["B_train_fit"], main)
    weights = []
    for w in protocol["fusion"]["student_weights"]:
        f = fuse(dev_base, ds, w)
        weights.append((evaluate(COSINE(f[dq], f[dg]), dqm, dgm)["ranking_full_gallery"]["mAP"], -w))
    weight = -max(weights)[1]
    devs = {"baseline": dev_base, "student": ds, "fusion": fuse(dev_base, ds, weight), "ablation": head_numpy(v["B_train_fit"], ablation)}
    thresholds = {name: {mode: calibrate(scores(x[dq], x[dg], mode), dqm, dgm) for mode in ["cosine", "KR"]} for name, x in devs.items()}
    dump(out / "selection.json", {"fusion_student_weight": weight, "dev_grid": [{"weight": -w, "mAP": m} for m,w in weights],
                                  "thresholds": thresholds, "dev_query": len(dq), "dev_gallery": len(dg), "selected_before_external_evaluation": True})
    # Canonical release vectors and row count are frozen independently; compare
    # reconstruction too, so shuffled source rows cannot hide behind dimensions.
    canonical = np.load(files["canonical_embeddings"], allow_pickle=False)
    qn = len(meta["val_query"])
    if canonical.shape != (qn+len(meta["val_gallery"]), 512):
        raise ValueError("canonical shape")
    release_lw = dict(np.load(files["whitening_release"], allow_pickle=False))
    raw_ens = l2(np.concatenate([v["A_val_query"]+v["B_val_query"], v["A_val_gallery"]+v["B_val_gallery"]]))
    reconstruction = l2((raw_ens - release_lw["m"]) @ release_lw["P"].T)
    max_delta = float(np.max(np.abs(canonical-reconstruction)))
    if max_delta > 1e-6:
        raise ValueError(f"canonical row/model mismatch {max_delta}")
    query_b, gallery_b = v["B_val_query"], v["B_val_gallery"]
    student = np.concatenate([head_numpy(query_b, main), head_numpy(gallery_b, main)])
    datasets = {"baseline": canonical, "student": student, "fusion": fuse(canonical, student, weight),
                "ablation": np.concatenate([head_numpy(query_b, ablation), head_numpy(gallery_b, ablation)])}
    all_metrics, results = {}, {}
    qm, gm = meta["val_query"], meta["val_gallery"]
    official_thresholds = {"cosine": 0.5141976914190476, "KR": 0.5282812306342437}
    for name, x in datasets.items():
        all_metrics[name], results[name] = {}, {}
        for mode in ["cosine", "KR"]:
            matrix = scores(x[:qn], x[qn:], mode)
            r = evaluate(matrix, qm, gm, thresholds[name][mode])
            no_cam = evaluate(matrix, qm, gm, thresholds[name][mode], fake_cameras=True)
            metrics = summary(r)
            metrics["camera_gap_mAP"] = no_cam["ranking_full_gallery"]["mAP"] - metrics["mAP"]
            metrics["by_camera"] = {c: {"known_queries": len(rows), "mAP": float(np.mean([z["ap"] for z in rows])), "Rank-1": float(np.mean([z["rank1"] for z in rows]))}
                 for c in sorted(set(row["camera_id"] for row in qm))
                 if (rows := [z for i,z in enumerate(r["per_query"]) if qm[i]["camera_id"] == c and z["status"] == "known"])}
            if name == "baseline":
                metrics["official_threshold_result"] = summary(evaluate(matrix, qm, gm, official_thresholds[mode]))
                expected = protocol["evaluation"]["baseline_expected"][f"{mode}_mAP"]
                if abs(metrics["mAP"]-expected) > 1e-14:
                    raise AssertionError(f"baseline not reproduced {mode}: {metrics['mAP']} != {expected}")
            all_metrics[name][mode], results[name][mode] = metrics, r
            print(json.dumps({"model": name, "mode": mode, **{k:metrics[k] for k in ["mAP", "Rank-1", "F1", "TNR"]}}), flush=True)
    comparisons = {}
    for name in ["student", "fusion", "ablation"]:
        comparisons[name] = {}
        for mode in ["cosine", "KR"]:
            a, b = results[name][mode]["per_query"], results["baseline"][mode]["per_query"]
            known = [i for i, z in enumerate(b) if z["status"] == "known"]
            compare = paired_bootstrap([a[i]["ap"] for i in known], [b[i]["ap"] for i in known], [qm[i]["vehicle_id"] for i in known])
            compare["fixed_query_ids"] = [qm[i]["image_id"] for i in known if a[i]["rank1"] and not b[i]["rank1"]]
            compare["broken_query_ids"] = [qm[i]["image_id"] for i in known if b[i]["rank1"] and not a[i]["rank1"]]
            ae = np.array([not a[i]["rank1"] for i in known], float)
            be = np.array([not b[i]["rank1"] for i in known], float)
            compare["top1_error_correlation"] = float(np.corrcoef(ae, be)[0,1])
            comparisons[name][mode] = compare
    family = sorted([(comparisons[n][m]["p"], n,m) for n in ["student", "fusion"] for m in ["cosine", "KR"]])
    running = 0.
    for i, (p,n,m) in enumerate(family):
        running = max(running, min(1., p*(len(family)-i)))
        comparisons[n][m]["p_Holm"] = running
    dump(out / "evaluation.json", {"metrics": all_metrics, "comparisons": comparisons,
         "canonical_reconstruction_max_abs": max_delta, "external_eval_used_for_selection": False,
         "scope": "repeatedly used historical validation; not an untouched holdout", "fusion_dimensions": 1024,
         "student_dimensions":512, "view_strata": "NOT MEASURED: no independent view labels"})
    np.savez(out / "export_validation.npz", input=np.concatenate([query_b,gallery_b]), output=student)
    dump(out / "artifact_hashes.json", {p.name: {"sha256":sha(p),"bytes":p.stat().st_size} for p in out.glob("*.npz")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["train", "evaluate"])
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / "protocol.json").read_text())
    files = setup(args.repo, protocol)
    data = load_data(files, protocol)
    if args.phase == "train":
        train(args, protocol, files, data)
    else:
        evaluate_run(args, protocol, files, data)


if __name__ == "__main__":
    main()
