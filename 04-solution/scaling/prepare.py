"""Freeze inputs without writing into the service; verify model provenance."""
import csv
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("REID_REPO", "/home/artem/projects/hackathon-lct-vehicle-reid"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    subprocess.run(["free", "-m"], check=True)
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)
    files = {
        "reid_metrics.py": "04-solution/eval/reid_metrics.py",
        "scope_metrics.py": "04-solution/eval/scope_metrics.py",
        "rerank.py": "04-solution/service/app/core/rerank.py",
        "ranking.py": "04-solution/service/app/core/ranking.py",
    }
    for part in ("query", "gallery"):
        files[f"val_{part}.npy"] = f"04-solution/postproc/out/val_{part}.npy"
        files[f"val_{part}.ids"] = f"04-solution/postproc/out/val_{part}.ids"
        files[f"val_{part}.csv"] = f"04-solution/split/files/val_{part}.csv"
    hashes = {}
    for dest, src in files.items():
        shutil.copyfile(REPO / src, inputs / dest)
        hashes[dest] = {"source": src, "sha256": sha(inputs / dest)}
    service = REPO / "04-solution/service"
    service_hashes = {str(p.relative_to(service)): sha(p) for p in service.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts}
    (HERE / "service_before.json").write_text(json.dumps(service_hashes, indent=2))
    sys.path.insert(0, str(REPO / "04-solution/baseline/scripts"))
    from extract_embeddings import make_session, load_crop, l2norm
    model = service / "model/osnet_ain_x1_0_vehicle_reid.onnx"
    sess = make_session(model, 1)
    inp = sess.get_inputs()[0].name
    checks = {}
    for part in ("query", "gallery"):
        rows = list(csv.DictReader((inputs / f"val_{part}.csv").open()))
        ids = (inputs / f"val_{part}.ids").read_text().split()
        assert ids == [r["image_id"] for r in rows]
        emb = np.load(inputs / f"val_{part}.npy")
        assert emb.shape == (len(rows), 512) and np.isfinite(emb).all()
        picks = sorted(set([0, len(rows)//2, len(rows)-1] +
                           np.random.default_rng(17).choice(len(rows), 3, replace=False).tolist()))
        checks[part] = []
        for i in picks:
            r = rows[i]
            crop = load_crop(REPO / "data/images", (r["image_id"], *(int(r[k]) for k in ("x","y","w","h"))), 0, False)
            vec = l2norm(sess.run(None, {inp: crop[None]})[0].astype(np.float64))[0]
            cosine = float(vec @ emb[i].astype(np.float64))
            assert cosine >= .9999, (part, i, cosine)
            checks[part].append({"row": i, "image_id": r["image_id"], "cosine": cosine,
                                 "max_abs_diff": float(np.max(np.abs(vec-emb[i])))})
    meta = {"repo": str(REPO), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "files": hashes, "model_sha256": sha(model), "reextraction_checks": checks,
            "packages": {n: md.version(n) for n in ("numpy", "faiss-cpu", "packaging", "onnxruntime", "pillow")},
            "python": sys.version, "ids_match_csv": True}
    (HERE / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"ids_match_csv": True, "reextracted": sum(map(len, checks.values())), "packages": meta["packages"]}))


if __name__ == "__main__":
    main()
