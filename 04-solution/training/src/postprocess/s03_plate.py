#!/usr/bin/env python3
"""job_48, шаг 3: абляция зоны номерной пластины для новой модели.

Код и схема контроля — дословно 04-solution/plate-ablation/scripts/{extract_variants,
eval_variants}.py (маски детектора кешируются, модельнонезависимы — берутся из
job_45b/out/boxes.json). Варианты сокращены до обязательного ядра + консервативной
проверки: base, plate_ring, shift_ring (основной контраст), platepad_ring,
shiftpad_ring (бокс +12%), plate_gray127, shift_gray127 (робастность заливки).
Порог тот же, что у бейзлайна; бутстрэп парный по per-query AP, 4000, seed 20260915.
"""
from __future__ import annotations

import csv, json, sys, time, hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HOME = Path.home() / "lct-reid"
REPO = HOME / "repo"
JOB = HOME / "jobs" / "job_48"
OUT = JOB / "out" / "plate"
SPLIT = REPO / "04-solution" / "split" / "files"
DATA = HOME / "data"
BOXES = HOME / "jobs" / "job_45b" / "out" / "boxes.json"
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(JOB.glob("job48_combined_s*.onnx"))[0]
PAD = 0.12
VARIANTS = ["base", "plate_ring", "shift_ring", "platepad_ring", "shiftpad_ring",
            "plate_gray127", "shift_gray127"]
SEED_BOX = 1000  # как в оригинальном extract_variants (seed=1000+s+j)

sys.path.insert(0, str(REPO / "04-solution" / "plate-ablation" / "scripts"))
from mask_ops import fill, controls  # noqa: E402
sys.path.insert(0, str(REPO / "04-solution" / "eval"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402


def pad_box(b, shape, f=PAD):
    H, W = shape; x, y, w, h = b
    dx, dy = int(round(w * f)), int(round(h * f))
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(W, x + w + dx), min(H, y + h + dy)
    return (x0, y0, x1 - x0, y1 - y0)


def make_variants(arr, box, seed):
    out = {"base": arr}
    if box is None:
        for v in VARIANTS[1:]:
            out[v] = arr
        return out
    ctl = controls(box, arr.shape[:2], seed)
    pbox = pad_box(box, arr.shape[:2])
    pctl = controls(pbox, arr.shape[:2], seed)
    out["plate_ring"] = fill(arr, box, "ring")
    out["shift_ring"] = fill(arr, ctl["shift"], "ring")
    out["platepad_ring"] = fill(arr, pbox, "ring")
    out["shiftpad_ring"] = fill(arr, pctl["shift"], "ring")
    out["plate_gray127"] = fill(arr, box, "gray127")
    out["shift_gray127"] = fill(arr, ctl["shift"], "gray127")
    return out


def to_input(a):
    im = Image.fromarray(a).resize((208, 208), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1)


def extract():
    OUT.mkdir(parents=True, exist_ok=True)
    boxes = json.load(open(BOXES))
    o = ort.SessionOptions(); o.log_severity_level = 3
    o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(MODEL), sess_options=o, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    info = {"model": str(MODEL), "sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            "variants": VARIANTS, "pad": PAD}
    print(json.dumps(info), flush=True)
    for split in ("val_query", "val_gallery"):
        rows = [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
                for r in csv.DictReader(open(SPLIT / f"{split}.csv", newline=""))]
        bx = boxes[split]
        assert len(bx) == len(rows), (len(bx), len(rows))
        acc = {v: np.empty((len(rows), 512), np.float32) for v in VARIANTS}
        t0 = time.perf_counter(); B = 24
        for s in range(0, len(rows), B):
            chunk = rows[s:s + B]
            per = {v: [] for v in VARIANTS}
            for j, (iid, x, y, w, h) in enumerate(chunk):
                with Image.open(DATA / "images" / f"{iid}.jpg") as im:
                    a = np.asarray(im.convert("RGB").crop((x, y, x + w, y + h)))
                b = bx[s + j]
                vs = make_variants(a, tuple(b[:4]) if b else None, seed=SEED_BOX + s + j)
                for v in VARIANTS:
                    per[v].append(to_input(vs[v]))
            for v in VARIANTS:
                acc[v][s:s + len(chunk)] = sess.run(None, {inp: np.stack(per[v])})[0]
            if s % 480 == 0:
                print(f"  {split} {s}/{len(rows)} {time.perf_counter()-t0:.0f}s", flush=True)
        for v in VARIANTS:
            m = acc[v].astype(np.float64); m /= np.linalg.norm(m, axis=1, keepdims=True)
            np.save(OUT / f"{split}_{v}.npy", m.astype(np.float32))
        print(f"{split}: {len(rows)}x{len(VARIANTS)} за {time.perf_counter()-t0:.0f}s", flush=True)
    (OUT / "extract_info.json").write_text(json.dumps(info, indent=1) + "\n")


def evaluate_all():
    qm = list(csv.DictReader(open(SPLIT / "val_query.csv", newline="")))
    gm = list(csv.DictReader(open(SPLIT / "val_gallery.csv", newline="")))
    common = dict(query_ids=[r["vehicle_id"] for r in qm],
                  gallery_ids=[r["vehicle_id"] for r in gm],
                  query_cameras=[r["camera_id"] for r in qm],
                  gallery_cameras=[r["camera_id"] for r in gm],
                  known_absent=np.array([r.get("has_mate", "1") == "0" for r in qm]),
                  threshold=0.0, camera_policy="market", refusal_mode="presence")
    res, aps = {}, {}
    for v in VARIANTS:
        q = np.load(OUT / f"val_query_{v}.npy"); g = np.load(OUT / f"val_gallery_{v}.npy")
        r = evaluate(scores_from_embeddings(q, g, metric="cosine"), **common)
        f = r["ranking_full_gallery"]
        res[v] = {"mAP": f["mAP"], "Rank-1": f["Rank-1"], "Rank-5": f["Rank-5"], "mINP": f["mINP"],
                  "valid_queries": f["num_valid_queries"]}
        aps[v] = np.array([row["ap"] for row in r["per_query"] if row["status"] == "known"])
        print(f"{v:16s} mAP={f['mAP']:.4f} R1={f['Rank-1']:.4f}", flush=True)
    n = len(aps["base"])
    assert all(len(a) == n for a in aps.values())
    rng = np.random.default_rng(20260915)
    idx = rng.integers(0, n, size=(4000, n))
    boots = {}
    for plate, ctl in (("plate_ring", "shift_ring"),
                       ("platepad_ring", "shiftpad_ring"),
                       ("plate_gray127", "shift_gray127")):
        da = aps[plate][idx].mean(1) - aps[ctl][idx].mean(1)
        boots[f"{plate}_vs_{ctl}"] = {
            "map_plate": float(aps[plate].mean()), "map_control": float(aps[ctl].mean()),
            "delta_plate_minus_control": float(aps[plate].mean() - aps[ctl].mean()),
            "ci95": [float(np.quantile(da, 0.025)), float(np.quantile(da, 0.975))],
            "p_two_sided": float(2 * min((da <= 0).mean(), (da >= 0).mean()))}
    out = {"metrics": res, "contrasts": boots, "n_valid": n,
           "cos_to_base": {v: float((np.load(OUT / f"val_query_{v}.npy") *
                                     np.load(OUT / f"val_query_base.npy")).sum(1).mean())
                           for v in VARIANTS if v != "base"}}
    (OUT / "plate_ablation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(boots, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "eval":
        evaluate_all()
    else:
        extract()
