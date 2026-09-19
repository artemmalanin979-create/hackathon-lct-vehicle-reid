#!/usr/bin/env python3
"""job_49: проверки контура (задание 3).
(а) полный verify.py в КОПИИ каталога eval (репозиторий не трогаем),
    без cli_and_comparator (example.npz отсутствует в зеркале репо);
(б) три подмены на реальных скорах d1:
    T1  согласованная перестановка строк галереи -> метрика НЕ меняется (инвариантность);
    T1b случайная перестановка СТОЛБЦОВ скоров (рассинхрон с метками) -> обвал;
    T2  подмена vehicle_id галереи -> обвал до ~0,01;
    T3  все камеры одинаковые -> market-фильтр выкидывает все позитивы (mAP None/обвал);
    T3b все камеры разные (нет исключения) -> рост до ~0,876 (проверка исключения same-cam).
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HOME = Path.home() / "lct-reid"
REPO = HOME / "repo"
S = REPO / "04-solution"
EVAL = S / "eval"
WORK = HOME / "jobs/job_49"
COPY = WORK / "eval_check"
OUT = WORK / "out"

# ---------- (а) verify в копии ----------
if COPY.exists():
    shutil.rmtree(COPY)
shutil.copytree(EVAL, COPY, ignore=shutil.ignore_patterns("__pycache__"))
runner = COPY / "_run_all_but_cli.py"
runner.write_text("""
import json, traceback
import verify
report = {}
report["baseline"] = None
baseline, log = verify.test_run(verify.metrics)
open(verify.ROOT / "test_results.log", "w").write(log)
report["baseline"] = baseline
steps = ["differential_ranking", "differential_tied_external", "differential_refusal",
         "differential_all_fields", "differential_new_fields", "random_level",
         "mutations", "compatibility_edges"]
report["differential_all_fields"] = None
import oracle_checks, test_protocols
report["differential_all_fields"] = oracle_checks.differential_all_fields(verify.metrics)
report["differential_new_fields"] = test_protocols.differential_extensions()
for name in ["differential_ranking", "differential_tied_external", "differential_refusal",
             "random_level", "compatibility_edges"]:
    report[name] = getattr(verify, name)()
report["mutations"] = verify.mutation_check()
json.dump(report, open("/home/fedora/lct-reid/jobs/job_49/out/verify_copy_results.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print("MUTANTS:", report["mutations"]["killed"], "/", report["mutations"]["total"])
print("ALL-STEPS-OK")
""")
r = subprocess.run([sys.executable, "-B", str(runner)], cwd=COPY, capture_output=True, text=True)
print(r.stdout[-2000:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-3000:])
    raise SystemExit("verify-copy failed")

# ---------- (б) подмены на скорах d1 ----------
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(WORK / "scripts"))
from reid_metrics import evaluate  # noqa: E402
from recalc_d1 import (l2n, read_meta, learn_whitening, apply_whitening,  # noqa: E402
                       kreciprocal, SPLIT, A2, J42, J45)

qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
tf = read_meta(SPLIT / "train_fit.csv")
vq = {t: l2n(np.load(A2 / f"val_query_{t}.npy")) for t in ("osnet", "ainv2")}
vg = {t: l2n(np.load(A2 / f"val_gallery_{t}.npy")) for t in ("osnet", "ainv2")}
Xt = {"osnet": l2n(np.load(J42 / "train_fit_osnet.npy")),
      "ainv2": l2n(np.load(J45 / "train_fit_ainv2.npy"))}
vid_t = np.array([r["vehicle_id"] for r in tf]); cam_t = np.array([r["camera_id"] for r in tf])
lw = learn_whitening(l2n(Xt["osnet"] + Xt["ainv2"]), vid_t, cam_t, 0.5)
dq = apply_whitening(l2n(vq["osnet"] + vq["ainv2"]), lw)
dg = apply_whitening(l2n(vg["osnet"] + vg["ainv2"]), lw)
scores = -kreciprocal(dq, dg)

qids = [r["vehicle_id"] for r in qm]; gids = [r["vehicle_id"] for r in gm]
qc = [r["camera_id"] for r in qm]; gc = [r["camera_id"] for r in gm]
absent = np.array([r.get("has_mate", "1") == "0" for r in qm])


def run(sc, tag, qids_=qids, gids_=gids, qc_=qc, gc_=gc):
    res = evaluate(sc, query_ids=qids_, gallery_ids=gids_, query_cameras=qc_,
                   gallery_cameras=gc_, known_absent=absent, threshold=0.0,
                   camera_policy="market", refusal_mode="presence")
    f = res["ranking_full_gallery"]
    print(f"{tag:36s} mAP={f['mAP']} R1={f['Rank-1']} valid={f['num_valid_queries']}", flush=True)
    return f


tamper = {}
tamper["T0_canon"] = run(scores, "T0: эталон d1 (без подмен)")

# T1: согласованная перестановка строк галереи (скоры+метки)
rng = np.random.default_rng(7)
perm = rng.permutation(scores.shape[1])
sc1 = scores[:, perm]
tamper["T1_consistent_perm"] = run(sc1, "T1: согласованная перестановка G",
                                   gids_=[gids[j] for j in perm], gc_=[gc[j] for j in perm])

# T1b: перестановка только столбцов скоров (рассинхрон)
tamper["T1b_scores_only_perm"] = run(sc1, "T1b: переставлены только скоры")

# T2: подмена vehicle_id галереи
gids_sh = list(np.random.default_rng(8).permutation(np.array(gids, dtype=object)))
tamper["T2_gallery_vid_subst"] = run(scores, "T2: vehicle_id галереи переставлены",
                                     gids_=gids_sh)

# T3: все камеры одинаковые
tamper["T3_all_cams_equal"] = run(scores, "T3: все камеры = 'X'",
                                  qc_=["X"] * len(qm), gc_=["X"] * len(gm))

# T3b: все камеры разные (исключение same-cam отключено)
tamper["T3b_all_cams_distinct"] = run(
    scores, "T3b: все камеры уникальны",
    qc_=[f"q{i}" for i in range(len(qm))], gc_=[f"g{j}" for j in range(len(gm))])

json.dump(tamper, open(OUT / "tamper.json", "w"), indent=1)
print("TAMPER-OK")
