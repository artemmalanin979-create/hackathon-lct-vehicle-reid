#!/usr/bin/env python3
"""Compare documented metric tables directly with their unrounded JSON sources."""
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[3]
CHECKS = []


def ref(file, *keys):
    path = REPO / "04-solution" / file
    obj = json.loads(path.read_text())
    for key in keys:
        obj = obj[key]
    return obj, str(path.relative_to(REPO)) + "#" + "/".join(map(str, keys))


def scores(file, keys, fields):
    return [ref(file, *keys, field) for field in fields]


def table(file, marker, values, offset=0):
    matches = [(i, line) for i, line in enumerate((REPO / file).read_text().splitlines(), 1)
               if line.startswith("|") and
               (line.split("|")[1].strip() == marker[1:] if marker.startswith("=")
                else marker in line.split("|")[1])]
    assert len(matches) == 1, (file, marker, matches)
    lineno, line = matches[0]
    tokens = re.findall(r"[+−-]?\d+[.,]\d+", "|".join(line.split("|")[2:]))[offset:]
    check_tokens(file, lineno, tokens, values)


def check_tokens(file, lineno, tokens, values):
    assert len(tokens) >= len(values), (file, lineno, tokens, values)
    for token, (value, source) in zip(tokens, values):
        normalized = token.replace(",", ".").replace("−", "-").lstrip("+")
        digits = len(normalized.split(".")[1])
        expected = f"{value:.{digits}f}"
        CHECKS.append({"file": file, "line": lineno, "claim": token, "source": source,
                       "raw": value, "expected": expected, "ok": normalized == expected})


def main():
    rank = ["mAP", "Rank-1", "Rank-5", "mINP"]
    combined = "training/combined/s02_metrics.json"
    baseline = "baseline/out/metrics_summary.json"
    final = "postproc/out/final_val.json"
    headline = "service/calib-d1_j48/headline.json"
    for marker, key in (("d1_j48, косинус", "cos"), ("d1_j48, переранжирование", "kr")):
        table("SOLUTION.md", marker, scores(combined, ["d1_j48", key], rank + ["mAP@10"]))
    table("SOLUTION.md", "Прежняя сдаваемая конфигурация", scores(combined, ["osnet(repro)", "kr"], rank + ["mAP@10"]))
    for marker, key in (("Косинус d1_j48", "cosine_rule_reproduced"), ("Переранжирование d1_j48", "rerank_point")):
        table("SOLUTION.md", marker, scores(headline, [key], ["f1", "tnr", "recall", "auc_pr"]))
    for marker, key in (("Полная галерея", "kr"), ("Косинус (HTTP", "cos")):
        table("04-solution/README.md", marker, scores(combined, ["d1_j48", key], rank[:3]))
    table("04-solution/README.md", "Усечение до 10", [ref(combined, "d1_j48", "kr", "mAP@10")])
    table("04-solution/README.md", "Без исключения", scores(baseline, ["base_no_camera_excl", "full_gallery"], rank[:2]))
    for doc in ("04-solution/baseline/README.md", "04-solution/baseline/REPORT.md"):
        if doc.endswith("README.md"):
            markers = [("OSNet-AIN, честный", "base_market"), ("Случайные векторы", "random_mean"),
                       ("Без исключения", "base_no_camera_excl")]
        else:
            markers = [("OSNet, market (наше", "base_market"), ("Случайные векторы 512", "random_mean"),
                       ("OSNet, **без**", "base_no_camera_excl"), ("OSNet, all_same_camera", "base_all_same_camera")]
        for marker, key in markers:
            table(doc, marker, scores(baseline, [key, "full_gallery"], rank))
    table("04-solution/baseline/README.md", "По усечению", scores(baseline, ["base_market", "top_k10"], rank[:3]))
    for marker, key in (("Наше заявленное", "base"), ("с параметрами прямо из статьи", "rerank_paper"),
                        ("с подобранными параметрами", "rerank_tuned")):
        table("04-solution/baseline/review/README.md", marker, scores(final, [key], rank[:2]))
    grid_file = "postproc/out/grid_val_ttacrit.json"
    points, _ = ref(grid_file, "points")
    chosen = next(i for i, p in enumerate(points) if (p["k1"], p["k2"], p["lam"]) == (6, 3, 0.3))
    tta4 = scores(grid_file, ["points", chosen], rank[:2])
    table("04-solution/baseline/review/README.md", "Вместе", tta4)
    table("04-solution/baseline/review/README.md", "Усреднение по четырём входам",
          scores("postproc/out/tta_val.json", ["runs", 8], rank[:2]))
    for lineno, line in enumerate((REPO / "04-solution/baseline/README.md").read_text().splitlines(), 1):
        if line.startswith("**mAP 0,") and "и Rank-1" in line:
            check_tokens("04-solution/baseline/README.md", lineno, re.findall(r"0,\d+", line), tta4)
    for marker, key in (("база", "base"), ("+ переранжирование (6", "rerank_tuned"),
                        ("с параметрами из статьи", "rerank_paper"), ("по трём входам", "tta3_rerank_tuned")):
        table("04-solution/postproc/README.md", marker, scores(final, [key], rank[:2]))
    table("04-solution/postproc/README.md", "со второй моделью", scores("postproc/out/fusion.json", ["fusion_rerank_val"], rank[:2]))
    for marker, file, keys in (
        ("оптимум, подобранный", "postproc/out/grid_val_base.json", ["best"]),
        ("параметры, подобранные честно", final, ["rerank_tuned"]),
        ("«всё вместе», подобранное", "postproc/out/grid_val_ttacrit.json", ["best"]),
        ("«всё вместе» честно", final, ["tta3_rerank_tuned"])):
        table("04-solution/postproc/README.md", marker, scores(file, keys, ["mAP"]))
    for marker, key in (("база (OSNet", "base"), ("+ rerank из статьи (", "rerank_paper"),
                        ("+ rerank подобр.", "rerank_tuned"), ("TTA2 (", "tta2"),
                        ("TTA2 + rerank", "tta2_rerank_tuned"), ("TTA3 (", "tta3"),
                        ("TTA3 + rerank (", "tta3_rerank_tuned"), ("TTA3 + rerank из статьи", "tta3_rerank_paper")):
        table("04-solution/postproc/REPORT.md", marker, scores(final, [key], ["mAP", "Rank-1", "mAP@10"]), offset=1)
    for marker, key in (("fast-reid R50", "fr_alone_val"), ("слияние TTA3", "fusion_val"), ("слияние + rerank", "fusion_rerank_val")):
        table("04-solution/postproc/REPORT.md", marker, scores("postproc/out/fusion.json", [key], ["mAP", "Rank-1", "mAP@10"]), offset=1)
    for marker, key in (("d1_j48, `--no-rerank`", "cos"), ("d1_j48, по умолчанию", "kr")):
        table("04-solution/service/README.md", marker, scores(combined, ["d1_j48", key], ["mAP", "Rank-1", "mAP@10"]))
    for marker, key in (("=по умолчанию (переранжирование)", "rerank_point"), ("`--no-rerank`", "cosine_rule_reproduced")):
        # The second marker must select the refusal row, not the earlier d1_j48 table.
        if key == "cosine_rule_reproduced": marker = "=`--no-rerank`"
        table("04-solution/service/README.md", marker, scores(headline, [key], ["threshold", "f1", "tnr"]))
    for doc in ("04-solution/reproduce/README.md", "04-solution/reproduce/evidence/README.md"):
        for marker, mode, refusal in (("cosine", "cos", "cosine_rule_reproduced"), ("rerank", "kr", "rerank_point")):
            values = scores(combined, ["d1_j48", mode], rank[:2]) + scores(headline, [refusal], ["f1", "tnr"])
            if "/evidence/" in doc:
                values += [ref("reproduce/evidence/validation-metrics.json", "calibration" if mode == "cos" else "rerank_calibration", "threshold")]
            table(doc, marker, values)
    for marker, key in (("a. OSNet", "a"), ("b. OSNet", "b"), ("c. ансамбль", "c_w1.0"),
                        ("d1. ансамбль", "d1_w1.0"), ("d2. ансамбль", "d2_w1.0"), ("d3. whitening", "d3_w1.0")):
        table("04-solution/postproc/whitening/final/README.md", marker,
              scores("postproc/whitening/final/s02_configs.json", ["table", key, "kr"], ["mAP", "Rank-1", "mAP@10"]))
    for key in ("AB", "ABD", "ABCD", "ABC", "ABDE", "ABE", "ABCE"):
        file = f"training/branches3/{key}_metrics.json"
        table("04-solution/training/branches3/README.md", "AB — сдаём" if key == "AB" else f" {key} ",
              scores(file, ["kr"], ["mAP", "Rank-1", "mINP"]) + [ref(file, "camera_gap_kr", "mAP"),
              ref("training/branches3/cpu_batch1_benchmark.json", "results", key, "mean_ms")])
    for marker, keys in (("d1_j48 @ 208", ["baseline", "d1"]), ("d1 @ 256", ["d1_256"]),
                         ("одиночная combined_v1", ["baseline", "single"]), ("одиночная дообученная", ["single256"]),
                         ("одиночная **старые**", ["old_single256"])):
        file = "training/input256/eval_all_configs.json"
        table("04-solution/training/input256/README.md", marker,
              scores(file, keys + ["kr"], rank[:2]) + [ref(file, *keys, "camera_gap_kr", "mAP")])
    for marker, key in (("ничего", "base"), ("зона пластины", "plate_ring"), ("сдвиг по вертикали", "shift_ring"),
                        ("зеркально", "mirror_ring"), ("вбок", "side_ring")):
        table("04-solution/plate-ablation/README.md", marker, [ref("plate-ablation/out/ablation.json", "metrics", key, "mAP")])
    for field in ("mAP", "Rank-1", "mINP"):
        file = "split/out/evalrun/degenerate_results.json"
        table("04-solution/split/README.md", f" {field} ", [ref(file, "random", 0, "expected", field)] +
              [ref(file, "random", i, "observed", field) for i in range(3)])
    errors = [c for c in CHECKS if not c["ok"]]
    print(json.dumps({"checks": len(CHECKS), "documents": len({c['file'] for c in CHECKS}),
                      "errors": errors, "claims": CHECKS}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
