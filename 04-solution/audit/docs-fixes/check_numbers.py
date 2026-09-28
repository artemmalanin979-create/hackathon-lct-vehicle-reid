#!/usr/bin/env python3
"""Check numeric claims against raw JSON, including prose and repeated deltas.

Every fractional/scientific token in tracked README/entry documents is inventoried.
Unbound occurrences are explicitly not verified; --require-complete rejects
such incomplete coverage instead of treating a selected sample as a full audit.
"""
import argparse
from decimal import Decimal, ROUND_HALF_EVEN
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
CHECKS = []
NUMBER = r"[+−-]?(?:\d+[.,]\d+(?:[eE][+−-]?\d+)?|\d+[eE][+−-]?\d+)"
# A sentence-final dot is punctuation; a dot followed by a digit is a version
# component. Treating both alike silently loses numeric claims in prose.
NUMBER_RE = re.compile(r"(?<![\w.])" + NUMBER + r"(?!\w|\.\d)")


def ref(file, *keys):
    path = REPO / "04-solution" / file
    obj = json.loads(path.read_text(), parse_float=Decimal)
    for key in keys:
        obj = obj[key]
    return obj, str(path.relative_to(REPO)) + "#" + "/".join(map(str, keys))


def difference(left, right):
    """Subtract raw source values, never already rounded documentation tokens."""
    return left[0] - right[0], left[1] + " - " + right[1]


def scores(file, keys, fields):
    return [ref(file, *keys, field) for field in fields]


def table(file, marker, values, offset=0):
    matches = [(i, line) for i, line in enumerate((REPO / file).read_text().splitlines(), 1)
               if line.startswith("|") and
               (line.split("|")[1].strip() == marker[1:] if marker.startswith("=")
                else marker in line.split("|")[1])]
    assert len(matches) == 1, (file, marker, matches)
    lineno, line = matches[0]
    start = line.index("|", 1) + 1
    matches = list(NUMBER_RE.finditer(line, start))[offset:]
    check_tokens(file, lineno, [m.group() for m in matches], values,
                 [m.start() + 1 for m in matches])


def check_tokens(file, lineno, tokens, values, columns=None):
    assert len(tokens) >= len(values), (file, lineno, tokens, values)
    line = (REPO / file).read_text().splitlines()[lineno - 1]
    cursor = 0
    for index, (token, (value, source)) in enumerate(zip(tokens, values)):
        normalized = token.replace(",", ".").replace("−", "-").lstrip("+")
        claim = Decimal(normalized)
        expected = Decimal(value).quantize(Decimal(1).scaleb(claim.as_tuple().exponent),
                                           rounding=ROUND_HALF_EVEN)
        column = columns[index] if columns else line.index(token, cursor) + 1
        cursor = column - 1 + len(token)
        CHECKS.append({"file": file, "line": lineno, "column": column,
                       "claim": token, "source": source, "raw": str(value),
                       "expected": str(expected), "ok": claim == expected})


def prose(file, patterns, value):
    """Bind by surrounding meaning, not by the expected or known wrong digits.

    Each pattern has a named `claim` group. Scan ALL occurrences (including
    bullets/repetitions); a missing selector is a broken binding, not a pass.
    """
    for pattern in patterns:
        count = 0
        for lineno, line in enumerate((REPO / file).read_text().splitlines(), 1):
            for match in re.finditer(pattern, line):
                check_tokens(file, lineno, [match['claim']], [value],
                             [match.start('claim') + 1])
                count += 1
        assert count, (file, "No numeric claim matched", pattern)


def numeric_inventory(files, checks):
    checked = {(c['file'], c['line'], c['column']) for c in checks}
    inventory = []
    for file in files:
        fenced = False
        for lineno, line in enumerate((REPO / file).read_text().splitlines(), 1):
            if line.lstrip().startswith(('```', '~~~')):
                fenced = not fenced
            for match in NUMBER_RE.finditer(line):
                key = (file, lineno, match.start() + 1)
                inventory.append({'file': file, 'line': lineno, 'column': key[2],
                                  'claim': match.group(), 'text': line,
                                  'kind': 'code' if fenced else 'table' if line.startswith('|') else 'prose',
                                  'status': 'checked' if key in checked else 'unbound'})
    return inventory


def coverage_status(inventory, require_complete):
    return 2 if require_complete and any(c['status'] == 'unbound' for c in inventory) else 0


def check_additional_claims():
    """Extend the original table sample with the critic's independent cases."""
    baseline = 'baseline/out/metrics_summary.json'
    rank = ['mAP', 'Rank-1', 'Rank-5', 'mINP']
    doc = '04-solution/baseline/REPORT.md'
    for marker, key in [('**OSNet, market**', 'base_market'),
                        ('Случайные векторы, среднее', 'random_mean'),
                        ('OSNet, без исключения', 'base_no_camera_excl')]:
        table(doc, marker, scores(baseline, [key, 'top_k10'], rank[:3]))
    table(doc, '=OSNet, market', scores(baseline, ['base_market'],
                                      ['threshold', 'F1', 'precision', 'recall', 'TNR', 'auc_pr']))
    for marker, key in [('База (без', 'base_market'), ('Низ 30% закрыт', 'ablate_mask30'),
                        ('Обесцвечено', 'ablate_gray'), ('Низ 30% +', 'ablate_mask30gray')]:
        values = []
        for field in rank:
            value = ref(baseline, key, 'full_gallery', field)
            values.append(value)
            if field in rank[:2] and key != 'base_market':
                values.append(difference(value, ref(baseline, 'base_market', 'full_gallery', field)))
        table(doc, marker, values + [ref(baseline, key, 'top_k10', 'mAP')])
    doc = '04-solution/refusal/README.md'
    source = 'refusal/results/summary.json'
    for marker, field in [('F1', 'f1'), ('TNR', 'tnr'), ('Recall', 'recall')]:
        left = ref(source, 'market_absolute_presence', 'selected', 'best_f1', field)
        right = ref(source, 'market_absolute_presence', 'selected', 'robust_balanced', field)
        # This table is identified by its header; later historical tables may
        # repeat row labels. Bind the selected rows under this header.
        lines = (REPO / doc).read_text().splitlines()
        header = lines.index('| | Было (max F1) | Стало | Изменение |')
        lineno, line = next((i + 1, line) for i, line in enumerate(lines)
                            if i > header and line.startswith('|')
                            and line.split('|')[1].strip() == marker)
        matches = list(NUMBER_RE.finditer(line))
        check_tokens(doc, lineno, [m.group() for m in matches],
                     [left, right, difference(right, left)], [m.start() + 1 for m in matches])
    for size in (208, 256, 320, 384):
        table('04-solution/cloud/bigres/results/README.md', str(size),
              [ref('cloud/bigres/results/results.json', 'configs', f'{model}_{size}', 'metrics', 'kr', 'mAP')
               for model in ('osnet', 'combined_v1', 'd1_j48')])
    number = '(?P<claim>' + NUMBER + ')'
    prose('04-solution/training/README.md',
          [r'Формально порог превышен на\s+' + number,
           r'\*\*' + number + r'\s+меньше разброса измерения',
           r'парный бутстрэп по \d+ значениям AP: Δ =\s*' + number],
          ref('training/attempt-2/out/boot_ainv2.json', 'ainv2rr_minus_osnetrr', 'delta'))
    combined = 'training/combined/s02_metrics.json'
    prose('SOLUTION.md',
          [r'Исторические full-query KR mAP/Rank и прирост\s+' + number],
          difference(ref(combined, 'd1_j48', 'kr', 'mAP'),
                     ref(combined, 'd1_j48', 'cos', 'mAP')))
    official = 'reproduce/evidence/official-validation-20260929.json'
    comparison = 'training/owncore-distill-20260926/official-evaluation-20260929.json'
    prose('SOLUTION.md', [r'с cosine равна\s+' + number],
          difference(ref(official, 'results', 'ranking', 'mAP@10'),
                     ref(comparison, 'results', 'baseline', 'cosine', 'ranking', 'mAP@10')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true',
                        help='Exit 2 if any inventoried numeric occurrence lacks a source binding')
    args = parser.parse_args()
    CHECKS.clear()
    rank = ["mAP", "Rank-1", "Rank-5", "mINP"]
    combined = "training/combined/s02_metrics.json"
    baseline = "baseline/out/metrics_summary.json"
    final = "postproc/out/final_val.json"
    headline = "service/calib-d1_j48/headline.json"
    official = "reproduce/evidence/official-validation-20260929.json"
    comparison = "training/owncore-distill-20260926/official-evaluation-20260929.json"
    manifest = "service/artifacts-final/manifest.json"
    number = '(?P<claim>' + NUMBER + ')'
    release_ranking = scores(official, ["results", "ranking"], ["mAP@10", "Rank-1", "Rank-5"])
    cosine_ranking = scores(comparison, ["results", "baseline", "cosine", "ranking"],
                            ["mAP@10", "Rank-1", "Rank-5"])
    table("SOLUTION.md", "d1_j48, независимый top-50 KR (сдаваемый)", release_ranking)
    table("SOLUTION.md", "d1_j48, cosine (`--no-rerank`)", cosine_ranking)
    table("Readme.md", "Сдаваемое ранжирование: независимый top-50 k-reciprocal",
          release_ranking[:2])
    table("Readme.md", "Та же модель, только cosine", cosine_ranking[:2])
    table("04-solution/README.md", "Независимый top-50 KR, сдаваемый", release_ranking)
    table("04-solution/README.md", "Только cosine", cosine_ranking)
    table("04-solution/service/README.md", "Независимый top-50 KR, сдаваемый", release_ranking)
    table("04-solution/service/README.md", "Cosine (`--no-rerank`)", cosine_ranking)
    prose("SOLUTION.md", [r'full-ranking mAP \*\*' + number],
          ref(official, "results", "full_ranking", "mAP_full"))
    prose("SOLUTION.md", [r'mINP \*\*' + number],
          ref(official, "results", "full_ranking", "mINP"))
    prose("SOLUTION.md", [r'Он дал full-ranking mAP\s+' + number],
          ref(combined, "d1_j48", "kr", "mAP"))
    prose("04-solution/service/README.md", [r'^' + number + r'\. \*\*Сдаваемый порядок'],
          ref(combined, "d1_j48", "kr", "mAP"))
    prose("SOLUTION.md", [r'локальное mAP@10\s+' + number],
          ref(combined, "d1_j48", "kr", "mAP@10"))
    prose("SOLUTION.md", [r'Rank-1\s+' + number + r'\s+и прежнее'],
          ref(combined, "d1_j48", "kr", "Rank-1"))
    prose("SOLUTION.md", [r'^\*\*' + number + r'\*\*, TNR'],
          ref(official, "results", "candidates", "F1"))
    prose("Readme.md", [r'F1\s+\*\*' + number], ref(official, "results", "candidates", "F1"))
    prose("Readme.md", [r'TNR\s+\*\*' + number], ref(official, "results", "candidates", "TNR"))
    prose("04-solution/README.md", [r'^\*\*' + number + r'\*\*, TNR'],
          ref(official, "results", "candidates", "F1"))
    prose("04-solution/README.md", [r'TNR — \*\*' + number],
          ref(official, "results", "candidates", "TNR"))
    prose("04-solution/service/README.md", [r'F1 \*\*' + number],
          ref(official, "results", "candidates", "F1"))
    prose("04-solution/service/README.md", [r'^\*\*' + number + r'\*\*\. При оценке'],
          ref(official, "results", "candidates", "TNR"))
    prose("SOLUTION.md", [r'PR-AUC\s+\*\*' + number],
          ref(official, "results", "candidates", "PR-AUC"))
    prose("SOLUTION.md", [r'TNR \*\*' + number + r'\*\*, PR-AUC'],
          ref(official, "results", "candidates", "TNR"))
    prose("Readme.md", [r'mAP@10 \*\*' + number + r'\*\* против'],
          ref(comparison, "results", "core", "top50", "ranking", "mAP@10"))
    prose("Readme.md", [r'против \*\*' + number + r'\*\* у релиза'],
          ref(official, "results", "ranking", "mAP@10"))
    prose("04-solution/README.md", [r'^\*\*' + number + r'\*\* против'],
          ref(comparison, "results", "core", "top50", "ranking", "mAP@10"))
    prose("04-solution/training/owncore-distill-20260926/README.md",
          [r'запросам\]\([^)]*\):\s*' + number],
          ref(comparison, "results", "core", "top50", "ranking", "mAP@10"))
    prose("04-solution/training/owncore-distill-20260926/README.md",
          [r'CrossViewCore против\s*' + number],
          ref(official, "results", "ranking", "mAP@10"))
    prose("SOLUTION.md", [r'd1_j48 mAP@10\s+' + number],
          ref(official, "results", "ranking", "mAP@10"))
    prose("SOLUTION.md", [r'CrossViewCore после combined-продолжения\s+' + number],
          ref(comparison, "results", "core", "top50", "ranking", "mAP@10"))
    prose("SOLUTION.md", [r'CrossViewCore после combined-продолжения.*? /\s*' + number],
          ref(comparison, "results", "core", "top50", "ranking", "Rank-1"))
    prose("04-solution/service/artifacts-final/README.md",
          [r'`' + number + r'`, сравнение до округления'],
          ref(manifest, "run_info", "threshold"))
    for before, after, field in ((r'извлечение векторов заняло \*\*', '', 'embed_elapsed_s'),
                                 (r'^\*\*', r' с\*\*, весь batch', 'rank_elapsed_s'),
                                 (r'весь batch \*\*', '', 'total_elapsed_s')):
        prose("04-solution/service/artifacts-final/README.md", [before + number + after],
              ref(manifest, "run_info", field))
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
    chosen = next(i for i, p in enumerate(points) if (p["k1"], p["k2"], p["lam"]) == (6, 3, Decimal('0.3')))
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
    service_doc = "04-solution/service/README.md"
    prose(service_doc, [r'контуром тот же порог давал F1\s+' + number],
          ref(headline, "cosine_rule_reproduced", "f1"))
    prose(service_doc, [r'/ TNR\s+' + number + r': смысл'],
          ref(headline, "cosine_rule_reproduced", "tnr"))
    prose(service_doc, [r'^' + number + r' и F1'],
          ref(headline, "rerank_point", "threshold"))
    prose(service_doc, [r'и F1\s+' + number + r' / TNR'],
          ref(headline, "rerank_point", "f1"))
    prose(service_doc, [r'/ TNR\s+' + number + r' —'],
          ref(headline, "rerank_point", "tnr"))
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
    check_additional_claims()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0')
    files = [f for f in tracked if f.endswith('.md') and
             (Path(f).name.startswith('README') or f == 'Readme.md' or f in
              ('SOLUTION.md', '04-solution/baseline/REPORT.md', '04-solution/postproc/REPORT.md'))]
    inventory = numeric_inventory(files, CHECKS)
    unbound = sum(c['status'] == 'unbound' for c in inventory)
    errors = [c for c in CHECKS if not c["ok"]]
    print(json.dumps({"checks": len(CHECKS), "documents": len({c['file'] for c in CHECKS}),
                      "errors": errors, "claims": CHECKS,
                      "coverage": {"documents_scanned": len(files), "numeric_occurrences": len(inventory),
                                   "unbound_occurrences": unbound, "complete": not unbound,
                                   "note": "Unbound numbers include metrics, parameters, versions and timings; none are declared verified."},
                      "inventory": inventory}, ensure_ascii=False, indent=2))
    return 1 if errors else coverage_status(inventory, args.require_complete)


if __name__ == "__main__":
    sys.exit(main())
