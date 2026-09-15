"""Reproduce all checks offline: python -B verify.py. Writes evidence locally."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import types
import unittest

import numpy as np

import reid_metrics as metrics
import test_metrics
from compare_results import compare

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def test_run(module):
    original = test_metrics.metrics
    test_metrics.metrics = module
    stream = io.StringIO()
    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_metrics.MetricTests)
        names = [test.id().split(".")[-1] for test in suite]
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        failed = [test.id().split(".")[-1] for test, _ in result.failures]
        errors = [test.id().split(".")[-1] for test, _ in result.errors]
        return dict(tests=result.testsRun, failed=failed, errors=errors,
                    passed=[n for n in names if n not in failed + errors]), stream.getvalue()
    finally:
        test_metrics.metrics = original


def mutation_check():
    source = (ROOT / "reid_metrics.py").read_text()
    specs = [
        ("reverse_sort", "Сортировка близости по возрастанию", [
            ('np.argsort(-values[candidates], kind="stable")', 'np.argsort(values[candidates], kind="stable")', 1)]),
        ("no_camera_filter", "Удаление фильтра камеры", [
            ('return ~(same_id & same_camera)', 'return np.ones_like(same_id, dtype=bool)', 1),
            ('return ~same_camera', 'return np.ones_like(same_camera, dtype=bool)', 1)]),
        ("mean_all_queries", "mAP делится на все запросы", [
            ('_mean([row["ap"] for row in valid])', '_ratio(math.fsum(row["ap"] or 0.0 for row in rows), total_queries)', 1)]),
        ("mean_weighted_positives", "mAP взвешивается числом верных", [
            ('_mean([row["ap"] for row in valid])', '_ratio(math.fsum(row["ap"] * row["num_relevant"] for row in valid), sum(row["num_relevant"] for row in valid))', 1)]),
        ("ap_divide_gallery", "AP делится на длину списка", [
            ('ap = float(np.mean(at_hit))', 'ap = float(np.sum(at_hit) / len(relevant))', 1)]),
        ("ignore_junk", "Не исключается gallery_junk", [
            ('_camera_keep(same_id, same_camera, camera_policy) & ~junk', '_camera_keep(same_id, same_camera, camera_policy)', 1)]),
        ("cmc_is_recall", "CMC подменена recall@k", [
            ('rank1=float(ranks[0] <= 1), rank5=float(ranks[0] <= 5)',
             'rank1=float(np.count_nonzero(ranks <= 1) / count), rank5=float(np.count_nonzero(ranks <= 5) / count)', 1)]),
        ("inp_first", "INP использует первое совпадение", [
            ('inp=float(count / ranks[-1])', 'inp=float(1.0 / ranks[0])', 1)]),
        ("ties_reverse", "Обратный порядок внутри ничьих", [
            ('candidates = tie_order[eligible[tie_order]]', 'candidates = tie_order[eligible[tie_order]][::-1]', 1)]),
        ("threshold_strict", "Строгое сравнение порога", [
            ('>= cutoff', '> cutoff', 2)]),
        ("identity_as_presence", "Ошибочная идентичность считается TP", [
            ('if refusal_mode == "top1":', 'if False:', 1)]),
        ("pr_split_ties", "Ничьи PR разрываются по одному кандидату", [
            ('ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])', 'ends = np.arange(scores.size)', 1)]),
        ("pr_area_alias", "Трапеции PR подменяются AP step", [
            ('auc_pr_trapezoid=float(trap_area)', 'auc_pr_trapezoid=float(step_area)', 1)]),
        ("rank5_off_by_one", "Rank-5 использует ранг <5", [
            ('rank5=float(ranks[0] <= 5)', 'rank5=float(ranks[0] < 5)', 1)]),
        ("filtered_become_fn", "Исключённые известные запросы добавляются в FN", [
            ('else known_count\n    escores', 'else known_count + filtered_count\n    escores', 1)]),
    ]
    directory = ROOT / "mutants"
    directory.mkdir(exist_ok=True)
    records = []
    for name, description, changes in specs:
        mutated = source
        for old, new, count in changes:
            if mutated.count(old) != count:
                raise AssertionError(f"mutation {name}: source pattern changed")
            mutated = mutated.replace(old, new)
        path = directory / f"{name}.py"
        path.write_text("# INTENTIONALLY BROKEN: " + description + "\n" + mutated)
        module = types.ModuleType(f"mutant_{name}")
        exec(compile(mutated, str(path), "exec"), module.__dict__)
        outcome, log = test_run(module)
        (directory / f"{name}.log").write_text(log)
        # A runtime crash alone is insufficient evidence of numerical detection.
        if not outcome["failed"]:
            raise AssertionError(f"mutant {name} survived all assertion tests")
        records.append(dict(name=name, description=description, **outcome))
    write_json(ROOT / "mutation_results.json", records)
    lines = ["# Фактическая матрица мутационных проверок", "",
             "Числа и имена получены запуском неизменённых аналитических тестов против каждого сломанного модуля. Ошибки исполнения учитываются отдельно от assertion failures.", "",
             "| Мутация | Нарушено проверок | Ошибки исполнения | Поймали | Не поймали |", "|---|---:|---:|---|---|"]
    for row in records:
        lines.append(f'| `{row["name"]}` — {row["description"]} | {len(row["failed"])} | {len(row["errors"])} | ' +
                     ", ".join(f"`{n}`" for n in row["failed"]) + " | " +
                     ", ".join(f"`{n}`" for n in row["passed"]) + " |")
    (ROOT / "mutation_matrix.md").write_text("\n".join(lines) + "\n")
    return dict(total=len(records), killed=sum(bool(r["failed"]) for r in records),
                runtime_errors=sum(len(r["errors"]) for r in records))


def external_rank_function(filename):
    # Only the real eval_market1501 function is compiled; no package imports,
    # dependency installation, Cython compilation or replacement of its body.
    source = (ROOT / "sources" / filename).read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "eval_market1501")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace["eval_market1501"]


def differential_ranking():
    manifest = json.loads((ROOT / "sources" / "manifest.json").read_text())
    for record in manifest["sources"]:
        actual = hashlib.sha256((ROOT / "sources" / record["file"]).read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise AssertionError("source checksum mismatch: " + record["file"])
    fast = external_rank_function("fastreid_rank.py")
    torch = external_rank_function("torchreid_rank.py")
    rng = np.random.default_rng(61704)
    maxima = dict(fastreid_map=0.0, torchreid_map=0.0, fastreid_minp=0.0, cmc_float32=0.0)
    for trial in range(200):
        ng, nq = 96, int(rng.integers(2, 15))
        gids = np.arange(ng) % 8
        gcams = np.arange(ng) // 8 % 3
        qids = rng.integers(0, 8, nq)
        qids[-1] = 999  # true unknown must be skipped by the ranking evaluators
        qcams = rng.integers(0, 3, nq)
        scores = rng.random((nq, ng))
        ours = metrics.evaluate(scores, qids, gids, qcams, gcams, qids == 999,
                                threshold=.5, camera_policy="market", refusal_mode="top1")["ranking"]
        fcmc, fap, finp = fast(-scores, qids, gids, qcams, gcams, 5)
        tcmc, tmap = torch(-scores, qids, gids, qcams, gcams, 5)
        differences = dict(fastreid_map=abs(ours["mAP"] - float(np.mean(fap))),
                           torchreid_map=abs(ours["mAP"] - float(tmap)),
                           fastreid_minp=abs(ours["mINP"] - float(np.mean(finp))),
                           cmc_float32=max(abs(ours[k] - float(c[r]))
                                           for c in [fcmc, tcmc] for k, r in [("Rank-1", 0), ("Rank-5", 4)]))
        for name, diff in differences.items():
            maxima[name] = max(maxima[name], diff)
            if diff > (1e-7 if name == "cmc_float32" else 1e-12):
                raise AssertionError((trial, name, diff))
    return dict(trials=200, seed=61704, max_absolute_error=maxima,
                scope="Python eval_market1501, 96 gallery, unique continuous scores; Cython/MATLAB not executed")


def slow_refusal(scores, qids, gids, qcams, gcams, absent, junk, excluded, mode, camera, threshold):
    """Independent direct counting with Python loops and exact Fraction areas."""
    query_data, events = [], []
    known = 0
    total_pairs = 0
    for i, qid in enumerate(qids):
        candidates = []
        for j, gid in enumerate(gids):
            same_cam = qcams[i] == gcams[j]
            if junk[j] or excluded[i][j]:
                continue
            if same_cam and (camera == "all_same_camera" or gid == qid):
                continue
            candidates.append(j)
        if not absent[i] and not any(gids[j] == qid for j in candidates):
            continue
        candidates = sorted(candidates, key=lambda j: (-scores[i][j], j))
        query_data.append((i, candidates))
        known += int(not absent[i])
        total_pairs += sum(gids[j] == qid for j in candidates)
        if mode == "pairwise":
            events += [(scores[i][j], gids[j] == qid) for j in candidates]
        elif candidates:
            j = candidates[0]
            label = not absent[i] if mode == "presence" else gids[j] == qid
            events.append((scores[i][j], label))
    positives = total_pairs if mode == "pairwise" else known
    def counts(t):
        tp = sum(label and value >= t for value, label in events)
        fp = sum(not label and value >= t for value, label in events)
        return int(tp), int(fp), int(positives - tp)
    tp, fp, fn = counts(threshold)
    unknown = sum(bool(absent[i]) for i, _ in query_data)
    tn = sum(bool(absent[i]) and not any(scores[i][j] >= threshold for j in candidates)
             for i, candidates in query_data)
    thresholds = sorted({value for value, _ in events}, reverse=True)
    prev_p, prev_r = Fraction(1), Fraction(0)
    step = trapezoid = Fraction(0)
    for t in thresholds:
        hit, false, _ = counts(t)
        p = Fraction(hit, hit + false)
        if positives:
            r = Fraction(hit, positives)
            step += (r - prev_r) * p
            trapezoid += (r - prev_r) * (p + prev_p) / 2
            prev_p, prev_r = p, r
    ratio = lambda a, b: a / b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn_unknown=int(tn), fp_unknown=int(unknown - tn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, positives),
                f1=ratio(2 * tp, 2 * tp + fp + fn), tnr=ratio(tn, unknown),
                auc_pr=float(trapezoid) if positives else None,
                ap_pr_step=float(step) if positives else None)


def differential_refusal():
    rng = np.random.default_rng(78401)
    maximum = 0.0
    comparisons = 0
    for _ in range(100):
        nq, ng = 10, 12
        scores = rng.integers(-2, 4, (nq, ng)).astype(float)
        qids, gids = rng.integers(0, 5, nq), np.arange(ng) % 4
        qcams, gcams = rng.integers(0, 3, nq), rng.integers(0, 3, ng)
        absent = qids == 4
        junk, excluded = rng.random(ng) < .08, rng.random((nq, ng)) < .1
        for camera in ["market", "all_same_camera"]:
            for mode in ["presence", "top1", "pairwise"]:
                ours = metrics.evaluate(scores, qids, gids, qcams, gcams, absent,
                                        gallery_junk=junk, exclude_mask=excluded,
                                        threshold=1.0, camera_policy=camera, refusal_mode=mode)["refusal"]
                reference = slow_refusal(scores.tolist(), qids.tolist(), gids.tolist(),
                                         qcams.tolist(), gcams.tolist(), absent.tolist(), junk.tolist(),
                                         excluded.tolist(), mode, camera, 1.0)
                for name, expected in reference.items():
                    if expected is None:
                        if ours[name] is not None:
                            raise AssertionError((name, ours[name], expected))
                    else:
                        diff = abs(ours[name] - expected)
                        maximum = max(maximum, diff)
                        if diff > 1e-12:
                            raise AssertionError((mode, camera, name, ours[name], expected))
                comparisons += 1
    return dict(comparisons=comparisons, seed=78401, max_absolute_error=maximum,
                scope="All 3 refusal units x 2 camera masks, discrete tied scores, junk, excluded queries, exact rational PR oracle")


def random_level():
    seed, nq, n, m = 20260915, 5000, 100, 4
    rng = np.random.default_rng(seed)
    scores = rng.random((nq, n))
    gids = np.arange(n) + 100
    gids[:m] = 1
    result = metrics.evaluate(scores, [1] * nq, gids, [0] * nq, [1] * n,
                              [False] * nq, threshold=.5,
                              camera_policy="market", refusal_mode="top1")
    harmonic = math.fsum(1 / r for r in range(1, n + 1))
    expected = {
        "mAP": harmonic / n + (m - 1) * (n - harmonic) / (n * (n - 1)),
        "Rank-1": m / n,
        "Rank-5": 1 - math.comb(n - m, 5) / math.comb(n, 5),
        "mINP": math.fsum(m / r * math.comb(r - 1, m - 1) / math.comb(n, m) for r in range(m, n + 1)),
    }
    mapping = {"mAP": "ap", "Rank-1": "rank1", "Rank-5": "rank5", "mINP": "inp"}
    measurements = {}
    for key, query_key in mapping.items():
        values = np.array([row[query_key] for row in result["per_query"]])
        se = float(values.std(ddof=1) / np.sqrt(nq))
        observed = result["ranking"][key]
        if abs(observed - expected[key]) > 5 * se:
            raise AssertionError((key, observed, expected[key], se))
        measurements[key] = dict(observed=observed, expected=expected[key], standard_error=se,
                                 z=(observed - expected[key]) / se)
    if result["ranking"]["mAP"] >= .15:
        raise AssertionError("random mAP is implausibly flattering")
    return dict(seed=seed, queries=nq, gallery=n, positives_per_query=m,
                acceptance="absolute difference <= 5 empirical standard errors; mAP < .15",
                metrics=measurements)


def compatibility_edges():
    records = []
    for filename in ["fastreid_rank.py", "torchreid_rank.py"]:
        function = external_rank_function(filename)
        cases = [
            ("all_queries_unknown", AssertionError,
             (np.array([[0., 1.]]), np.array([99]), np.array([1, 2]), np.array([0]), np.array([1, 1]), 5)),
            ("short_gallery_ragged_after_filter", ValueError,
             (np.array([[0., 1., 2.], [2., 1., 0.]]), np.array([1, 2]), np.array([1, 1, 2]),
              np.array([0, 0]), np.array([0, 1, 1]), 5)),
        ]
        for name, expected_error, args in cases:
            try:
                with redirect_stdout(io.StringIO()):
                    function(*args)
            except expected_error as error:
                records.append(dict(source=filename, case=name,
                                    exception=type(error).__name__, message=str(error)))
            else:
                raise AssertionError(f"unexpected external edge behavior: {filename}, {name}")
    write_json(ROOT / "compatibility_edge_results.json", records)
    return dict(reproduced=len(records))


def cli_and_comparator():
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "reid_metrics.py"), str(ROOT / "example.npz"),
         "--threshold", "0.5", "--camera-policy", "market", "--refusal-mode", "top1",
         "--output", str(ROOT / "example_result.json")],
        check=True, capture_output=True, text=True,
    )
    original = json.loads((ROOT / "example_result.json").read_text())
    for key, expected in [("mAP", 2 / 3), ("mINP", 2 / 3), ("Rank-1", 1 / 3), ("Rank-5", 1)]:
        if abs(original["ranking"][key] - expected) > 1e-12:
            raise AssertionError(("CLI ranking", key))
    for key, expected in [("precision", 1 / 3), ("recall", 1 / 3), ("f1", 1 / 3),
                          ("tnr", .5), ("auc_pr", 1 / 3), ("ap_pr_step", 1 / 3)]:
        if abs(original["refusal"][key] - expected) > 1e-12:
            raise AssertionError(("CLI refusal", key))
    identity = compare(original, deepcopy(original))
    changed = deepcopy(original)
    changed["per_query"][0]["positive_ranks"] = [2]
    per_query = compare(original, changed)
    changed = deepcopy(original)
    changed["ranking"]["mAP"] += 1e-9
    tolerance = compare(original, changed)
    changed["protocol"]["refusal_mode"] = "presence"
    protocol = compare(original, changed)
    if not (identity["equal"] and tolerance["equal"] and not per_query["equal"] and not protocol["equal"]):
        raise AssertionError("comparison smoke checks failed")
    if per_query["differences"][0]["path"] != "$.per_query[0].positive_ranks[0]":
        raise AssertionError("wrong per-query difference path")
    write_json(ROOT / "comparison_check.json", dict(identity=identity, per_query_changed=per_query,
                                                    float_tolerance=tolerance, protocol_changed=protocol))
    return dict(cli="passed", comparator_checks=4)


def main():
    baseline, log = test_run(metrics)
    (ROOT / "test_results.log").write_text(log)
    if baseline["failed"] or baseline["errors"]:
        raise AssertionError("baseline tests failed; see test_results.log")
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  baseline=baseline, exhaustive_configurations=247)
    report["differential_ranking"] = differential_ranking()
    report["differential_refusal"] = differential_refusal()
    report["random_level"] = random_level()
    report["mutations"] = mutation_check()
    report["compatibility_edges"] = compatibility_edges()
    report["cli_and_comparator"] = cli_and_comparator()
    write_json(ROOT / "verification_results.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "baseline"}, ensure_ascii=False, indent=2))
    print(f"PASS: {baseline['tests']} tests; {report['mutations']['killed']} / {report['mutations']['total']} mutants detected")


if __name__ == "__main__":
    main()
