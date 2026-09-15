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


def test_run(module, include_protocols=True):
    modules = [test_metrics]
    if include_protocols and (ROOT / "test_protocols.py").exists():
        import test_protocols
        modules.append(test_protocols)
    originals = [m.metrics for m in modules]
    stream = io.StringIO()
    try:
        suite = unittest.TestSuite()
        names = []
        for m in modules:
            m.metrics = module
            loaded = unittest.defaultTestLoader.loadTestsFromModule(m)
            def collect(tests):
                for test in tests:
                    if isinstance(test, unittest.TestSuite):
                        collect(test)
                    else:
                        names.append(test.id())
            collect(loaded)
            suite.addTests(loaded)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        failed = sorted({test.id() for test, _ in result.failures})
        errors = sorted({test.id() for test, _ in result.errors})
        return dict(tests=result.testsRun, failed=failed, errors=errors,
                    passed=[n for n in names if n not in failed + errors]), stream.getvalue()
    finally:
        for m, original in zip(modules, originals):
            m.metrics = original


def mutation_check():
    source = (ROOT / "reid_metrics.py").read_text()
    specs = json.loads((ROOT / "mutation_specs.json").read_text())
    directory = ROOT / "evidence" / "mutants_current"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in specs:
        name = spec["name"]
        mutated = source
        for old, new, count in spec["changes"]:
            if mutated.count(old) != count:
                # Failure to inject is a harness error, NEVER a killed mutant.
                raise AssertionError(f"mutation {name}: cannot inject ({old!r})")
            mutated = mutated.replace(old, new)
        path = directory / f"{name}.py"
        path.write_text("# INTENTIONALLY BROKEN: " + spec["description"] + "\n" + mutated)
        record = {k: spec[k] for k in ["name", "description", "origin", "artifact"]}
        for campaign, code, filename in [
            ("current", mutated, path),
            ("original", (ROOT / spec["artifact"]).read_text(), ROOT / spec["artifact"]),
        ]:
            module = types.ModuleType(f"mutant_{campaign}_{name}")
            exec(compile(code, str(filename), "exec"), module.__dict__)
            outcome, log = test_run(module, include_protocols=campaign == "current")
            (directory / f"{name}.{campaign}.log").write_text(log)
            record[campaign] = dict(**outcome, sha256=hashlib.sha256(code.encode()).hexdigest(),
                                    detected=bool(outcome["failed"]))
        records.append(record)
    write_json(ROOT / "mutation_results.json", records)
    lines = ["# Фактическая матрица мутационных проверок", "",
             "Команда: `python -B verify.py`. Исходные мутанты не изменяются. "
             "Каждый запускается как приложенный файл и как такая же поломка новой реализации. "
             "Засчитываются только assertion failures, не ошибки API/исполнения/внедрения.", "",
             "| Поломка | Проверка исходного файла | Проверка новой реализации |", "|---|---|---|"]
    for row in records:
        cells = []
        for campaign in ["original", "current"]:
            failures = row[campaign]["failed"]
            cells.append(", ".join(f"`{n}`" for n in failures) or "**НЕ ПОЙМАНА**")
        lines.append(f'| `{row["name"]}` | {cells[0]} | {cells[1]} |')
    (ROOT / "mutation_matrix.md").write_text("\n".join(lines) + "\n")
    summary = dict(total=len(records),
                   killed=sum(r["current"]["detected"] for r in records),
                   original_killed=sum(r["original"]["detected"] for r in records),
                   runtime_errors=sum(len(r[c]["errors"]) for r in records for c in ["current", "original"]),
                   survivors=[r["name"] for r in records
                              if not r["current"]["detected"] or not r["original"]["detected"]])
    if summary["survivors"]:
        raise AssertionError(f"Uncaught mutants: {summary['survivors']}; see mutation_results.json")
    return summary


def external_rank_function(filename, stable_ties=False):
    # Only the real eval_market1501 function is compiled; no package imports,
    # dependency installation, Cython compilation or replacement of its body.
    source = (ROOT / "sources" / filename).read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "eval_market1501")
    class StableSortNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        def argsort(self, values, axis=-1):
            return np.argsort(values, axis=axis, kind="stable")

    # The body stays byte-for-byte as vendored. A separate, explicitly labelled
    # adapter changes ONLY its unspecified tie order, never metric arithmetic.
    namespace = {"np": StableSortNumpy() if stable_ties else np}
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


def differential_tied_external():
    from review.oracle import oracle_evaluate
    rng = np.random.default_rng(61705)
    native = {name: external_rank_function(name) for name in ["fastreid_rank.py", "torchreid_rank.py"]}
    stable = {name: external_rank_function(name, stable_ties=True) for name in native}
    maximum = 0.
    disagreements = 0
    first_difference = None
    trials = 160

    def check(output, reference, source, max_rank):
        nonlocal maximum
        cmc, ap, *inp = output
        expected = reference["ranking"]
        values = [(float(np.mean(ap)), expected["mAP"])]
        if inp:
            values.append((float(np.mean(inp[0])), expected["mINP"]))
            for observed, row in zip(ap, reference["per_query"]):
                values.append((float(observed), row["ap"]))
        for r in range(1, max_rank + 1):
            expected_cmc = np.mean([row["positive_ranks"][0] <= r for row in reference["per_query"]])
            if abs(float(cmc[r - 1]) - expected_cmc) > 1e-7:
                raise AssertionError((source, "CMC", r, cmc[r - 1], expected_cmc))
        for observed, wanted in values:
            delta = abs(observed - wanted)
            maximum = max(maximum, delta)
            if delta > 1e-12:
                raise AssertionError((source, observed, wanted))

    for trial in range(trials):
        nq, ng, max_rank = 4, 48, 7
        values = rng.integers(0, 4, (nq, ng)).astype(float)
        if trial % 8 == 0:
            values.fill(1.)
        gids, qids = np.arange(ng) % 4, np.arange(nq)
        gcams, qcams = rng.integers(0, 3, ng), rng.integers(0, 2, nq)
        gcams[:8] = 2
        junk = rng.random(ng) < .1
        junk[:8] = False
        keys = rng.permutation(ng)
        kind = "similarity" if trial % 2 == 0 else "distance"
        raw = values if kind == "similarity" else -values
        ours = metrics.evaluate(raw, qids, gids, qcams, gcams, [False] * nq,
                                threshold=0., score_kind=kind, camera_policy="market",
                                refusal_mode="top1", gallery_keys=keys, gallery_junk=junk,
                                rank_ks=range(1, max_rank + 1), include_rankings=True)
        # Upstream has no keys/junk API: remove global junk and put columns in
        # canonical key order before calling it. Its own camera mask is executed.
        columns = np.asarray([j for j in np.argsort(keys) if not junk[j]])
        utility, ids, cams = values[:, columns], gids[columns], gcams[columns]
        args = (qids, ids, qcams, cams, max_rank)
        for name in native:
            direct = native[name](-utility, *args)
            adapted = stable[name](-utility, *args)
            # Verify the unmodified upstream's actual tie order against the
            # Fraction oracle as well; do not falsely assume quicksort is stable.
            order = np.argsort(-utility, axis=1)
            ordinal = np.empty_like(utility)
            for i in range(nq):
                ordinal[i, order[i]] = np.arange(len(columns), 0, -1)
            reference = oracle_evaluate(ordinal.tolist(), qids.tolist(), ids.tolist(),
                                        qcams.tolist(), cams.tolist(), [False] * nq,
                                        threshold=0., camera_policy="market", refusal_mode="top1")
            check(direct, reference, name + "/native", max_rank)
            check(adapted, ours, name + "/stable-tie adapter", max_rank)
            for rank in range(1, max_rank + 1):
                if abs(float(adapted[0][rank - 1]) - ours["ranking"][f"Rank-{rank}"]) > 1e-7:
                    raise AssertionError(("all requested CMC ranks", rank))
            native_map, stable_map = float(np.mean(direct[1])), float(np.mean(adapted[1]))
            if abs(native_map - stable_map) > 1e-12:
                disagreements += 1
                if first_difference is None:
                    first_difference = dict(trial=trial, source=name,
                                            native_mAP=native_map, stable_mAP=stable_map)
    return dict(trials=trials, source_calls=trials * 4, seed=61705,
                max_absolute_metric_error=maximum,
                native_vs_stable_mAP_disagreements=disagreements,
                first_tie_policy_difference=first_difference,
                scope="Actual tied inputs; unmodified upstream vs Fraction in its native order; "
                      "stable-argsort adapter vs evaluator; 2 measures; keys+junk+camera; CMC ranks 1..7")


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
    alternate_path = ROOT / "evidence" / "cli_alternative_result.json"
    alternate_path.parent.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-B", str(ROOT / "reid_metrics.py"), str(ROOT / "example.npz"),
                    "--threshold", "0.5", "--camera-policy", "all_same_camera",
                    "--refusal-mode", "pairwise", "--score-kind", "distance",
                    "--ap-method", "market_matlab", "--top-k", "2",
                    "--query-average", "all_queries_zero", "--ap-denominator", "retrieved_positives",
                    "--filtered-positive-policy", "as_unknown", "--incomplete-inp", "zero_if_incomplete",
                    "--rank-beyond-k", "clamp_to_k", "--rank-ks", "1", "3", "5", "12",
                    "--truncation-order", "filter_then_top_k", "--output", str(alternate_path)],
                   check=True, capture_output=True, text=True)
    with np.load(ROOT / "example.npz", allow_pickle=False) as data:
        direct = metrics.evaluate(**{k: data[k] for k in data.files}, threshold=.5,
                                  camera_policy="all_same_camera", refusal_mode="pairwise",
                                  score_kind="distance", ap_method="market_matlab", top_k=2,
                                  query_average="all_queries_zero", ap_denominator="retrieved_positives",
                                  filtered_positive_policy="as_unknown", incomplete_inp="zero_if_incomplete",
                                  rank_beyond_k="clamp_to_k", rank_ks=[1, 3, 5, 12],
                                  truncation_order="filter_then_top_k", include_rankings=True)
    if json.loads(alternate_path.read_text()) != metrics._json_safe(direct):
        raise AssertionError("CLI switches differ from Python API")
    return dict(cli="passed (default and every nondefault convention switch)", comparator_checks=4)


def main():
    baseline, log = test_run(metrics)
    (ROOT / "test_results.log").write_text(log)
    if baseline["failed"] or baseline["errors"]:
        raise AssertionError("baseline tests failed; see test_results.log")
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  baseline=baseline, exhaustive_configurations=247)
    report["differential_ranking"] = differential_ranking()
    report["differential_tied_external"] = differential_tied_external()
    report["differential_refusal"] = differential_refusal()
    from oracle_checks import differential_all_fields
    import test_protocols
    report["differential_all_fields"] = differential_all_fields(metrics)
    report["differential_new_fields"] = test_protocols.differential_extensions()
    report["random_level"] = random_level()
    report["mutations"] = mutation_check()
    report["compatibility_edges"] = compatibility_edges()
    report["cli_and_comparator"] = cli_and_comparator()
    write_json(ROOT / "verification_results.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "baseline"}, ensure_ascii=False, indent=2))
    print(f"PASS: {baseline['tests']} tests; {report['mutations']['killed']} / {report['mutations']['total']} mutants detected")


if __name__ == "__main__":
    main()
