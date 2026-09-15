"""Try to break reid_metrics.py so that the author's test_metrics.py passes.

For each mutant: apply textual patch, run the author's full unittest suite
in-process, report which tests fail. Survivors are the coverage holes."""
import io
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "run"))
import test_metrics

SOURCE = (Path(__file__).parent / "run" / "reid_metrics.py").read_text()

MUTANTS = [
    ("M1_keys_filter_mask", "gallery_keys: маска eligible применяется к tie_order позиционно (ранжируется не то множество)", [
        ("candidates = tie_order[eligible[tie_order]]",
         "candidates = tie_order[eligible]", 1)]),
    ("M1b_keys_inverse_perm", "gallery_keys: используется обратная перестановка ключей (незаметно при 2 элементах)", [
        ("tie_order = np.asarray(sorted(range(ng), key=lambda j: keys[j]), dtype=int)",
         "tie_order = np.argsort(np.asarray(sorted(range(ng), key=lambda j: keys[j]), dtype=int))", 1)]),
    ("M2_pr_thresholds_sign", "pr_curve.thresholds не переводятся обратно в шкалу расстояний", [
        ('curve["thresholds"] = [None if t is None else t * sign for t in curve["thresholds"]]',
         'curve["thresholds"] = list(curve["thresholds"])', 1)]),
    ("M3_top_score_sign", "top_score отдаёт внутреннюю полезность (знак) вместо исходной оценки", [
        ("top_score=float(raw[i, best]) if best is not None else None",
         "top_score=float(utility[i, best]) if best is not None else None", 1)]),
    ("M4_eligible_count", "eligible_count всегда равен размеру галереи", [
        ('row = dict(query_index=i, status=status, eligible_count=int(order.size),',
         'row = dict(query_index=i, status=status, eligible_count=int(ng),', 1)]),
    ("M5_relevant_pairs", "counts.relevant_pairs подменён числом известных запросов", [
        ("relevant_pairs=pair_positive_count),",
         "relevant_pairs=known_count),", 1)]),
    ("M6_positive_ranks_0based", "positive_ranks отдаются с нуля, а не с единицы", [
        ("positive_ranks=ranks.tolist(), num_relevant=count)",
         "positive_ranks=(ranks - 1).tolist(), num_relevant=count)", 1)]),
    ("M7_num_relevant", "num_relevant отдаёт ранг последнего совпадения вместо числа совпадений", [
        ("positive_ranks=ranks.tolist(), num_relevant=count)",
         "positive_ranks=ranks.tolist(), num_relevant=int(ranks[-1]))", 1)]),
    ("M8_float32", "оценки и порог квантуются во float32 перед сравнением и сортировкой", [
        ("utility, cutoff = raw * sign, threshold * sign",
         "utility, cutoff = (raw * sign).astype(np.float32).astype(np.float64),"
         " float(np.float64(np.float32(threshold * sign)))", 1)]),
    ("M9_accepted_strict", "порог принятия строгий только в per_query/TNR (tp/fp остаются нестрогими)", [
        ("accepted = best is not None and bool(utility[i, best] >= cutoff)",
         "accepted = best is not None and bool(utility[i, best] > cutoff)", 1)]),
    ("M10_max_recall", "pr_curve.max_recall всегда достижимая единица при наличии позитивов", [
        ("max_recall=recall[-1], positive_count=int(positive_count),",
         "max_recall=(1.0 if positive_count else None), positive_count=int(positive_count),", 1)]),
]


def run_suite(module):
    original = test_metrics.metrics
    test_metrics.metrics = module
    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_metrics.MetricTests)
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        return ([t.id().split(".")[-1] for t, _ in result.failures],
                [t.id().split(".")[-1] for t, _ in result.errors], result.testsRun)
    finally:
        test_metrics.metrics = original


def build(name, changes):
    mutated = SOURCE
    for old, new, count in changes:
        assert mutated.count(old) == count, (name, old)
        mutated = mutated.replace(old, new)
    module = types.ModuleType(f"mutant_{name}")
    exec(compile(mutated, name, "exec"), module.__dict__)
    return module, mutated


if __name__ == "__main__":
    survivors = []
    for name, desc, changes in MUTANTS:
        module, mutated = build(name, changes)
        failed, errors, total = run_suite(module)
        verdict = "SURVIVED" if not failed and not errors else "killed"
        if verdict == "SURVIVED":
            survivors.append(name)
            Path(f"mutant_{name}.py").write_text(
                f"# СЛОМАНО НАМЕРЕННО ({desc}); все {total} авторских тестов проходят\n" + mutated)
        print(f"{name}: {verdict}  failed={failed} errors={errors}")
    print("\nsurvivors:", survivors)
