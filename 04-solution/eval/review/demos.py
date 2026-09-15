"""Reproducible demos: for each surviving mutant, a tiny input where the
mutant (which passes ALL author checks) returns a different number than the
correct implementation. Run: python3 -B demos.py"""
import sys
from pathlib import Path

sys.path.insert(0, "run")
import numpy as np
import reid_metrics as correct
from mutants import MUTANTS, build

MOD = {name: build(name, changes)[0] for name, _, changes in MUTANTS}


def show(title, keys, good, bad):
    print(f"\n=== {title} ===")
    for k in keys:
        g, b = good, bad
        for part in k.split("."):
            part = int(part) if part.isdigit() else part
            g, b = g[part], b[part]
        mark = "  <-- РАСХОЖДЕНИЕ" if g != b else ""
        print(f"  {k}: верно={g!r}  мутант={b!r}{mark}")


# --- M1: gallery_keys + junk: ранжируется не то множество объектов -------
kw = dict(threshold=0.5, camera_policy="market", refusal_mode="top1",
          gallery_keys=["c", "a", "b"], gallery_junk=[False, True, False],
          include_rankings=True)
args = ([[5.0, 9.0, 1.0]], [7], [8, 7, 7], [0], [1, 1, 1], [False])
show("M1: keys+junk — мутант ранжирует junk-объект и теряет допустимый",
     ["ranking.mAP", "ranking.Rank-1", "per_query.0.ranking",
      "per_query.0.top_gallery_index"],
     correct.evaluate(*args, **kw), MOD["M1_keys_filter_mask"].evaluate(*args, **kw))

# --- M1b: обратная перестановка ключей (>=3 ключей) ----------------------
kw = dict(threshold=0.5, camera_policy="market", refusal_mode="top1",
          gallery_keys=["b", "c", "a"], include_rankings=True)
args = ([[1.0, 1.0, 1.0]], [7], [8, 8, 7], [0], [1, 1, 1], [False])
show("M1b: 3 ключа, все оценки равны — обратная перестановка меняет ранжирование",
     ["ranking.mAP", "ranking.Rank-1", "per_query.0.ranking"],
     correct.evaluate(*args, **kw), MOD["M1b_keys_inverse_perm"].evaluate(*args, **kw))
kw2 = dict(kw, gallery_keys=["b", "a"])
args2 = ([[1.0, 1.0]], [7], [8, 7], [0], [1, 1], [False])
a = correct.evaluate(*args2, **kw2); b = MOD["M1b_keys_inverse_perm"].evaluate(*args2, **kw2)
print(f"  (при 2 ключах, как в тестах автора, разницы нет: mAP {a['ranking']['mAP']} == {b['ranking']['mAP']})")

# --- M2: знак thresholds в PR-кривой при score_kind=distance --------------
kw = dict(threshold=2.0, camera_policy="market", refusal_mode="presence",
          score_kind="distance")
args = ([[1.0, 3.0]], [1], [1, 2], [0], [1, 1], [False])
show("M2: distance — thresholds PR-кривой в чужой шкале (отрицательные расстояния)",
     ["refusal.pr_curve.thresholds"],
     correct.evaluate(*args, **kw), MOD["M2_pr_thresholds_sign"].evaluate(*args, **kw))

# --- M3: top_score при distance -------------------------------------------
show("M3: distance — top_score с перевёрнутым знаком",
     ["per_query.0.top_score", "per_query.0.accepted"],
     correct.evaluate(*args, **kw), MOD["M3_top_score_sign"].evaluate(*args, **kw))

# --- M4: eligible_count ----------------------------------------------------
kw = dict(threshold=0.5, camera_policy="market", refusal_mode="top1")
args = ([[0.9, 0.8]], [1], [1, 1], [0], [0, 1], [False])
show("M4: eligible_count игнорирует фильтрацию той же камеры",
     ["per_query.0.eligible_count"],
     correct.evaluate(*args, **kw), MOD["M4_eligible_count"].evaluate(*args, **kw))

# --- M5: counts.relevant_pairs --------------------------------------------
args = ([[0.9, 0.8, 0.1]], [1], [1, 1, 2], [0], [1, 1, 1], [False])
show("M5: counts.relevant_pairs = число запросов вместо числа пар",
     ["counts.relevant_pairs"],
     correct.evaluate(*args, **kw), MOD["M5_relevant_pairs"].evaluate(*args, **kw))

# --- M6/M7: per_query.positive_ranks / num_relevant -----------------------
args = ([[0.9, 0.8, 0.7, 0.6]], [1], [2, 1, 3, 1], [0], [1, 1, 1, 1], [False])
show("M6: positive_ranks с нуля (ранг '0' невозможен по определению)",
     ["per_query.0.positive_ranks"],
     correct.evaluate(*args, **kw), MOD["M6_positive_ranks_0based"].evaluate(*args, **kw))
show("M7: num_relevant = ранг последнего совпадения вместо их числа",
     ["per_query.0.num_relevant"],
     correct.evaluate(*args, **kw), MOD["M7_num_relevant"].evaluate(*args, **kw))

# --- M8: float32-квантование ----------------------------------------------
args = ([[0.6, 0.6 + 2e-8]], [1], [1, 2], [0], [1, 1], [False])
show("M8: float32 — истинный порядок (дистрактор выше) схлопывается в ничью",
     ["ranking.mAP", "ranking.Rank-1", "per_query.0.ranking"],
     correct.evaluate(*args, include_rankings=True, **kw),
     MOD["M8_float32"].evaluate(*args, include_rankings=True, **kw))
kw8 = dict(threshold=0.5 + 6e-9, camera_policy="market", refusal_mode="top1")
args8 = ([[0.5]], [9], [1], [0], [1], [True])
show("M8: float32 порога — отказ (TNR=1) превращается в ложное принятие (TNR=0)",
     ["refusal.tnr", "refusal.fp_unknown", "per_query.0.accepted"],
     correct.evaluate(*args8, **kw8), MOD["M8_float32"].evaluate(*args8, **kw8))

# --- M9: строгий порог только в accepted/TNR (ловится лишь verify.py) -----
kw9 = dict(threshold=0.5, camera_policy="market", refusal_mode="top1")
args9 = ([[0.5]], [9], [1], [0], [1], [True])
show("M9: score == threshold у unknown-запроса — accepted/TNR расходятся с tp/fp",
     ["refusal.tnr", "refusal.fp_unknown", "per_query.0.accepted"],
     correct.evaluate(*args9, **kw9), MOD["M9_accepted_strict"].evaluate(*args9, **kw9))
