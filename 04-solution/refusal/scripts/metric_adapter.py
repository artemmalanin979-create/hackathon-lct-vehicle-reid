"""Reuse the existing evaluator's metric expressions, without reimplementing them.

The public evaluator reranks Q x G for every threshold. For a sweep, its own
_pr_events supplies counts and its unchanged AST assignments supply metrics.
There is intentionally no F1, precision, recall, TNR or PR-area formula here.
Integer repetition implements exact class-prior changes, not new observations.
"""
from __future__ import annotations

import ast
import copy
import inspect
import math
from fractions import Fraction

import numpy as np


def metric_kernel(module):
    original = ast.parse(inspect.getsource(module._evaluate_full)).body[0]
    wanted = ("positives", "fn", "refusal")
    assignments = {}
    for node in original.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                assignments[target.id] = copy.deepcopy(node)
    if set(assignments) != set(wanted):
        raise RuntimeError("Evaluator aggregation changed; review the adapter")
    tree = ast.parse(
        "def reuse_metrics(refusal_mode, threshold, known_count, pair_positive_count, "
        "tp, fp, unknown_tn, unknown_count, curve):\n    pass\n"
    )
    fn = tree.body[0]
    fn.body = [assignments[name] for name in wanted]
    fn.body.append(ast.Return(value=ast.Name(id="refusal", ctx=ast.Load())))
    ast.fix_missing_locations(tree)
    namespace = dict(module.__dict__)
    exec(compile(tree, module.__file__ + ":unchanged_metric_assignments", "exec"), namespace)
    return namespace["reuse_metrics"], ast.unparse(tree) + "\n"


class Sweep:
    def __init__(self, module, kernel, scores, known, correct_top1, mode="presence",
                 prior=None, multiplicity=None):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.known = np.asarray(known, dtype=bool)
        self.correct_top1 = np.asarray(correct_top1, dtype=bool)
        self.mode, self.kernel = mode, kernel
        counts = np.ones(len(self.scores), dtype=np.int64) if multiplicity is None else np.asarray(multiplicity, dtype=np.int64)
        if np.any(counts < 0):
            raise ValueError("negative multiplicities")
        self.original_known = int(counts[self.known].sum())
        self.original_unknown = int(counts[~self.known].sum())
        if not self.original_known or not self.original_unknown:
            raise ValueError("both classes are needed")
        wk = wu = 1
        if prior is not None:
            fraction = Fraction(str(prior))
            wk = (fraction.denominator - fraction.numerator) * self.original_unknown
            wu = fraction.numerator * self.original_known
            divisor = math.gcd(wk, wu)
            wk, wu = wk // divisor, wu // divisor
        self.class_weights = {"known": wk, "unknown": wu}
        weighted = counts * np.where(self.known, wk, wu)
        self.positive_count = int(weighted[self.known].sum())
        self.unknown_count = int(weighted[~self.known].sum())
        correct = self.known if mode == "presence" else self.correct_top1
        self.curve = module._pr_events(
            np.repeat(self.scores, weighted), np.repeat(correct, weighted), self.positive_count
        )
        # Repeating the unknown events is unnecessary: its common class weight
        # cancels in TNR. The PR-event implementation still does all counting.
        unknown_scores = np.repeat(self.scores[~self.known], counts[~self.known])
        self.unknown_curve = module._pr_events(
            unknown_scores, np.ones(len(unknown_scores), dtype=bool), len(unknown_scores)
        )
        self.unknown_weight = wu
        self.thresholds = np.asarray(self.curve["thresholds"][1:])
        self.unknown_thresholds = np.asarray(self.unknown_curve["thresholds"][1:])
        self.unknown_scores_sorted = np.sort(unknown_scores)
        self.cuts = np.r_[np.nextafter(self.thresholds[0], np.inf), self.thresholds]
        self.rows = [self.at(float(t)) for t in self.cuts]

    def at(self, threshold):
        threshold = float(threshold)
        index = int(np.searchsorted(-self.thresholds, -threshold, side="right"))
        unknown_index = int(np.searchsorted(-self.unknown_thresholds, -threshold, side="right"))
        fp_unknown = int(self.unknown_curve["tp"][unknown_index]) * self.unknown_weight
        result = self.kernel(
            refusal_mode=self.mode, threshold=threshold,
            known_count=self.positive_count, pair_positive_count=0,
            tp=int(self.curve["tp"][index]), fp=int(self.curve["fp"][index]),
            unknown_tn=self.unknown_count - fp_unknown, unknown_count=self.unknown_count,
            curve=self.curve,
        )
        return {k: v for k, v in result.items() if k != "pr_curve"}

    def best_f1(self):
        return max(self.rows, key=lambda r: (r["f1"], r["tnr"], -r["threshold"]))

    def balanced(self):
        return max(self.rows, key=lambda r: (min(r["f1"], r["tnr"]), r["f1"], r["tnr"]))

    def target_tnr(self, target):
        if target == 0:
            cutoff = float(np.min(self.scores))
        else:
            rank = math.ceil(float(target) * len(self.unknown_scores_sorted))
            cutoff = float(np.nextafter(self.unknown_scores_sorted[rank - 1], np.inf))
        return self.at(cutoff)
