# INTENTIONALLY BROKEN: positive_ranks отдаются с нуля, а не с единицы
"""Independent, offline NumPy evaluator. Protocol choices are explicit.

See REPORT.md for definitions, sources and unresolved organizer conventions.
All results are fractions in [0, 1]; undefined values are None, never NaN.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _ids(values, size, name):
    result = np.asarray(values, dtype=object)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if any(isinstance(x, (bool, np.bool_)) or
           not isinstance(x, (str, int, np.integer)) for x in result):
        raise ValueError(f"{name} must contain only non-null integer/string IDs")
    return result


def _bools(values, shape, name):
    result = np.asarray(values)
    if result.shape != shape or (result.size and result.dtype.kind != "b"):
        raise ValueError(f"{name} must be a boolean array of shape {shape}")
    return result.astype(bool)


def _matrix(values, name):
    result = np.asarray(values)
    if result.ndim != 2 or result.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric matrix")
    result = result.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _mean(values):
    return float(math.fsum(values) / len(values)) if len(values) else None


def _camera_keep(same_id, same_camera, policy):
    if policy == "market":
        return ~(same_id & same_camera)
    return ~same_camera


def _rank_indices(values, eligible, tie_order):
    # tie_order is either the original column order or ascending unique keys.
    candidates = tie_order[eligible[tie_order]]
    return candidates[np.argsort(-values[candidates], kind="stable")]


def _rank_statistics(relevant, method):
    ranks = np.flatnonzero(relevant) + 1  # one-based, AFTER junk removal
    count = len(ranks)
    if count == 0:
        return None
    hits = np.arange(1, count + 1, dtype=np.float64)
    at_hit = hits / ranks
    if method == "step":
        ap = float(np.mean(at_hit))
    else:
        # MATLAB compute_AP updates precision even at non-relevant positions.
        before_hit = np.ones(count, dtype=np.float64)
        later = ranks > 1
        before_hit[later] = (hits[later] - 1) / (ranks[later] - 1)
        ap = float(np.mean((before_hit + at_hit) / 2))
    return dict(ap=ap, inp=float(count / ranks[-1]),
                rank1=float(ranks[0] <= 1), rank5=float(ranks[0] <= 5),
                positive_ranks=(ranks - 1).tolist(), num_relevant=count)


def _aggregate_ranking(rows, total_queries):
    # total_queries is intentionally not the denominator: only valid queries.
    valid = [row for row in rows if row["ap"] is not None]
    return {
        "mAP": _mean([row["ap"] for row in valid]),
        "Rank-1": _mean([row["rank1"] for row in valid]),
        "Rank-5": _mean([row["rank5"] for row in valid]),
        "mINP": _mean([row["inp"] for row in valid]),
        "num_valid_queries": len(valid),
    }


def _pr_events(scores, correct, positive_count):
    """Threshold events, grouped ties; top1 may have unreachable positives.

    Curves start with (recall=0, precision=1), then sweep high to low.
    The area outside attainable recall is zero; no artificial endpoint at R=1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    precision = [1.0]
    recall = [0.0] if positive_count else [None]
    thresholds = [None]  # sentinel: no accepted candidates
    tp_counts, fp_counts = [0], [0]
    step_area = trap_area = 0.0
    if scores.size:
        order = np.argsort(-scores, kind="stable")
        scores, correct = scores[order], correct[order]
        cumulative = np.cumsum(correct, dtype=np.int64)
        ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
        for end in ends:
            accepted = int(end + 1)
            tp = int(cumulative[end])
            p = tp / accepted
            r = tp / positive_count if positive_count else None
            if r is not None:
                delta = r - recall[-1]
                step_area += delta * p
                trap_area += delta * (precision[-1] + p) / 2
            thresholds.append(float(scores[end]))
            precision.append(float(p))
            recall.append(r)
            tp_counts.append(tp)
            fp_counts.append(accepted - tp)
    return dict(
        auc_pr_trapezoid=float(trap_area) if positive_count else None,
        ap_pr_step=float(step_area) if positive_count else None,
        max_recall=recall[-1], positive_count=int(positive_count),
        thresholds=thresholds, precision=precision, recall=recall,
        tp=tp_counts, fp=fp_counts,
    )


def _evaluate_full(
    scores, query_ids, gallery_ids, query_cameras, gallery_cameras,
    known_absent, *, threshold, camera_policy, refusal_mode,
    ap_method="step", score_kind="similarity", gallery_keys=None,
    gallery_junk=None, exclude_mask=None,
    query_frames=None, gallery_frames=None, include_rankings=False,
    filtered_positive_policy="drop",
):
    """Evaluate full Q x G scores without network or heavyweight dependencies.

    camera_policy: 'market' removes same ID AND same camera; 'all_same_camera'
      removes ALL same-camera items. Both are evaluation masks using truth.
    refusal_mode: 'presence' (binary mate existence), 'top1' (correct identity),
      'pairwise' (all accepted image pairs). Required to prevent silent mixing.
    known_absent: exact boolean truth for identity absence in the RAW gallery.
      Contradictory flags raise ValueError, including unflagged absent IDs.
      A known query whose positives are all excluded keeps its status; refusal
      either drops it or treats it as unknown, controlled explicitly.
    score_kind: similarity accepts score >= threshold; distance accepts <=.
    Exclusions: gallery_junk[G], exclude_mask[Q,G]; optional paired frame IDs
      exclude equal frame ID AND equal camera. No magic ID (-1 etc.) is assumed.
    Ties: ascending original column order, or ascending unique gallery_keys.
    No valid queries / no positives / empty TNR denominator return None.
    """
    if camera_policy not in {"market", "all_same_camera"}:
        raise ValueError("camera_policy must be market or all_same_camera")
    if refusal_mode not in {"presence", "top1", "pairwise"}:
        raise ValueError("refusal_mode must be presence, top1 or pairwise")
    if ap_method not in {"step", "market_matlab"}:
        raise ValueError("ap_method must be step or market_matlab")
    if score_kind not in {"similarity", "distance"}:
        raise ValueError("score_kind must be similarity or distance")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ) or math.isnan(float(threshold)):
        raise ValueError("threshold must be a number other than NaN")
    threshold = float(threshold)
    raw = _matrix(scores, "scores")
    nq, ng = raw.shape
    qids = _ids(query_ids, nq, "query_ids")
    gids = _ids(gallery_ids, ng, "gallery_ids")
    qcams = _ids(query_cameras, nq, "query_cameras")
    gcams = _ids(gallery_cameras, ng, "gallery_cameras")
    absent = _bools(known_absent, (nq,), "known_absent")
    junk = np.zeros(ng, dtype=bool) if gallery_junk is None else _bools(
        gallery_junk, (ng,), "gallery_junk")
    excluded = None if exclude_mask is None else _bools(
        exclude_mask, (nq, ng), "exclude_mask")
    if (query_frames is None) != (gallery_frames is None):
        raise ValueError("query_frames and gallery_frames must be supplied together")
    qframes = gframes = None
    if query_frames is not None:
        qframes = _ids(query_frames, nq, "query_frames")
        gframes = _ids(gallery_frames, ng, "gallery_frames")
    tie_order = np.arange(ng)
    if gallery_keys is not None:
        keys = _ids(gallery_keys, ng, "gallery_keys")
        if len(set(keys.tolist())) != ng:
            raise ValueError("gallery_keys must be unique")
        try:
            tie_order = np.asarray(sorted(range(ng), key=lambda j: keys[j]), dtype=int)
        except TypeError as exc:
            raise ValueError("gallery_keys must have mutually comparable types") from exc
    sign = 1.0 if score_kind == "similarity" else -1.0
    utility, cutoff = raw * sign, threshold * sign
    rows, event_scores, event_correct = [], [], []
    known_count = unknown_count = filtered_count = unknown_tn = 0
    pair_positive_count = 0
    for i in range(nq):
        same_id = gids == qids[i]
        raw_has_match = bool(same_id.any())
        if bool(absent[i]) == raw_has_match:
            raise ValueError(f"known_absent[{i}] contradicts identity presence in raw gallery")
        same_camera = gcams == qcams[i]
        eligible = _camera_keep(same_id, same_camera, camera_policy) & ~junk
        if excluded is not None:
            eligible &= ~excluded[i]
        if qframes is not None:
            eligible &= ~(same_camera & (gframes == qframes[i]))
        order = _rank_indices(utility[i], eligible, tie_order)
        relevant = same_id[order]
        stats = _rank_statistics(relevant, ap_method)
        status = "unknown" if absent[i] else "known" if stats else "filtered_positive"
        row = dict(query_index=i, status=status, eligible_count=int(order.size),
                   ap=None, inp=None, rank1=None, rank5=None,
                   positive_ranks=[], num_relevant=0)
        if stats:
            row.update(stats)
        best = int(order[0]) if order.size else None
        accepted = best is not None and bool(utility[i, best] >= cutoff)
        row.update(top_gallery_index=best,
                   top_score=float(raw[i, best]) if best is not None else None,
                   accepted=accepted)
        if include_rankings:
            row["ranking"] = order.tolist()
        rows.append(row)
        if status == "filtered_positive":
            filtered_count += 1
            if filtered_positive_policy == "drop":
                continue
        if status in {"unknown", "filtered_positive"}:
            unknown_count += 1
            unknown_tn += int(not accepted)
        else:
            known_count += 1
        pair_positive_count += int(relevant.sum())
        if refusal_mode == "pairwise":
            event_scores.extend(utility[i, order].tolist())
            event_correct.extend(relevant.tolist())
        elif best is not None:
            event_scores.append(float(utility[i, best]))
            correct = status == "known"
            if refusal_mode == "top1":
                correct = correct and bool(same_id[best])
            event_correct.append(correct)
    positives = pair_positive_count if refusal_mode == "pairwise" else known_count
    escores = np.asarray(event_scores, dtype=np.float64)
    ecorrect = np.asarray(event_correct, dtype=bool)
    selected = escores >= cutoff
    tp = int(np.count_nonzero(selected & ecorrect))
    fp = int(np.count_nonzero(selected & ~ecorrect))
    fn = int(positives - tp)
    curve = _pr_events(escores, ecorrect, positives)
    curve["thresholds"] = [None if t is None else t * sign for t in curve["thresholds"]]
    refusal = dict(
        mode=refusal_mode, threshold=threshold,
        tp=tp, fp=fp, fn=fn, tn_unknown=unknown_tn,
        fp_unknown=unknown_count - unknown_tn,
        precision=_ratio(tp, tp + fp), recall=_ratio(tp, positives),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
        tnr=_ratio(unknown_tn, unknown_count),
        auc_pr=curve["auc_pr_trapezoid"],
        auc_pr_trapezoid=curve["auc_pr_trapezoid"],
        ap_pr_step=curve["ap_pr_step"], pr_curve=curve,
    )
    if refusal_mode == "presence":
        refusal["tn"] = unknown_tn
    return dict(
        protocol=dict(camera_policy=camera_policy, ap_method=ap_method,
                      refusal_mode=refusal_mode, score_kind=score_kind,
                      tie_policy="gallery_keys" if gallery_keys is not None else "column_index",
                      filtered_positive_policy=filtered_positive_policy, empty_denominator=None,
                      threshold_rule=">=" if sign == 1 else "<=",
                      ranking_scope="full_gallery", auc_pr_method="trapezoid"),
        counts=dict(queries=nq, gallery=ng, known_queries=known_count,
                    unknown_queries=unknown_count, filtered_queries=filtered_count,
                    refusal_queries=known_count + unknown_count,
                    relevant_pairs=pair_positive_count),
        ranking=_aggregate_ranking(rows, nq), refusal=refusal, per_query=rows,
    )


def evaluate(
    scores, query_ids, gallery_ids, query_cameras, gallery_cameras,
    known_absent, *, threshold, camera_policy, refusal_mode,
    ap_method="step", score_kind="similarity", gallery_keys=None,
    gallery_junk=None, exclude_mask=None,
    query_frames=None, gallery_frames=None, include_rankings=False,
    top_k=10, ap_denominator="all_gallery_positives",
    query_average="valid_queries", filtered_positive_policy="drop",
    incomplete_inp="undefined_if_incomplete", rank_beyond_k="undefined",
    rank_ks=(1, 5), truncation_order="top_k_then_filter",
):
    """Always report full-gallery AND top-K metrics, with all convention branches.

    `ranking` is a compatibility alias for `ranking_full_gallery`; both have an
    explicit scope label. `ranking_top_k` selects from two filter-order branches.
    AP denominator: all eligible gallery positives, or only retrieved positives.
    mAP average: valid queries, or all queries with missing AP assigned zero.
    Censored INP: undefined (propagated to the mean), or zero for incomplete lists.
    Rank-k beyond the submitted K: undefined, or Rank-min(k,K).
    Filter order: raw top-K then evaluation filters (submission semantics), or
    evaluation filters then top-K (eligible-list semantics).
    Both refusal policies, drop / as_unknown, are reported with their counts.
    Defaults are explicit in `protocol`; see REPORT.md for formulas and examples.
    Scores/thresholds are compared in float64 without tolerance or quantization.
    Already equal float32 inputs remain ties, ordered by keys or column index.
    """
    from scope_metrics import (AP_DENOMINATORS, QUERY_AVERAGES, INCOMPLETE_INP,
                               RANK_BEYOND_K, FILTER_ORDERS, full_summary, top_k_summaries)

    for name, value, choices in [
        ("ap_denominator", ap_denominator, AP_DENOMINATORS),
        ("query_average", query_average, QUERY_AVERAGES),
        ("filtered_positive_policy", filtered_positive_policy, ("drop", "as_unknown")),
        ("incomplete_inp", incomplete_inp, INCOMPLETE_INP),
        ("rank_beyond_k", rank_beyond_k, RANK_BEYOND_K),
        ("truncation_order", truncation_order, FILTER_ORDERS),
    ]:
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices}")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    try:
        ranks = list(rank_ks)
    except TypeError as exc:
        raise ValueError("rank_ks must be an iterable of positive integers") from exc
    if not ranks or any(isinstance(k, (bool, np.bool_)) or
                        not isinstance(k, (int, np.integer)) or k < 1 for k in ranks):
        raise ValueError("rank_ks must contain positive integers")
    ranks = sorted({1, 5, *(int(k) for k in ranks)})
    top_k = int(top_k)
    common = dict(threshold=threshold, camera_policy=camera_policy,
                  refusal_mode=refusal_mode, ap_method=ap_method, score_kind=score_kind,
                  gallery_keys=gallery_keys, gallery_junk=gallery_junk,
                  exclude_mask=exclude_mask, query_frames=query_frames,
                  gallery_frames=gallery_frames, include_rankings=True)
    raw = _matrix(scores, "scores")
    branches = {policy: _evaluate_full(raw, query_ids, gallery_ids, query_cameras,
                                      gallery_cameras, known_absent,
                                      filtered_positive_policy=policy, **common)
                for policy in ("drop", "as_unknown")}
    result = branches[filtered_positive_policy]
    rows = result["per_query"]
    full = full_summary(result["ranking"], rows, ranks, query_average)
    summaries, truncated_rows = top_k_summaries(
        rows, raw, query_ids, gallery_ids, k=top_k, ranks=ranks,
        gallery_keys=gallery_keys, score_kind=score_kind, ap_method=ap_method,
        ap_denominator=ap_denominator, query_average=query_average,
        incomplete_inp=incomplete_inp, rank_beyond_k=rank_beyond_k,
        include_rankings=include_rankings)
    result.update(ranking=full, ranking_full_gallery=full,
                  ranking_top_k=summaries[truncation_order],
                  ranking_top_k_by_filter_order=summaries,
                  per_query_top_k=truncated_rows[truncation_order],
                  per_query_top_k_by_filter_order=truncated_rows,
                  refusal_by_filtered_positive_policy={policy: dict(
                      counts=branch["counts"], refusal=branch["refusal"])
                      for policy, branch in branches.items()})
    result["protocol"].update(
        schema_version=2, ranking_scopes=["full_gallery", "top_k"],
        ranking_alias="ranking_full_gallery", top_k=top_k,
        ap_denominator=ap_denominator, query_average=query_average,
        incomplete_inp=incomplete_inp, rank_beyond_k=rank_beyond_k,
        rank_ks=ranks, truncation_order=truncation_order,
        rank_and_inp_query_average="valid_queries", refusal_scope="full_eligible_gallery",
        score_precision="float64_no_downcast", threshold_precision="float64_no_downcast",
        score_input_dtype=str(np.asarray(scores).dtype),
        tie_equality="exact_represented_value", embedding_accumulation="float64",
        missing_ap_all_queries="zero", empty_retrieved_ap_for_valid_query=0.,
        incomplete_inp_mean="propagate_undefined_or_assign_zero; never_drop_query")
    if not include_rankings:
        for row in rows:
            row.pop("ranking")
    return result


def scores_from_embeddings(query_embeddings, gallery_embeddings, metric="cosine"):
    """Cosine / negative L2: cast embeddings to float64 BEFORE all arithmetic.

    Float32 coordinates are represented exactly; lost upstream precision cannot
    be recovered. Dot products/norms accumulate in float64, preserving distinct
    scores that would collapse if the result were rounded back to float32.
    """
    q = _matrix(query_embeddings, "query_embeddings")
    g = _matrix(gallery_embeddings, "gallery_embeddings")
    if q.shape[1] != g.shape[1] or q.shape[1] == 0:
        raise ValueError("embedding dimensions must match and be nonzero")
    if metric == "cosine":
        def normalize(x):
            scale = np.max(np.abs(x), axis=1, keepdims=True)
            if np.any(scale == 0):
                raise ValueError("cosine is undefined for a zero embedding")
            x = x / scale
            return x / np.linalg.norm(x, axis=1, keepdims=True)
        result = np.clip(normalize(q) @ normalize(g).T, -1.0, 1.0)
    elif metric == "negative_l2":
        # Per-row differences avoid cancellation for nearly identical vectors.
        result = np.empty((len(q), len(g)), dtype=np.float64)
        for i, vector in enumerate(q):
            with np.errstate(over="ignore", invalid="ignore"):
                result[i] = -np.linalg.norm(g - vector, axis=1)
    else:
        raise ValueError("metric must be cosine or negative_l2")
    if not np.isfinite(result).all():
        raise ValueError("embedding score overflow; rescale embeddings")
    return result


def validate_identity_split(train_ids, test_ids):
    """Explicit helper; evaluation alone cannot detect train/test identity leakage."""
    train = _ids(train_ids, len(train_ids), "train_ids")
    test = _ids(test_ids, len(test_ids), "test_ids")
    overlap = set(train.tolist()) & set(test.tolist())
    if overlap:
        raise ValueError(f"train/test identity overlap: {len(overlap)} IDs")
    return dict(train_identities=len(set(train)), test_identities=len(set(test)), overlap=0)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ; see REPORT.md for array names")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--camera-policy", choices=["market", "all_same_camera"], required=True)
    parser.add_argument("--refusal-mode", choices=["presence", "top1", "pairwise"], required=True)
    parser.add_argument("--ap-method", choices=["step", "market_matlab"], default="step")
    parser.add_argument("--score-kind", choices=["similarity", "distance"], default="similarity")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ap-denominator", choices=["all_gallery_positives", "retrieved_positives"], default="all_gallery_positives")
    parser.add_argument("--query-average", choices=["valid_queries", "all_queries_zero"], default="valid_queries")
    parser.add_argument("--filtered-positive-policy", choices=["drop", "as_unknown"], default="drop")
    parser.add_argument("--incomplete-inp", choices=["undefined_if_incomplete", "zero_if_incomplete"], default="undefined_if_incomplete")
    parser.add_argument("--rank-beyond-k", choices=["undefined", "clamp_to_k"], default="undefined")
    parser.add_argument("--rank-ks", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--truncation-order", choices=["top_k_then_filter", "filter_then_top_k"], default="top_k_then_filter")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = ["scores", "query_ids", "gallery_ids", "query_cameras", "gallery_cameras", "known_absent"]
    optional = ["gallery_keys", "gallery_junk", "exclude_mask", "query_frames", "gallery_frames"]
    with np.load(args.input, allow_pickle=False) as data:
        missing = set(required) - set(data.files)
        if missing:
            parser.error(f"missing arrays: {sorted(missing)}")
        result = evaluate(
            **{k: data[k] for k in required},
            **{k: data[k] for k in optional if k in data.files},
            threshold=args.threshold, camera_policy=args.camera_policy,
            refusal_mode=args.refusal_mode, ap_method=args.ap_method,
            score_kind=args.score_kind, include_rankings=True,
            top_k=args.top_k, ap_denominator=args.ap_denominator,
            query_average=args.query_average, filtered_positive_policy=args.filtered_positive_policy,
            incomplete_inp=args.incomplete_inp, rank_beyond_k=args.rank_beyond_k,
            rank_ks=args.rank_ks, truncation_order=args.truncation_order,
        )
    args.output.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
