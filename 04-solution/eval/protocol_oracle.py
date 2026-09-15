"""Fraction reference for the NEW conventions, separate from production code.

The reviewer's original oracle supplies full-gallery truth and event counting.
Truncation is independently enumerated from the submitted indices, using exact
rational precision contributions. No evaluator helpers are imported.
"""
from fractions import Fraction as F

from review.oracle import oracle_evaluate


def expected_extensions(scores, qids, gids, qcams, gcams, absent, **options):
    legacy_names = ("threshold", "camera_policy", "refusal_mode", "score_kind", "ap_method",
                    "gallery_keys", "gallery_junk", "exclude_mask", "query_frames", "gallery_frames")
    old_kw = {k: options[k] for k in legacy_names if k in options}
    old_kw.update(include_rankings=True)
    base = oracle_evaluate(scores, qids, gids, qcams, gcams, absent, **old_kw)
    total = len(qids)
    valid_count = base["ranking"]["num_valid_queries"]
    k = options.get("top_k", 10)
    ranks = sorted({1, 5, *options.get("rank_ks", (1, 5))})
    average = options.get("query_average", "valid_queries")
    denom = options.get("ap_denominator", "all_gallery_positives")
    inp_policy = options.get("incomplete_inp", "undefined_if_incomplete")
    rank_policy = options.get("rank_beyond_k", "undefined")
    filter_order = options.get("truncation_order", "top_k_then_filter")
    filtered_policy = options.get("filtered_positive_policy", "drop")
    method = options.get("ap_method", "step")
    kind = options.get("score_kind", "similarity")
    sign = 1 if kind == "similarity" else -1
    keys = options.get("gallery_keys")
    keys = list(range(len(gids))) if keys is None else keys
    expose = options.get("include_rankings", False)

    def quotient(a, b):
        return float(a / b) if b else None

    def mean(values):
        if not values or any(x is None for x in values):
            return None
        return float(sum(values, F()) / len(values))

    def means(values):
        return {"valid_queries": quotient(sum(values, F()), valid_count),
                "all_queries_zero": quotient(sum(values, F()), total)}

    full = dict(base["ranking"], scope="full_gallery", mAP_query_average=average,
                mAP_denominator=valid_count if average == "valid_queries" else total)
    # Base APs come from Fraction before final conversion; rounding the aggregate
    # once more here is covered with absolute tolerance, never rank tolerance.
    full["mAP_by_query_average"] = {
        "valid_queries": base["ranking"]["mAP"],
        "all_queries_zero": (base["ranking"]["mAP"] or 0.) * valid_count / total if total else None}
    full["mAP"] = full["mAP_by_query_average"][average]
    for r in ranks:
        full[f"Rank-{r}"] = mean([F(int(row["positive_ranks"][0] <= r))
                                   for row in base["per_query"] if row["status"] == "known"])

    summaries, all_rows = {}, {}
    for selection in ("top_k_then_filter", "filter_then_top_k"):
        out_rows = []
        ap_values = {d: [] for d in ("all_gallery_positives", "retrieved_positives")}
        inp_values = {p: [] for p in ("undefined_if_incomplete", "zero_if_incomplete")}
        cmc_values = {p: {f"Rank-{r}": [] for r in ranks} for p in ("undefined", "clamp_to_k")}
        complete_count = 0
        for i, original in enumerate(base["per_query"]):
            eligible_order = original["ranking"]
            submitted = sorted(range(len(gids)), key=lambda j: (-sign * scores[i][j], keys[j]))[:k]
            included = ([j for j in submitted if j in eligible_order] if selection == "top_k_then_filter"
                        else eligible_order[:k])
            positive_ranks = [n for n, j in enumerate(included, 1) if gids[j] == qids[i]]
            R, H = original["num_relevant"], len(positive_ranks)
            is_valid = original["status"] == "known"
            contributions = []
            for hit, rank in enumerate(positive_ranks, 1):
                value = F(hit, rank)
                if method == "market_matlab":
                    value = (value + (F(hit - 1, rank - 1) if rank > 1 else F(1))) / 2
                contributions.append(value)
            numerator = sum(contributions, F())
            ap = {"all_gallery_positives": numerator / R if is_valid else None,
                  "retrieved_positives": (numerator / H if H else F()) if is_valid else None}
            complete = is_valid and H == R
            complete_count += int(complete)
            inp = {"undefined_if_incomplete": F(R, positive_ranks[-1]) if complete else None,
                   "zero_if_incomplete": (F(R, positive_ranks[-1]) if complete else F()) if is_valid else None}
            cmc = {p: {} for p in cmc_values}
            for p in cmc:
                for r in ranks:
                    value = (None if not is_valid or (r > k and p == "undefined") else
                             F(int(any(position <= min(r, k) for position in positive_ranks))))
                    cmc[p][f"Rank-{r}"] = float(value) if value is not None else None
                    if is_valid:
                        cmc_values[p][f"Rank-{r}"].append(value)
            for name, value in ap.items():
                if is_valid:
                    ap_values[name].append(value)
            for name, value in inp.items():
                if is_valid:
                    inp_values[name].append(value)
            row = dict(query_index=i, status=original["status"], candidate_count=len(included),
                       num_relevant_gallery=R, num_relevant_retrieved=H,
                       positive_ranks=positive_ranks,
                       all_positives_retrieved=complete if is_valid else None,
                       ap_by_denominator={n: float(v) if v is not None else None for n, v in ap.items()},
                       inp_by_incomplete_policy={n: float(v) if v is not None else None for n, v in inp.items()},
                       rank_by_beyond_k_policy=cmc)
            if expose:
                row["ranking"] = included
                if selection == "top_k_then_filter":
                    row["submitted_ranking_before_filter"] = submitted
            out_rows.append(row)
        aps = {d: means(values) for d, values in ap_values.items()}
        inps = {p: mean(values) for p, values in inp_values.items()}
        cmcs = {p: {n: mean(v) for n, v in values.items()} for p, values in cmc_values.items()}
        summaries[selection] = dict(
            scope="top_k", k=k, filter_order=selection, ap_denominator=denom,
            mAP_query_average=average, incomplete_inp_policy=inp_policy,
            rank_beyond_k_policy=rank_policy, mAP=aps[denom][average], mINP=inps[inp_policy],
            **cmcs[rank_policy], num_valid_queries=valid_count,
            mAP_denominator=valid_count if average == "valid_queries" else total,
            num_incomplete_queries=valid_count - complete_count, num_complete_queries=complete_count,
            mAP_by_ap_denominator_and_query_average=aps,
            mINP_by_incomplete_policy=inps, ranks_by_beyond_k_policy=cmcs)
        all_rows[selection] = out_rows

    # Re-express filtered queries as absent IDs, while explicitly retaining ONLY
    # their original eligible candidates. This independently exercises every
    # refusal mode and every PR event, not only the TNR scalar.
    transformed_ids = list(qids)
    sentinel = -1
    for i, row in enumerate(base["per_query"]):
        if row["status"] == "filtered_positive":
            while sentinel in qids or sentinel in gids:
                sentinel -= 1
            transformed_ids[i] = sentinel
            sentinel -= 1
    transformed_absent = [qid not in gids for qid in transformed_ids]
    other_kw = dict(old_kw, exclude_mask=[
        [j not in row["ranking"] for j in range(len(gids))] for row in base["per_query"]])
    other = oracle_evaluate(scores, transformed_ids, gids, qcams, gcams,
                            transformed_absent, **other_kw)
    other["counts"]["filtered_queries"] = base["counts"]["filtered_queries"]
    refusal_branches = {p: {n: b[n] for n in ("counts", "refusal")}
                        for p, b in [("drop", base), ("as_unknown", other)]}
    return dict(ranking=full, ranking_full_gallery=full,
                ranking_top_k=summaries[filter_order], ranking_top_k_by_filter_order=summaries,
                per_query_top_k=all_rows[filter_order], per_query_top_k_by_filter_order=all_rows,
                refusal_by_filtered_positive_policy=refusal_branches,
                **refusal_branches[filtered_policy])
