"""Explicit full-gallery / submitted-list conventions; no refusal logic."""
import math

import numpy as np


AP_DENOMINATORS = ("all_gallery_positives", "retrieved_positives")
QUERY_AVERAGES = ("valid_queries", "all_queries_zero")
INCOMPLETE_INP = ("undefined_if_incomplete", "zero_if_incomplete")
RANK_BEYOND_K = ("undefined", "clamp_to_k")
FILTER_ORDERS = ("top_k_then_filter", "filter_then_top_k")


def mean(values):
    return math.fsum(values) / len(values) if values else None


def ap_means(values, total_queries):
    total = math.fsum(values)
    return {"valid_queries": total / len(values) if values else None,
            "all_queries_zero": total / total_queries if total_queries else None}


def full_summary(base, rows, ranks, query_average):
    valid = [row for row in rows if row["status"] == "known"]
    # Preserve the independently tested core aggregation for the default branch.
    by_average = ap_means([row["ap"] for row in valid], len(rows))
    by_average["valid_queries"] = base["mAP"]
    result = dict(base, scope="full_gallery", mAP_by_query_average=by_average,
                  mAP_query_average=query_average,
                  mAP_denominator=len(valid) if query_average == "valid_queries" else len(rows))
    result["mAP"] = by_average[query_average]
    for rank in ranks:
        if rank not in (1, 5):
            result[f"Rank-{rank}"] = mean([float(row["positive_ranks"][0] <= rank)
                                          for row in valid])
    return result


def list_statistics(full_row, order, qid, gids, ap_method, ranks, k, include_rankings):
    positives = [i for i, j in enumerate(order, 1) if gids[j] == qid]
    total, found = full_row["num_relevant"], len(positives)
    valid = full_row["status"] == "known"
    numerator = []
    for hit, rank in enumerate(positives, 1):
        precision = hit / rank
        before = (hit - 1) / (rank - 1) if rank > 1 else 1.
        numerator.append(precision if ap_method == "step" else (before + precision) / 2)
    numerator = math.fsum(numerator)
    # A valid query retrieving zero positives scores zero under both conventions;
    # it never disappears from the valid-query denominator.
    ap = {"all_gallery_positives": numerator / total if valid else None,
          "retrieved_positives": (numerator / found if found else 0.) if valid else None}
    complete = valid and found == total
    observed_inp = found / positives[-1] if complete else None
    inp = {"undefined_if_incomplete": observed_inp,
           "zero_if_incomplete": (observed_inp if complete else 0.) if valid else None}
    cmc = {policy: {f"Rank-{rank}":
                   (None if not valid or (rank > k and policy == "undefined") else
                    float(bool(positives) and positives[0] <= min(rank, k)))
                   for rank in ranks} for policy in RANK_BEYOND_K}
    result = dict(query_index=full_row["query_index"], status=full_row["status"],
                  candidate_count=len(order), num_relevant_gallery=total,
                  num_relevant_retrieved=found, positive_ranks=positives,
                  all_positives_retrieved=complete if valid else None,
                  ap_by_denominator=ap, inp_by_incomplete_policy=inp,
                  rank_by_beyond_k_policy=cmc)
    if include_rankings:
        result["ranking"] = list(order)
    return result


def top_k_summaries(rows, raw, qids, gids, *, k, ranks, gallery_keys, score_kind,
                    ap_method, ap_denominator, query_average, incomplete_inp,
                    rank_beyond_k, include_rankings):
    sign = 1 if score_kind == "similarity" else -1
    keys = list(range(len(gids))) if gallery_keys is None else list(gallery_keys)
    tie_order = np.asarray(sorted(range(len(gids)), key=lambda j: keys[j]), dtype=int)
    per_order = {policy: [] for policy in FILTER_ORDERS}
    for i, row in enumerate(rows):
        full_order = row["ranking"]
        eligible = set(full_order)
        submitted = tie_order[np.argsort(-raw[i, tie_order] * sign, kind="stable")][:k]
        orders = {"filter_then_top_k": full_order[:k],
                  "top_k_then_filter": [int(j) for j in submitted if int(j) in eligible]}
        for policy, order in orders.items():
            stats = list_statistics(row, order, qids[i], gids, ap_method, ranks, k,
                                    include_rankings)
            if include_rankings and policy == "top_k_then_filter":
                stats["submitted_ranking_before_filter"] = submitted.tolist()
            per_order[policy].append(stats)
    summaries = {}
    for policy, query_rows in per_order.items():
        valid = [row for row in query_rows if row["status"] == "known"]
        aps = {den: ap_means([row["ap_by_denominator"][den] for row in valid], len(rows))
               for den in AP_DENOMINATORS}
        inps = {}
        for choice in INCOMPLETE_INP:
            values = [row["inp_by_incomplete_policy"][choice] for row in valid]
            # Undefined means undefined for the whole population, NOT a silently
            # improved mean of the easy complete queries.
            inps[choice] = None if any(v is None for v in values) else mean(values)
        cmcs = {}
        for choice in RANK_BEYOND_K:
            cmcs[choice] = {}
            for rank in ranks:
                name = f"Rank-{rank}"
                values = [row["rank_by_beyond_k_policy"][choice][name] for row in valid]
                cmcs[choice][name] = None if any(v is None for v in values) else mean(values)
        censored = sum(not row["all_positives_retrieved"] for row in valid)
        summaries[policy] = dict(
            scope="top_k", k=k, filter_order=policy,
            ap_denominator=ap_denominator, mAP_query_average=query_average,
            incomplete_inp_policy=incomplete_inp, rank_beyond_k_policy=rank_beyond_k,
            mAP=aps[ap_denominator][query_average], mINP=inps[incomplete_inp],
            **cmcs[rank_beyond_k], num_valid_queries=len(valid),
            mAP_denominator=len(valid) if query_average == "valid_queries" else len(rows),
            num_incomplete_queries=censored, num_complete_queries=len(valid) - censored,
            mAP_by_ap_denominator_and_query_average=aps,
            mINP_by_incomplete_policy=inps, ranks_by_beyond_k_policy=cmcs)
    return summaries, per_order
