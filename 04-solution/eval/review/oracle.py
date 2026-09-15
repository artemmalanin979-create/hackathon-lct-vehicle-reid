"""Independent exact-arithmetic oracle for reid_metrics.evaluate.

Written from the metric definitions (AP step / Market MATLAB AP, INP, CMC,
grouped-ties PR curve) and the subject's *documented* protocol, without
reusing any subject code. Pure Python + Fraction; floats only at the edge.
"""
from fractions import Fraction


def _f(x):
    return float(x)


def oracle_evaluate(scores, qids, gids, qcams, gcams, absent, *, threshold,
                    camera_policy, refusal_mode, ap_method="step",
                    score_kind="similarity", gallery_keys=None,
                    gallery_junk=None, exclude_mask=None,
                    query_frames=None, gallery_frames=None,
                    include_rankings=False):
    nq, ng = len(qids), len(gids)
    junk = gallery_junk or [False] * ng
    excl = exclude_mask or [[False] * ng for _ in range(nq)]
    sign = 1 if score_kind == "similarity" else -1
    cutoff = threshold * sign

    # tie rank of gallery item j: position in ascending-key order, else column j
    if gallery_keys is not None:
        by_key = sorted(range(ng), key=lambda j: gallery_keys[j])
        tie_rank = {j: pos for pos, j in enumerate(by_key)}
    else:
        tie_rank = {j: j for j in range(ng)}

    rows = []
    exact = []             # (ap, inp, r1, r5) Fractions for valid queries
    events = []            # (utility_float, correct_bool) refusal events
    known = unknown = filtered = tn_unknown = 0
    relevant_pairs = 0
    for i in range(nq):
        raw_has = any(g == qids[i] for g in gids)
        assert bool(absent[i]) != raw_has, f"absent[{i}] contradiction"
        elig = []
        for j in range(ng):
            if junk[j] or excl[i][j]:
                continue
            same_cam = gcams[j] == qcams[i]
            same_id = gids[j] == qids[i]
            if camera_policy == "market" and same_id and same_cam:
                continue
            if camera_policy == "all_same_camera" and same_cam:
                continue
            if query_frames is not None and same_cam and \
                    gallery_frames[j] == query_frames[i]:
                continue
            elig.append(j)
        order = sorted(elig, key=lambda j: (-scores[i][j] * sign, tie_rank[j]))
        pos_ranks = [r + 1 for r, j in enumerate(order) if gids[j] == qids[i]]
        R = len(pos_ranks)
        relevant_here = R

        row = dict(query_index=i, status=None, eligible_count=len(order),
                   ap=None, inp=None, rank1=None, rank5=None,
                   positive_ranks=pos_ranks if R else [], num_relevant=R)
        if R:
            if ap_method == "step":
                ap = sum(Fraction(k, r) for k, r in
                         enumerate(pos_ranks, 1)) / R
            else:
                total = Fraction(0)
                for k, r in enumerate(pos_ranks, 1):
                    before = Fraction(k - 1, r - 1) if r > 1 else Fraction(1)
                    total += (before + Fraction(k, r)) / 2
                ap = total / R
            row.update(ap=_f(ap), inp=_f(Fraction(R, pos_ranks[-1])),
                       rank1=float(pos_ranks[0] <= 1),
                       rank5=float(pos_ranks[0] <= 5))
            exact.append((ap, Fraction(R, pos_ranks[-1]),
                          Fraction(int(pos_ranks[0] <= 1)),
                          Fraction(int(pos_ranks[0] <= 5))))
        row["status"] = ("unknown" if absent[i]
                         else "known" if R else "filtered_positive")
        best = order[0] if order else None
        accepted = best is not None and scores[i][best] * sign >= cutoff
        row.update(top_gallery_index=best,
                   top_score=float(scores[i][best]) if best is not None else None,
                   accepted=accepted)
        if include_rankings:
            row["ranking"] = list(order)
        rows.append(row)

        if row["status"] == "filtered_positive":
            filtered += 1
            continue
        if row["status"] == "unknown":
            unknown += 1
            tn_unknown += int(not accepted)
        else:
            known += 1
        relevant_pairs += relevant_here
        if refusal_mode == "pairwise":
            events += [(scores[i][j] * sign, gids[j] == qids[i]) for j in order]
        elif best is not None:
            good = not absent[i]
            if refusal_mode == "top1":
                good = good and gids[best] == qids[i]
            events.append((scores[i][best] * sign, good))

    positives = relevant_pairs if refusal_mode == "pairwise" else known
    tp = sum(1 for v, c in events if c and v >= cutoff)
    fp = sum(1 for v, c in events if not c and v >= cutoff)
    fn = positives - tp

    # PR curve: sweep unique utilities descending, ties grouped.
    precision, recall = [1.0], [Fraction(0) if positives else None]
    thresholds, tps, fps = [None], [0], [0]
    step = trap = Fraction(0)
    prev_p, prev_r = Fraction(1), Fraction(0)
    for v in sorted({v for v, _ in events}, reverse=True):
        atp = sum(1 for u, c in events if c and u >= v)
        acc = sum(1 for u, _ in events if u >= v)
        p = Fraction(atp, acc)
        thresholds.append(float(v) * sign)
        precision.append(_f(p))
        tps.append(atp)
        fps.append(acc - atp)
        if positives:
            r = Fraction(atp, positives)
            step += (r - prev_r) * p
            trap += (r - prev_r) * (p + prev_p) / 2
            prev_p, prev_r = p, r
            recall.append(_f(r))
        else:
            recall.append(None)
    if positives:
        recall[0] = 0.0
    curve = dict(auc_pr_trapezoid=_f(trap) if positives else None,
                 ap_pr_step=_f(step) if positives else None,
                 max_recall=recall[-1], positive_count=positives,
                 thresholds=thresholds, precision=precision, recall=recall,
                 tp=tps, fp=fps)

    def ratio(a, b):
        return _f(Fraction(a, b)) if b else None

    refusal = dict(mode=refusal_mode, threshold=float(threshold),
                   tp=tp, fp=fp, fn=fn, tn_unknown=tn_unknown,
                   fp_unknown=unknown - tn_unknown,
                   precision=ratio(tp, tp + fp), recall=ratio(tp, positives),
                   f1=ratio(2 * tp, 2 * tp + fp + fn),
                   tnr=ratio(tn_unknown, unknown),
                   auc_pr=curve["auc_pr_trapezoid"],
                   auc_pr_trapezoid=curve["auc_pr_trapezoid"],
                   ap_pr_step=curve["ap_pr_step"], pr_curve=curve)
    if refusal_mode == "presence":
        refusal["tn"] = tn_unknown

    def mean(idx):
        return _f(sum(e[idx] for e in exact) / len(exact)) if exact else None
    ranking = {"mAP": mean(0), "Rank-1": mean(2), "Rank-5": mean(3),
               "mINP": mean(1), "num_valid_queries": len(exact)}
    counts = dict(queries=nq, gallery=ng, known_queries=known,
                  unknown_queries=unknown, filtered_queries=filtered,
                  refusal_queries=known + unknown,
                  relevant_pairs=relevant_pairs)
    protocol = dict(camera_policy=camera_policy, ap_method=ap_method,
                    refusal_mode=refusal_mode, score_kind=score_kind,
                    tie_policy="gallery_keys" if gallery_keys is not None else "column_index",
                    filtered_positive_policy="drop", empty_denominator=None,
                    threshold_rule=">=" if sign == 1 else "<=",
                    ranking_scope="full_gallery", auc_pr_method="trapezoid")
    return dict(protocol=protocol, counts=counts, ranking=ranking,
                refusal=refusal, per_query=rows)


def diff(a, b, path="$", atol=1e-9, out=None):
    """Recursive exact/tolerant differ; returns list of (path, a, b)."""
    if out is None:
        out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                diff(a[k], b[k], f"{path}.{k}", atol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}.len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", atol, out)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            out.append((path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                out.append((path, a, b))
        elif a != b and not abs(a - b) <= atol:
            out.append((path, a, b))
    elif type(a) is not type(b) or a != b:
        out.append((path, a, b))
    return out
