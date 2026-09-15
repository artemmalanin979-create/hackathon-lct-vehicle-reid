"""Reuse the reviewer's unchanged Fraction oracle; check every legacy field."""
import numpy as np

from review.oracle import oracle_evaluate, diff


def project(actual, expected):
    # New protocol fields are checked separately by test_protocols. Never omit
    # any field defined by the original independent oracle.
    if isinstance(expected, dict):
        return {k: project(actual[k], v) for k, v in expected.items()}
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(("list length", len(actual), len(expected)))
        return [project(a, b) for a, b in zip(actual, expected)]
    return actual


def check_legacy(module, scores, qids, gids, qcams, gcams, absent, **options):
    kw = dict(threshold=.5, camera_policy="market", refusal_mode="top1",
              include_rankings=True)
    kw.update(options)
    raw = np.asarray(scores).reshape(len(qids), len(gids))
    expected = oracle_evaluate(raw.tolist(), list(qids), list(gids),
                               list(qcams), list(gcams), list(absent), **kw)
    implementation_kw = dict(kw)
    if kw.get("exclude_mask") is not None:
        implementation_kw["exclude_mask"] = np.asarray(kw["exclude_mask"], dtype=bool).reshape(len(qids), len(gids))
    actual = module.evaluate(raw, qids, gids, qcams, gcams, absent, **implementation_kw)
    differences = diff(project(actual, expected), expected, atol=1e-12)
    if differences:
        raise AssertionError(differences[:12])
    return actual


def differential_all_fields(module, trials=6000, seed=20260916):
    """Discrete/near-float32 values, both measures and all exclusion channels."""
    rng = np.random.default_rng(seed)
    counts = dict(distance=0, similarity=0, keys_with_filters=0, empty_gallery=0,
                  empty_queries=0, exact_threshold=0, near_float32=0)
    for trial in range(trials):
        nq, ng = int(rng.integers(0, 5)), int(rng.integers(0, 9))
        gids, qids = rng.integers(0, 4, ng).tolist(), rng.integers(0, 6, nq).tolist()
        qcams, gcams = rng.integers(0, 3, nq).tolist(), rng.integers(0, 3, ng).tolist()
        absent = [q not in gids for q in qids]
        raw = rng.integers(0, 5, (nq, ng)).astype(float) / 4
        if trial % 4 == 0:
            raw = .6 + raw * 2e-8
            counts["near_float32"] += 1
        if trial % 7 == 0:
            raw = raw.astype(np.float32)
        if raw.size and trial % 3:
            threshold = float(raw.flat[int(rng.integers(raw.size))])
            counts["exact_threshold"] += 1
        else:
            threshold = [float("inf"), -float("inf"), .5 + 6e-9][trial % 3]
        keys = rng.permutation(ng).tolist() if trial % 3 else None
        if keys is not None and trial % 2:
            keys = [f"key-{v:03d}" for v in keys]
        kind = ["similarity", "distance"][trial % 2]
        counts[kind] += 1
        counts["keys_with_filters"] += int(keys is not None)
        counts["empty_gallery"] += int(ng == 0)
        counts["empty_queries"] += int(nq == 0)
        try:
            check_legacy(module, raw, qids, gids, qcams, gcams, absent,
                         score_kind=kind, threshold=threshold,
                         camera_policy=["market", "all_same_camera"][(trial // 2) % 2],
                         refusal_mode=["presence", "top1", "pairwise"][(trial // 4) % 3],
                         ap_method=["step", "market_matlab"][(trial // 12) % 2],
                         gallery_keys=keys,
                         gallery_junk=(rng.random(ng) < .2).tolist(),
                         exclude_mask=(rng.random((nq, ng)) < .2).tolist(),
                         query_frames=rng.integers(0, 3, nq).tolist(),
                         gallery_frames=rng.integers(0, 3, ng).tolist())
        except AssertionError as exc:
            raise AssertionError(f"seed={seed}, trial={trial}: {exc}") from exc
    return dict(trials=trials, seed=seed, mismatches=0, coverage=counts,
                scope="Every legacy field: protocol/counts/ranking/per_query/refusal/pr_curve")
