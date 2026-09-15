"""Differential fuzz: subject reid_metrics.evaluate vs independent oracle.

Covers ties, junk, exclude_mask, frames, gallery_keys, distance mode, exact
threshold hits, empty galleries, filtered queries. Compares the ENTIRE output
(protocol, counts, ranking, refusal incl. pr_curve, per_query)."""
import itertools
import random
import sys

sys.path.insert(0, "run")
import numpy as np
import reid_metrics as metrics
from oracle import oracle_evaluate, diff

random.seed(20260915)
FAILS = 0


def one_trial(t):
    global FAILS
    nq = random.randint(1, 4)
    ng = random.choice([0, 1, 2, 3, 4, 5, 6])
    if ng == 0:
        gids, gcams = [], []
    else:
        gids = [random.randint(0, 3) for _ in range(ng)]
        gcams = [random.randint(0, 2) for _ in range(ng)]
    qids = [random.choice([0, 1, 2, 3, 99]) for _ in range(nq)]
    qcams = [random.randint(0, 2) for _ in range(nq)]
    absent = [q not in gids for q in qids]
    denom = random.choice([1, 2, 4])
    scores = [[random.randint(0, 4) / denom for _ in range(ng)]
              for _ in range(nq)]
    if random.random() < 0.25 and ng:
        scores = np.asarray(scores, dtype=np.float32).astype(float).tolist()
    flat = sorted({v for row in scores for v in row}) or [0.0]
    threshold = random.choice(
        flat + [random.choice(flat) + 0.125, -np.inf, np.inf, 0.5])
    junk = [random.random() < 0.2 for _ in range(ng)] \
        if random.random() < 0.5 else None
    excl = [[random.random() < 0.15 for _ in range(ng)] for _ in range(nq)] \
        if random.random() < 0.5 else None
    if random.random() < 0.5 and ng:
        qframes = [random.randint(0, 2) for _ in range(nq)]
        gframes = [random.randint(0, 2) for _ in range(ng)]
    else:
        qframes = gframes = None
    if random.random() < 0.5 and ng:
        keys = list(range(100, 100 + ng))
        random.shuffle(keys)
        if random.random() < 0.5:
            keys = [f"k{v}" for v in keys]
    else:
        keys = None
    policy = random.choice(["market", "all_same_camera"])
    mode = random.choice(["presence", "top1", "pairwise"])
    apm = random.choice(["step", "market_matlab"])
    kind = random.choice(["similarity", "distance"])
    kw = dict(threshold=threshold, camera_policy=policy, refusal_mode=mode,
              ap_method=apm, score_kind=kind, gallery_keys=keys,
              gallery_junk=junk, exclude_mask=excl, query_frames=qframes,
              gallery_frames=gframes, include_rankings=True)
    got = metrics.evaluate(np.asarray(scores, dtype=float).reshape(nq, ng),
                           qids, gids, qcams, gcams, absent, **kw)
    want = oracle_evaluate(scores, qids, gids, qcams, gcams, absent, **kw)
    problems = diff(want, got)
    if problems:
        FAILS += 1
        print(f"== trial {t}: {len(problems)} diffs ==")
        print(dict(scores=scores, qids=qids, gids=gids, qcams=qcams,
                   gcams=gcams, absent=absent, **{k: v for k, v in kw.items()
                                                  if k != 'include_rankings'}))
        for p, a, b in problems[:12]:
            print(f"  {p}: oracle={a!r} subject={b!r}")
        if FAILS >= 8:
            sys.exit(1)


for t in range(6000):
    one_trial(t)
print(f"done: 6000 trials, {FAILS} mismatching")
