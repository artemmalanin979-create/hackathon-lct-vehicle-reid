# Bbox margin pilot — frozen before inference

Frozen 2026-09-28, branch `ml/bbox-margin-20260928`, starting SHA
`a0404df` (`main`). This is an **experimental input-preprocessing check**, not
the released `d1_j48` and not a change to the evaluator or reference metrics.
The previously used LCT validation set is **not an untouched holdout**.

## Hypothesis and variants

The detector boxes may crop distinctive vehicle edges too tightly, or include
background that harms cross-camera matching. Test exactly three deterministic
transforms on **both** query and gallery original frames:

1. `original`: CSV `(x, y, w, h)` unchanged.
2. `inward_3pct`: `dx = floor(0.03*w + 0.5)`, `dy = floor(0.03*h + 0.5)`;
   `(x+dx, y+dy, w-2*dx, h-2*dy)`.
3. `outward_3pct`: the same deltas; `(x-dx, y-dy, w+2*dx, h+2*dy)`.

All boxes must remain positive and intersect the frame. Do **not** clamp an
outward box: retain the service's existing PIL black-padding behavior at image
boundaries. The crop is RGB, PIL bilinear 208×208, float32 0..255 NCHW; both
released ONNX branches and the frozen whitening receive it unchanged.

This is a **global setting**, not a query-dependent crop oracle. No additional
percentages, anisotropic changes, separate query/gallery policies, or TTA are
allowed in this pilot.

## Data and fixed inputs

The authorized organizer images reside outside Git at
`/home/artem/projects/hackathon-lct-vehicle-reid/data/images`. The predeclared
selection set is `04-solution/postproc/tune/`, derived from train-fit identities:
1110 query and 732 gallery. The one confirmation set is
`04-solution/split/files/val_*.csv`: 1110 query and 750 gallery. Vehicle IDs
are disjoint across these sets by the existing split design; the tune
identities did participate in model/whitening fitting, a limitation. The val
has been examined repeatedly in this project and is only an exploratory
confirmation, not an independent external generalization estimate.

| Input | CSV SHA-256 | Aggregate image SHA-256 (sorted CSV row order; `name + NUL + raw-file SHA-256 bytes` per image) |
|---|---|---|
| tune query | `17b03ff171a82a42fb3733b293cd8c1be7f111076613d2c0c8041564b201dce6` | `798f4af4728073d6dc4937c8552b770602b7f6347ad7159b90c18ed16a3beea8` |
| tune gallery | `224fe2e21ddcc991b8a88a2a2e95d36d9ea3ac0e8e58439d9da2fe5175bcd518` | `9554594bc6a39f742653e604025e8fcd05d03a72990b12612d1d59bb3de89358` |
| val query | `0988668d730e2599f487e25dac8913d718e3b5405442a73ca61123cc8b1a50f6` | `f25eb108bff73fe78f361bb30a0a9f0ee0a2224f4fa1625823d069046f112a80` |
| val gallery | `6c1a8d65e89dc5e898d12c90f02c6db6d53b031817a6acaf648bcacc1b532a9d` | `ca2f3b16f0a8b42a1c541e87ed7157f53ab4d0872242e17c93d5bbd435a257b0` |

Released weights: OSNet-AIN `4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f`,
combined_v1 `b1ba5021275b34079a1653608bdfd215fc9404306dc909852e4cdfa03402efb2`,
whitening `eb4433ffd5e38d3751d5cb04e234090060a83be6d1720b47bcf2fa1274b3c5b5`.
Model output remains 512D and L2-normalized.

## Selection, confirmation, stopping

Freeze all variants and all criteria before running inference. Run `original`
and both ±3% variants on tune. Use the unchanged evaluator
`04-solution/eval/reid_metrics.py` with `market` exclusion,
`presence` refusal and full-gallery mAP over valid queries, plus the released
cosine scorer and KR `(k1=6,k2=3,lambda=0.3)`. Positive control: a label-perfect
score matrix gives mAP 1; negative control: fixed-seed independent random
512D query/gallery vectors give a low, non-perfect mAP. Both controls use the
same labels and evaluator. Compare variants with paired bootstrap by
`vehicle_id`, seed 20260928, 2000 resamples.

Select **at most one** variant for val: the larger tune KR-mAP delta, with
`inward_3pct` winning an exact tie, only if tune delta is at least +0.015 and
the paired 95% bootstrap lower bound is above zero. Otherwise stop after tune
and report a negative pilot; do not examine val crop embeddings or scores.
This gate does not make the reused val pristine. No seed hunting, choice by val,
or post-hoc re-tuning of KR/whitening is allowed.

If the tune gate passes, calibrate that variant's cosine and KR refusal
thresholds **on tune only**, using the released robust-balanced rule: maximize
`min(TNR, F1 at absent priors 0.10/0.25/0.40)`, tie-break by measured F1,
then TNR. Freeze both thresholds before val inference. On val, compare against
the original crop with its released thresholds, using full-ranking mAP, Rank-1/5,
mINP, and presence F1/TNR. A candidate can replace the release only after:

* val KR full-gallery mAP delta ≥ +0.010, and paired bootstrap 95% lower
  bound by vehicle ID > 0;
* val presence F1 and TNR each no worse than original by more than 0.010,
  using tune-fitted candidate thresholds;
* batch-1 p95 image→descriptor latency on the same local CPU, same threads,
  30 warm + 100 measured **paired** items, no worse by more than 5%; same
  offline setup and 512D format;
* positive/negative controls pass and the candidate is independently reviewed.

This pilot cannot itself change release code. Even a pass requires integration,
full canonical reproduce, compatible gallery re-embedding/index rebuild and
independent review. Stop if resource use exceeds 30 minutes CPU inference on
the local laptop, any input hash differs, a control fails, or the evaluator
cannot reproduce the known original-crop baseline. No GPU or unrelated worker
process may be used. Logs, commands, exit codes, result counts and runtime
versions go in the follow-up report. Images stay outside Git.
