# ±3% bbox margin: negative tune pilot

2026-09-28. The isolated pilot started from release source `a0404df`.
The [protocol](PROTOCOL.md) was committed before inference (`acfa30c` in the
pilot branch, `a2df3fe` after integration into `main`); execution code and a
wording correction followed before inference (`ab35262`, then `cc26700` in
`main`). No release file, evaluator, weight, gallery or threshold was changed.

**Decision: stop at tune.** Neither fixed variant reached the predeclared
+0.015 KR-mAP tune gate, so there was **no val inference, no threshold fit,
no batch-1 latency comparison and no promotion**. The released `d1_j48`
remains the candidate. This closes the previously untested deterministic crop
idea from `04-solution/postproc/whitening/REPORT.md` §4 for these exact ±3%
variants; it does not exclude every possible crop policy.

| Crop on query and gallery | Cosine mAP | KR mAP | Δ KR mAP vs original | paired 95% CI by vehicle ID | Rank-1 KR |
|---|---:|---:|---:|---:|---:|
| Original CSV bbox | 0.788945 | 0.810447 | — | — | 0.748798 |
| Inward 3% per side | 0.766372 | 0.793618 | −0.016828 | [−0.029290; −0.005447] | 0.729567 |
| Outward 3% per side | 0.781068 | 0.810216 | −0.000230 | [−0.012327; +0.012306] | 0.759615 |

These are **train-fit tune identities**, not the 1110×750 released validation
score and not an external test. The table uses 1110 tune queries × 732 tune
gallery, 832 queries with an eligible cross-camera positive, full-gallery
`market` mAP, released cosine/KR (6,3,0.3) code, 2000 fixed-seed paired
vehicle-ID bootstrap replicates. The tune identities helped fit the model and
whitening; these numbers can be optimistic. The LCT val has already been used
many times, and was deliberately not opened for this failed pilot.

Inputs were checked before inference: all 1842 tune images exist, the four
predeclared tune/val CSVs and associated image sets have the SHA-256 values in
the protocol, and the three release weight hashes match `app/core/config.py`.
The actual tune script only opens the two tune CSVs and their images. All
variants used the same RGB/PIL bilinear 208 preprocessing and the same
image→512D ensemble/whitening. `original` was cross-checked against the
release `load_crop` on a real frame. No images or feature arrays entered Git.

Controls through the unchanged evaluator passed: label-perfect scores gave
mAP **1.000000** on all 832 valid queries; independent random 512D vectors
(seed 20260928) gave **0.009255**. The repeated run exactly reproduced the
first interrupted run's original and inward mAP values. The first tool shell
lost its output pipe and ended 143 before producing JSON; it is **not** an
additional experimental attempt or a model failure. Its partial log is
[`tune-output-interrupted.txt`](tune-output-interrupted.txt).

Execution: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10
timeout 1800s /home/artem/projects/hackathon-lct-vehicle-reid/.venv/bin/python
-u 04-solution/training/bbox-margin-20260928/run_tune.py`, with stdout/stderr
redirected to [`tune-output.txt`](tune-output.txt). The completed run wrote
[`tune_results.json`](tune_results.json) in 465.56 s for extraction, ranking
and summary after input checks; OS: Fedora, CPU Intel i5-12450H, one ORT/BLAS
thread. Runtime: Python 3.13.13, NumPy 2.5.3, Pillow 12.3.0, ONNX Runtime
1.30.0. Because this was detached from the shell after the first pipe failure,
its **process exit code was not captured**; the final JSON and terminal line
were written and the process exited. The separate JSON/hash/arithmetic
verification returned exit **0**, 1/1 passed, 0 failed, 0 skipped:
[`verify-output.txt`](verify-output.txt), [`verify-exit.txt`](verify-exit.txt).
No timeout, OOM or evaluator skip occurred in the completed run.

Artifact SHA-256: protocol `971a68b7e24e3056d090e7c8f4d61623eb30f462e44ecf5467f7d0385eb1bc6f`,
script `bb8f2c58bf2c1269a927cb5af37d4a85c5df8f9f6de98f6fca049e79bfe36f5e`,
results `be71861245523b058496c77fa71cb62e08caa3c380eedcb27e16c68a9469974a`.

Status: tune controls **PASS**; both crop candidates **FAIL the selection
gate**; val/threshold/latency/release acceptance **NOT RUN** by the frozen
stopping rule. No further bbox percentages or post-hoc variants are planned.
