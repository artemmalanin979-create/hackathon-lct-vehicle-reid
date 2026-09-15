"""Reproduce every interpretation/precision example: python -B report_examples.py."""
import json
from pathlib import Path

import numpy as np

from reid_metrics import evaluate, scores_from_embeddings, _json_safe

ROOT = Path(__file__).resolve().parent


def run(scores, qids, gids, **kw):
    raw = np.asarray(scores)
    options = dict(threshold=.5, camera_policy="market", refusal_mode="top1", include_rankings=True)
    options.update(kw)
    qcams = options.pop("qcams", [0] * len(qids))
    gcams = options.pop("gcams", [1] * len(gids))
    return evaluate(raw, qids, gids, qcams, gcams, [q not in gids for q in qids], **options)


def main():
    cases = {
        "rank12": run([list(range(12, 0, -1))], [1], [2] * 11 + [1], rank_ks=[1, 5, 12]),
        "ap_denominator": run([[4, 3, 2, 1]], [1], [2, 1, 2, 1], top_k=2),
        "ap_denominator_matlab": run([[4, 3, 2, 1]], [1], [2, 1, 2, 1], top_k=2, ap_method="market_matlab"),
        "query_average": run([[1], [1]], [1, 9], [1]),
        "filtered_only": run([[1]], [1], [1], gcams=[0]),
        "filtered_mixed": run([[.1, .9], [.9, .1]], [9, 1], [1, 2], gcams=[0, 1]),
        "incomplete_inp": run([[2, 1], [1, 2]], [1, 1], [1, 2], top_k=1),
        "rank_beyond_k": run([[2, 1]], [1], [1, 2], top_k=2),
        "filter_order": run([[9, 8, 1]], [1], [1, 2, 1], gcams=[0, 1, 1], top_k=2),
        "near_float32": run([[.6, .6 + 2e-8]], [1], [1, 2]),
        "already_quantized_float32": run(np.array([[.6, .6 + 2e-8]], dtype=np.float32), [1], [1, 2]),
        "threshold_float64": run([[.5]], [9], [1], threshold=.5 + 6e-9),
        "threshold_explicitly_quantized_float32": run(np.array([[.5]], dtype=np.float32), [9], [1],
                                                       threshold=np.float32(.5 + 6e-9)),
    }
    q = np.array([[1., 0.]], dtype=np.float32)
    g = np.array([[.6, .8], [.6, np.nextafter(np.float32(.8), np.float32(0))]], dtype=np.float32)
    cosine = scores_from_embeddings(q, g)
    cases["float32_embeddings_float64_cosine"] = run(cosine, [1], [1, 2])
    cases["float32_embeddings_quantized_cosine"] = run(cosine.astype(np.float32), [1], [1, 2])
    metadata = dict(query_embeddings=q.tolist(), gallery_embeddings=g.tolist(),
                    cosine_float64=cosine.tolist(), cosine_quantized_float32=cosine.astype(np.float32).tolist(),
                    cosine_difference=float(cosine[0, 1] - cosine[0, 0]),
                    float32_spacing_at_point6=float(np.spacing(np.float32(.6))))
    directory = ROOT / "evidence"
    directory.mkdir(exist_ok=True)
    (directory / "examples.json").write_text(json.dumps(_json_safe(dict(cases=cases, precision=metadata)),
                                                        indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def number(value):
        return "null (не определено)" if value is None else f"{value:.12g}"

    rows = []
    def add(case, measure, left_label, left, right_label, right):
        rows.append(f"| `{case}` | {measure} | `{left_label}` = {number(left)} | `{right_label}` = {number(right)} |")

    c = cases["rank12"]
    add("rank12", "mAP, единственный позитив на ранге 12", "full_gallery", c["ranking_full_gallery"]["mAP"],
        "top_k (K=10)", c["ranking_top_k"]["mAP"])
    for name in ["ap_denominator", "ap_denominator_matlab"]:
        c = cases[name]["ranking_top_k"]["mAP_by_ap_denominator_and_query_average"]
        add(name, "mAP, позитивы на рангах 2 и 4; K=2", "all_gallery_positives", c["all_gallery_positives"]["valid_queries"],
            "retrieved_positives", c["retrieved_positives"]["valid_queries"])
    for scope in ["ranking_full_gallery", "ranking_top_k"]:
        c = cases["query_average"][scope]
        branches = (c["mAP_by_query_average"] if scope == "ranking_full_gallery" else
                    c["mAP_by_ap_denominator_and_query_average"]["all_gallery_positives"])
        add("query_average", f"mAP ({c['scope']}), AP=1 и unknown", "valid_queries", branches["valid_queries"],
            "all_queries_zero", branches["all_queries_zero"])
    for name in ["filtered_only", "filtered_mixed"]:
        b = cases[name]["refusal_by_filtered_positive_policy"]
        add(name, "TNR", "drop", b["drop"]["refusal"]["tnr"], "as_unknown", b["as_unknown"]["refusal"]["tnr"])
        add(name, "знаменатель TNR", "drop", b["drop"]["counts"]["unknown_queries"],
            "as_unknown", b["as_unknown"]["counts"]["unknown_queries"])
    c = cases["incomplete_inp"]["ranking_top_k"]["mINP_by_incomplete_policy"]
    add("incomplete_inp", "mINP, один полный и один неполный список", "undefined_if_incomplete", c["undefined_if_incomplete"],
        "zero_if_incomplete", c["zero_if_incomplete"])
    c = cases["rank_beyond_k"]["ranking_top_k"]["ranks_by_beyond_k_policy"]
    add("rank_beyond_k", "Rank-5 при K=2", "undefined", c["undefined"]["Rank-5"],
        "clamp_to_k", c["clamp_to_k"]["Rank-5"])
    c = cases["filter_order"]["ranking_top_k_by_filter_order"]
    add("filter_order", "mAP, K=2, лидер исключается камерой", "top_k_then_filter", c["top_k_then_filter"]["mAP"],
        "filter_then_top_k", c["filter_then_top_k"]["mAP"])
    add("near_float32", "mAP, входные оценки 0.6 и 0.6+2e-8", "float64", cases["near_float32"]["ranking"]["mAP"],
        "уже округлённый float32", cases["already_quantized_float32"]["ranking"]["mAP"])
    add("threshold_float64", "TNR, score=0.5, threshold=0.5+6e-9", "float64", cases["threshold_float64"]["refusal"]["tnr"],
        "явно округлённый порог", cases["threshold_explicitly_quantized_float32"]["refusal"]["tnr"])
    add("float32_embeddings", "mAP по косинусам float32-эмбеддингов", "накопление float64", cases["float32_embeddings_float64_cosine"]["ranking"]["mAP"],
        "косинусы округлены float32", cases["float32_embeddings_quantized_cosine"]["ranking"]["mAP"])
    table = "\n".join(["| Вход | Величина | Ветка A | Ветка B |", "|---|---|---|---|", *rows])
    (directory / "examples.md").write_text("# Воспроизводимые численные развилки\n\nКоманда: `python -B report_examples.py`. "
                                            "Полный вывод и входные эмбеддинги: `evidence/examples.json`.\n\n" + table + "\n")
    print(table)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
