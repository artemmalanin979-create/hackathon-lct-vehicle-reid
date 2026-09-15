# Воспроизводимые численные развилки

Команда: `python -B report_examples.py`. Полный вывод и входные эмбеддинги: `evidence/examples.json`.

| Вход | Величина | Ветка A | Ветка B |
|---|---|---|---|
| `rank12` | mAP, единственный позитив на ранге 12 | `full_gallery` = 0.0833333333333 | `top_k (K=10)` = 0 |
| `ap_denominator` | mAP, позитивы на рангах 2 и 4; K=2 | `all_gallery_positives` = 0.25 | `retrieved_positives` = 0.5 |
| `ap_denominator_matlab` | mAP, позитивы на рангах 2 и 4; K=2 | `all_gallery_positives` = 0.125 | `retrieved_positives` = 0.25 |
| `query_average` | mAP (full_gallery), AP=1 и unknown | `valid_queries` = 1 | `all_queries_zero` = 0.5 |
| `query_average` | mAP (top_k), AP=1 и unknown | `valid_queries` = 1 | `all_queries_zero` = 0.5 |
| `filtered_only` | TNR | `drop` = null (не определено) | `as_unknown` = 1 |
| `filtered_only` | знаменатель TNR | `drop` = 0 | `as_unknown` = 1 |
| `filtered_mixed` | TNR | `drop` = 0 | `as_unknown` = 0.5 |
| `filtered_mixed` | знаменатель TNR | `drop` = 1 | `as_unknown` = 2 |
| `incomplete_inp` | mINP, один полный и один неполный список | `undefined_if_incomplete` = null (не определено) | `zero_if_incomplete` = 0.5 |
| `rank_beyond_k` | Rank-5 при K=2 | `undefined` = null (не определено) | `clamp_to_k` = 1 |
| `filter_order` | mAP, K=2, лидер исключается камерой | `top_k_then_filter` = 0 | `filter_then_top_k` = 0.5 |
| `near_float32` | mAP, входные оценки 0.6 и 0.6+2e-8 | `float64` = 0.5 | `уже округлённый float32` = 1 |
| `threshold_float64` | TNR, score=0.5, threshold=0.5+6e-9 | `float64` = 1 | `явно округлённый порог` = 0 |
| `float32_embeddings` | mAP по косинусам float32-эмбеддингов | `накопление float64` = 0.5 | `косинусы округлены float32` = 1 |
