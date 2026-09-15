"""Build REPORT.md from executed verification and examples, never guessed counts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

WITNESSES = {
    "reverse_sort": "test_H02_single_rank2_step",
    "no_camera_filter": "test_H06_remove_same_id_camera",
    "mean_all_queries": "test_H08_invalid_queries_excluded",
    "mean_weighted_positives": "test_H07_macro_queries_not_positives",
    "ap_divide_gallery": "test_H02_single_rank2_step",
    "ignore_junk": "test_junk_frame_and_explicit_exclusions",
    "cmc_is_recall": "test_H04_cmc_first_inp_last",
    "inp_first": "test_H04_cmc_first_inp_last",
    "ties_reverse": "test_H10_ties_column_order",
    "threshold_strict": "test_threshold_inclusive",
    "identity_as_presence": "test_H13_wrong_identity_not_tp",
    "pr_split_ties": "test_H14_pr_ties_are_threshold_groups",
    "pr_area_alias": "test_H14_pr_ties_are_threshold_groups",
    "rank5_off_by_one": "test_H05_rank5_boundary",
    "filtered_become_fn": "test_all_positives_filtered",
    "M1_keys_filter_mask": "test_keys_combined_with_every_filter",
    "M1b_keys_inverse_perm": "test_keys_three_cycle_and_permutations",
    "M2_pr_thresholds_sign": "test_distance_pr_curve_all_fields",
    "M3_top_score_sign": "test_distance_top_score_original_units",
    "M4_eligible_count": "test_audit_counts_and_one_based_ranks",
    "M5_relevant_pairs": "test_audit_counts_and_one_based_ranks",
    "M6_positive_ranks_0based": "test_audit_counts_and_one_based_ranks",
    "M7_num_relevant": "test_audit_counts_and_one_based_ranks",
    "M8_float32": "test_float64_near_float32_ranking",
    "M9_accepted_strict": "test_unknown_threshold_equality_all_modes",
}

INTRO = """# Доработка измерительного контура после независимой проверки

## Что изменено и почему

Замечания проверки подтверждены и закрыты изменениями проверок и интерфейса.
Исходная арифметика полной галереи сохранена. `evaluate()` и CLI теперь всегда
возвращают **`ranking_full_gallery`** и **`ranking_top_k`** одновременно, каждый
со своим полем `scope`. Для усечённого списка также указаны `k`, `filter_order`
и выбранные трактовки. Совместимое поле `ranking` содержит полную галерею,
что явно записано в `protocol.ranking_alias` и в самом объекте результата.

- `reid_metrics.py`: общая точка входа, явные переключатели, обе ветки отказа,
  протокол точности и CLI. Прежний расчёт вынесен в `_evaluate_full`.
- `scope_metrics.py`: определения метрик списков, обе ветки каждого спорного
  показателя и знаменатели; усечение не превращает запрос с пропущенным
  позитивом в невалидный.
- `test_metrics.py`, `oracle_checks.py`: проверки слепых зон и сверка всех
  прежних полей с **неизменённым** `review/oracle.py` на Fraction.
- `test_protocols.py`, `protocol_oracle.py`: проверки новых веток по отдельному
  дробному оракулу, перебору списков и случаям из определения метрик.
- `verify.py`, `mutation_specs.json`: прежние мутанты и все приложенные чужие
  мутанты, исходные файлы и внедрение в актуальную реализацию; настоящие ничьи
  против внешних реализаций, сравнение всего аудита и CLI.
- `report_examples.py`, `build_report.py`: численные примеры и этот отчёт
  воспроизводятся командами. `journal.md` хранит ход работы.

Исходные `review/`, `mutants/` и `sources/` сохранены. Новые варианты поломок и
логи лежат в `evidence/mutants_current/`. Сети и установки библиотек проверка
не требует.

## Запуск

Команды выполняются из каталога `job_05`:

```bash
python -B -m unittest -v test_metrics test_protocols
python -B verify.py
python -B report_examples.py
python -B build_report.py
```

Для расчёта по NPZ:

```bash
python -B reid_metrics.py example.npz \\
  --threshold 0.5 --camera-policy market --refusal-mode top1 \\
  --top-k 10 --rank-ks 1 5 12 --output evaluation.json
```

Обязательные массивы: `scores[Q,G]`, `query_ids[Q]`, `gallery_ids[G]`,
`query_cameras[Q]`, `gallery_cameras[G]`, `known_absent[Q]`.
`known_absent` означает отсутствие идентичности в **исходной** галерее;
противоречащая разметка отклоняется. Опциональны `gallery_keys[G]` (уникальные,
взаимно сравнимые ключи), `gallery_junk[G]`, `exclude_mask[Q,G]`, парные
`query_frames[Q]` и `gallery_frames[G]`. Маски и `known_absent` — boolean.
`include_rankings=True` в Python и CLI публикуют индексы кандидатов.

## Определения и переключатели

Все ветки вычисляются при любом выборе переключателя. Выбор меняет основные
поля выбранного результата; соседние именованные ветки остаются в JSON.
Python-аргументы перечислены ниже; у CLI вместо `_` используется `-`.

| Параметр | По умолчанию | Другая ветка |
|---|---|---|
| `top_k` | `10` | Любое положительное целое K |
| `ap_denominator` | `all_gallery_positives` | `retrieved_positives` |
| `query_average` | `valid_queries` | `all_queries_zero` |
| `filtered_positive_policy` | `drop` | `as_unknown` |
| `incomplete_inp` | `undefined_if_incomplete` | `zero_if_incomplete` |
| `rank_beyond_k` | `undefined` | `clamp_to_k` |
| `truncation_order` | `top_k_then_filter` | `filter_then_top_k` |

Эти значения записываются в `protocol`. Для AP доступны `step` и
`market_matlab`; обе формулы проверены. Их выбор сохранён в `ap_method`.
`rank_ks` задаёт требуемые положительные ранги; Rank-1 и Rank-5 присутствуют всегда.

### AP по усечённому списку

R — число всех допустимых позитивов в полной галерее **после оценочных
исключений**. H — число позитивов, фактически сохранившихся в усечённом списке.
Их ранги r₁…r_H отсчитываются с единицы после удаления исключённых кандидатов.

Для step числитель S = Σ(i/rᵢ), i=1…H. Тогда:

- `all_gallery_positives`: AP = S/R. Пропущенные позитивы снижают результат.
- `retrieved_positives`: AP = S/H. Это нормировка на найденные позитивы;
  при неполной выдаче она может быть выше, поэтому всегда подписана.

Для `market_matlab` каждое i/rᵢ заменяется полусуммой precision до попадания
и в момент попадания: ((i−1)/(rᵢ−1) + i/rᵢ)/2; precision до первой позиции
принимается равным единице, как в приложенном `compute_AP.m`.

При R>0, H=0 **AP=0 в обеих ветках**, и запрос остаётся в усреднении.
Это явная договорённость для пустой найденной релевантности, а не деление на ноль.
При R=0 AP отдельного запроса не определён; дальнейшее усреднение управляется
`query_average`. Знаменатель R невозможно восстановить из одного усечённого
списка без разметки полной галереи; этот контур располагает полной матрицей и truth.

Полная таблица AP-веток:
`ranking_top_k.mAP_by_ap_denominator_and_query_average`.
Аудит отдельного запроса: `num_relevant_gallery`, `num_relevant_retrieved`,
`positive_ranks`, `ap_by_denominator` в `per_query_top_k`.

### Усреднение mAP

`valid_queries` усредняет по запросам с R>0; список без найденного позитива
при R>0 даёт ноль, а не исключается. `all_queries_zero` делит сумму AP на
число **всех исходных запросов**, включая unknown и filtered_positive с нулевым
вкладом. Пустая выборка остаётся `null`. Обе цифры присутствуют в полной
галерее (`mAP_by_query_average`) и в каждой AP-ветке top-K.
`mAP_denominator` и `num_valid_queries` публикуются отдельно.

Переключатель относится именно к mAP. CMC и mINP усредняются по R>0, что
обозначено в `protocol.rank_and_inp_query_average`.

### mINP, если последнее совпадение отсутствует

Полный INP = R/r_R. При H=R он вычисляется по последнему позитиву списка.
При H<R список не раскрывает положение последнего позитива полной галереи:

- `undefined_if_incomplete`: INP = `null`; если хотя бы один валидный запрос
  имеет неполный список, **mINP всей этой выборки также `null`**.
- `zero_if_incomplete`: неполному списку назначается ноль и усреднение
  выполняется по всем валидным запросам.

Сложные запросы не удаляются для улучшения среднего. H/r_H для видимых
позитивов не выдаётся за полный INP. Выведены `num_incomplete_queries`,
`num_complete_queries` и `mINP_by_incomplete_policy`. При отсутствии валидных
запросов обе ветки mINP — `null`.

### Rank-k при k>K

Для k≤K это доля валидных запросов с первым позитивом не глубже k.
Для k>K `undefined` возвращает `null`; `clamp_to_k` возвращает результат
по имеющемуся префиксу Rank-min(k,K), то есть учитывает отсутствие кандидатов
после K. Последний показатель не оценивает скрытые позиции полной галереи.
Обе ветки — в `ranks_by_beyond_k_policy`; выбор `undefined` действует и при
галерее короче K, если запрошенный ранг больше K.

### Усечение и фильтрация не переставляются местами

`top_k_then_filter` сначала формирует сырой top-K по оценке и правилу ничьих,
затем применяет оценочные исключения; освободившиеся места **не дозаполняются**.
Это выбранная по умолчанию трактовка сдаваемого префикса.
`filter_then_top_k` сначала фильтрует полную галерею, затем берёт K допустимых
кандидатов. Обе ветки присутствуют в `ranking_top_k_by_filter_order` и
`per_query_top_k_by_filter_order`. Во второй порядок кандидатов зависит от truth
и потому может быть недоступен при формировании реальной сдачи.

Для первого варианта `submitted_ranking_before_filter` показывает исходные
индексы сдаваемого префикса, `ranking` — оставшиеся после оценочных исключений.
Детерминированные ничьи разрешаются ключами или порядком столбцов до усечения;
никакого добавления «ещё всех кандидатов с тем же score» сверх K нет.

### Отфильтрованные запросы в отказе

`drop` исключает filtered_positive из отказа. `as_unknown` требует отказа
для такого запроса: он входит в знаменатель TNR, а принятые допустимые кандидаты
становятся FP. Это распространяется на camera/junk/mask/frame-исключения,
а не только на камеру. Статус `per_query.status=filtered_positive` сохраняется
для аудита; `known_absent` исходной галереи не переписывается.

`refusal_by_filtered_positive_policy` содержит **весь** результат отказа и
счётчики обеих веток, включая PR-кривую, для выбранного presence/top1/pairwise.
`counts.unknown_queries` — знаменатель TNR выбранной ветки; в `as_unknown`
он уже включает filtered_positive. Диагностический `filtered_queries` при этом
сохраняется и не является ещё одним непересекающимся слагаемым общего числа.

Отказ здесь явно имеет область `protocol.refusal_scope=full_eligible_gallery`.
Top-K-результаты описывают ранжирование; pairwise PR полной галереи не
объявляется PR усечённого submission. Пороги distance и `top_score` остаются
в исходной шкале расстояний; равенство порогу означает принятие.

## Числа обеих веток

Таблица ниже автоматически перенесена из результата `python -B report_examples.py`.
Входы определены в `report_examples.py`; все поля результатов — в
[`evidence/examples.json`](evidence/examples.json).

{examples}

## Решение по float32

Выбрано **считать в float64, сохраняя точные ничьи уже представленных входов**.
`scores_from_embeddings` переводит координаты float32 в float64 **до**
нормировки, норм и скалярных произведений и возвращает float64. Это повышает
точность арифметики над имеющимися координатами; потерянные при создании
эмбеддинга разряды не восстанавливаются.

Готовые scores и threshold также переводятся в float64 без промежуточного
сужения. Различные значения не объявляются ничьёй по произвольному epsilon.
Если вызывающая сторона уже передала совпавшие после float32-округления scores,
это честная точная ничья с документированным порядком. По округлённому входу
невозможно узнать, какое из двух исходных значений было больше.

Воспроизводимый пример из float32-координат дал косинусы {cosines},
их разность {cosine_delta}; шаг float32 около 0.6 — {spacing}.
После явного округления этих косинусов они становятся {rounded_cosines}.
Значения mAP/TNR до и после такого округления приведены в таблице выше.

Это проверяют `test_float64_near_float32_ranking`,
`test_float64_near_float32_unknown_threshold` (оба направления меры),
`test_float32_scores_are_honest_exact_ties` и
`test_float32_embeddings_accumulate_in_float64`. Последний независимо считает
нормировку через Decimal, затем проверяет итоговый порядок, а не только dtype.
Протокол публикует `score_input_dtype`, `score_precision`, `threshold_precision`,
`tie_equality` и `embedding_accumulation`.

## Проверки, полученные запуском

Команда: `python -B verify.py`. Полный машиночитаемый результат:
[`verification_results.json`](verification_results.json), журнал:
[`evidence/verify.log`](evidence/verify.log).

{verification}

### Как проверялись ничьи с внешними реализациями

Исполняются функции `eval_market1501` из приложенных исходников fastreid и
torchreid; SHA-256 сверяются с `sources/manifest.json`. MATLAB/Cython не
исполняются. Для ничьих используются настоящие дискретные scores, оба направления
меры, ключи, junk и камеры. Global junk удаляется, столбцы сортируются по ключам
перед вызовом внешних функций, поскольку у них нет этих аргументов.

**Штатная внешняя сортировка не обещает наш стабильный порядок ничьих.** Поэтому:

- Неизменённые внешние функции запускаются на настоящих ничьих; их результат
  сверяется с Fraction-оракулом в их фактическом порядке `np.argsort`.
- Дополнительный явно обозначенный адаптер подменяет только default argsort
  на stable argsort в NumPy namespace внешней функции. Тело функции и формулы
  остаются исходными. Этот результат сверяется с нашим стабильным порядком,
  включая все запрошенные CMC-ранги проверочного набора.

Зафиксировано {tie_disagreements} различий mAP между штатным и стабильным
порядком внешних функций. Первый пример: trial={tie_trial}, {tie_source},
mAP штатного порядка {tie_native}, стабильного {tie_stable}.
Это опубликованное различие протоколов, а не замаскированное совпадение эталонов.
Воспроизведение входа: seed и trial в `differential_tied_external()`.

## Все обязательные мутанты: поломка → ловящая проверка

Команда: `python -B verify.py`. Каждый приложенный исходный файл запускается
с прежними и новыми регрессионными тестами без требований к новому API. Затем
**та же текстовая поломка** внедряется в актуальный код и запускается весь набор,
включая новые ветки. Невозможность внедрения — ошибка стенда; исключение или
отсутствие нового API **не засчитываются** как обнаружение. В таблице только
проверки, фактически завершившиеся assertion failure в **обоих** запусках.

{mutations}

`mean_all_queries` — поломка, молча меняющая знаменатель ветки valid_queries;
явная ветка all_queries_zero сама по себе допустима. Для `M8_float32` отдельно
падает и `test_float64_near_float32_unknown_threshold`.

Полные списки упавших/прошедших тестов, SHA-256 исполняемых вариантов и итог
каждого запуска: [`mutation_results.json`](mutation_results.json) и
[`mutation_matrix.md`](mutation_matrix.md). Подробные traceback:
`evidence/mutants_current/<name>.original.log` и `<name>.current.log`.

## Что осталось незакрытым

- **Непойманные обязательные мутанты: {survivors}.** Успех ограничен проверенными
  поломками и конфигурациями; это не доказательство отсутствия других ошибок.
- Организатор не подтвердил знаменатель mAP, судьбу filtered_positive,
  нормировку AP@K, обработку неполного INP и порядок фильтрации/усечения.
  Теперь все эти различия видны числами и переключателями; выбранные значения
  не объявляются официальным протоколом.
- `dataset-readme.md`, на который ссылается BRIEF.md, в выданном каталоге
  отсутствует; скрипт организатора и его CSV-схема также не предоставлены.
  Требование top-10 взято из BRIEF.md. Контур рассчитывает префикс матрицы и
  публикует точные индексы; побайтовая сверка с реальным submission.csv и
  соответствие ещё неизвестному оценочному скрипту здесь не подтверждены.
- Для расчёта нужны truth-идентичности, камеры и признаки отсутствия. Если
  организатор скрывает их в test, участник может валидировать собственный
  размеченный сплит, но не воспроизвести скрытую оценку по одному submission.
- Потерянную до входа в контур точность float32 восстановить нельзя. Порядок
  точных ничьих зависит от стабильных ключей/колонок; произвольная их перестановка
  без постоянных ключей может изменить метрику. Тождество косинусов на любых
  BLAS/CPU не заявляется; доказан проверенный локальный численный режим.
- Проверка масштабной production-матрицы, MATLAB и Cython не проводилась;
  измерения относятся к перечисленным локальным наборам. Git-коммитов нет:
  предоставленный scratch-каталог не является репозиторием.
"""


def main():
    verification = json.loads((ROOT / "verification_results.json").read_text())
    examples = json.loads((ROOT / "evidence/examples.json").read_text())
    mutants = json.loads((ROOT / "mutation_results.json").read_text())
    assert not verification["baseline"]["failed"] and not verification["baseline"]["errors"]
    assert set(WITNESSES) == {r["name"] for r in mutants}
    rows = ["| Поломка | Проверка (test_metrics.py) | Падающих тестов: исходный / новый |",
            "|---|---|---:|"]
    for record in mutants:
        witness = WITNESSES[record["name"]]
        for campaign in ("original", "current"):
            assert any(name.endswith("." + witness) for name in record[campaign]["failed"])
        rows.append(f"| `{record['name']}` — {record['description']} | `{witness}` | "
                    f"{len(record['original']['failed'])} / {len(record['current']['failed'])} |")
    v = verification
    native = v["differential_tied_external"]
    checks = [
        ("Среда", f"Python {v['python']}, NumPy {v['numpy']}"),
        ("Unit tests", f"{v['baseline']['tests']} PASS; failures/errors отсутствуют"),
        ("Исходные обязательные мутанты", f"{v['mutations']['original_killed']} / {v['mutations']['total']} обнаружено"),
        ("Те же поломки в новом коде", f"{v['mutations']['killed']} / {v['mutations']['total']} обнаружено; runtime errors = {v['mutations']['runtime_errors']}"),
        ("Все прежние поля vs независимый Fraction", f"{v['differential_all_fields']['trials']} конфигураций, расхождений {v['differential_all_fields']['mismatches']}"),
        ("Все новые поля и развилки vs Fraction", f"{v['differential_new_fields']['trials']} конфигураций, расхождений {v['differential_new_fields']['mismatches']}"),
        ("Внешние реализации: непрерывные scores", f"{v['differential_ranking']['trials']} наборов; max |Δ mAP| = {v['differential_ranking']['max_absolute_error']['fastreid_map']:.12g}"),
        ("Внешние реализации: настоящие ничьи", f"{native['trials']} наборов, {native['source_calls']} вызовов; max |Δ AP/INP| при одинаковом порядке = {native['max_absolute_metric_error']:.12g}"),
        ("Дополнительный дробный дифференциал отказа", f"{v['differential_refusal']['comparisons']} сравнений; max |Δ| = {v['differential_refusal']['max_absolute_error']:.12g}"),
        ("CLI", v["cli_and_comparator"]["cli"]),
        ("Случайный уровень mAP", f"{v['random_level']['metrics']['mAP']['observed']:.12g}, ожидание {v['random_level']['metrics']['mAP']['expected']:.12g}, z={v['random_level']['metrics']['mAP']['z']:.12g}"),
    ]
    table = "\n".join(["| Проверка | Результат |", "|---|---|", *[f"| {a} | {b} |" for a, b in checks]])
    example_text = (ROOT / "evidence/examples.md").read_text()
    example_table = example_text[example_text.index("| Вход |"):].rstrip()
    p = examples["precision"]
    first = native["first_tie_policy_difference"]
    report = INTRO.format(
        examples=example_table, cosines=str(p["cosine_float64"][0]), cosine_delta=f"{p['cosine_difference']:.12g}",
        spacing=f"{p['float32_spacing_at_point6']:.12g}", rounded_cosines=str(p["cosine_quantized_float32"][0]),
        verification=table, tie_disagreements=native["native_vs_stable_mAP_disagreements"],
        tie_trial=first["trial"], tie_source=first["source"], tie_native=f"{first['native_mAP']:.12g}",
        tie_stable=f"{first['stable_mAP']:.12g}", mutations="\n".join(rows),
        survivors=", ".join(v["mutations"]["survivors"]) or "нет")
    (ROOT / "REPORT.md").write_text(report)
    print(f"REPORT.md: {len(mutants)} mutant witnesses validated against both executed campaigns")


if __name__ == "__main__":
    main()
