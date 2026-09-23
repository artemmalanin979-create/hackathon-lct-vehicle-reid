# job_93 — исправления F1 и F5

Дата: 23 сентября 2026, MSK. Репозиторий `/home/artem/projects/hackathon-lct-vehicle-reid`, ветка `main`. Исходный коммит `97dac47`; исправления `bce72e7` (F1) и `0b8a23d` (F5). Push не выполнялся.

## Итог

F1 устранён: нулевой знаменатель больше не порождает NaN; неконечный score/threshold отклоняется явно до сравнения/записи CSV. F5 обработан **контролируемым отказом всего некорректного набора до инференса**, без пропуска строк и без эмбеддинга чёрной подложки.

Полный новый прогон из **1860 оригинальных изображений** подтвердил точные метрики **mAP 0.7740915539438481 / Rank-1 0.7307692307692307**. Все **111 тестов** прошли (90 существующих + 21 новый). Перепроверены исходные краевые сценарии аудита; подробности и сравнение файлов ниже.

## Изменения и обоснование

| Место | Исправление |
|---|---|
| `04-solution/service/app/core/rerank.py` | Проверка конечности входных эмбеддингов. Только **строго нулевой** максимум столбца заменяется на 1 при нормировке: весь такой столбец равен нулю и остаётся нулевым. Epsilon, округление или изменение ненулевых дистанций не введены. |
| `app/core/ranking.py` | Общая проверка конечности score и threshold; сортировка и `accepted_candidates` больше не превращают NaN в отказ. Проверяется вся строка, включая кандидатов за пределами top-10. |
| `app/core/submission.py` | Полная матрица проверяется до открытия CSV на запись; NaN в поздней строке не оставляет частичный CSV и не затирает прежний файл. |
| `app/api/main.py` | Второй сервисный путь сравнения с порогом защищён тем же правилом: неконечный порог — HTTP 422; неконечный score из хранилища — HTTP 502, не обычный отказ. Тесты API используют контролируемый ответ хранилища, сеть/Qdrant не запускается. |
| `app/core/preprocess.py` | `validate_bbox` отклоняет нулевые/отрицательные размеры и bbox без пересечения с кадром. `crop_problems` полностью декодирует изображения, собирая проблемы обеих CSV с путём, ID и номером строки. Проверка повторяется в реальном чтении кропа. |
| `app/batch.py` | Неконечный threshold отклоняется до инференса. Ошибки изображений/геометрии — exit 2 до создания Embedder, с понятной диагностикой. Порча файла после проверки также даёт именованный контролируемый отказ. |

Модели, whitening, их SHA, алгоритм обычного rerank, k1/k2/λ, оба рабочих порога, `config.py`, `model.py`, входные CSV и исходные сдаваемые артефакты сохранены. [Хеши](evidence/unchanged-artifacts.json), [snapshot чистого дерева](evidence/snapshot.json).

**Почему выбран отказ всего набора.** По формату каждая строка `embeddings.npy` обязана соответствовать своей исходной позиции query → gallery. Пропуск одного кадра без нового согласованного формата нарушил бы эту связь; нулевой вектор или чёрный кроп выдали бы ошибку ввода за результат модели. Поэтому все изображения проверяются **до** дорогостоящей обработки валидных кадров: уже затраченная работа по ним не теряется. Пользователь исправляет указанный файл/bbox и повторяет команду. Частичный комплект не публикуется. Это не режим «пропустить битый кадр и продолжить».

Полное декодирование выполняется дважды (проверка и штатное чтение). В памяти — один кадр на стадии проверки; валидные кропы обрабатываются прежним PIL-конвейером. Частично выходящие bbox, крайний пиксель, узкие и широкие bbox сохраняют прежние пиксели. Настоящее полностью чёрное изображение остаётся допустимым: проверяется геометрия/читаемость, а не содержание снимка. CSV читается в строки один раз; pipe/FIFO не потребляются повторно.

## F1: команда и фактический вывод

Лёгкое численное воспроизведение (отдельно от полного прогона метрик):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B scripts/probe_f1.py
```

```text
before {"scores": "[[nan]]", "accepted": [], "warnings": ["invalid value encountered in divide"], "nan_score": [0], "nan_threshold": []}
after {"scores": "[[1.]]", "accepted": [0], "warnings": [], "nan_score": {"error": "scores содержат NaN или бесконечность"}, "nan_threshold": {"error": "порог threshold должен быть конечным (не NaN/Inf)"}}
```

[Полный вывод](evidence/f1-before-after.txt). Старый код взят из неизменённого snapshot аудита job_92; новый — из текущего репозитория. Единичные тесты дополнительно включают 20 одинаковых векторов, смешанный self-match, обе стороны порога и равные score. На обычных/смешанных данных результат побайтно равен исходной канонической реализации.

Искусственные fixture-JPEG из аудита прогнаны штатным CLI через обе настоящие ONNX-модели; это проверка поведения, не измерение качества ReID. Команда-пример внутри runtime worker:

```bash
cd /work/snapshot/repo/04-solution/service
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B -m app.batch --images-dir /work/fixtures/images \
  --query /work/fixtures/same_frame_01/query.csv \
  --gallery /work/fixtures/same_frame_01/gallery.csv \
  --out-dir /work/results/edge-cases/same_frame_01/r1/out --batch 32 --threads 2
```

| Сценарий JPEG | rc | score | Принято / отказов |
|---|---:|---:|---:|
| [same_frame_only](results/edge-cases/same_frame_only/r1/execution.json) | 0 | 0.7 | 1 / 0 |
| [same_frame_01](results/edge-cases/same_frame_01/r1/execution.json) | 0 | 1 | 1 / 0 |
| [same_frame_02](results/edge-cases/same_frame_02/r1/execution.json) | 0 | 1 | 1 / 0 |
| [same_frame_03](results/edge-cases/same_frame_03/r1/execution.json) | 0 | 0.7 | 1 / 0 |
| [same_frame_04](results/edge-cases/same_frame_04/r1/execution.json) | 0 | 1 | 1 / 0 |

В `same_frame_01`, `same_frame_02`, `same_frame_04` прежде были NaN, rc=0 и ложный отказ. После исправления нет предупреждения деления, score конечен, кандидат принят. Повторы каждого совпали побайтно по трём основным файлам. `same_frame_only` и `same_frame_03`, которые прежде возвращали конечный score, сохранены побайтно; ненулевые округлительные дистанции намеренно не менялись.

Инъекции NaN/±Inf в любом месте строки и в threshold явно отвергаются; writers проверяют матрицу до открытия файлов. В реальном CLI `--threshold nan` теперь exit 2 до инференса. Это неизбежно закрывает F8 из расширенного списка аудита как часть требования F1 «ни одного сравнения на NaN»; рабочие значения порогов не изменены.

## F5: команда и фактический вывод

```bash
cd /work/snapshot/repo/04-solution/service
python -B -m app.batch --images-dir /work/fixtures/images \
  --query /work/fixtures/corrupt_gallery/query.csv \
  --gallery /work/fixtures/corrupt_gallery/gallery.csv \
  --out-dir /work/results/edge-cases/corrupt_gallery/r1/out --batch 32 --threads 2
```

Код завершения: **2**. stderr:

```text
Входы не готовы: 1 проблем.
  /work/fixtures/corrupt_gallery/gallery.csv: строка 3, image_id=broken, файл /work/fixtures/images/broken.jpg: cannot identify image file '/work/fixtures/images/broken.jpg'
Исправьте изображения/bbox и повторите запуск; частичный комплект не создаётся, строки не пропускаются.
Вычисления не запущены.
```

| Вход | До (job_92) | После |
|---|---|---|
| [corrupt_query](results/edge-cases/corrupt_query/r1/execution.json) | rc=1, traceback, инференс мог начаться | rc=2, точная диагностика, новых файлов нет |
| [corrupt_gallery](results/edge-cases/corrupt_gallery/r1/execution.json) | rc=1, traceback, инференс мог начаться | rc=2, точная диагностика, новых файлов нет |
| [zero_width](results/edge-cases/zero_width/r1/execution.json) | rc=0, чёрный кроп и обычный эмбеддинг | rc=2, точная диагностика, новых файлов нет |
| [zero_height](results/edge-cases/zero_height/r1/execution.json) | rc=0, чёрный кроп и обычный эмбеддинг | rc=2, точная диагностика, новых файлов нет |
| [zero_both](results/edge-cases/zero_both/r1/execution.json) | rc=0, чёрный кроп и обычный эмбеддинг | rc=2, точная диагностика, новых файлов нет |
| [bbox_outside](results/edge-cases/bbox_outside/r1/execution.json) | rc=0, чёрный кроп и обычный эмбеддинг | rc=2, точная диагностика, новых файлов нет |
| [negative_width](results/edge-cases/negative_width/r1/execution.json) | rc=1, traceback, инференс мог начаться | rc=2, точная диагностика, новых файлов нет |
| [truncated_query](results/edge-cases/truncated_query/r1/execution.json) | Дополнительный контроль | rc=2, точная диагностика, новых файлов нет |
| [gallery_zero_width](results/edge-cases/gallery_zero_width/r1/execution.json) | Дополнительный контроль | rc=2, точная диагностика, новых файлов нет |
| [gallery_outside](results/edge-cases/gallery_outside/r1/execution.json) | Дополнительный контроль | rc=2, точная диагностика, новых файлов нет |
| [repaired_gallery](results/edge-cases/repaired_gallery/r1/execution.json) | Дополнительный контроль | rc=0; 3 query / 13 gallery, все строки сохранены |

Тесты подтверждают, что при повреждении query **или** gallery конструктор Embedder вообще не вызывается; ошибки двух CSV собираются за одну попытку. Усечённый JPEG отдельно проверен: `Image.open()` его принимает, полное декодирование отклоняет. После исправления одного повреждённого файла повтор того же входа проходит с сохранением всех строк — [прогон repaired_gallery](results/edge-cases/repaired_gallery/r1/execution.json).

При ошибке существующий out-dir не очищается: старый результат остаётся старым. Это прежняя граница F9, а не новый успешный комплект. Возврат 2 обязательно учитывать. Ошибка, возникшая после предварительной проверки из-за внешнего изменения входного файла, также не превращается в чёрный кроп; сохранение промежуточного инференса/возобновление не добавлялось.

## Полный прогон из изображений и точные метрики

SSH `worker-vm` был доступен; `Network is unreachable` не воспроизвелась. Тяжёлые вычисления шли только там: сохранённый runtime `localhost/job72-jury:6179b69`, Podman, **2 CPU / 4 GiB**, сеть `none`, исходники и изображения read-only. Python 3.13.15, NumPy 2.5.3, Pillow 12.3.0, ONNX Runtime 1.30.0. Snapshot 335 файлов совпал по SHA. Исходная долгоживущая SSH-сессия потеряла транспорт (`Connection reset by peer`), но runner продолжил работу: отдельное подключение подтвердило `ALL_CHECKS_COMPLETE` и `runner.exit=0`. В скрипт повторения добавлен SSH keepalive. Локально — чтение, лёгкие unit-проверки и четыре проверки документации, включая одностраничный render за 1.3 с; инференса не было.

Внутри контейнера:

```bash
python -B /work/snapshot/repo/04-solution/reproduce/run.py \
  --mode val --data-dir /data --out-dir /work/results/full-validation/out
python -B /work/snapshot/repo/04-solution/service/tools/eval_split.py \
  /work/results/full-validation/out/val/embeddings.npy
```

`run.py` сначала проверил размеры/SHA **1864 файлов** (1860 настоящих JPEG + четыре CSV сплита), затем вызвал изменённый штатный `app.batch` из изображений. Готовые векторы для извлечения признаков не использовались. Четыре граничные позиции эмбеддингов пересчитаны независимым инференсом. Последующий `eval_split.py` оценивает **service rerank**, а не только неизменённую исследовательскую копию формулы.

| Проверка | Получено |
|---|---|
| R-val / service eval | Оба rc=0 |
| Штатный batch | 1110 query + 750 gallery; embeddings [1860, 512] |
| Время batch / всей R-val | 769.59 с / 824.351 с |
| **mAP rerank** | **0.7740915539438481** — точно прежнее |
| **Rank-1 rerank** | **0.7307692307692307** — точно прежнее |
| Rank-5 / mAP@10 rerank | 0.8810096153846154 / 0.7684546226343101 |
| mAP / Rank-1 cosine | 0.7316093422310448 / 0.6850961538461539 |
| Проверка порядка независимым инференсом | 4 / 4, все совпали по критерию контура |
| Шкалы distance / score | Полное равенство метрик |

[Команда и код завершения](results/full-validation/execution.json), [полный stdout](results/full-validation/stdout.log), [входы с хешами](results/full-validation/out/inputs-val.json), [метрики/проверка порядка](results/full-validation/out/metrics.json), [числа именно service rerank](results/service-metrics/stdout.log). Точное равенство двум заданным числам проверено `assert ==`, не допуском.

## Перепроверка рабочих краевых сценариев

Для сравнения с job_92 сохранены те же fixture-изображения и CSV, runtime и ONNX threads=2; BLAS/OMP/MKL=2. В полном validation использовано рекомендованное окружение BLAS/OMP/MKL=1. Между этими режимами побайтовая идентичность не заявляется.

Команда проверки, включая реальные вызовы batch и независимую построчную сверку CSV с ранжированием:

```bash
python -B /work/scripts/verify_edges.py
```

| Сценарий | rc | Три основных файла против job_92 |
|---|---:|---|
| [small_queries](results/edge-cases/small_queries/r1/execution.json) | 0 | **Побайтно прежние** |
| [empty_gallery](results/edge-cases/empty_gallery/r1/execution.json) | 1 | Прежний отказ на пустом наборе; файлов нет |
| [empty_query](results/edge-cases/empty_query/r1/execution.json) | 1 | Прежний отказ на пустом наборе; файлов нет |
| [one_gallery](results/edge-cases/one_gallery/r1/execution.json) | 0 | **Побайтно прежние** |
| [duplicate_query_exact](results/edge-cases/duplicate_query_exact/r1/execution.json) | 0 | **Побайтно прежние** |
| [duplicate_query_bbox](results/edge-cases/duplicate_query_bbox/r1/execution.json) | 0 | **Побайтно прежние** |
| [duplicate_gallery_exact](results/edge-cases/duplicate_gallery_exact/r1/execution.json) | 0 | **Побайтно прежние** |
| [duplicate_gallery_bbox](results/edge-cases/duplicate_gallery_bbox/r1/execution.json) | 0 | **Побайтно прежние** |
| [bbox_boundary](results/edge-cases/bbox_boundary/r1/execution.json) | 0 | **Побайтно прежние** |
| [bbox_cross_boundary](results/edge-cases/bbox_cross_boundary/r1/execution.json) | 0 | **Побайтно прежние** |
| [bbox_outside](results/edge-cases/bbox_outside/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [zero_width](results/edge-cases/zero_width/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [zero_height](results/edge-cases/zero_height/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [zero_both](results/edge-cases/zero_both/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [negative_width](results/edge-cases/negative_width/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [corrupt_query](results/edge-cases/corrupt_query/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [corrupt_gallery](results/edge-cases/corrupt_gallery/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [wide_crop](results/edge-cases/wide_crop/r1/execution.json) | 0 | **Побайтно прежние** |
| [narrow_crop](results/edge-cases/narrow_crop/r1/execution.json) | 0 | **Побайтно прежние** |
| [same_frame_only](results/edge-cases/same_frame_only/r1/execution.json) | 0 | **Побайтно прежние** |
| [same_frame_mixed](results/edge-cases/same_frame_mixed/r1/execution.json) | 0 | **Побайтно прежние** |
| [more_than_ten](results/edge-cases/more_than_ten/r1/execution.json) | 0 | **Побайтно прежние** |
| [forced_refusal](results/edge-cases/forced_refusal/r1/execution.json) | 0 | **Побайтно прежние** |
| [threshold_nan](results/edge-cases/threshold_nan/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [same_frame_cosine](results/edge-cases/same_frame_cosine/r1/execution.json) | 0 | **Побайтно прежние** |
| [same_frame_01](results/edge-cases/same_frame_01/r1/execution.json) | 0 | Изменён только candidates.csv: устранён ложный отказ F1 |
| [same_frame_02](results/edge-cases/same_frame_02/r1/execution.json) | 0 | Изменён только candidates.csv: устранён ложный отказ F1 |
| [same_frame_03](results/edge-cases/same_frame_03/r1/execution.json) | 0 | **Побайтно прежние** |
| [same_frame_04](results/edge-cases/same_frame_04/r1/execution.json) | 0 | Изменён только candidates.csv: устранён ложный отказ F1 |
| [more_than_ten_threshold_zero](results/edge-cases/more_than_ten_threshold_zero/r1/execution.json) | 0 | **Побайтно прежние** |
| [threshold_equal_raw](results/edge-cases/threshold_equal_raw/r1/execution.json) | 0 | **Побайтно прежние** |
| [threshold_nextafter_above](results/edge-cases/threshold_nextafter_above/r1/execution.json) | 0 | **Побайтно прежние** |
| [supported_extensions](results/edge-cases/supported_extensions/r1/execution.json) | 0 | Дополнительный успешный контроль, построчная проверка пройдена |
| [truncated_query](results/edge-cases/truncated_query/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [gallery_zero_width](results/edge-cases/gallery_zero_width/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [gallery_outside](results/edge-cases/gallery_outside/r1/execution.json) | 2 | Ожидаемый ранний отказ F5/F1; файлов нет |
| [repaired_gallery](results/edge-cases/repaired_gallery/r1/execution.json) | 0 | Дополнительный успешный контроль, построчная проверка пройдена |

Всего **42 реальных запусков batch**, 37 разных сценария. Для пяти дополнительных сценариев выполнены два повтора. В каждом успешном прогоне проверены конечность эмбеддингов/score, порядок всех query и независимое построение всех строк submission/candidates.

«Побайтно прежние» означает совпадение `submission.csv`, `embeddings.npy`, `candidates.csv` с завершённым прогоном job_92, а не только одинаковые коды возврата. `run_info.json` содержит новые времена и не сравнивается целиком. Ни один удачный старый сценарий не выдан за исправление его известных проблем формата.

[Вся матрица и точные команды](results/edge-cases/summary.json), [stdout оркестратора](results/edges/stdout.log), [скрипт](scripts/verify_edges.py). Изменения ожидаемы только для F1, F5 и неконечного threshold как части защиты F1.

## Все тесты

| Набор | Команда (из указанного каталога) | Результат |
|---|---|---:|
| service | `04-solution/service`: `python -B -m unittest discover -s tests -v` | [21/21 OK](results/tests/service/stderr.log) |
| metrics | `04-solution/eval`: `python -B -m unittest -v test_metrics test_protocols` | [60/60 OK](results/tests/metrics/stderr.log) |
| inputs | `04-solution/reproduce`: `python -B -m unittest -v test_check_inputs test_cli_inputs` | [16/16 OK](results/tests/inputs/stderr.log) |
| numeric-audit | `04-solution/audit/docs-fixes`: `python -B -m unittest -v test_check_numbers` | [7/7 OK](results/tests/numeric-audit/stderr.log) |
| runtime-cli | `04-solution/reproduce`: `python -B -m unittest -v check_cli_runtime` | [3/3 OK](results/tests/runtime-cli/stderr.log) |
| documentation | `06-documentation`: `python3 -B -m unittest -v test_build_pdf check_build_pdf_render` | [4/4 OK](evidence/documentation-tests.txt) |

Итого **111/111**, без пропусков. Новые 21 тест: `service/tests/test_f1.py` (9), `test_api_scores.py` (3), `test_f5.py` (9). Существующие 90 тестов — весь текущий unittest-набор и обе явно запускаемые runtime-проверки. Исторические исследования/мутационные генераторы не названы дополнительными unit-тестами.

Новые тесты до исправления показали ошибки: [исходный красный прогон](evidence/new-tests-before.txt). Проверки NaN и bbox реально различают старое и новое поведение. Тесты с Mock проверяют только отсутствие вызова модели при неверном входе или обработку ответа хранилища; **метрики, CLI-аудит и полная валидация используют настоящие модели**.

## Известные границы, оставленные по заданию

- **F2:** галерея короче 10 по-прежнему даёт короткую строку submission при фиксированном заголовке; новых/фиктивных ID не добавляли.
- **F3:** пустой query/gallery по-прежнему exit 1; новый формат пустого результата не вводили.
- **F4:** дубликаты image_id/разные bbox одного кадра сохраняют прежнюю неоднозначность candidates; объектную идентичность не меняли.
- **F6:** confidence по-прежнему сериализуется до 6 знаков, сравнение идёт с полным score. Проверены точное равенство порогу и следующий float, результаты прежние.
- **F7:** candidates по-прежнему содержит всех прошедших порог (в проверке по 15 на запрос), submission — до 10.
- В расширенном аудите также были **F9/F10**: старый out-dir при ошибке и последние биты между разными вычислительными окружениями. Их исправление в эту работу не включено.

Восстановить тот же стенд и повторить проверки можно через [scripts/run_worker.sh](scripts/run_worker.sh); для сравнений нужны сохранённые fixtures/результаты job_92 на worker и оригинальные данные организатора `/home/fedora/lct-reid/data`. Полный ход — [journal.md](journal.md). Системные настройки, чужие процессы/VM и сеть оборудования не изменялись. Рабочее дерево по окончании чистое; два локальных коммита, без push.
