# job_89 — исправление команд, диагностики и документации

Исходный снимок: `3e2cc9a`, ветка `main`. Все пять пунктов BRIEF выполнены в заданной области. Push не выполнялся.

**Полный новый прогон из изображений подтвердил исходные метрики точно: mAP `0.7740915539438481`, Rank-1 `0.7307692307692307`.** Косинусный mAP — `0.7316093422310448`, mAP@10 после переранжирования — `0.7684546226343101`. [Результат](evidence/metrics.json), [параметры и версии прогона](evidence/validation-run-info.json).

## Находка → исправление → проверка

В командах ниже `python` — Python проверенного runtime; cwd указан именем каталога внутри `04-solution`. Полные argv, cwd, коды завершения и stdout/stderr сохранены в [23 положительных запусках](evidence/checks/commands.json) и [9 проверках полного набора](evidence/full/commands.json). Контейнерные вычисления выполнены на `worker-vm`, без внешней сети, с лимитом 2 CPU / 4 GiB. Транспорт Docker/Compose не переаудировался целиком: для исправленных Python entry point использовался тот же runtime в Podman с актуальными исходниками.

| Находка | Что сделано | Команда проверки | Результат |
|---|---|---|---|
| C049: цепочка `eval/review` падает на `reid_metrics` | Выяснено, что архивные скрипты требуют отсутствующий исторический `run/` и прежний API. README явно подписывает архив и даёт рабочую проверку текущего контура | Из корня: `cd 04-solution/eval && python3 -B verify.py` | Exit 0; 60 тестов, 25/25 мутантов, независимые оракулы и CLI PASS. [Лог](evidence/eval-verify.txt) |
| C038/C040: извлечение baseline падает без CSV/JPG | Проверка выбранного CSV, фактически нужных изображений и модели до NumPy/ORT; `--limit` учтён | `baseline`: `python scripts/extract_embeddings.py --csv … --images-dir … --model … --out …`; выполнены все 8 validation-вариантов и оба test-вызова | Без входов exit 2 без traceback; с изображениями все 10 запусков exit 0. [Реестр](evidence/checks/commands.json) |
| C039: baseline eval падает на `.ids`/`.npy` | Собирается список всех необходимых векторов, ID и CSV; сообщение указывает шаг извлечения | `baseline`: `python scripts/run_eval.py` | Exit 2 без входов; exit 0 на полном наборе, весь `metrics_summary.json` совпал. [Лог](evidence/full/baseline-eval-full.txt) |
| C041: сборка baseline submission без входов | Проверяются CSV, `.ids` и векторы до вычислений и создания выходного каталога | `baseline`: `python scripts/make_submission.py --threshold 0.34921352213815304` | Exit 2 без входов; exit 0 с данными; три полных артефакта побайтово совпали. [Сверка](evidence/research-comparison.json) |
| C042: проверка порядка падает на отсутствующих artifacts | Проверяются артефакт, CSV, ID, модель и изображения выбранных контрольных позиций; сообщение объясняет предыдущий шаг | `baseline`: `python scripts/check_embeddings_order.py` | Exit 2 без файлов; полный положительный контроль exit 0, 24 позиции, `VERDICT=OK`, минимальный cos 1.0. [Лог](evidence/full/baseline-order-full.txt) |
| C053: postproc baseline без векторов/ID | Ранняя проверка входов с указанием шага копирования из baseline | `postproc`: `python scripts/s01_verify_baseline.py` | Exit 2 без входов; exit 0 на полном наборе, числа совпали. [Лог](evidence/full/postproc-baseline-full.txt) |
| C054/C055: соседние шаги postproc проверяли входы поздно/с кодом 1 | Проверки перенесены до ML-импортов; учитываются выбранные наборы, варианты и кэш | `postproc`: `python scripts/s03_extract.py --sets val --variants 208,208f,256`; `python scripts/s05b_make_tta_vectors.py --sets val` | Exit 2 без входов; положительные запуски exit 0. Сверка препроцессинга `max\|diff\|=0`; кэшированные варианты работают без изображений. [Извлечение](evidence/checks/postproc-extract.txt), [кэш](evidence/checks/postproc-cached-without-images.txt) |
| C056: финальная таблица postproc падает на первом `.npy` | Проверяются все используемые CSV и комбинации `208/TTA/TTA2` до импортов | `postproc`: `python scripts/s06_final_val.py` | Exit 2 без входов; все 8 конфигураций полного набора совпали с исходным `final_val.json`, кроме времени. [Лог](evidence/full/postproc-final-full.txt) |
| C022/C071/C072/C073/C077/I036: batch без данных | Общая stdlib-проверка CSV, изображений и файлов модели до ML-импортов; данные произвольного закрытого теста не привязаны к manifest организатора | `service`: `python -m app.batch --images-dir … --query … --gallery … --out-dir …` — default, `--no-rerank`, `REID_RERANK=0` | Без входов exit 2; три режима на реальных моделях exit 0. Все три выходных файла побайтово совпали с версией до правки; JPEG/PNG и имена без расширения приняты. [SHA-256](evidence/checks/before-after.json) |
| C069/I010/I044: loader не объясняет нехватку данных | Те же проверки выполняются до импорта Qdrant-клиента, подключения и инференса | `service`: `python -m app.load_gallery --gallery … --images-dir … --url http://localhost:6333 --collection job89-fixture` | Без входов exit 2; отдельный Qdrant принял 10/10 точек, exit 0. [Лог](evidence/loader-positive.txt) |
| Устаревший PDF и ссылка на исключённый PPTX | PDF пересобран из актуального Markdown; презентация обозначена как передаваемая отдельно. В §12 дана ссылка на существующий PDF | `python3 06-documentation/build_pdf.py`; `python3 04-solution/audit/docs-fixes/check_pdf_links.py` | 23 страницы; 167 URI-аннотаций, все 126 локальных целей существуют в Git, битых нет. Ссылки на исключённый PPTX нет. [Сборка](evidence/pdf-build.json), [ссылки](evidence/pdf-links.json) |
| Свежесть PDF раньше проверял только аудит | Добавлен `--check`, читающий SHA из самого PDF через `pdfinfo`; свежесть автоматически проверяется и после сборки | `python3 -B -S 06-documentation/build_pdf.py --check`; `python3 -B -m unittest discover -s 06-documentation -v` | Старый PDF отвергнут с кодом 2; новый принят с кодом 0. 3 теста: свежий, изменённый источник, отсутствующий/повреждённый PDF. [Старый](evidence/pdf-stale-before.txt), [новый](evidence/pdf-freshness.json), [тесты](evidence/pdf-tests.txt) |
| Ошибки округления | Исправлены 6 вхождений в 4 документах: два `0,664 → 0,663`, три `0,682 → 0,681`, одно `0.6547 → 0.6546` | `python3 04-solution/audit/docs-fixes/check_numbers.py` | 237 числовых утверждений в 15 документах сверены с 21 исходным JSON; до правок 6 ошибок, после 0. [До](evidence/numbers-before.json), [после](evidence/numbers.json) |
| C032/I029, два из трёх `НЕТОЧНО`: команда установки якобы гарантирует полный freeze | Разделены четыре закреплённые прямые зависимости и исторический снимок транзитивных; отмечено различие protobuf | `.venv/bin/pip install onnxruntime==1.30.0 numpy==2.5.3 pillow==12.3.0 onnx==1.22.0`; `.venv/bin/pip freeze --all` | Обе команды exit 0. Установка повторена без сети на уже имеющихся версиях (`PIP_NO_INDEX=1`); нового resolver-прогона здесь нет. [Установка](evidence/research-install.txt), [freeze](evidence/research-freeze.txt) |
| C081, третий `НЕТОЧНО`: буквальное обещание размеров `docker images` | Вывод подписан по движку/типу хранилища, целостность проверяется SHA архивов; учтены config ID и manifest ID | `docker images \| grep -E 'python\|qdrant'` после загрузки обоих архивов | Настоящий Docker 29.7.2, отдельный daemon без сети: exit 0; фактические ID/размеры записаны. [Вывод](evidence/docker-images.txt), [inspect](evidence/docker-image-inspect.txt) |
| Противоречивые статусы Docker и старые числа камеры рядом с d1_j48 | Удалены устаревшие заявления о непроверенном Docker, включая вводный абзац; контроли OSNet подписаны историческими. Уточнено различие параметров максимумов grid_val/train_fit | Сопоставление с исходным аудитом и `baseline/out/metrics_summary.json`, `postproc/out/grid_val_base.json` | Числа разных конфигураций и исторические результаты больше не выдаются за результат текущего d1_j48 |

## Метрики, тесты и отсутствие изменений алгоритма

- Новая полная валидация: **1110 query + 750 gallery**, все 1864 входных файла прошли проверку размеров/SHA; batch и `tools/eval_split.py` завершились с кодом 0. Время batch — 692,326 с на worker. [Входы](evidence/validation-inputs.json), [полный журнал](evidence/full-run.txt).
- mAP/Rank-1 совпали **точно**, не только до округления. Шкалы similarity/distance согласованы; параметры KR остались `[6,3,0.3]`, порог `0.5282812306342437`.
- **65 существовавших unit-тестов проходят:** 60 тестов метрик и 5 тестов `check_inputs`. Добавлены 5 регрессий CLI/preflight и 3 регрессии PDF; всего **73 unit-теста PASS**. Исходные тесты не изменены. `verify.py` дополнительно поймал 25/25 мутантов и прошёл дифференциальные проверки. [Метрики](evidence/eval-verify.txt), [входы](evidence/input-tests.txt), [PDF](evidence/pdf-tests.txt).
- 11 вариантов команд проверены через `python -S` на дереве без данных и site-packages: код 2, понятное сообщение, без traceback и выходных файлов. Дополнительно проверены отсутствующие отдельные изображения, `--limit` и поддерживаемые расширения. Положительные проверки используют настоящие ONNX-модели.
- 23 файла вычислительного ядра, evaluator, моделей, конфигурации, Compose и готовых сдаваемых артефактов побайтово соответствуют исходному коммиту. [Git blob-сверка](evidence/protected-files.json). Сохранённые JSON метрик также не редактировались.
- Для чисел дополнительно составлена [инвентаризация](evidence/decimal-inventory.json): 40 README/входных документов, 727 вхождений дробных чисел. Машинная проверка 237 утверждений привязывает каждое к конкретному JSON-полю, а не просто ищет похожее число; пороги разных прогонов и режимов сохранены с их подписями.
- Итоговый SHA-256 `SOLUTION.md` и Subject PDF: `dcc7ed8205403b43373f931d4919d93e0491c29090459b46fc135d699996f97c`. SHA-256 PDF: `3bd08e6e0d558cf553e5b397de080d8c572e44616536472d8e01975a8dbd062e`.

## Что осталось из исходного реестра

В пределах пяти пунктов BRIEF незавершённых исправлений нет. **Исходный реестр 173 команд не переписан:** это свидетельство аудита старого снимка. В нём ещё есть 12 сбоев вне перечисленных в BRIEF entry point:

| ID | Что осталось | Почему не менялось |
|---|---|---|
| C092–C094 | Абсолютные авторские пути в историческом `split/` | BRIEF выделяет `eval/review`, `baseline/`, `postproc/` и batch; перенос split-generator — отдельная работа |
| C104, C108–C112 | Диагностика отсутствующих тренировочных массивов/checkpoint, включая Windows-chain | Обучающие entry point не входят в заданную область; модель и конфигурация обучения сохранены |
| C030 | Inline-manifest изображений без каталога `images/` | Вспомогательный фрагмент вне выбранных CLI; для штатного воспроизведения уже есть `check_inputs.py` |
| I018 | Калибровка без предварительных validation-векторов | Отдельный исследовательский инструмент; полный validation/batch теперь проверен, алгоритм калибровки не менялся |
| I046 | bigres без `payload/manifest.json` | Исторический облачный эксперимент, его подготовка не входит в BRIEF |

C033 остаётся историческим ограничением стенда systemd/Docker из прежнего отчёта. Архивные `review/fuzz.py`, `mutants.py`, `demos.py`, `tz_checks.py` не объявлены рабочими standalone-командами: актуальный и выполненный рецепт — `verify.py`. Внешние HTTP-ссылки повторно по сети не проверялись; проверены все локальные цели PDF и отсутствие ссылок на исключённые файлы.

## Изменённые файлы и коммиты

Полный список — [changed-files.txt](changed-files.txt). Основные группы:

- `baseline/scripts/{inputs,extract_embeddings,run_eval,make_submission,check_embeddings_order}.py`;
- `postproc/scripts/{inputs,s01_verify_baseline,s03_extract,s05b_make_tta_vectors,s06_final_val}.py`;
- `service/app/{input_checks,batch,load_gallery}.py`, `reproduce/test_cli_inputs.py`;
- `eval/review/README.md`, `04-solution/README.md`, baseline README и review README, postproc README и REPORT, service README, offline README и wheels README;
- `SOLUTION.md`, `06-documentation/{build_pdf.py,test_build_pdf.py,SOLUTION.pdf}`;
- `audit/docs-fixes/`: этот отчёт, журнал, воспроизводимые проверочные скрипты, результаты и логи.

Логические коммиты: `ac87cba` (review), `835993d` (preflight), `e519dae` (числа/описания/PDF), `2c8a164` (документация зависимости pdfinfo), `1b85e70` (оставшиеся повторения и актуальный PDF). Отчёт и доказательства фиксируются отдельным завершающим коммитом. Все изменения локальные, **push не выполнялся**.

Рабочие стенды: `worker-vm:~/tmp/job_89` и локальный `~/tmp/job_89-docker`. Собственные временные контейнеры завершены, отдельный Docker daemon остановлен. Журнал — [journal.md](journal.md).
