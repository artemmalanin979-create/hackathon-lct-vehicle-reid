# job_91 — исправление регрессий и закрытие выживших мутаций

Все пункты BRIEF выполнены. **90 тестов прошли без пропусков, 8/8 выживших мутаций пойманы.**
Новый полный инференс из изображений дал **mAP 0.7740915539438481 / Rank-1 0.7307692307692307**.
Работа начата на `main/95d228b`, итоговый HEAD — `1eedc0b8d3d1c4f572da8947b23f9ebf62ba1a1e`. Рабочее дерево чистое. Push не выполнялся.

## Находки, исправления и проверки

Команды suites запускаются из указанного каталога репозитория. Пути `scripts/*.py` и `evidence/`
относятся к рабочему каталогу `/home/artem/tmp/lct-jobs/job_91`. Полные argv, cwd, коды и длительности
worker-команд записаны в [commands.jsonl](evidence/runtime/commands.jsonl).

| Находка | Исправление | Команда проверки | Результат и вывод |
|---|---|---|---|
| F1: pipe/FIFO отвергался как отсутствующий файл; повторное чтение опустошало поток | `require_files` различает отсутствующий путь, каталог и обычный файл через `stat`. Существующий поток не открывается preflight. `require_dataset` читает заранее только обычные CSV; поток один раз читает настоящий загрузчик | Из `04-solution/reproduce`: `python -B -S -m unittest -v test_cli_inputs.CliInputTests.test_stream_preflight_does_not_consume_or_open_csv`; в runtime: `python -B -m unittest -v check_cli_runtime.RuntimeCliTests.test_batch_pipe_and_fifo_match_regular_csv` | Оба теста `ok`. FIFO проверяется также до появления writer. Настоящий batch с обычным CSV, Bash process substitution и FIFO вернул 0; все три выходных файла совпали побайтно. [Лёгкие тесты](evidence/runtime/cli-tests.log), [ONNX-тесты](evidence/runtime/runtime-tests.log) |
| F2: `python -m` не находил `inputs` | У baseline extraction и s03 введён относительный импорт при наличии `__package__`; у s03 так же исправлены ленивые импорты `common`. Прямой запуск файла сохранён | Из `04-solution/reproduce`: `python -B -m unittest -v check_cli_runtime.RuntimeCliTests.test_baseline_module_and_direct_match_with_limit check_cli_runtime.RuntimeCliTests.test_cached_postproc_module_and_direct_without_images` | Оба теста `ok`. Baseline через модуль с `--limit 1` и прямой запуск из другого cwd с `--limit -1` дали идентичные `.npy`; неиспользуемое изображение отсутствовало. Кэшированный s03 прошёл обоими способами без каталога исходных изображений. [Вывод](evidence/runtime/runtime-tests.log) |
| Восемь неразличаемых штатными тестами поломок | Проверки отдельных отсутствующих входов, каталога, готового кэша и гонки изменения Markdown при настоящем render | `python -B scripts/check_mutations.py --repo /home/artem/projects/hackathon-lct-vehicle-reid --out evidence/mutations --only M05 M06 M07 M11 M12 M17 M18`; M10 — тот же script в runtime worker с `--only M10` | Все восемь: контроль exit 0, мутант exit 1 тестового suite. [Локальный вывод](evidence/mutations-local.log), [полная таблица](evidence/mutations/all-results.json), подробности ниже |
| Два неверных `0,0063` в training README, строки 28 и 30 | Оба исправлены на `0,0064`; первичный JSON и расчёт не менялись | Из корня repo: `python -B 04-solution/audit/docs-fixes/check_numbers.py` | Exit 0, `checks: 305`, `errors: []`. [Результат](evidence/numbers-after.json) |
| Прежняя сверка исключала прозаические повторы и дельты | Семантические правила для всех повторов в прозе/списках, Decimal из первоисточников, инвентаризация каждого дробного/экспоненциального токена и явное непроверенное покрытие | `python -B scripts/recheck_old_roundings.py`; из корня repo: `python -B -S -m unittest discover -v -s 04-solution/audit/docs-fixes` | Финальный checker на прежнем README: exit 1, ровно две ошибки из 305 утверждений. Метод: `Ran 7 tests ... OK`. [Красная проверка](evidence/numbers-regression.log), [тесты метода](evidence/number-tests.log) |

F1 сознательно отдаёт проверку читаемости потокового устройства самому загрузчику: отдельное открытие
FIFO может заблокироваться или забрать единственного writer. Поэтому для потокового CSV не гарантируется
ранняя диагностика отсутствующих изображений; сохраняется работавший ввод и применяется принцип BRIEF
«ложное срабатывание хуже отсутствия проверки». Каталоги по-прежнему отклоняются.

## Восемь выживших мутаций

Каждая исходная мутация критика внесена **отдельно** в копию текущего кода. Перед ней выполняется
тот же новый тест на немутированной копии. Все восемь мутаций содержательно вредны; безобидных исключений нет.

| Выжившая мутация | Новый тест | Ловит |
|---|---|---|
| M05 — Проверять только первый CSV | `test_cli_inputs.CliInputTests.test_missing_second_csv_is_reported` | Да: контроль 0, мутант 1. [Контроль](evidence/mutations/M05/control.log), [мутация](evidence/mutations/M05/mutant.log), [патч](evidence/mutations/M05/mutation.patch) |
| M06 — Пропустить MODEL2 в batch | `test_cli_inputs.CliInputTests.test_each_service_weight_is_required_before_ml_import` | Да: контроль 0, мутант 1. [Контроль](evidence/mutations/M06/control.log), [мутация](evidence/mutations/M06/mutant.log), [патч](evidence/mutations/M06/mutation.patch) |
| M07 — Пропустить whitening в batch | `test_cli_inputs.CliInputTests.test_each_service_weight_is_required_before_ml_import` | Да: контроль 0, мутант 1. [Контроль](evidence/mutations/M07/control.log), [мутация](evidence/mutations/M07/mutant.log), [патч](evidence/mutations/M07/mutation.patch) |
| M10 — Требовать изображения при готовом кэше | `check_cli_runtime.RuntimeCliTests.test_cached_postproc_module_and_direct_without_images` | Да: контроль 0, мутант 1. [Контроль](evidence/mutations/M10/control.log), [мутация](evidence/mutations/M10/mutant.log), [патч](evidence/mutations/M10/mutation.patch) |
| M11 — Пропустить обязательный TTA2 | `test_cli_inputs.CliInputTests.test_each_tta2_part_is_required_before_ml_import` | Да: контроль 0, мутант 1. [Контроль](evidence/mutations/M11/control.log), [мутация](evidence/mutations/M11/mutant.log), [патч](evidence/mutations/M11/mutation.patch) |
| M12 — Принять каталог вместо файла | `test_cli_inputs.CliInputTests.test_directory_is_not_an_input_file` | Да: контроль 0, мутант 1. [Контроль](evidence/mutations/M12/control.log), [мутация](evidence/mutations/M12/mutant.log), [патч](evidence/mutations/M12/mutation.patch) |
| M17 — Убрать проверку свежести после render | `check_build_pdf_render.PdfRenderTests.test_source_change_during_render_is_rejected` | Да: контроль 0, мутант 1. [Контроль](evidence/mutations/M17/control.log), [мутация](evidence/mutations/M17/mutant.log), [патч](evidence/mutations/M17/mutation.patch) |
| M18 — Пропустить MODEL2 в loader | `test_cli_inputs.CliInputTests.test_each_service_weight_is_required_before_ml_import` | Да: контроль 0, мутант 1. [Контроль](evidence/mutations/M18/control.log), [мутация](evidence/mutations/M18/mutant.log), [патч](evidence/mutations/M18/mutation.patch) |

Для M05 раньше проверялись только полностью отсутствующие входы. M06/M07/M18 теперь получают все входы,
кроме одного заданного файла весов; проверяются отдельно обе модели и whitening в batch и loader.
M11 проверяет отсутствие каждого из двух TTA2-массивов по очереди. Нужен именно exit **2**, диагностическое
сообщение о конкретном пути и отсутствие traceback до ML-импортов; произвольное падение не считается успехом.

M10 проверяет настоящий запуск s03 с двухстрочными `.npy`-кэшами, CSV и реальной ONNX-сессией:
контроль возвращает 0, мутант требует изображения и возвращает 2. Проверяются оба способа запуска и
неизменность файлов кэша. M12 требует отклонить каталог. Для M17 WeasyPrint действительно формирует документ,
после `render()` Markdown меняется; отдельный `pdfinfo` подтверждает устаревший SHA в PDF.
Исправный build возвращает 2, мутант — 0. Нормальная сборка без изменения источника тоже проходит.

## Почему числа были пропущены и что изменилось в методе

Указанные в BRIEF **231** утверждение — ранний объём проверки. Финальный checker автора проверял
**237**: это подтверждено его новым запуском до наших изменений ([вывод](evidence/numbers-before.json)).
Он возвращал 0 при наличии обеих ошибок.

Причина — выбор вручную перечисленных **табличных строк**, плюс одно специальное правило для строки
baseline README. Training README не входил в карту источников. Два ошибочных значения — повторённый
**прирост/дельта в абзаце и пункте списка**, а не значения из выбранной таблицы. До округляющей формулы
они вообще не доходили. В третьем повторе рядом уже стояло правильное число.

Исходник: `04-solution/training/attempt-2/out/boot_ainv2.json#ainv2rr_minus_osnetrr/delta`
= `0.006361571811840161`, округление до четырёх знаков — `0.0064`.
Разность исходных mAP `0.7000200442705277 - 0.6936584724586875` даёт то же округление.

Новый метод:

1. Привязывает числа к смысловому окружению, а не к строке `0,0063` или предполагаемому правильному значению.
   Проверяет все совпадения, включая повторение внутри одной строки. Исчезновение привязки вызывает ошибку.
2. Читает неокруглённые JSON-числа как `Decimal`, вычисляет производные разности до округления и явно
   использует `ROUND_HALF_EVEN`. Перенесены дополнительные независимые случаи критика: итого **305** привязок.
3. Обходит **48** отслеживаемых README/входных документов, включая
   `README-репозитория.md`: **1591** дробных/экспоненциальных вхождений
   с координатой «файл/строка/колонка», типом текста и статусом `checked` либо `unbound`.
   Поддерживаются знак, запятая/точка, значения больше единицы, экспонента и точка конца предложения.
4. Отдельно показывает **1286** непривязанных вхождений. Это в том числе
   параметры, версии, времена и исторические метрики; они **не объявлены проверенными**.
   `python -B 04-solution/audit/docs-fixes/check_numbers.py --require-complete` возвращает ожидаемый **2**
   с `coverage.complete: false` ([вывод](evidence/numbers-complete.json)). Успешная выборка больше
   не маскируется под доказательство всех чисел документации.

Семь тестов метода проверяют оба старых повтора, другое произвольное ошибочное значение, правильные повторы,
исчезнувшую привязку, арифметику сырых дельт, расширенную лексику и отказ объявлять неполный охват полным.
[Финальный checker на прежнем README](evidence/numbers-before-final-method.json): **305 проверок, 2 ошибки**.
[На исправленном README](evidence/numbers-after.json): **305 проверок, 0 ошибок**.

## Полный прогон и неизменность метрик

Worker отвечал по SSH; `Network is unreachable` не наблюдался. Использован отдельный каталог
`worker-vm:/home/fedora/tmp/job_91`, runtime-образ `localhost/reid-d1j48:clean`.
Контейнеру выделено **2 CPU / 4 GiB**, сеть отключена, checkout и данные смонтированы read-only.
[Фактические cgroup-лимиты](evidence/resource-limits.log): `cpu.max = 200000 100000`,
`memory.max = 4294967296`, `network=none`.

Перед запуском проверены SHA-256 всех **175 файлов снимка**
([манифест](evidence/source-manifest.json), [сверка](evidence/runtime/source-verification.json)).
Штатный `check_inputs.py --mode val` подтвердил размеры и SHA-256 **1864 обязательных файлов**
([JSON](evidence/runtime/input-preflight.json), [вывод](evidence/runtime/input-preflight.log)).

Из `/repo/04-solution/service` в контейнере выполнено:

```bash
python -B -m app.batch --images-dir /data/images \
  --query /repo/04-solution/split/files/val_query.csv \
  --gallery /repo/04-solution/split/files/val_gallery.csv \
  --out-dir /out/full --threads 2
```

**Exit 0, 711.136 с.** Каталог `/out/full` перед командой отсутствовал — это проверяется
скриптом. Обработаны **1110 запросов + 750 элементов галереи из исходных изображений организатора**,
обе ONNX-модели и whitening. Готовые векторы не использовались как вход полного прогона.
[Вывод batch](evidence/runtime/full-batch.log), [программа запуска](scripts/validate_runtime.py).

Затем из `/repo` выполнено:

```bash
python -B 04-solution/service/tools/eval_split.py /out/full/embeddings.npy
```

| Показатель | Новый результат |
|---|---:|
| mAP | **0.7740915539438481** |
| Rank-1 | **0.7307692307692307** |
| Rank-5 | 0.8810096153846154 |
| mAP@10 | 0.7684546226343101 |
| Допустимых запросов | 832 |

Exit 0; similarity/distance совпали. Значения mAP/Rank-1 сверены точным равенством Python float,
без допуска: `0x1.8c55ba6898e85p-1` / `0x1.7627627627627p-1`.
[Полные метрики](evidence/runtime/full-metrics.log), [проверка значений и SHA](evidence/runtime/full-verification.json).

Все три новых файла совпали побайтно с независимым полным результатом job_90:

| Файл | SHA-256 |
|---|---|
| `embeddings.npy` | `1d85f346359e8acdee1c439c2750e39135ef1fefd23033a50243a18179e9dab0` |
| `submission.csv` | `bfd372b2d8f16b7c56bb9fca344475ae3989791f158d38be924a765fed95d83f` |
| `candidates.csv` | `d1364538e5650f301a2f11d447903c291026fe9bc51fb409dfff84f64b1eba66` |

## Все существующие тесты и границы изменений

| Проверка | Команда из соответствующего каталога | Вывод |
|---|---|---|
| Входы, CLI | `python -B -S -m unittest -v test_check_inputs test_cli_inputs` | [16 tests, OK](evidence/runtime/cli-tests.log) |
| Настоящие ONNX CLI | `python -B -m unittest -v check_cli_runtime` | [3 tests, OK](evidence/runtime/runtime-tests.log) |
| Метрики и протоколы | `python -B -m unittest -v test_metrics test_protocols` | [60 tests, OK](evidence/runtime/metric-tests.log) |
| Существующие проверки PDF | `python -B -S 06-documentation/test_build_pdf.py -v` | [3 tests, OK](evidence/pdf-existing.log) |
| Настоящий render и гонка | `python -B -m unittest discover -s 06-documentation -p check_build_pdf_render.py -v` | [1 test, OK](evidence/pdf-render.log) |
| Метод сверки чисел | `python -B -S -m unittest discover -s 04-solution/audit/docs-fixes -v` | [7 tests, OK](evidence/number-tests.log) |
| Полный проверяющий контур метрик | Из `04-solution/eval`: `python -B verify.py` | [Exit 0](evidence/runtime/metric-verify.log) |

Итого **90/90** unit/runtime-тестов без пропусков, включая все прежние **73**.
`verify.py` дополнительно поймал **25/25** мутаций метрик, без ошибок исполнения и выживших
([полный результат](evidence/runtime/metric-verification-results.json)).
Локально выполнялись короткие stdlib-проверки и render одной страницы; полный инференс и тяжёлый `verify.py` — на worker.

Первый вызов `verify.py` не смог записать свой `test_results.log` в read-only checkout.
Это ошибка подготовки стенда: [первоначальный лог](evidence/runtime/metric-verify-initial-fixture-error.log)
сохранён. Скопирован неизменённый `eval/` в доступный для записи `/out/metric-verify-tree`, затем повторена
только `verify.py`, успешно. Полный инференс и оценка к тому времени уже прошли все численные проверки;
их результаты сохранены. [Команда завершения проверки](scripts/resume_verify.py).

**131 защищённый файл** вычислительного кода, метрик, весов, конфигурации и первичных JSON сохранил
Git blob ID относительно `95d228b` ([сверка](evidence/protected-files.json)). После упаковки runtime-снимка
изменились только инструкции reproduce README и docstring теста PDF; исполняемые исправления F1/F2
в снимке соответствуют итоговому коду. Алгоритмы, пороги и веса не менялись.

Локальные логические коммиты:

- `f6796ae` — потоковые CSV, модульные импорты и регрессии CLI;
- `ba666ee` — тест гонки PDF и документированные команды новых suites;
- `1eedc0b` — два округления, метод числовой проверки и его тесты.

Незакрытых пунктов BRIEF нет. Полная семантическая проверка всех остальных числовых вхождений документации
не заявляется: их список теперь явно доступен для дальнейшего связывания с первоисточниками.
Прежние логи и отчёты аудитов сохранены. [Журнал работы](journal.md).
