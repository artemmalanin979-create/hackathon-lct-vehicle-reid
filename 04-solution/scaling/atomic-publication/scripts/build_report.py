"""Build the final report only after both real-image runs and byte comparison."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
comparison = json.loads((root / 'evidence/bitwise-comparison.json').read_text())
assert comparison['all_four_byte_equal']
before = json.loads((root / 'evidence/j81-full-before.validation.json').read_text())
after = json.loads((root / 'evidence/j81-full-after.validation.json').read_text())
manual = json.loads((root / 'evidence/manual-batch.json').read_text())
filesystem = json.loads((root / 'evidence/filesystem.json').read_text())
mutations = json.loads((root / 'evidence/mutations.json').read_text())
assert all(r['passed'] for r in manual + filesystem)
assert all(not r['extended']['success'] for r in mutations)
hash_table = '\n'.join(f"| `{r['file']}` | `{r['before_sha256']}` | `{r['after_sha256']}` | да |"
                       for r in comparison['files'])
mutant_table = '\n'.join(
    f"| {r['id'][:3]} — {r['description']} | `{r['new_test']}` | 9/9 проходят | "
    f"**ловит** ([выделенный запуск]({r['dedicated']['log']})) |"
    for r in mutations if r['new_test'])

report = f'''# job_81 — только атомарная публикация в main

Перенос выполнен в `main`, коммит кода **46bc5ee92c49bde42d555adfa914185d01e47ff9**.
Исходный main — **7800f222094df0da26387399e23e9bf6f3c60dd4**.
`guard/memory` остаётся **de1c68cf97b23ad9732f6da06dcf693495618676**; ветка и её рабочее дерево не изменены.
Push не выполнялся.

**Все четыре файла полных прогонов совпали побайтно и по SHA-256. Для `run_info.json`
это проверено при воспроизведении показаний reporting clock только в измерительной обвязке.**
В production часы не менялись; настоящие длительности сохранены отдельно. Обычные два
запуска с настоящими часами не обязаны давать одинаковые elapsed-поля — это условие
проверки не скрывается и не выдаётся за естественную детерминированность времени.

## Что перенесено

- `app/batch_safety.py`: `_check_destination`, `_publish`, `atomic_output`, `OUTPUT_FILES`
  и только необходимые импорты стандартной библиотеки. Тела трёх функций дословно
  совпадают с веткой-источником.
- `app/batch.py`: весь цикл от создания Embedder до записи `run_info.json` находится
  внутри `atomic_output`. Четыре файла пишутся в sibling `.out.incomplete-*`; успешный
  новый каталог публикуется одним rename, прежний комплект заменяется одним
  `renameat2(RENAME_EXCHANGE)`. Печать успешного результата происходит после публикации.
- Перенесены все **9** тестов публикации из исходного `test_batch_safety.py`.
  Добавлены **4** теста на выжившие мутации; список ожидаемых файлов в тестах независим
  от production-константы.

**Намеренно не перенесены:** `estimate_memory`, `available_memory`, `check_memory`,
их вызовы, `_unescape_mount`, `MIB/GIB`, `re`, чтение `/proc`/cgroup, вся арифметика
бюджета и все 21 тест проверки памяти. Из десяти выживших мутаций исходного аудита
M02/M07/M09/M10/M11/M19 относятся к этой исключённой части; для них здесь ничего не добавлено.

AST вычислительной части `batch.main` совпадает с исходным после удаления только
обвязки публикации и нормализации выходного пути. `core/`, модели, пороги, конфигурация,
сплит и eval не изменялись. [Машинная проверка переноса](evidence/transfer-integrity.json),
[скрипт](scripts/verify_transfer.py).

## Полный прогон: настоящие изображения и ONNX

Два новых последовательных запуска на **worker-vm**, по одному закреплённому CPU,
4 GiB RAM, без swap и сети, UID 1000, **private cgroup namespace**. Использован
сохранённый runtime `localhost/job72-jury:6179b69` (`f04d50ee90c8`), Python
{before['versions']['python']}, NumPy {before['versions']['numpy']}, ONNX Runtime
{before['versions']['onnxruntime']}. Снимки исходников смонтированы read-only поверх
runtime; выполнялся именно `app.batch.main` до/после изменения, а не старый код образа.

В каждом запуске прочитаны **1110 query + 750 gallery**, заново выполнены обе ONNX-сети,
whitening и штатное переранжирование. Готовые embeddings для инференса не использовались.
Все **1860 оригинальных изображений** (1 221 141 691 байт) и оба CSV сверены с manifest;
все хеши совпали. **107** защищённых файлов модели/алгоритмов/сплита/eval совпадают
между снимками. [Полная сверка входов](evidence/input-verification.json).

| Файл | SHA-256 до | SHA-256 после | Побайтно равны |
| --- | --- | --- | --- |
{hash_table}

Сверка читает реальные файлы целиком и дополнительно проверяет `before == after`;
выходы не редактировались, JSON не нормализовался, файлы между прогонами не копировались.
[Результат](evidence/bitwise-comparison.json), [скрипт](scripts/compare_outputs.py).

Метрики независимо пересчитаны из нового `embeddings.npy` каждого запуска тем же
eval-кодом, с camera policy `market` и refusal mode `presence`:

| Метрика | До | После |
| --- | ---: | ---: |
| mAP | {before['metrics']['mAP']} | {after['metrics']['mAP']} |
| Rank-1 | {before['metrics']['Rank-1']} | {after['metrics']['Rank-1']} |
| Настоящее время batch, с | {before['batch_wall_s']:.3f} | {after['batch_wall_s']:.3f} |
| Пик RSS процесса, MiB | {before['rss_peak_bytes'] / 1024**2:.1f} | {after['rss_peak_bytes'] / 1024**2:.1f} |

Пик RSS относится ко всему дочернему процессу, включая последующий расчёт метрик.
Оба controller завершились с exit **0**, OOM не было.
[До](evidence/j81-full-before.validation.json), [после](evidence/j81-full-after.validation.json),
[команда до](evidence/j81-full-before.controller.json),
[команда после](evidence/j81-full-after.controller.json).

### Как проверялся run_info.json

Обвязка первого прогона записывает пять настоящих вызовов `batch.time.perf_counter`.
Второй прогон возвращает только этому модулю те же пять чисел, а параллельно записывает
свои реальные показания. `time` у ONNX, NumPy и остальных модулей не подменяется.
Так проверяется исходная сериализация всех четырёх файлов при одинаковых внешних
показаниях часов. Продакшен-файлы не содержат replay/freeze и продолжают сообщать
настоящие времена. Равенство отчётных часов — явное условие этой проверки.
[Обвязка](scripts/full_validation.py), реальные часы — `actual_clock` и `actual_elapsed_s`
в JSON обоих запусков. Это не доказательство побитового равенства времён двух обычных запусков.

## Выжившие мутации публикации → новый тест → ловит

Использованы **точные изменения** независимого критика, каждое в отдельной копии
модуля и отдельном процессе Python с `-B`. Production-код не мутировался.
Для каждой из четырёх мутаций заново показаны зелёные исходные 9 тестов публикации,
красный выделенный новый тест и красный расширенный набор из 13 тестов.

| Мутация | Новый тест | Исходные тесты | Новый тест |
| --- | --- | --- | --- |
{mutant_table}

M18 проверяется настоящим дочерним процессом: наблюдение на границе файлового rename
или сразу после exchange, затем настоящий **SIGKILL**. Проверяется содержимое публичного
пути с независимым набором четырёх имён. Удаление старого каталога и перенос файлов
по одному обнаруживаются как пустой/частичный комплект.

Дополнительно заново убиты M12, M13, M17. Всего **7/7 мутаций публикации убиты**.
[Полные результаты](evidence/mutations.json), [точные мутации и происхождение](evidence/critic-publication-mutations.json),
[скрипт запуска](scripts/run_mutations.py). Ошибки импорта не считаются доказательством:
выделенные новые тесты дают именно assertion failures.

## Ручные аварийные сценарии уже на main

Сквозной harness исполняет настоящий `app.batch.main`, настоящие модели и оригинальные
изображения. Контроль старого комплекта — **2 query × 1 gallery**, нового — **1 × 2**,
поэтому старые и новые файлы различаются. Для сравнения контрольных комплектов
зафиксированы только reporting clock и точки наблюдения/паузы файлового I/O.
Чужие процессы не останавливались; SIGKILL посылался исключительно созданным harness дочерним процессам.

| Сценарий | Без прежнего результата | С прежним результатом |
| --- | --- | --- |
| SIGKILL в середине первой записи `embeddings.npy` | Final отсутствует; 64 байта только в staging | Старый комплект побайтно цел; 64 байта только в staging |
| SIGKILL после первого закрытого файла | Final отсутствует; полный `embeddings.npy` только в staging | Старый комплект побайтно цел |
| SIGKILL после всех четырёх записей, до публикации | Final отсутствует; новый полный набор только в staging | Старый комплект побайтно цел |
| SIGKILL сразу после rename/exchange, до cleanup | Final — полный новый комплект | Final — полный новый; прежний комплект остался в staging |
| Настоящая нехватка места в tmpfs 2 MiB | Ошибка записи, final отсутствует, staging убран | Ошибка записи, старый комплект побайтно цел, staging убран |

Итого **8/8 SIGKILL** и оба опыта нехватки места прошли. При ENOSPC NumPy сообщает
короткую запись `1536 requested and 992 written`; дополнительная прямая запись
стандартной библиотекой в отдельном tmpfs подтверждает настоящий **errno 28**.

Также проверены:

- новый, заранее пустой и прежний непустой output: успешный полный комплект, staging убран;
- посторонний `notes.txt`: отказ до создания Embedder, файл и старый комплект сохранены;
- symlink самого output и symlink файла внутри него: отказ до создания Embedder, цель сохранена;
- реальный bind mount в роли final: **EBUSY (16)**, старые файлы сохранены;
- намеренно другой FS для staging: настоящий **EXDEV (18)**, старые файлы сохранены, staging убран.

Сквозной набор — **16/16**; отдельные syscall-опыты — **4/4**.
[Сквозные результаты](evidence/manual-batch.json), [syscall-результаты](evidence/filesystem.json),
[harness](scripts/manual_batch.py), [контейнер](evidence/j81-manual-batch.controller.json),
[mount namespace](scripts/filesystem_checks.py). Mounts создавались только в отдельном
контейнере/namespace; настройки файловых систем хоста не менялись.

## Все существующие тесты

| Набор | Результат |
| --- | --- |
| `04-solution/eval/test_metrics.py`, `test_protocols.py` | **60/60** ([лог](evidence/eval-tests.txt)) |
| `04-solution/reproduce/test_check_inputs.py` | **5/5** ([лог](evidence/reproduce-tests.txt)) |
| `04-solution/service/tests/test_batch_safety.py` | **13/13** ([локальный лог](evidence/publication-tests.txt), [worker](logs/worker-publication-tests.txt)) |

Всего **78/78**. `git diff --check` пройден. Полный ONNX-прогон и вся тяжёлая работа
выполнялись на worker-vm; локально запускались только лёгкие unit/mutation проверки.

## Контракт перенесённой публикации и сохранённые границы

Для container output монтируется **родительский** каталог, а `--out-dir` задаётся
внутри него; корень bind mount обменять нельзя. Например:

```bash
docker run --rm --network none \\
  -v /путь/к/данным:/data:ro -v /путь/к/выводу:/out \\
  vehicle-reid-service python -m app.batch \\
  --images-dir /data/images --query /data/test_query.csv \\
  --gallery /data/test_gallery.csv --out-dir /out/result
```

Выходы будут в `/путь/к/выводу/result`. Прежние примеры прямого `app.batch --out-dir /out`
с примонтированным `/out` несовместимы с этой схемой; основной reproducer уже использует
вложенные `val/` и `test/`. В рамках буквального переноса команды прежней документации
не переписывались; рабочая команда дана выше.

Гарантия относится к смерти процесса и публичному пути. Сохраняются известные свойства
взятой реализации: нужен writable parent и Linux `renameat2` для замены; новый каталог
получает режим `0700` от `mkdtemp`; SIGKILL оставляет явно незавершённый sibling; fsync-протокола
против потери питания нет. Выходной каталог должен принадлежать одному писателю:
повторная проверка не устраняет гонку после самой последней проверки. Cleanup после
успешного exchange может отдельно завершиться ошибкой, когда новый результат уже виден.
Полнота staging проверяется по именам; штатные serializers создают обычные файлы.
Эти свойства не расширялись и не выдаются за исправленные.

## Воспроизведение и файлы

Рабочий каталог на worker: `/home/fedora/lct-reid/jobs/job_81-atomic-20260923`.
Точные исходные команды и ограничения находятся в controller JSON. Скрипт
`launch_validation.py` отказывается повторно запускать тот же каталог через
`validation.started`; для независимого повторения нужен свежий каталог со снимками
`snapshots/before` и `snapshots/after`, оригинальными данными и тем же runtime.
Скрипты не меняют production-файлы.

Для повторения мутаций:

```bash
python3 -B scripts/run_mutations.py --repo /путь/к/репозиторию --work-dir /путь/к/новому/опыту
python3 -B scripts/verify_transfer.py --repo /путь/к/репозиторию --output /путь/к/проверке.json
```

Для файловой сверки полного прогона:

```bash
python3 -B scripts/compare_outputs.py --root /путь/к/каталогу/прогонов
```

В workspace сохранены оба полных комплекта `artifacts/j81-full-before` и
`artifacts/j81-full-after`, включая embeddings. В Git включены исходники harness,
хеши, результаты и логи; промежуточные embeddings остаются воспроизводимыми артефактами.

Изменены три файла кода/тестов:

- `04-solution/service/app/batch.py`;
- `04-solution/service/app/batch_safety.py` (новый);
- `04-solution/service/tests/test_batch_safety.py` (новый).

Добавлен пакет `04-solution/scaling/atomic-publication/`: этот отчёт,
`journal.md`, `scripts/`, `evidence/`, `logs/`. Полный пофайловый список —
[changed-files.txt](evidence/changed-files.txt). [Журнал работы](journal.md).
'''
(root / 'REPORT.md').write_text(report)
print('REPORT.md written')
