# job_73 — исправление воспроизводимости сдаваемой d1_j48

Работа выполнена в `/home/artem/projects/hackathon-lct-vehicle-reid`, ветка `main`.
Коммитов и push не делалось. Сдаваемые веса, пороги, алгоритмы, backend и четыре
исходных test-артефакта сохранены. Из существующего исполняемого кода сервиса
изменены только видимые строки статического интерфейса; в `fetch_model.sh` —
только комментарии. Новый проверочный контур и адаптеры подготовки обучения
находятся вне backend сервиса.

Начальный HEAD — `fbc92fd`.
Во время работы другой процесс добавил коммит `3fc553f` с архивом базовых образов.
Исполнитель job_73 этого коммита не создавал. При заключительной проверке
содержимое архива оказалось ошибочным: один Python-образ с тегами Python и
Qdrant. Эта находка учтена в документации и отчёте; архив не перезаписывался.

Главный результат: сторонний исполнитель с **тем же набором организатора**
может пересчитать головные validation-числа из изображений, не имея test-части.
Это проверено фактическим запуском R-val: mAP **0.7740915539438481**, Rank-1
**0.7307692307692307**, F1 **0.8090787716955942**, TNR **0.7841726618705036**.
Фактические исходники combined_v1 включены в рабочее дерево с происхождением
каждого файла, конфигами, командами и проверенными снапшотами данных.

## Что перенесено и чем подтверждено

В `04-solution/training/src/` перенесены **28 архивных файлов**, без изменения
содержимого. Для каждого проверены SHA-256 на источнике и после переноса;
список полных путей и хешей приведён ниже. `provenance.json` фиксирует также
время изменения и независимые копии, где они сохранились. `SHA256SUMS` проверяется
обычным `sha256sum -c`.

- Windows `worker:D:\lct-reid`: `reid_train5.py`, `job48_chain.py`, архитектура,
  export, восстановление initial PT и две библиотеки метрик. Независимые
  сохранённые копии job_68 совпали. Export исходного stage2.pt заново дал
  SHA сдаваемого ONNX, в том числе с синтетическим probe.
- `worker-vm:~/lct-reid/jobs/job_44`: действительный сборщик combined, build.log
  и исторические download-скрипты. Последние архивируются как свидетельства,
  а не предлагаются в качестве нового рецепта: в них есть также неиспользованные
  попытки загрузки других датасетов.
- VM `job_48`, `job_45`, `job_42`, `job_55`: реальные extraction/evaluation/plate
  scripts, исходные whitening/bootstrap-библиотеки, проверка f64/f32, конфиги
  этапов 1/2, epoch-логи и журнал chain. Полный порядок обучения записан в README.

Важное уточнение: экспортирована **последняя эпоха 15 этапа 2**, dev mAP 0.77211.
Лучшая эпоха 13 (0.77719) использовалась в воротах; отдельного best-checkpoint
код не сохранял. Этап 3 пропущен по росту CH, его config/log не существовали.
Seed 20260916, torch seeds `SEED+stage`, dev seed `SEED`; `cudnn.benchmark=True`.
Исторически зафиксированы Python 3.12 / torch 2.5.1+cu118, текущая сохранившаяся
среда подробно снята: Python 3.12.8, CUDA 11.8, cuDNN 90100, driver 538.18.
Полный freeze 21.09 не выдается за lock с момента тренировки 19.09.

Полный RoundaboutHD ZIP найден и проверен: **8 075 314 494 байта**,
`fad611b70065bb67dbd7eb6a1c03280b8c1252fe46c291537ccd0b9230f1dd54`.
CARLA ZIP: **2 529 020 889 байт**,
`55153a24d2c6c95eb88d9eb44b93b03f7cddc05f28dbc58e03c4c4a5e913908d`.
По **каждому из 13 328 внешних кадров** сверены исходный ZIP-member, извлечённый
JPEG, packed JPEG и raw-208 относительно combined; сохранён полный
`external-selection.jsonl`. Восстановлены точные glob, порядок сортировки,
round-robin камер, максимум 12 кадров на ID и смещения меток. Выбор охватывал
CARLA train/query/gallery и весь RoundaboutHD_Reid_subset. `--cap` в старом
сборщике только сообщает превышение, не обрезает — это отражено в инструкции.

Три новых адаптера **явно обозначены новыми**:

- `initialize_from_onnx.py`: неизвестный исторический одноразовый скрипт не
  подменяется похожим. Новый экспорт 559 инициализаторов дал точный SHA исходного
  NPZ; все тензоры совпали и с preserved initial PT.
- `prepare_own.py`: вызывает неизменённую старую функцию crop и делает raw-208
  через decode сохранённого JPEG-256. На узле полный запуск остановился до
  обработки: нет 7248 исходных train_fit-JPG. Готовые own NPZ/raw и их точное
  совпадение с own-префиксом combined подтверждены; новый полный crop не заявляется.
- `rebuild_whitening.py`: вызывает исходную `lib45.learn_lw`. Пересчёт из
  исторических train-векторов дал точные f64/f32 SHA; 32 кропа на каждой модели
  новым extractor дали max|Δ|=0.0. Полное повторное извлечение 14 496 векторов
  не выполнялось. Историческая одноразовая shell-команда f64→f32 не найдена;
  восстановленная операция дала тот же итоговый SHA.

## D1–D10

| Пункт | Статус и сделанное исправление | Доказательство / граница |
|---|---|---|
| D1 | Исправлено описание поставки: оба ONNX и whitening входят в Git; комментарий fetch_model согласован с фактом | Файлы отслеживаются, три SHA сохранены; логика fetch_model не менялась |
| D2 | Таблица service README заменена на d1_j48: cosine 0.7316/0.6851, rerank 0.7741/0.7308; старая OSNet подписана исторической | Новый R-val подтвердил числа из изображений |
| D3 | Убрано обещание воспроизвести всё из одного чистого дерева; дан отдельный контур R-val/R-test, все входы, получение, manifest и ранняя проверка | R-val без test-входов прошёл; отсутствующие данные дают exit 2 до ML-импортов |
| D4 | Восстановлен полный SHA RoundaboutHD, подтверждён CARLA, опубликован точный отбор каждого внешнего кадра | Полные архивы измерены; все 13 328 кадров проверены против combined |
| D5 | Перенесены фактические источники, конфиги/логи, текущие среды; удалена ссылка на несуществующий input256/reference | Архивные SHA и повтор export/whitening подтверждены; исторический source-lock и полный freeze момента обучения не восстановлены, новое обучение не выполнено |
| D6 | Во всех исправляемых схемах добавлена L2 среднего перед whitening | Формула согласована с неизменённым model.py; run_info остаётся исходным артефактом |
| D7 | UI показывает OSNet-AIN OMZ и «Косинус OSNet-AIN», README объясняет отличие от d1_j48 | Только строки UI, node --check прошёл; расчёт объяснения/поиска не менялся |
| D8 | Убрано заявление о полной изоляции по одному --network none; разделены RUN-сеть, pull и сеть сборщика; Docker --pull=never исправлен, в том числе в новом offline/README | Документирован проверенный Podman-путь; дополнительно обнаружен ошибочный архив Python/Qdrant; Docker Engine/BuildKit отдельно не проверен |
| D9 | В сетевой ветви явно требуется wheels/ из-за безусловного COPY; добавлен mkdir -p wheels | Соответствует существующему Dockerfile, который не менялся |
| D10 | Разведены смена конфигурации 20.09 и выпуск артефактов 21.09; добавлены manifest.json и SHA256SUMS | Все четыре текущих файла проверены; manifest ретроспективный, а не новый выпуск |

D1–D10 исправлены в пределах поставленной задачи. Это **не утверждение, что все
пробелы исторической воспроизводимости закрыты**: ограничения D5 и полного
офлайн-стенда перечислены далее. Исторический отчёт job_72 не переписывался.

## Ограничение порога и объяснимости

Документация связывает `t_rr` с полным calibration-сплитом **1110×750**,
d1_j48, KR(6,3,0.3), `market` + `presence`; минимальный безопасный размер
галереи не установлен. Даже новый набор того же размера требует проверки.
Сохранён факт аудита: на 4×10 приняты чужие автомобили с 0.541908 / 0.768959
и серый кадр с 0.801990. Для малых наборов предлагается cosine (API или
`--no-rerank`) и собственная калибровка для повторного включения rerank;
F1/TNR большого сплита для малого не обещаются. Автоматического fallback нет.

Объяснение помечено как **первая OSNet-AIN OMZ**, поиск — **d1_j48**, две модели
с whitening. Числа одной пары 0.053297 (поиск) и 0.131823 (explain) указаны со
ссылкой на аудит. UI показывает модель и её шкалу прямо рядом с картами.

Изменять код сервиса для выполнения brief не потребовалось. Если координатор
решит вводить автоматическое отключение rerank на малых наборах, сначала нужна
размеченная проверка размеров и состава галереи: обоснованной границы сейчас нет.

## Проверки

Полные результаты: [README проверки](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/evidence/README.md),
[сводный JSON](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/evidence/reproduction-summary.json).
Изображения не копировались в Git. Переизвлечение делалось только на worker-vm.

| Проверка | Фактический результат |
|---|---|
| R-val из 1860 исходных JPG, без test-части | Exit 0; cosine mAP 0.7316093422310448 / Rank-1 0.6850961538461539; rerank mAP 0.7740915539438481 / Rank-1 0.7307692307692307 |
| Повтор калибровки | t_cos 0.5141977018197194, t_rr 0.5282812332956895; абсолютные отклонения от сдаваемых порогов 1.040067176827364e-08 и 2.6614458414897513e-09, меньше допуска 1e-6 |
| F1/TNR | cosine 0.7684996605566871 / 0.7302158273381295; rerank 0.8090787716955942 / 0.7841726618705036 |
| Независимая калибровка через штатные ranking/rerank сервиса | 187 сверок, max_abs_error=0.0 |
| R-test из отдельного test-only набора | Exit 0; все три сдаваемых файла совпали по размеру и SHA-256; 1076 принятых запросов / 34 отказа |
| run_info test | Различаются только embed_elapsed_s, rank_elapsed_s, total_elapsed_s |
| 60 тестов метрик | Все пройдены в runtime-образе на worker-vm |
| 5 проверок новых preflight/режимов | Все пройдены; test не требуется для val, split не требуется для test |
| Клон без организаторских данных | Под python -S — exit 2, полный список 1860 пропусков, понятное сообщение; ML-импорты и инференс не начались |
| Перенос источников | 28 пар source/transferred SHA совпали, sha256sum -c успешен |
| Независимый export сохранённого checkpoint | Точный SHA сдаваемого combined_v1; синтетический probe также дал тот же ONNX |
| Dataset/whitening | Все 13 328 внешних кадров проверены; обе whitening NPZ совпали по SHA; extractor первых 32 кропов на модель — max|Δ|=0.0 |
| UI | node --check прошёл; изменены только видимые подписи |
| Финальная статика | 19 Python-файлов разобраны AST, все локальные ссылки изменённых Markdown существуют, git diff --check чист |
| Границы изменения | Backend, Dockerfile, Compose, требования, веса не менялись; в fetch_model совпадают все исполняемые строки |
| Новый архив base-images.tar.gz | SHA правильный, содержимое неправильное: один Config с двумя RepoTags; дефект явно отражён, бинарный архив не заменён |

Среда проверки: Python 3.13.15, NumPy 2.5.3, ONNX Runtime 1.30.0;
образ `localhost/job72-jury:6179b69`, image ID
`f04d50ee90c845b60e4ce2e20820d006dee893024e1a1fcca347c374e494ae43`.
Validation — 2 CPU / 4 GiB, test — 1 CPU / 4 GiB, threads=2 / batch=32,
`--network none`, read-only mount исходников. Полные команды сохранены в
validation-run.txt и test-run.txt. 140 исполняемых файлов/весов/входов снимка
сверены с текущим деревом по SHA. Это изолированный снимок текущих изменений
поверх существующего runtime-образа; **не свежая сборка и не новый Git-клон**.
Времена batch: validation extraction 903.599 с / rank 3.709 с;
test extraction 1093.573 с / rank 2.702 с. Из-за фоновой нагрузки и разных
квот это не сравнительный бенчмарк. Протокол validation — market/presence.

Test-SHA, совпавшие с выпуском 21.09:

- `submission.csv` — `9f08068971d10efe815f6c5817d3f2adce59d343b1a3feb130aac1d22407bca8`.
- `embeddings.npy` — `a64fecfe3d2c21d2e27142260bb12749987f6f4d663ab81c8192a0be3575cc57`.
- `candidates.csv` — `204559a5c9b50a6130801c2141ec2d5a305bbcf0ae1b28c84d20c1bf98aaf0b1`.


## Что остаётся незакрытым

1. Нет сохранённых SHA исходников и полного freeze **на момент старта** 19.09.
   Текущие копии связаны с обучением журналами, независимыми копиями и повторным
   export; это сильное свидетельство, но не доказательство отсутствия изменений
   trainer после запуска. Повторное CUDA-обучение не выполнено и не гарантирует
   побитовую идентичность при недетерминированных операциях.
2. Старый одноразовый ONNX→NPZ-скрипт и отдельная shell-команда f64→f32 не найдены.
   Новые адаптеры обозначены честно, их результаты проверены по сохранённым SHA.
3. Нет полного повторного own-preparation из 7248 train_fit-JPG и полного нового
   extraction train-векторов. Сохранившиеся производные проверены, рецепты даны,
   но эти два полных этапа в job_73 не пройдены.
4. Изображения организатора, original probe и test-CSV не публикуются. Доступ к
   набору указан через кабинет; отдельный проверенный канал выдачи жюри неизвестен.
   Индивидуальные SHA позволяют однозначно проверить, получен ли тот же набор.
5. **Новый блокирующий дефект полного офлайн-стенда:** добавленный параллельно
   `service/offline/base-images.tar.gz` содержит только Python с двумя тегами,
   настоящего Qdrant нет. SHA `cde3ec69…6bb3c` совпадает с приложенным, но в manifest
   ровно один Config `51cce855…210f`. Нельзя загружать его как готовый комплект:
   он назначит тег Qdrant образу Python. Сохранён полный manifest/config в
   `reproduce/evidence/base-archive-inspection.json`; `offline/README.md` теперь
   содержит точный диагноз, проверку разных image ID и правильные команды
   Docker/Podman. Архив не заменялся: восстановление поставки остаётся
   координатору, задача job_73 исправляет документацию и сообщает неполноту.
   R-val/R-test используют готовый образ аудита job_72 и прошли; Docker Engine,
   новая сборка и Compose/Qdrant не перепроверены. После коммита координатора
   нужен свежий проход опубликованного Git-клона с исправленным архивом.
6. Некоторые промежуточные `.npy/.ids`, разметка и `work/` для bootstrap/абляций
   остаются только в авторских каталогах. R-val от них не зависит; повтор всех
   исследовательских отчётов не обещается.
7. Полная версия и происхождение данных предобучения внешней OSNet неизвестны;
   данные и условия сторонних источников описаны с прежними оговорками. Исторические
   URL с `main` не являются immutable — обязательна проверка SHA архивов.

## Пройдёт ли посторонний до головных чисел

**Да, если у него есть тот же выданный организатором набор и рабочий runtime-образ
или возможность собрать его по документации.** Для validation достаточно ровно
1860 нужных train-JPG; CSV сплита, оба ONNX, whitening и код уже в репозитории.
Он запускает preflight, затем R-val из SOLUTION §7. Такой путь в job_73 пройден
на отдельном read-only снимке, без test-части и без кэшей эмбеддингов.

Если перед ним только Git-клон без данных, головных чисел не будет: вместо
непонятного FileNotFoundError он получает количество пропусков, имена, каталог,
способ получения и полный JSON-список. Если ему нужно переобучить combined_v1
с нуля либо поднять целиком офлайн Docker/Compose, степень подтверждения ниже:
код/рецепт теперь доступны, но перечисленные выше пробелы не скрыты.

## Происхождение каждого перенесённого файла

Все назначения ниже — относительно `04-solution/training/src/`. Полное
свидетельство каждого файла и предел доказательства — в `provenance.json`.

| Назначение | Источник | Полный SHA-256 (источник = копия) | Связь с прогоном |
|---|---|---|---|
| `windows/reid_train5.py` | `worker:D:\lct-reid\reid_train5.py` | `17df40eb016c39eeeea17a609a1bcbb07a16f50c35a17378ec275f2d90e17e9c` | Chain/config/epoch-логи; SHA на старте тренировки не записан; отдельная копия job_68 совпала |
| `windows/job48_chain.py` | `worker:D:\lct-reid\job48_chain.py` | `746df226cabc9e674360a0f940cacb5afc00da7288670daaae0d09664fe21924` | Chain/config/epoch-логи; SHA на старте тренировки не записан; отдельная копия job_68 совпала |
| `windows/export_onnx2.py` | `worker:D:\lct-reid\export_onnx2.py` | `eaaa39c1714b10160db78927619d0d8e4e57403b287a85d35c912f3858479009` | Повторный export stage2.pt дал точный ONNX SHA; отдельная копия job_68 совпала |
| `windows/osnet_ain.py` | `worker:D:\lct-reid\osnet_ain.py` | `6fd22d609f94798adf1748a4989cceb5b8a8261edfec3620eeb12507c667746b` | Повторный export stage2.pt дал точный ONNX SHA; отдельная копия job_68 совпала |
| `windows/build_from_onnx.py` | `worker:D:\lct-reid\build_from_onnx.py` | `f03adef28652026767a7f3a0a15cfc82b24e346a19d251ab29ff3a70cb0a8fdb` | 559 тензоров исходного ONNX, initial NPZ/PT равны |
| `windows/reid_metrics.py` | `worker:D:\lct-reid\reid_metrics.py` | `79fba7051bc1fd6e856230f6a256334196c7ae7f14767ad85175aaeabe37ffec` | Chain/config/epoch-логи; SHA на старте тренировки не записан |
| `windows/scope_metrics.py` | `worker:D:\lct-reid\scope_metrics.py` | `6dd489984a531ad2d07b58b1e963d61f015e5d4ee780187de90b424b71e463df` | Chain/config/epoch-логи; SHA на старте тренировки не записан |
| `datasets/build_combined.py` | `worker-vm:/home/fedora/lct-reid/jobs/job_44/build_combined.py` | `3e335088615202067c0970565a835bb1c7693adad12cba4b6c6bc0c373a87a04` | Исторический build.log и полная проверка выбранных кадров |
| `postprocess/s01_extract.py` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/scripts/s01_extract.py` | `ff9589ad7863e83422dc37c4bd8dfb657ef8f00fe79fb2b37b7741ea04e65e3b` | Пути и результаты postprocess в журналах job_48/job_55; отдельная копия job_68 совпала |
| `postprocess/s02_eval.py` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/scripts/s02_eval.py` | `80f3a7820e1f1184ce4a7e71dd930d121f544259c281d216c2b651e062449830` | Пути и результаты postprocess в журналах job_48/job_55; отдельная копия job_68 совпала |
| `postprocess/s03_plate.py` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/scripts/s03_plate.py` | `ad6f0de4a8a234445226a29c804217131e8c8931ca6b5cc150a21ef70f0aa059` | Пути и результаты postprocess в журналах job_48/job_55; отдельная копия job_68 совпала |
| `postprocess/lib45.py` | `worker-vm:/home/fedora/lct-reid/jobs/job_45/scripts/lib45.py` | `8a03c39489e463529ea5814753b6be435989fe14ea972d2848851bad09fb6983` | Исходный learn_lw повторно дал обе точные NPZ; отдельная копия job_68 совпала |
| `postprocess/lib42.py` | `worker-vm:/home/fedora/lct-reid/jobs/job_42/lib42.py` | `940ce0f7e8a2b8883973d135d76f76ece9ae3aa964b9840cc6e85d8a5a7cd8da` | Пути и результаты postprocess в журналах job_48/job_55 |
| `postprocess/delta.py` | `worker-vm:/home/fedora/lct-reid/jobs/job_55/delta.py` | `603993b51d015120ffa839ad611eb04090f590e796402bf8f5fb2c5b87dab8d6` | Пути и результаты postprocess в журналах job_48/job_55 |
| `evidence/combined_v1/chain.log.txt` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/runs/combined_v1/chain.log` | `cfc2b7811294109d14d91800e6bff1994a4b52ebdff9fd4b27625e721bac8d3b` | Подлинная запись из каталога runs/combined_v1 |
| `evidence/combined_v1/chain_s1.out.txt` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/runs/combined_v1/chain_s1.out.txt` | `fff3d5c059034ac80418d1d8ca1c0e5a19a8f6b1ee46c2d2d9a4918651cb13a1` | Подлинная запись из каталога runs/combined_v1 |
| `evidence/combined_v1/chain_s2.out.txt` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/runs/combined_v1/chain_s2.out.txt` | `bc97626a8e1e0a00b5144b7ae3cd540e6bd7da610c80073b652effc17f8194e6` | Подлинная запись из каталога runs/combined_v1 |
| `evidence/combined_v1/chain_status.json` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/runs/combined_v1/chain_status.json` | `153040f27361b6b20c1a188cbe687b31e33b13510ac38ea955d84b0f1991d5ec` | Подлинная запись из каталога runs/combined_v1 |
| `evidence/combined_v1/stage1_config.json` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/runs/combined_v1/stage1_config.json` | `0d005396af4daebeb676e065e7050ebd4f8431c8d65b12d37da77265ddf84927` | Подлинная запись из каталога runs/combined_v1 |
| `evidence/combined_v1/stage2_config.json` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/runs/combined_v1/stage2_config.json` | `52c76d157a8a0adb0639ef347d3b3f796289a3331d16ec5d09dfccf0be98a4f5` | Подлинная запись из каталога runs/combined_v1 |
| `evidence/combined_v1/stage1_log.jsonl` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/runs/combined_v1/stage1_log.jsonl` | `757a9670008c3da3cad4a9aa88bfa21eb6a3d382bdca9c46a0636e46221b7344` | Подлинная запись из каталога runs/combined_v1 |
| `evidence/combined_v1/stage2_log.jsonl` | `worker-vm:/home/fedora/lct-reid/jobs/job_48/runs/combined_v1/stage2_log.jsonl` | `fd46a80198d217acb6a88714f6ab15cebc9b6626ad6ec04bee727331c1ec2088` | Подлинная запись из каталога runs/combined_v1 |
| `evidence/datasets/build.log.txt` | `worker-vm:/home/fedora/lct-reid/jobs/job_44/build.log` | `bca6edd21336ed8c2ee49585f377975afb42ec72837629bfcb20a417c832639a` | Исторический build.log и полная проверка выбранных кадров |
| `evidence/datasets/download.sh` | `worker-vm:/home/fedora/lct-reid/jobs/job_44/download.sh` | `69b31cd7967802cab580f0035950c8e220158cb6ba2449c988d36ca654494911` | Исторический build.log и полная проверка выбранных кадров |
| `evidence/datasets/download2.sh` | `worker-vm:/home/fedora/lct-reid/jobs/job_44/download2.sh` | `da94f0ca1e25247159c416a6b75eb67602ddc9899a1897d97e692f92062e2409` | Исторический build.log и полная проверка выбранных кадров |
| `evidence/datasets/download3.sh` | `worker-vm:/home/fedora/lct-reid/jobs/job_44/download3.sh` | `8369d53ce3e25bb8e77d0fce9cce3aada9751357a417d4fa204ec55873c65a0e` | Исторический build.log и полная проверка выбранных кадров |
| `evidence/datasets/finish_carla.sh` | `worker-vm:/home/fedora/lct-reid/jobs/job_44/finish_carla.sh` | `a2b02f22d568a311e7b23aa712f6bfc1f10ae1e67266a747490aa117cc4544f8` | Исторический build.log и полная проверка выбранных кадров |
| `evidence/datasets/dl_roundabout.sh` | `worker-vm:/home/fedora/lct-reid/jobs/job_44/dl_roundabout.sh` | `d09350d3fc15453bffd7d48399e56fb57509101245aede315a97b924f2d2605a` | Исторический build.log и полная проверка выбранных кадров |


## Изменённые и добавленные файлы

Изменены 13 существующих файлов, добавлены 66 файлов; всего 79.

Изменены:

- `04-solution/README.md`
- `04-solution/service/README.md`
- `04-solution/service/app/static/app.js`
- `04-solution/service/app/static/index.html`
- `04-solution/service/artifacts-final/README.md`
- `04-solution/service/model/fetch_model.sh`
- `04-solution/service/offline/README.md`
- `04-solution/service/wheels/README.md`
- `04-solution/training/README.md`
- `04-solution/training/combined/REPORT.md`
- `04-solution/training/input256/REPORT.md`
- `README.md`
- `SOLUTION.md`

Добавлены:

- `04-solution/reproduce/README.md`
- `04-solution/reproduce/check_inputs.py`
- `04-solution/reproduce/evidence/README.md`
- `04-solution/reproduce/evidence/base-archive-inspection.json`
- `04-solution/reproduce/evidence/calibration-headline.json`
- `04-solution/reproduce/evidence/calibration-verification.json`
- `04-solution/reproduce/evidence/checks.json`
- `04-solution/reproduce/evidence/data-layout.json`
- `04-solution/reproduce/evidence/executed-snapshot.json`
- `04-solution/reproduce/evidence/final-artifacts.txt`
- `04-solution/reproduce/evidence/input-tests.txt`
- `04-solution/reproduce/evidence/inputs-test.json`
- `04-solution/reproduce/evidence/inputs-val.json`
- `04-solution/reproduce/evidence/metric-tests.txt`
- `04-solution/reproduce/evidence/missing-validation.txt`
- `04-solution/reproduce/evidence/reproduction-summary.json`
- `04-solution/reproduce/evidence/source-transfer.txt`
- `04-solution/reproduce/evidence/test-report.json`
- `04-solution/reproduce/evidence/test-run-info.json`
- `04-solution/reproduce/evidence/test-run.txt`
- `04-solution/reproduce/evidence/ui-syntax.txt`
- `04-solution/reproduce/evidence/validation-metrics.json`
- `04-solution/reproduce/evidence/validation-run-info.json`
- `04-solution/reproduce/evidence/validation-run.txt`
- `04-solution/reproduce/inputs-manifest.json`
- `04-solution/reproduce/run.py`
- `04-solution/reproduce/test_check_inputs.py`
- `04-solution/service/artifacts-final/SHA256SUMS`
- `04-solution/service/artifacts-final/manifest.json`
- `04-solution/training/src/README.md`
- `04-solution/training/src/SHA256SUMS`
- `04-solution/training/src/evidence/combined_v1/chain.log.txt`
- `04-solution/training/src/evidence/combined_v1/chain_s1.out.txt`
- `04-solution/training/src/evidence/combined_v1/chain_s2.out.txt`
- `04-solution/training/src/evidence/combined_v1/chain_status.json`
- `04-solution/training/src/evidence/combined_v1/stage1_config.json`
- `04-solution/training/src/evidence/combined_v1/stage1_log.jsonl`
- `04-solution/training/src/evidence/combined_v1/stage2_config.json`
- `04-solution/training/src/evidence/combined_v1/stage2_log.jsonl`
- `04-solution/training/src/evidence/dataset-verification.json`
- `04-solution/training/src/evidence/external-selection.jsonl`
- `04-solution/training/src/evidence/initialization-verification.json`
- `04-solution/training/src/evidence/postprocess-environment.json`
- `04-solution/training/src/evidence/remote-artifacts.json`
- `04-solution/training/src/evidence/synthetic-probe-export.json`
- `04-solution/training/src/evidence/whitening-extraction-smoke.json`
- `04-solution/training/src/evidence/whitening-verification.json`
- `04-solution/training/src/evidence/windows-environment.json`
- `04-solution/training/src/evidence/windows-freeze-20260921.txt`
- `04-solution/training/src/initialize_from_onnx.py`
- `04-solution/training/src/postprocess/delta.py`
- `04-solution/training/src/postprocess/lib42.py`
- `04-solution/training/src/postprocess/lib45.py`
- `04-solution/training/src/postprocess/s01_extract.py`
- `04-solution/training/src/postprocess/s02_eval.py`
- `04-solution/training/src/postprocess/s03_plate.py`
- `04-solution/training/src/prepare_own.py`
- `04-solution/training/src/provenance.json`
- `04-solution/training/src/rebuild_whitening.py`
- `04-solution/training/src/windows/build_from_onnx.py`
- `04-solution/training/src/windows/export_onnx2.py`
- `04-solution/training/src/windows/job48_chain.py`
- `04-solution/training/src/windows/osnet_ain.py`
- `04-solution/training/src/windows/reid_metrics.py`
- `04-solution/training/src/windows/reid_train5.py`
- `04-solution/training/src/windows/scope_metrics.py`

Отдельно в каталоге задания: `REPORT.md`, `journal.md`, скрипты сбора/проверки и исходные логи. В Git эти служебные файлы задания не добавлялись.


## Текст добавленных разделов

Ниже дословные тексты новых основных инструкций и разделов итогового рабочего
дерева. Полные изменения существующих документов остаются в `git diff`.

### SOLUTION.md, §7

````markdown
## 7. Как повторить наши числа

В Git есть код, веса и CSV сплита, **но нет изображений организатора**.
Validation-R требует ровно **1860 исходных JPG** по `image_id` из
`04-solution/split/files/val_query.csv` (1110) и `val_gallery.csv` (750).
Список **каждого имени, размера и SHA-256** —
[reproduce/inputs-manifest.json](04-solution/reproduce/inputs-manifest.json),
ключ `data.val`; отпечатки CSV — `repo_files` и
[split/files/manifest.json](04-solution/split/files/manifest.json).
Test-R отдельно требует `test_query.csv`, `test_gallery.csv` и 1860 других кадров
(`data.test` того же manifest). Отсутствие test не препятствует validation.

Участник получает исходный набор через личный кабинет
[задачи №7](https://i.moscow/cabinet/hackaton/lct/contest/4e03d57b5c1d4ef987d8966258c03bd8).
Жюри использует **тот же выданный участникам набор** из своих материалов либо
получает его у постановщика. Отдельного проверенного канала выдачи жюри и
публичного прямого URL архива у нас нет. Если кабинет недоступен, данные нужно
запросить у организатора; из этого репозитория восстановить изображения нельзя.
Изображения не публикуются в Git. Закрытый тест жюри не является данным
validation-сплитом и не воспроизводит наши validation-числа.

```text
$REPO/04-solution/split/files/{val_query,val_gallery,train_fit,split_assignment}.csv
$DATA_DIR/
  images/<image_id>.jpg       # имена из manifest, без дополнительной папки images/images
  test_query.csv              # только для R-test / R-all
  test_gallery.csv            # только для R-test / R-all
  train.csv, README.md        # для подготовки обучения; R-val их не требует
```

SHA-256 test_query.csv — `97e1ed21942bae9c95b1ce2e5d339d9f635bf49fa484367e6b19349789bb9b4c`,
test_gallery.csv — `a64ed21fa39bbfd8c7a172468d043415800500b6ed695cdb8162188cf9005496`.
Полный исходный набор и агрегированный отпечаток изображений — §9.

### Сначала проверить входы

На хосте достаточно Python 3.10+, без NumPy/torch/ONNX Runtime:

```bash
python3 04-solution/reproduce/check_inputs.py --mode val --data-dir "$DATA_DIR" \
  --report "$OUT_DIR/inputs-val.json"
```

Проверяются **все** нужные файлы, размеры и SHA-256. При нехватке данных — exit 2,
число отсутствующих файлов, примеры имён и каталог назначения; полный список
записывается в JSON. Инференс не начинается. Для test замените `val` на `test`.
Эта же проверка встроена в каждую команду R ниже; флаг `--check-only` запускает
только её. Подробнее — [README контура](04-solution/reproduce/README.md).

### R-val: валидация, метрики и оба порога

После сборки/импорта образа из §3–4, из корня репозитория:

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -B /repo/04-solution/reproduce/run.py \
  --mode val --data-dir /data --out-dir /out
```

На SELinux добавьте `--security-opt label=disable` (§3). Скрипт использует
исходники из смонтированного репозитория и зависимости образа; модели проверяет
неизменённый Embedder. Переопределения `REID_*` для эталонного прогона запрещены.
Он заново извлекает признаки, вызывает штатный контур метрик и правило выбора
порогов, проверяет порядок векторов отдельным инференсом граничных строк.
Готовые `.npy` ему не нужны. Выход: `val/`, `inputs-val.json`, `metrics.json`.
Поля и арифметика validation сохранены из прежней команды R; допуск для mAP,
Rank-1/5 и обоих порогов — `1e-6`. F1/TNR даны для `market` + `presence`.

Ожидаемые основные строки (последние цифры порога зависят от CPU/потоков):

```text
cosine: mAP 0.7316093422310448, Rank-1 0.6850961538461539,
        threshold ≈ 0.5141976914190476, F1 0.7684996605566871, TNR 0.7302158273381295
rerank: mAP 0.7740915539438481, Rank-1 0.7307692307692307,
        threshold ≈ 0.5282812306342437, F1 0.8090787716955942, TNR 0.7841726618705036
```

Источник эталонов: [s02_metrics.json](04-solution/training/combined/s02_metrics.json)
и [calib-d1_j48/](04-solution/service/calib-d1_j48/). R не восстанавливает
камерные эксперименты, абляции и bootstrap всех исследовательских отчётов (§8, §11).

### R-test: только сдаваемые файлы выданного теста

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -B /repo/04-solution/reproduce/run.py \
  --mode test --data-dir /data --out-dir /out
```

Выход: `test/{submission.csv,embeddings.npy,candidates.csv,run_info.json}`,
`inputs-test.json`, `test-report.json` с хешами. Метрики качества test не считаются:
его `vehicle_id`/`camera_id` не выданы. Manifest финального выпуска **21.09.2026** —
[artifacts-final/manifest.json](04-solution/service/artifacts-final/manifest.json).
`run_info.json` содержит время, поэтому его хеш пересчёта будет другим.
Для общего прежнего пути используется `--mode all`; он заранее требует обе
части и пишет `metrics.json` с обеими batch-записями. Для произвольного закрытого
теста используйте обычный `app.batch` из §4: manifest R-test намеренно фиксирует
именно выданный участникам набор, а не любой набор того же размера.

### T: тесты контура; V: версии

```bash
docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -w /repo/04-solution/eval \
  vehicle-reid-service python -B -m unittest -q test_metrics test_protocols
docker run --rm --network none vehicle-reid-service python -m pip freeze --all
```

Дополнительно [tools/calibrate_threshold.py](04-solution/service/tools/calibrate_threshold.py)
выполняет 187 сверок точек с контуром (§8); [tools/eval_split.py](04-solution/service/tools/eval_split.py)
оценивает уже извлечённые validation-векторы. Эти инструменты читаются из Git,
Dockerfile не копирует их в образ.
````

### 04-solution/reproduce/README.md

````markdown
# Как повторить наши числа

Этот контур заново извлекает признаки **d1_j48** из исходных изображений и
пересчитывает validation-метрики и оба порога. Он вынесен из команды R в
[SOLUTION.md §7](../../SOLUTION.md), расчёт сервиса и протокол метрик не менялись.
Обучать модель для проверки опубликованных чисел не нужно: веса входят в Git.
Обучение описано отдельно в [training/src/](../training/src/README.md).

## Где взять данные

В Git **нет изображений организатора** и исходных test-CSV. Участник скачивает
набор через личный кабинет [задачи №7](https://i.moscow/cabinet/hackaton/lct/contest/4e03d57b5c1d4ef987d8966258c03bd8).
Жюри использует тот же набор, выданный участникам, из материалов организатора;
если его нет — запрашивает у постановщика. Проверенного отдельного канала выдачи
жюри и прямого публичного URL архива у команды нет. Архив не распространяется
этим репозиторием. Закрытый тест жюри не подменяет нашу валидацию.

| Режим | CSV | Изображения | Результат |
|---|---|---|---|
| `val` (по умолчанию) | `../split/files/val_query.csv`, `val_gallery.csv`; проверяются также `train_fit.csv` и `split_assignment.csv`, все в Git | 1860 JPG из train организатора, перечислены в `data.val` manifest | mAP, Rank-1/5, mAP@10, mINP, оба порога и F1/TNR |
| `test` | `$DATA_DIR/test_query.csv`, `test_gallery.csv`, выдаются организатором | 1860 других JPG, `data.test` manifest | Сдаваемые файлы; качество неизвестно, меток нет |
| `all` | Обе части | Все входы выше | Обе ветви прежней команды R |

Для `val` **не нужны** `train.csv`, test-CSV и test-кадры. Для `test` не нужны
validation-изображения и split-CSV. Дополнительные файлы в `images/` не мешают.

```text
repo/
  04-solution/split/files/{val_query,val_gallery,train_fit,split_assignment}.csv
  data/                       # DATA_DIR может быть вне repo
    images/
      64a7b342e5654ac38e16ab1cc2d58037.jpg
      ...                     # точные имена перечислены в manifest
    test_query.csv            # только test/all
    test_gallery.csv          # только test/all
    train.csv                 # для повторной подготовки обучения, не для R-val
    README.md                 # описание исходного набора
```

## Идентификация входов

[inputs-manifest.json](inputs-manifest.json) содержит **для каждого файла**
относительный путь, размер и полный SHA-256. Он измерен 21.09.2026 на
`worker-vm:~/lct-reid/data` по тому же сплиту; изображения не копировались в Git.
`repo_files` проверяются относительно корня репозитория, `data.val` и `data.test`
относительно `DATA_DIR`. `optional_dataset_metadata` — справочные отпечатки
train.csv и README исходного набора, не обязательные входы R.

| CSV | SHA-256 |
|---|---|
| val_query.csv | `0988668d730e2599f487e25dac8913d718e3b5405442a73ca61123cc8b1a50f6` |
| val_gallery.csv | `6c1a8d65e89dc5e898d12c90f02c6db6d53b031817a6acaf648bcacc1b532a9d` |
| test_query.csv | `97e1ed21942bae9c95b1ce2e5d339d9f635bf49fa484367e6b19349789bb9b4c` |
| test_gallery.csv | `a64ed21fa39bbfd8c7a172468d043415800500b6ed695cdb8162188cf9005496` |

Имена/байты фиксируют именно выданный набор. Не переименовывайте и не
перекодируйте JPG; другой набор тех же размеров проверку не пройдёт.
Для произвольного закрытого теста используйте обычный `app.batch`
([SOLUTION.md §4](../../SOLUTION.md)), он не привязан к этому manifest.

## Проверка до инференса

Из корня репозитория, Python 3.10+, только стандартная библиотека:

```bash
export REPO="$(pwd -P)"
export DATA_DIR="$REPO/data"
export OUT_DIR="$REPO/outputs/reproduce"
mkdir -p "$OUT_DIR"
python3 04-solution/reproduce/check_inputs.py --mode val --data-dir "$DATA_DIR" \
  --report "$OUT_DIR/inputs-val.json"
```

Пример для клона без данных: `Входы val не готовы: нет файлов — 1860`,
далее первые отсутствующие имена и `Положите исходные файлы … в …/data/images`.
Полный список отсутствующих, изменённых и нечитаемых файлов — в JSON; exit **2**.
При успехе — `OK`, все размеры/SHA-256 совпали, exit **0**. Модель и ML-библиотеки
не импортируются. Та же проверка встроена в `run.py`; `--check-only` позволяет
вызвать её через runtime-контейнер.

## Запуск

Соберите или импортируйте `vehicle-reid-service` по [SOLUTION.md §3–4](../../SOLUTION.md).
Для офлайн-сборки нужен локальный Python base, для этих batch-команд Qdrant не нужен.
Текущий [base-images.tar.gz](../service/offline/README.md) содержит один Python с
двумя тегами и требует исправления перед использованием как комплекта Python/Qdrant.

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -B /repo/04-solution/reproduce/run.py \
  --mode val --data-dir /data --out-dir /out
```

На SELinux добавьте `--security-opt label=disable`. Для R-test замените
`--mode val` на `--mode test`, для старого совместного пути — на `--mode all`.
В окружении с установленными runtime-зависимостями можно вызвать тот же
`python 04-solution/reproduce/run.py --mode val --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"`.
Переопределения `REID_*` отвергаются, чтобы не посчитать другой режим под именем эталона.

Validation пишет `val/`, `inputs-val.json`, `metrics.json`. Последний содержит
вычисленные метрики, новую калибровку, хеши весов, версии, проверку порядка строк
отдельным batch=1 на границах и хеши выходов. Проверки с допуском **1e-6** сверяют
mAP/Rank-1/Rank-5 с [s02_metrics.json](../training/combined/s02_metrics.json) и
пороги с [calib-d1_j48/](../service/calib-d1_j48/). mAP считается по 832 запросам
с парой, отказ — `presence`, камеры — `market`; точное правило в SOLUTION.md §6.

| Режим | mAP | Rank-1 | F1 | TNR |
|---|---:|---:|---:|---:|
| cosine | 0.7316093422310448 | 0.6850961538461539 | 0.7684996605566871 | 0.7302158273381295 |
| rerank | 0.7740915539438481 | 0.7307692307692307 | 0.8090787716955942 | 0.7841726618705036 |

Test пишет `test/{submission.csv,embeddings.npy,candidates.csv,run_info.json}`,
`inputs-test.json`, `test-report.json`. `metrics.json` validation не перезаписывается
отдельным test-запуском. Эталонный выпуск test от 21.09 —
[artifacts-final/manifest.json](../service/artifacts-final/manifest.json).
Время в run_info меняется; побайтовое совпадение векторов на другой машине не гарантировано.

R не воспроизводит все исследовательские абляции/бутстрэпы и не переобучает модель.
Порог rerank относится к полному сплиту; его нельзя переносить на малую галерею
без калибровки ([границы применения](../service/README.md#где-применим-порог-переранжирования)).

Исправленный R-val проверен 21.09.2026 из исходных изображений на отдельном
снимке текущих исходников, с read-only mount и без сети; все числа таблицы
совпали. Validation-каталог содержал только её 1860 JPG, без test-CSV и
test-кадров. [Логи, версии, входы и полученные метрики](evidence/README.md).
Использован сохранённый образ аудита job_72; новая сборка и полный Compose
этой проверкой не утверждаются.
Отдельный R-test с test-данными тоже прошёл: все три файла сдачи совпали
побайтово, в run_info изменилось только время. Дополнительный расчёт порогов
через код сервиса прошёл 187 сверок с max_abs_error=0.0.

Проверки раннего отказа и разделения режимов:

```bash
cd 04-solution/reproduce
python3 -B -m unittest -v test_check_inputs
```
````

### 04-solution/training/src/README.md

````markdown
# Фактические исходники обучения combined_v1

Это исходники **сдаваемой модели 208**, перенесённые с вычислительных узлов
21.09.2026. Архивные файлы сохранены побайтово; алгоритмы и конфигурация не
исправлялись при переносе. Оценочный контур готовых весов —
[../../reproduce/](../../reproduce/README.md), он не требует обучения.

## Происхождение и предел доказательства

[provenance.json](provenance.json) перечисляет **каждый перенесённый файл**:
хост, полный исходный путь, размер, время изменения, исходный SHA-256 и SHA-256
после переноса, подтверждающие записи. Все пары SHA совпали. Проверка локальных
копий: `sha256sum -c SHA256SUMS` из этого каталога.

| Файлы | Откуда | Подтверждение связи с combined_v1 |
|---|---|---|
| `windows/reid_train5.py`, `job48_chain.py` | `worker:D:\lct-reid\` | Имена в журнале job_48, команды и параметры в chain/config; копии job_68 совпали. SHA исходников непосредственно при старте 19.09 не записывался |
| `windows/osnet_ain.py`, `export_onnx2.py` | Тот же Windows-каталог | Повторный экспорт сохранённого stage2.pt дал **тот же ONNX SHA-256**, что в Git и chain.log |
| `windows/build_from_onnx.py` | Тот же каталог | Все 559 тензоров preserved initial NPZ и initial PT совпали с инициализаторами сдаваемого исходного OMZ ONNX |
| `windows/reid_metrics.py`, `scope_metrics.py` | Тот же каталог | Самодостаточные импорты trainer и его dev-контур; перенесены именно узловые копии |
| `datasets/build_combined.py` | `worker-vm:~/lct-reid/jobs/job_44/` | Исторический build.log с SHA двух выходов; повторно проверены **все 13 328** выбранных внешних кадров против архивов, packed JPEG и raw массивов |
| `postprocess/s01_extract.py`, `s02_eval.py`, `s03_plate.py` | VM `jobs/job_48/scripts/` | Пути выходов и шаги в журнале; совпадение с отдельно сохранённой копией job_68 |
| `postprocess/lib45.py`, `lib42.py` | VM `jobs/job_45/scripts/`, `jobs/job_42/` | Исторические реализации whitening/bootstrap; `lib45.learn_lw` повторно дал **идентичные f64 и f32 NPZ** из исходных train_fit-векторов |
| `postprocess/delta.py` | VM `jobs/job_55/` | Исходный расчёт сравнения f64/f32, связанный с `training/combined/f32_delta.json` |
| `evidence/combined_v1/*` | VM `jobs/job_48/runs/combined_v1/` | Подлинные config этапов 1/2, все epoch-логи, chain/status; `.log` назван `.log.txt`, содержимое сохранено |
| `evidence/datasets/*` | VM `jobs/job_44/` | Подлинные build/download-записи. Download-скрипты содержат также неиспользованные попытки VeRi/CityFlow; это архив свидетельств, запускать их вместо рецепта ниже не надо |

Сводные проверки: [export и Windows-среда](evidence/windows-environment.json),
[инициализация](evidence/initialization-verification.json),
[whitening](evidence/whitening-verification.json),
[выбор внешних кадров](evidence/dataset-verification.json),
[хеши больших исходных артефактов](evidence/remote-artifacts.json).
Это сильная связь с сохранёнными результатами, **но не криптографическое
доказательство неизменности trainer после 19.09**: source-lock при старте не было.
Полное новое обучение для этого исправления не запускалось.

## Среда и параметры

Исторические `chain.log.txt`/config относятся к 19.09.2026; обучение выполнялось
на Windows, Quadro M2000M 4 ГБ. Записанное тогда окружение — Python 3.12,
torch 2.5.1+cu118. Снимок сохранившейся среды 21.09:

| Компонент | Версия |
|---|---|
| Python | 3.12.8, Windows 10 build 19045 |
| torch / CUDA runtime | 2.5.1+cu118 / 11.8 |
| cuDNN / NVIDIA driver | 90100 / 538.18 |
| NumPy / Pillow / ONNX | 2.1.3 / 11.0.0 / 1.17.0 |
| CPU-постобработка job_45 сейчас | Python 3.14.3, NumPy 2.4.6, ONNX Runtime 1.30.0, Pillow 12.3.0 |
| Сборщик данных job_44 сейчас | Python 3.14.3, NumPy 2.5.3, Pillow 12.3.0 |

Полные установленные пакеты, включая транзитивные, —
[windows-freeze-20260921.txt](evidence/windows-freeze-20260921.txt) и
[postprocess-environment.json](evidence/postprocess-environment.json).
В Windows freeze есть прямые ссылки на локальные torch/torchvision wheels:
это запись факта с SHA колёс, не переносимый `pip install -r`.
Снимок содержит пакеты других экспериментов; их историческое наличие 19.09
не утверждается. Полный freeze **на момент запуска обучения неизвестен**.

Конфиги: [stage1](evidence/combined_v1/stage1_config.json),
[stage2](evidence/combined_v1/stage2_config.json). Seed **20260916**;
torch и torch.Generator инициализируются `SEED + stage`, dev-split — `SEED`.
Прочие seeds и использование NumPy RNG видны в `reid_train5.py`.
`cudnn.benchmark=True`; детерминизм всех CUDA-операций не включён, поэтому
новое обучение не обещает побитовое совпадение весов даже при том же seed.

Batch C4×P4×K4=64, own_k=2; MCNL m1=m2=0.1 + CE label smoothing 0.1;
MixStyle p=0.5, alpha=0.1, flip/pad-crop 10; без Random Erasing;
Adam wd=5e-4, foreach=False, warmup 2, cosine внутри этапа.
Из combined 20 576 кадров dev — 100 own-ID, 372 query / 237 gallery;
train — 19 967 кадров, 2187 классов. Внешняя validation по ID отделена.

## Снапшоты и точный отбор данных

| Архив | Байты | SHA-256 |
|---|---:|---|
| RoundaboutHD.zip | 8 075 314 494 | `fad611b70065bb67dbd7eb6a1c03280b8c1252fe46c291537ccd0b9230f1dd54` |
| VeRi_CARLA_dataset.zip | 2 529 020 889 | `55153a24d2c6c95eb88d9eb44b93b03f7cddc05f28dbc58e03c4c4a5e913908d` |

Хеши полных локальных архивов заново измерены 21.09. Исторически RoundaboutHD
ReID-subset извлекался `7z x` из частично скачанного ZIP; теперь полный ZIP
найден. Все выбранные из него файлы сверены с текущим полным архивом побайтово.
Номер ревизии HuggingFace не сохранялся. URL с `main` изменяемый: **обязательна
проверка SHA**, другой архив нельзя считать тем же снапшотом.
Условия источников и ограничения предобучения раскрыты в [SOLUTION.md §9](../../../SOLUTION.md);
прежний исследовательский отчёт — [datasets-open](../../../02-research/datasets-open/REPORT.md).

```bash
export WORK="$PWD/outputs/combined-rebuild"  # новый каталог, не авторские runs
export DATA_DIR="$PWD/data"                # полный набор организатора
mkdir -p "$WORK/datasets/carla" "$WORK/datasets/roundabout"
curl -fL -o "$WORK/datasets/RoundaboutHD.zip" \
  https://huggingface.co/datasets/yl4300/RoundaboutHD/resolve/main/RoundaboutHD.zip
curl -fL -o "$WORK/datasets/VeRi_CARLA_dataset.zip" \
  'https://www.dropbox.com/s/cg1etrs22y2xb62/VeRi_CARLA_dataset.zip?dl=1'
# Сверьте два SHA-256 с таблицей до распаковки.
sha256sum "$WORK/datasets/"*.zip
unzip -q "$WORK/datasets/VeRi_CARLA_dataset.zip" -d "$WORK/datasets/carla"
unzip -q "$WORK/datasets/RoundaboutHD.zip" 'RoundaboutHD/RoundaboutHD_Reid_subset/*' \
  -d "$WORK/datasets/roundabout"
```

`WORK` — новый рабочий каталог с местом для архивов и массивов; команды сборки
выполняются на Linux CPU, обучение — нативно Windows CUDA. Организаторские
train-изображения берутся из того же полного набора, что в SOLUTION.md §7.
Команды ниже выполняются из корня репозитория. Новые CPU-adapter требуют
Python 3.11+ (`hashlib.file_digest`), NumPy/Pillow; для инициализации нужен ONNX,
для whitening — ONNX Runtime. Проверенные версии CPU-среды приведены выше.

Порядок **own → CARLA → RoundaboutHD** важен. Сборщик сортирует полный glob путей
лексикографически; сохраняет порядок первого появления ID/камер, берёт
по одному следующему кадру с каждой камеры по кругу, максимум **12 на ID**.
Рандома при этом нет. CARLA glob включает `image_train`, `image_gallery`,
`image_query`; Roundabout — весь `RoundaboutHD_Reid_subset`, включая его
train/test-папки. `--cap=1171` в историческом коде только печатает превышение,
не отрезает ID; фактические 1116 внешних ID укладываются в cap.
CARLA: ID+200000, camera+2000; Roundabout: ID+100000, camera+1000.

[external-selection.jsonl](evidence/external-selection.jsonl) задаёт для
**каждого из 13 328 внешних кадров** индекс в combined, исходный member ZIP,
SHA-256 изображения и получившиеся image_id/vehicle_id/camera_id.
CARLA: 7260 кадров / 605 ID, Roundabout: 6068 / 511; own: 7248 / 1171.
Найдено 55 196 и 65 522 JPG соответственно. Поле `files` старого build.log
считает группы камер, а не JPG — его не надо использовать как число изображений.

Own: исходный bbox → RGB → bilinear 256×256 → JPEG q95, subsampling=0,
упаковка в `train_crops.npz`; затем decode этого JPEG → bilinear 208×208
в `crops_208.npy`. Это **не** прямой resize исходного bbox в 208.
Внешние raw 208 получены прямо из внешнего JPEG; packed JPEG отдельно resize256/q95
(у него исторический default subsampling). Источники — существующий
[prep_train_crops.py](../scripts/prep_train_crops.py) и неизменённый
[build_combined.py](datasets/build_combined.py).

Переносимый adapter для own добавлен **в job_73**, вызывает старую функцию
нарезки без изменений и работает последовательно:

```bash
python 04-solution/training/src/prepare_own.py \
  --images-dir "$DATA_DIR/images" --out-dir "$WORK/own"
python 04-solution/training/src/datasets/build_combined.py \
  --own-npz "$WORK/own/train_crops.npz" --own-raw "$WORK/own/crops_208.npy" \
  --add "carla:$WORK/datasets/carla/**/*.jpg:carla:200000:2000" \
  --add "roundabout:$WORK/datasets/roundabout/**/*.jpg:roundabout:100000:1000" \
  --out-npz "$WORK/combined_train.npz" --out-raw "$WORK/combined_crops_208.npy"
```

Ожидаемые отпечатки:

| Файл | SHA-256 |
|---|---|
| train_crops.npz | `a910e8a3a10986845fb83f3a3fcd401bfc8213ac697d50864e36b8c84fefff9d` |
| crops_208.npy | `ea77290d1ad1ae6c245bc33e2ba19bc7c676a8185eda69f9028fe4d1b9e05057` |
| combined_train.npz | `c1ee49dee0873a5748e7b799613fd1e14b4d3689bcd93b3cf51d24d08c42652c` |
| combined_crops_208.npy | `babd228fbe649999a09e9b2d83f3701de9c1b17424a72e23331940dcca3521cc` |

Новый запуск полного own-preparation на VM остановился **до обработки**: там нет
7248 исходных train_fit-JPG. Готовые own NPZ/raw есть, их SHA и побайтовое
совпадение с own-префиксом combined подтверждены. Полное восстановление own из
кадров новым adapter в этой проверке не выполнено. При другой версии JPEG/Pillow
байты могут отличаться; adapter в таком случае завершится ошибкой по SHA.

## Инициализация, обучение и экспорт — в порядке выполнения

Архивные Windows-скрипты содержат `ROOT=D:\lct-reid`; для запуска без их изменения
на новой Windows-машине требуется такая структура:
`py312/python.exe`, все `windows/*.py` в ROOT, `data/combined_train.npz`,
`data/combined_crops_208.npy`, initial PT и probe NPZ в ROOT.
`runs/combined_v1` должен отсутствовать: trainer автоматически продолжает уже
имеющийся checkpoint. Этот рецепт не запускайте поверх авторских сохранённых runs.

1. Восстановить initial NPZ из исходного ONNX. Одноразовый исторический скрипт
   этой операции **не найден**; `initialize_from_onnx.py` — явно новый adapter.
   Все 559 тензоров и полный SHA получившегося NPZ совпали с сохранённым оригиналом
   `35ca9af6702642f63b437df867db14da955669feab5f98fb47a00c95415292c7`.

   ```bash
   python 04-solution/training/src/initialize_from_onnx.py \
     --onnx 04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx \
     --out "$WORK/osnet_ain_init.npz"
   ```

2. Перенести этот NPZ, оба combined-массива и `windows/*.py` на новую Windows-машину
   по структуре выше. В PowerShell, из `D:\lct-reid`, в среде Python 3.12.8 /
   torch 2.5.1+cu118 с NumPy/ONNX из таблицы:

   ```powershell
   .\py312\python.exe .\build_from_onnx.py --npz .\osnet_ain_init.npz --save .\osnet_ain_start.pt
   ```

   Исторический `osnet_ain_start.pt`: 8 958 188 байт,
   SHA `c0a84cbf4d80035dcf9df6275a9b188786da82b3a269813d1b809ae81ca563ea`.
   Все его тензоры совпали с NPZ; формат сериализации PT может менять файловый SHA.

3. Исходный `probe64.npz` содержит организаторские кропы, в Git не переносится.
   Экспортёр использует его только для дополнительного Torch-выхода и размерности;
   ONNX экспортирует на отдельном dummy 1×3×208×208. Для нового запуска можно
   создать синтетический probe (это не проверка качества на реальных кадрах):

   ```powershell
   .\py312\python.exe -c "import numpy as np; np.savez('probe64.npz', x=np.zeros((2,3,208,208),dtype=np.float32))"
   .\py312\python.exe .\job48_chain.py
   ```

4. **Исторический фактический порядок**, подтверждённый `chain.log.txt`, внутри chain:

   ```powershell
   .\py312\python.exe .\reid_train5.py --run combined_v1 --stage 1 --epochs 5 --freeze-fc --lr 3.5e-4
   .\py312\python.exe .\reid_train5.py --run combined_v1 --stage 2 --epochs 15 --resume-from stage1.pt --lr 3.5e-5 --cls-lr 3.5e-4
   .\py312\python.exe .\export_onnx2.py --weights .\runs\combined_v1\stage2.pt --out .\job48_combined_s2.onnx
   ```

   Это раскрытие выполненных команд, **не второе выполнение после chain**.
   Этап 3 был пропущен: CH последних измерений 3.0706 < 3.0717 < 3.0731.
   Его config/log/checkpoint не существовали и здесь не придуманы.
   Экспортирован **конечный stage2.pt (эпоха 15, dev mAP 0.77211)**. Лучшая
   dev mAP 0.77719 на эпохе 13 использовалась для ворот; её веса не сохранены
   отдельным best-checkpoint. При выборе s2/s3 код сравнивает максимумы по этапам,
   но экспортирует конечное состояние выбранного этапа.

Сохранённый исторический `stage2.pt` (SHA `826a7e7a6d826b9077bf6cf14203649a826ea129800baf1f193a2e3faad58965`)
повторно экспортирован 21.09 без тренировки: **8 742 779 байт**, SHA
`b1ba5021275b34079a1653608bdfd215fc9404306dc909852e4cdfa03402efb2`,
совпадает с `service/model/osnet_ain_combined_v1.onnx`.
Отдельно повторён экспорт с синтетическим probe из двух нулевых кропов:
он тоже дал этот SHA ([протокол](evidence/synthetic-probe-export.json)).
Таким образом, закрытый `probe64.npz` не нужен для получения того же ONNX
из сохранённого checkpoint; эта проверка не заменяет повторное обучение.

## Постобработка и whitening

Исторический порядок после переноса ONNX на VM: `s01_extract.py` → `s02_eval.py`
→ `s03_plate.py` (журнал combined). Первые два лежат в `postprocess/` вместе с
`lib45.py`; в архивных entry point сохранены пути `~/lct-reid/...`.
`s02_eval.py` содержит сравнение с прежней ain_v2 и bootstrap, поэтому для
его **полного** запуска нужны старые `.npy/.ids/npz` из job_42/45/attempt-2,
не включённые в Git. Это не скрытая зависимость R-validation.

Для построения именно сдаваемой матрицы добавлен новый переносимый entry point:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python 04-solution/training/src/rebuild_whitening.py \
  --crops "$WORK/own/crops_208.npy" --out-dir "$WORK/whitening"
```

Он извлекает обе модели только на own train_fit, нормирует каждый вектор,
нормирует их сумму, вызывает **исходную `lib45.learn_lw`** с rho=0.5 и
сохраняет P/m в f64 и f32. По умолчанию берёт сдаваемые ONNX из Git;
для нового обучения передайте его ONNX через `--model2`.
Whitening использует все **13 485 межкамерных пар одного ID** только train_fit,
не validation. Перед применением `(x-m) @ P.T` обязательно L2 среднего.

Повтор вычисления из исторических train_fit-векторов дал точные SHA:
f64 `4b02f158fa54d89a54f9ad30d7e471f1baa14d7019ea9ea26b42ea7c267eb625`,
f32 `eb4433ffd5e38d3751d5cb04e234090060a83be6d1720b47bcf2fa1274b3c5b5`.
Это проверка алгоритма/конверсии; все 14 496 train-признаков заново не извлекались.
Дополнительно новый extractor на первых 32 собственных кропах для каждой модели
дал max|Δ|=0.0 против сохранённых векторов
([whitening-extraction-smoke.json](evidence/whitening-extraction-smoke.json)).
Историческая одноразовая shell-команда f64→f32 отдельно не сохранена;
операция `np.savez(..., P=P.astype(np.float32), m=m.astype(np.float32))`
восстановлена и дала исходный SHA, а не выдана за найденный старый файл.

Оставшиеся ограничения: полного source-lock/freeze от момента тренировки нет;
исторический одноразовый extraction ONNX→NPZ не найден; организаторские
изображения/probe не публикуются; некоторые входы сравнений/абляций остаются
в авторских каталогах; происхождение данных предобучения внешней OSNet известно
не полностью. Публикация этих исходников не равна независимому повторному обучению.
````

### service/README.md — область порога

````markdown
### Где применим порог переранжирования

`t_rr=0.5282812306342437` калиброван только на полном validation-сплите:
**1110 запросов × 750 объектов галереи**, 832 запроса с парой / 278 без,
d1_j48, KR(6,3,0,3), `market`, `presence`. Перенос на выданный test той же
размерности — рабочая настройка; его F1/TNR неизвестны. Даже одинаковый размер
другой галереи не подтверждает перенос порога: важен весь состав query/gallery.
Минимального «безопасного размера» не установлено.

На контроле **4 запроса × 10 объектов** batch с этим порогом принял два чужих
автомобиля (0,541908 / 0,768959) и серый кадр (0,801990); косинусный поиск их
отклонил. Источник: [jury-path-2/REPORT.md](../audit/jury-path-2/REPORT.md).
Для малой/произвольной галереи используйте HTTP API или `batch --no-rerank`
с косинусной шкалой и `t_cos=0.5141976914190476`. Этот fallback устраняет
зависимость оценки от соседних запросов, но не гарантирует опубликованные F1/TNR
на новом наборе. Для применения rerank к нему требуется отдельная калибровка на
представительных размеченных положительных и отрицательных запросах.
Автоматического переключения режима по размеру галереи в коде нет.
````

### service/README.md — объяснимость

````markdown
## Объяснимость в сервисе

`POST /api/explain` раскладывает **косинус первой модели OSNet-AIN (OMZ)**
по позициям её карты признаков. Поиск использует **d1_j48: OSNet-AIN +
combined_v1 + whitening**. Поэтому косинус объяснения не является численной
оценкой поиска; это показано рядом с картами в интерфейсе. На контрольной паре
аудита поиск дал 0,053297, объяснение OSNet — 0,131823
([аудит, D7](../audit/jury-path-2/REPORT.md)). Это не приближение: после глобального пулинга голова сети аффинна,
поэтому `cos = Σ вкладов + свободный член` — тождество, проверяемое на каждом
вызове (невязка возвращается в ответе и показывается оператору). Метод, его
проверки и иллюстрации — в `04-solution/explainability/`.

Расчёт в сервисе не тянет новых зависимостей: `app/core/explain.py` читает
нужные веса и собирает копию графа с дополнительным выходом минимальным
разбором protobuf, на стандартной библиотеке. Сессия создаётся лениво, при
первом разборе, — обычный поиск за необязательную возможность не платит.

Для показа кропов и карт каталог кадров монтируется в контейнер `api` на
чтение (`DATA_DIR`, внутри — `/data/images`). Без него поиск и API работают
как прежде, клиент показывает идентификаторы и оценки без картинок.

### Повторная загрузка галереи (другой каталог данных)

```bash
DATA_DIR=/абсолютный/путь/к/data docker compose run --rm loader
```

Шаг идемпотентен: коллекция пересоздаётся, повторный запуск даёт то же
состояние. Загрузчик сам ждёт готовности хранилища (`--wait`, по умолчанию
120 с), поэтому его можно запускать сразу после `up`.

> Podman вместо Docker: `podman-compose up -d` запускает тот же набор сервисов
> (профилей в файле больше нет); при SELinux (Fedora) каталогу данных нужна
> метка контейнера (`chcon -Rt container_file_t "$DATA_DIR"`) либо монтирование
> с `:z`. Ожидание `condition: service_healthy` podman-compose не выполняет —
> его закрывает собственное ожидание внутри загрузчика.
````

### service/offline/README.md — уточнение нового архива

````markdown
# Базовые образы для сборки без интернета: текущий архив неполон

**Проверка job_73 21.09.2026 выявила ошибку архива:** в `manifest.json` только
один образ — Python, config ID `51cce855bb6e44a8ff6ed0f46ded8850f246bfc7549801459f83fc34b80c210f`.
На него навешены оба тега: Python и Qdrant. Настоящего Qdrant внутри нет.
Архив 45 621 040 байт имеет заявленный SHA-256 `cde3ec69c78e1da5881dc8c42afb06a487a37f9944e8384d397fb1e85996bb3c`,
поэтому одна проверка SHA эту ошибку не обнаруживает.
[Manifest и конфигурация образа](../../reproduce/evidence/base-archive-inspection.json).
Не загружайте этот архив как готовую двухобразную поставку: он назначит тег
Qdrant образу Python. В job_73 архив не заменялся.

Колёса Python лежат в `../wheels/`, веса — в `../model/`. Для полной поставки
нужны следующие **два разных** образа, на которые ссылаются Dockerfile/Compose:

| Образ | Назначение | Digest |
|---|---|---|
| `python:3.13-slim` | базовый образ сервиса | `sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285` |
| `qdrant/qdrant:v1.15.5` | векторная база | `sha256:0fb8897412abc81d1c0430a899b9a81eb8328aa634e7242d1bc804c1fe8fe863` |

## Проверка и загрузка после исправления архива

```bash
sha256sum -c base-images.tar.gz.sha256
python3 - <<'PY'
import json, tarfile
with tarfile.open('base-images.tar.gz', 'r:gz') as archive:
    manifest = json.load(archive.extractfile('manifest.json'))
print([(item['Config'], item['RepoTags']) for item in manifest])
assert len({item['Config'] for item in manifest}) == 2, 'Нужны два разных образа; загрузка запрещена'
PY
```

На текущем архиве последняя проверка **завершается ошибкой**. После пересборки,
проверки тегов/различных image ID и успешной проверки выше:

```bash
docker load -i base-images.tar.gz     # или: podman load -i base-images.tar.gz
```

Ранее был зафиксирован exit 0 у `podman load`, но он подтверждал только чтение
архива; два тега не доказывают наличие двух разных образов.

После загрузки собирать и запускать с запретом обращения к реестру:

```bash
podman build --pull=never --network none -t vehicle-reid-service ..
```

Для полной изоляции отключают также внешнюю сеть процесса сборки.
`--pull=never` здесь — флаг **Podman**. У Docker `--pull` булев: после импорта
base используют `docker build --pull=false --network none -t vehicle-reid-service ..`
при отключённой внешней сети daemon/BuildKit. `--pull=false` не запрещает
получить отсутствующий base; Docker-путь ещё требует отдельной проверки.

## Как пересобрать архив

```bash
docker save python:3.13-slim qdrant/qdrant:v1.15.5 | gzip -9 > base-images.tar.gz
```

Для Podman обязательно `podman save --multi-image-archive`: без этого флага
может сохраниться один образ с двумя тегами. Перед экспортом убедитесь, что
теги Python и Qdrant указывают на разные правильные image ID: если ошибочный
архив уже загружали, тег Qdrant нужно сначала восстановить из проверенного источника.
Отпечаток архива после исправления нужно вычислить заново. Исправленный размер
здесь не заявляется; исходная команда создания нынешнего ошибочного архива
в job_73 не установлена, выявлено его фактическое содержимое.

Оба образа распространяются свободно: `python:3.13-slim` — официальный образ Docker
на Debian с Python под лицензией PSF, `qdrant/qdrant` — Apache 2.0.

Архив и его отпечаток обновляются вместе с версиями в `Dockerfile` и
`docker-compose.yml`; при расхождении верным считается `Dockerfile`, а архив надо
пересобрать приведённой выше командой.
````

### Новые строки интерфейса

```text
На что смотрела OSNet-AIN
Объяснение первой модели OSNet-AIN (OMZ): сумма вкладов областей и свободного члена равна её косинусу. Поиск использует d1_j48 — две модели и whitening, поэтому его оценка может отличаться от числа ниже. Красным — то, что поддерживает совпадение, синим — то, что ему мешает.
Модель объяснения: OSNet-AIN · OMZ
Косинус OSNet-AIN: <значение data.cos>
```

