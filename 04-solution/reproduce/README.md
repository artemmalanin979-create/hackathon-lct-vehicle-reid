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
