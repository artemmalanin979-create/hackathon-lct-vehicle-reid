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
