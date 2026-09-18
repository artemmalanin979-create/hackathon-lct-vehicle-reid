# REPORT — job_45: складываются ли whitening и ансамбль, и проходят ли они абляцию номера

Статус: **в работе** (2026-09-17). Узел `worker-vm`, файлы `~/lct-reid/jobs/job_45/` (скрипты `scripts/`,
результаты `out/`, логи `logs/`). Метрики — только контуром `04-solution/eval/reid_metrics.py` (импорт без
изменений); KR — `04-solution/postproc/scripts/common.py::rerank` (6, 3, 0,3); абляция — неизменённые
скрипты `04-solution/plate-ablation/scripts/` (побайтовые копии в `mirror/*/`, sha256 —
`out/mirror_scripts_sha256.txt`); отказ — `refusal/scripts/metric_adapter.py` (импорт). Сплит
`04-solution/split/files/` (1110 / 750; 832 запроса с парой, 278 без).

## 0. Что переиспользовано и что проверено до счёта

| Что | Откуда | Проверка |
|---|---|---|
| val-векторы OSNet и ain_v2 | `04-solution/training/attempt-2/out/val_{query,gallery}_{osnet,ainv2}.npy` | порядок `.ids` = CSV сплита; контур на них даёт ровно 0,6936584724586873 (KR) — `out/s02_configs.json → table.a.kr.mAP` |
| матрица whitening OSNet (ρ = 0,5, train_fit) | `~/lct-reid/jobs/job_42/out/lw_P_rho0.5.npy`, `lw_m_rho0.5.npy` (float32 — как пойдёт в решение) | (b) воспроизводит 0,7366 job_42 — см. табл. 1 |
| train_fit-векторы OSNet (7248, путь crops_208) | `~/lct-reid/jobs/job_42/out/train_fit_osnet.npy` | ids = `train_fit.csv` |
| train_fit-векторы ain_v2 (7248) | **новые**, `out/train_fit_ainv2.npy` (`scripts/s01_extract_ainv2_train.py`; job_43 извлёк только 2313 строк) | max\|Δ\| на 2313 строках job_43 — `out/s01_extract_info.json` |
| кропы val из кадров | `~/lct-reid/data/images/` по рамкам `val_{query,gallery}.csv`, кодом `extract_variants.py` (`to_input`) | **первые 64 кропа: cos ≥ 0,99999999999 против готовых векторов (OSNet и ain_v2), max\|Δ\| 2,8·10⁻⁷ / 2,3·10⁻⁷** — `out/s00_check64.json` |
| детектор пластины, маски, контроли | `cache_boxes.py --procs 4`, `extract_variants.py`, `eval_variants.py` — без правок | `out/boxes.json`; воспроизведение базовой абляции (ожидается −0,003 [−0,011; +0,004]) — табл. 3 |

Оговорка по запуску (не правка кода): `extract_variants.py` создаёт сессию ORT с потоками по умолчанию —
на узле это 8 потоков под cgroup-квоту 4 ядра, 3,5 кроп/с; с `intra_op=4` — 10 кроп/с
(`logs/bench_threads.log`). Скрипт запускался неизменённым через `scripts/run_stage.py` (runpy, подмена
только конструктора `InferenceSession`: intra_op 4, inter_op 1). На результат влияет лишь на уровне
порядка суммирования (≤ 10⁻⁶ на векторе).
