# Воспроизведение job_66

Оригинальная выполненная установка и веса находятся на `worker-vm` в `~/lct-reid/jobs/job_66/`. Исходный корпус и исторические результаты Qwen лежат здесь в `baseline/`; менять их не нужно. Данные не отправляются во внешние inference API.

Перед каждым обращением к узлу проверяется маршрут `ip route get 100.117.38.29`: нужен `dev tailscale0 table 52`. Локальный `remote.sh` реализует проверку и указанное в BRIEF исправление правила.

Для независимого повторения создайте новый каталог результатов на worker-vm, скопируйте туда `baseline/`, `scripts/`, `sources/hf_model_api.json`, `protocol.json`, `dev_claims_en.json` и сохранённый lockfile. Не повторяйте holdout в исходной папке job_66: харнес специально запрещает это. Новый повтор — отдельный эксперимент, не часть исходной однократной оценки.

Установка использует Python 3.12.14 и отдельный venv:

```bash
uv venv --python 3.12.14 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
uv pip install --python .venv/bin/python -r sources/requirements.lock.txt
nice -n 15 ionice -c 3 taskset -c 6,7 python3 scripts/download_model.py
```

`download_model.py` загружает конкретную HF revision через hf-mirror, затем использует HF как резерв. Каждый файл проверяется: веса по SHA-256/LFS-oid из оригинального API, небольшие файлы по git blob SHA-1 и размеру. Архив PyPI и его метаданные сохранены в `sources/`.

```bash
bash scripts/run_phase.sh audit > logs/audit.log 2>&1
bash scripts/run_phase.sh dev > logs/dev.log 2>&1
bash scripts/run_phase.sh holdout > logs/holdout.log 2>&1
```

Нужны существующие каталоги `logs/` и `results/`. Сначала завершается dev: три фиксированные инструкции × 20 исходных примеров, затем выбор по заранее заданному правилу и отдельные 20 dev-переводов. `results/selection.json` фиксирует выбор и хеши кода/протокола до holdout. Holdout читает только выбранную конфигурацию, создаёт `holdout.started.json`, делает ровно 10 вызовов и создаёт `holdout.completed.json`. Повторный запуск поверх существующих результатов запрещён.

После обоих этапов `python3 scripts/analyze_results.py` без inference повторно считает метрики, проверяет сплит, протокол, входы, число вызовов holdout, идентичность evidence в языковой паре и формирует `results/comparison.csv`, `errors.csv`, `language.csv`, `analysis.json`, `validation.json`.

Используется `laya.load(local_model, device='cpu')`, затем `Router.attach('typed-decisions', agent)` и `Router.predict(..., model='typed-decisions')`. Другие модели семейства не загружаются. Вызов содержит только `claim`, пути и выдержки; gold-метки, id и название сплита в модель не передаются.

Ограничения: affinity CPU 6,7, `nice=15`, `ionice=idle`, 2 intra-op потока и 1 inter-op поток; CUDA не используется. Контекст 1024 включает заголовок и варианты noul. В `token_audit` каждого результата сохранены точный вход, длина и каждое сокращение выдержки. Инструкции не обрезаются.

ECE считается по уверенности выбранного ответа, 10 равных бинов, как в исходном Qwen-харнесе. Предсказание «да» при `noul > 0.5`, при равенстве — «нет». Вероятности берутся из SDK без новой калибровки. Исторические результаты Qwen повторно прогонять через модель не требуется.
