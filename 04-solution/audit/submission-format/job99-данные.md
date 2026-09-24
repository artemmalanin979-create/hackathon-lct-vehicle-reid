# journal — job_99

## 2026-09-24

- Прочитал BRIEF.md. Репозиторий `/home/artem/projects/hackathon-lct-vehicle-reid`, main @ `18ae861` — совпадает.
- Регрессия локализована: `04-solution/service/app/input_checks.py::image_candidates` —
  `if "" in suffixes and Path(image_id).suffix:` трактует ЛЮБОЕ имя с точкой как буквальное
  полное имя (суффикс `.v1` у `frame.v1` не пуст), поэтому кандидат ровно один: `frame.v1`,
  а файл на диске — `frame.v1.jpg` → «Нет изображения», SystemExit(2).
- План правки: буквальная трактовка только когда суффикс — известное расширение изображения
  (`.jpg`/`.jpeg`/`.png`, без учёта регистра). Иначе — прежний детерминированный перебор
  `.jpg, .jpeg, .png, ""`. Это сохраняет:
  - защиту от подмены `named.png` → `named.png.jpg` (и при наличии, и при отсутствии,
    и при повреждении точного файла — как зафиксировано тестами N2);
  - тест приоритета для имён без расширения (`case0.jpg` раньше `case0`);
  - контракт `suffixes=('.jpg',)` исследовательского префлайта (`""` нет в списке — ветка
    не срабатывает, как и раньше).
- Фикстуры критика на месте: `~/tmp/lct-jobs/job_98/fixtures/minimal-dotted/` (frame.v1.jpg, q.csv),
  доказательства в `~/tmp/lct-jobs/job_98/evidence/minimal-dotted*.json`.

- `ssh worker-vm` РАБОТАЕТ (ожидание из брифа «Network is unreachable» не подтвердилось):
  `sespel-worker.local`, 4 vCPU, 31 ГиБ RAM, есть podman и образы reid. Тяжёлый полный
  прогон (reproduce/run.py --mode val, 1860 изображений) сделаю там.
- Воспроизведение «до» на HEAD 18ae861: обе фикстуры падают с кодом 2 —
  критикова (`job_98/fixtures/minimal-dotted`) и моя расширенная
  (`fixtures/dotted`: frame.v1, cam.2026.09.24, alias.v2→symlink).
  Свидетельства: `evidence/before-critic.stderr`, `evidence/before-dotted.stderr`.
- Правка внесена: `input_checks.py::image_candidates` — буквальная трактовка только для
  суффиксов `.jpg/.jpeg/.png` (без учёта регистра); README сервиса уточнён; +3 теста
  в `tests/test_image_names.py` (точечные ID, symlink, сохранение защиты от подмены
  для полного имени с точками: `frame.v1.png` ≠ `frame.v1.png.jpg`).
- Тестовые окружения: у репозитория есть `.venv` (numpy/PIL/onnxruntime, без fastapi);
  у job_98 — `unit-env` с полным набором. Целевые метрики — rerank mAP/Rank-1 из
  таблицы reproduce/README.md: 0.7740915539438481 / 0.7307692307692307.

- Матрица разрешения имён (12 случаев): после правки 12/12 OK
  (`evidence/name-matrix-after.json`); на 18ae861 — 3 FAIL: символическая ссылка,
  одна точка, несколько точек (`evidence/name-matrix-before.json`); обе защиты от
  подмены OK в обеих версиях.
- CLI после правки: обе фикстуры проходят, exit 0, артефакты записаны
  (`evidence/after-critic.stderr`, `evidence/after-dotted.*`, `results/after-*`).
- Тесты: 7 наборов = 146 (52 service вкл. 3 новых + 60 metrics + 16 inputs +
  7 numeric-audit + 3 runtime-cli + 5 documentation + 3 presentation), все OK,
  плюс отдельный check_build_pdf_render (1) OK. Существующих 144 (49 service на
  18ae861 + 94 остальных + 1 render) — все проходят. Логи: `results/tests/`.
- Коммит правки: 36d08f9 в main. Push не делаю.
- Полный прогон: rsync репозитория (без .git/.venv/data/outputs) на
  worker-vm:~/lct-reid/jobs/job_99/repo; затем podman offline 2cpu/4g,
  образ localhost/job72-jury:6179b69, reproduce/run.py --mode val,
  данные ~/lct-reid/data (3720 изображений на месте).
- Дополнительный коммит c459c26: докстринг resolve_image_path приведён к новому правилу
  (только комментарий, поведение не меняет).
- Передача на worker: прямой rsync шёл ~100 КБ/с (≈1,5 ч на 590 МБ) — остановлен.
  Вместо этого каталог засеян жёсткими ссылками из старой копии ~/lct-reid/repo на самом
  worker (5,6 ГБ мгновенно), затем дельта-rsync с --checksum передаёт только изменённое.
- Проверил: другой логики подбора расширений в app/ нет — image_candidates единственная
  точка правды, префлайт и чтение кропа используют её же.
- Образ podman на worker есть: localhost/job72-jury:6179b69; скрипт worker_val_run.sh
  загружен (offline, --cpus 2 --memory 4g, /repo и /data только чтение).
