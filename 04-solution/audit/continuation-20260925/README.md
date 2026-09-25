# Проверка сдаваемой основы 25.09.2026

База `8e7a20580613f5a784c6ee47270a209cf1923e0e`. Изолированная ветка `delivery/20260925`. Численный inference, ранжирование, пороги и финальные test-артефакты не менялись.

Независимый read-only аудит release_audit подтвердил SHA четырёх test-артефактов и трёх model-файлов. M5 остаётся пробелом теста research suffixes: теперь `car.png` с `suffixes=('.jpg',)` принимает `car.png.jpg` и отвергает только литеральный `car.png`. Для обычного сервиса тот же литеральный PNG допустим. Продуктовый код resolver не менялся.

M8 уже защищён существующим `reproduce/test_cli_inputs.py::test_stream_preflight_does_not_consume_or_open_csv`; предыдущий прогон запускал только service suite. Добавления второго FIFO-теста нет.

## Свежие проверки

| Проверка | Результат | Свидетельство в audit/continuation-20260925/evidence |
|---|---|---|
| Исходный service suite | PASS, 52 tests, 0 skip | baseline-service-actual.txt |
| Service после теста M5 | PASS, 53 tests, 0 skip | service-after-M5.txt |
| CLI input suite | PASS, 16 tests, 0 skip | baseline-inputs.txt |
| Документация/PDF | PASS, 5 tests, 0 skip | baseline-docs.txt |
| M5 | GREEN 9 → FAIL (1 failure, 1 error) → GREEN 9 | M5*.log, M5.diff |
| M8 | GREEN 1 (pipe и FIFO) → FAIL (1 failure, 1 error) → GREEN 1 | M8*.log, M8.diff |

Оба изменения применены ровно один раз, diff непустой, после каждой проверки восстановлен SHA `app/input_checks.py`. Мутации запускались только в временной копии внутри Podman `--network=none`, с 1 CPU/512 MiB, отдельным временным pycache, без dataset/БД. M8 блокируется именно на чтении FIFO до появления writer; это детерминированный сценарий теста, ограниченный его subprocess timeout 10 s. Дополнительно pipe даёт неверный отказ, поскольку мутант прочитал поток в preflight. Внешний timeout контейнера 75 s не срабатывал и не считается доказательством.

Команда воспроизведения из корня:

```bash
python3 04-solution/audit/continuation-20260925/check_input_mutations.py \
  --image localhost/lct-continuation:baseline \
  --out-dir outputs/continuation-20260925/mutations
```

Runtime собран штатным Dockerfile из локальных wheels с `--pull=never --network=none`; image ID `fd00bb90471c650466ff10dbea3cc79f678d0b92fe100085d048359a770d9c0a`, Python3.13/ORT1.30.0/NumPy2.5.3. Metadata, команды, тестовые хеши, exit code и время каждого запуска — `evidence/result.json`. Первую ошибочную попытку запуска до готовности image не считаем тестом; она отдельно сохранена в outputs/baseline-service.log и завершилась инфраструктурной ошибкой localhost registry. Фактический suite запущен после сборки с `--pull=never`.

`pdfinfo` присутствует (25.02.0), старый BLOCKED закрыт выполнением проверок. Полный validation inference не повторялся: его входы и граф не изменены. Задокументированные границы маленькой/пустой gallery, повторных ID, округления, количества candidates и batch KR сохраняются. Полной готовности сдачи эта проверка не доказывает: UI/deck/ML и внешняя публикация проверяются отдельно.
