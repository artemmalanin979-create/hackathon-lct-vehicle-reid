# CrossViewCore на внешнем KPNEUMA — протокол до инференса

Зафиксировано 28.09.2026 **до запуска CrossViewCore на KPNEUMA**. Основной
checkout: `ed028e82702ee7d80904b5a49cd5cc5c23b5505c`. Это проверка уже
обученной модели, без дообучения, выбора checkpoint, веса fusion или порога.
Сдаваемая модель `d1_j48` не меняется.

## Модель

Берём **combined** CrossViewCore: самостоятельный MobileNetV3-Small с
global/local головой, `RGB image → 512D L2 descriptor`, без teacher при
инференсе. ONNX находится вне Git в старом worktree
`lct-owncore-distill-20260926/outputs/owncore-distill-20260926/assessment-combined-d94d3fd/core.onnx`.
SHA-256 ONNX:
`4cb47b31cf2abf3eabb9fdf30f8de5bc4f06cfe2306670a01fe1bc9e95961df6`.
Источник — checkpoint
`outputs/owncore-distill-20260926/train-combined-d94d3fd/checkpoint.pt`,
SHA-256 `30d98c3f4b8c4a99421dc4feb4f835f8e467b00f306bbe78ba3ab5b8cfd51467`.
Проверенный export manifest — `assessment-combined-d94d3fd/export.json`;
его frozen training protocol SHA-256
`648f60b5e3a2a0004bf33aab2d432eba0e751b26393ff83f66ff84f7239458ac`.

Вход: весь исходный PNG-кроп, `PIL.convert("RGB")`, bilinear resize до 208×208,
`float32 NCHW 0..255`. Дополнительная нормализация происходит **внутри** ONNX.
Декодирование и bbox используют штатный `app.core.preprocess.load_crop`.

## Данные, метрика и фиксированные проверки

Ровно тот же внешний тест, что и у релиза: 318 query и 318 gallery PNG из
`KPNEUMA`, без train-части. Архив `data.tar.gz` SHA-256
`c55c90b1b2e87e86991d47e18a52ce39b7839f2a17a8142be90320a4b9821332`.
Отсортированные `имя + NUL + SHA-256(содержимое)` имеют SHA-256
`9c2f87f21ce2db2087ce904b0e3d1066ab31e1e60886e00390b8607d6d23d7ae`
для query и
`19a071fee864e02c9f3eda15873979d35b2f469fb63c2cee04a6ef8f9d4d3ea1`
для gallery. Имена задают ID и виртуальный детектор; все 318 query должны
иметь ровно одну gallery-пару другого детектора. Это проверяется и записывается
отдельным preflight **до инференса**.

Сортировка по полному имени файла. `camera_policy=market`: совпадение того же
ID и детектора исключается. mAP, Rank-1/5, mINP — полный gallery ranking через
`04-solution/eval/reid_metrics.py`. Cosine и KR `k1=6,k2=3,lambda=0.3` из тех
же признаков, без настройки по результату. Положительный контроль — идеальная
матрица ID, отрицательный — случайные нормированные векторы с seed `20260928`.
Bootstrap по 318 ID: 2000 повторов, seed `20260928`. Пороги отказа, F1/TNR
не считаются: в этом наборе нет запросов без пары. Числа релиза для сравнения
берутся из `04-solution/eval/external-kpneuma-20260928/results.json`, SHA-256
`6ed7bb8d8f87daab1f9e08697fcefa8d8ec8ca3dcab22559ff5f2a59b4e2f0ab`.

Привязанные реализации, SHA-256: `preprocess.py`
`4a6a203feb059d1626ab3a683dc505956e36c3280d746a96a0abd593238af686`,
`ranking.py` `92aebe6c65dd54b9375abbb56c26cafd4e6344c8ef5c27a638dde4d10c19c6b9`,
`rerank.py` `ac946451b2e2ae901e4597667b4b5bfe9a77d85d94caf2e369636cee876feb76`,
`reid_metrics.py` `79fba7051bc1fd6e856230f6a256334196c7ae7f14767ad85175aaeabe37ffec`,
`scope_metrics.py` `6dd489984a531ad2d07b58b1e963d61f015e5d4ee780187de90b424b71e463df`,
внешний release runner `run.py`
`138558c7c4bb711ead6ddad7b74bfa4f0dcfed40ab033181ec71592a7365490a`.

CPU-only, ORT 1 thread, batch 16, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`,
`nice -n 10`. Замер полного прохода не является парным benchmark с релизом:
для сравнения скорости брать только уже проведённый парный тест на val.

Граница вывода: этот набор — аэросъёмка и виртуальные детекторы, не закрытый
тест ЛЦТ. Он не использовался для обучения нового ядра, но результат релиза
уже был известен при фиксации этого протокола. Изображения остаются локально;
лицензия датасета в источнике не указана, в Git их не добавлять.
