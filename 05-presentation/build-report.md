# Опубликованная презентация — 28.09.2026

Собраны 16-слайдовый редактируемый PPTX, полный PDF и отдельный
**contact-free technical preview** на 11 страниц. Релизная модель на всех
слайдах — d1_j48. Отдельное обученное CrossViewCore показано как отрицательный
исследовательский результат: быстрее в ограниченном CPU-замере, но KR mAP и отказ
не прошли заранее зафиксированный критерий. Повторно использованная val не
объявляется новым holdout.

Обязательные страницы 7–11 шаблона остаются итоговыми 1–5; слайд 3 содержит
контакты только в локальном полном PPTX/PDF. Файлы с контактами, локальный
`team-data.md`, проверки и рендеры исключены из Git. `render_review.py` удалил
ровно 10 contact fields из отдельной копии для просмотра, исходный PPTX не менял.
В Git входят только код сборки, notes, источники схем, screenshot разрешённого
демокадра и машиночитаемое резюме ML без персональных данных.

| Локальный артефакт | SHA-256 |
|---|---|
| `ЛЦТ2026-задача7-ПРОСВЕТ.pptx` | `1ef32e53d1b24f663fcd9444ba1902bf30323253f02a644c6a43a1a25fd097f3` |
| `checks/ЛЦТ2026-задача7-ПРОСВЕТ.pdf`, 16 страниц | `46ced0396e4ead71f0ce496ac55a2e31935b4b66997f9bf2d35a7bb18188a2b4` |
| `checks/content_preview.pdf`, 11 технических страниц, без контактов | `544aa32fe556025a8166031d54e35d69b42884333bfd279512d50b6681047afe` |
| `assets/ui-search.png` | `c737fd4bb7d924f9da0be07e7e584af03f9798ee3e6c464411cef0ed7c99beba` |
| `assets/owncore-distill-summary.json` | `a53d4d923ac1db215d6e85594fd60eacb15bdc62c949d398db1d3103c1744bfa` |
| `slides.md`, побайтное зеркало Obsidian | `2fac5b19c3fc3f9a7304abdcd45a48d118d920aa3b97a2af8ff00ef53b292d32` |

UI source: интегрированный checkout `6e8c1c9d7be015073252f1d30da181737894067c`;
развёрнутый API d1_j48 с 750 объектами. Демонстрационный кадр прошёл настоящий
API без готовой выдачи. `assets/ui-provenance.json` закрепляет хеши static,
каталога demo, сценария захвата, proxy и PNG. Новый кадр `e03daa7c…` имеет полный
bbox и отличающиеся первые кандидаты; у закрытого test нет identity/camera labels.
Gate свежести выполнен штатными build/validate в интегрированном checkout.

ML source: `d94d3fd`; хеши первичных локальных Holm8, evaluation и benchmark
зафиксированы в `assets/owncore-distill-summary.json` и notes слайда 14.
На повторно использованной val 1110×750, 832 query с парой: d1_j48 KR mAP
0,774092, CrossViewCore 0,327450, fusion 0,762604. Для core ΔmAP −0,446641,
CI95 [−0,492405; −0,400641], p_Holm(8)=0,003999. Для fusion CI95 дельты
[−0,026954; +0,003198], доказанного выигрыша нет. Отказ core TNR=0,0.
На одном Linux CPU, 2 потока, 40 парных кадров, p95 готовый RGB208 tensor→descriptor:
43,86 против 1,73 мс. JPEG, API, Qdrant и KR вне границы этого замера; исторические
времена с другого узла с ним не сравниваются.

## Проверки

Команды запущены 28.09 из интегрированного корня репозитория; все перечисленные
результаты **PASS**, exit 0. Логи и машинные результаты лежат в ignored
`05-presentation/checks/`.

| Команда | Проверенное свойство | Лог |
|---|---|---|
| `python3 05-presentation/check_ui_provenance.py` | PNG и 7 исходников совпали с SHA | `checks/deck-live-provenance.json` |
| `python3 05-presentation/build_presentation.py` | 16 slides, 18 illustrations, 184 text checks, 0 fit warnings | `checks/deck-live-build.log` |
| `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 05-presentation/validate_presentation.py` | 57 template geometries, 18 pixel comparisons, 16 full / 11 preview PDF pages, 258/181 text blocks, 0 layout failures | `checks/deck-live-validate.log`, `checks/validation.json` |
| `python3 05-presentation/test_submission_contract.py` | 3 passed, 0 failed, 0 skipped | `checks/deck-live-contract.log` |
| Office XML/package validator с `--original` на фактический шаблон | 0 issues | `checks/deck-live-office-schema.log` |
| `python3 05-presentation/render_review.py` | 10 contact fields удалены только из review-копии | `checks/deck-live-redact.log` |
| LibreOffice → PDF, затем 1600×900 JPG | 16 обезличенных страниц просмотрены автором, включая 7/10/16 в масштабе 1:1 | `checks/deck-live-redacted-render.log`, `/home/artem/tmp/lct-deck-live-final-all/` |

Ранние исправления слайдов 11–14 сохранены. В этой сборке обновлены реальный
скриншот слайда 7 и статус HTTPS слайда 16; обязательные страницы 1–5 остались
по шаблону. Автор просмотрел все 16 обезличенных страниц, без наложений и
обрезки. Числа и границы измерений остаются в заметках. Итоговые хеши относятся
к опубликованной версии PPTX/PDF, а не к прежнему предварительному deck.

Toolchain: Python 3.13.13, python-pptx 1.0.2, Pillow 12.3.0,
lxml 5.3.2, LibreOffice 25.2.7.2, Poppler 25.02.0, локальный Montserrat.

Независимая визуальная приёмка текущей пересборки — **PASS**: критик осмотрел
все 16 обезличенных страниц, отдельно 7/10/16, не нашёл видимых дефектов,
сверил 11 страниц technical preview с полным PDF по тексту. Контакты страницы 3
он видел только замаскированными; полный вариант проверяет validator.
Регламент выступления в минутах не подтверждён;
репетиционный сценарий около шести минут. Отключённый от сети резервный прогон
**PENDING**. PPTX/PDF опубликованы с HTTPS, их SHA-256 сверены с локальными и
серверными файлами; сохранение ссылки в форме организатора этим отчётом не
подтверждается.
