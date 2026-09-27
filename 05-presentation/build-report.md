# Локальный кандидат презентации — 26.09.2026

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
| `ЛЦТ2026-задача7-ПРОСВЕТ.pptx` | `8c48f4f254245d0cb4557da262e7247fd9f8c6640592087c0548114b2eee40dc` |
| `checks/ЛЦТ2026-задача7-ПРОСВЕТ.pdf`, 16 страниц | `acbbd740b2aa07c40919e821d1ea27615f2a9110073f0cf7289915a363e2cfb8` |
| `checks/content_preview.pdf`, 11 технических страниц, без контактов | `a85a7494824fb0d6e7f347f25d81914fb513a35eae202f6ece78b3cca4cc1c38` |
| `assets/ui-search.png` | `12e4a0eb0b7b6886dce014977716d777e48431beb2c344c606b1d34d687affea` |
| `assets/owncore-distill-summary.json` | `a53d4d923ac1db215d6e85594fd60eacb15bdc62c949d398db1d3103c1744bfa` |
| `slides.md`, побайтное зеркало Obsidian | `0f2826cbcaecd006c25d197e9c2293e8fac19e08c735e72f3d36ec51aeaa6196` |

UI source: интегрированный checkout `d94d3fdc33c720796574000accf2c1ce75132280`;
API d1_j48 с 750 объектами. Демонстрационный кадр прошёл настоящий read-only API,
а не отрисован вручную. `assets/ui-provenance.json` закрепляет хеши static,
сценария захвата, proxy и PNG. `LCT_PRESENTATION_UI_REPO` указывает на этот
checkout при сборке из отдельного deck worktree. Gate свежести выполняется
штатными build/validate; без переменной в этом старом checkout он ожидаемо STALE.
После интеграции deck-коммитов в checkout с актуальным UI переменная не нужна.

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

Команды запущены из корня `lct-presentation-20260926`; все перечисленные ниже
результаты **PASS**, exit 0. Логи и машинные результаты лежат в ignored
`05-presentation/checks/owncore-final/` и `05-presentation/checks/validation.json`.

| Команда | Проверенное свойство | Лог |
|---|---|---|
| `LCT_PRESENTATION_UI_REPO=… python3 05-presentation/check_ui_provenance.py` | PNG и пять исходников совпали с закреплёнными SHA | терминал; `assets/ui-provenance.json` |
| `python3 05-presentation/test_ui_provenance.py -v` | 1 passed, 0 failed, 0 skipped; stale script даёт отказ | терминал |
| `LCT_PRESENTATION_UI_REPO=… python3 05-presentation/build_presentation.py` | 16 slides, 18 illustrations, 184 text checks, 0 fit warnings | `checks/owncore-final/build.log` |
| `LCT_PRESENTATION_UI_REPO=… OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 05-presentation/validate_presentation.py` | 57 template geometries, 18 pixel comparisons, 16 full / 11 preview PDF pages, 258/181 text blocks, 0 layout failures | `checks/owncore-final/validate.log`, `checks/validation.json` |
| `python3 05-presentation/test_submission_contract.py` | 3 passed, 0 failed, 0 skipped | `checks/owncore-final/contract.log` |
| Office XML/package validator с `--original` на фактический шаблон | 0 issues | `checks/owncore-final/office.log` |
| `python3 05-presentation/render_review.py` | 10 contact fields удалены только из review-копии | `checks/owncore-final/redaction.log` |
| LibreOffice → PDF, `pdftoppm -scale-to 1600 -png` | 16 обезличенных страниц; автор визуально проверил каждую | `checks/owncore-final/render.log`, `checks/owncore-final/pages/`, `slides-report.md` |

После первого визуального рендера единственное найденное наложение — длинная
подпись слайда 12 рядом с номером страницы. Подпись сокращена без изменения
смысла; build, validate, контракт, Office validator и PDF повторены после этой
правки, слайд 12 просмотрен вновь. Итоговые хеши в таблице относятся к этой
последней сборке. Полный текст и embedded notes слайда 14 содержат точные числа,
границы метрик и ссылки на источники. Исходный PPTX и PDF не передавались на
внешний сервис.

Toolchain: Python 3.13.13, python-pptx 1.0.2, Pillow 12.3.0,
lxml 5.3.2, LibreOffice 25.2.7.2, Poppler 25.02.0, локальный Montserrat.

Новая независимая визуальная приёмка обновлённого слайда 14 — **NOT RUN**;
авторский осмотр её не заменяет. Предыдущий независимый PASS на deck 25.09
относился к другим ML-цифрам. Регламент выступления в минутах не подтверждён;
репетиционный сценарий около шести минут. Отключённый от сети резервный прогон
и внешний доступ жюри отдельно **PENDING**. План аренды VPS — 27–28.09;
публикация и загрузка материалов этим отчётом не объявляются выполненными.
