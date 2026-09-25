# Финальный локальный кандидат — 25.09.2026

**PPTX и полный PDF собраны, машинные проверки пройдены, все 16 страниц просмотрены автором и независимым критиком.**
На исходном deck `fcea8f5` критик дал визуальный PASS и указал одну неточную подпись P3:
27 % — снижение p95, а не относительное ускорение. Подпись исправлена, слайд 14
пересобран и просмотрен отдельно. Слайд 7 содержит настоящий поиск нового UI через API;
student снизил задержку, но не прошёл заранее принятый критерий качества, поэтому релиз — d1_j48.

## Артефакты и версии

- `ЛЦТ2026-задача7-ПРОСВЕТ.pptx`: редактируемый полный deck, 16 слайдов.
- `checks/ЛЦТ2026-задача7-ПРОСВЕТ.pdf`: полный PDF, **16 страниц**, локально с контактами.
- `checks/content_preview.pptx` / `.pdf`: 11 технических страниц без контактов.
- `checks/review_redacted.pptx` / `.pdf`: полный просмотр с удалёнными контактами.
- `checks/final/slide-01.png`…`slide-16.png`: все 16 просмотренных страниц.
- `assets/pipeline.dot` / `.svg`: редактируемый источник pipeline; в PPTX схема,
  график и таблица нативные. `slides.md` и embedded notes содержат источники измерений.

| Артефакт | SHA-256 |
|---|---|
| Полный PPTX | `581554874f9a55778bcf8f1d24d3b31042bc702d3045621d44d22c05785d28ac` |
| Полный PDF | `c6211998891669f5b4efba368ef1737c9e01fd38b0435b4d058de80fbbeb7c72` |
| Технический PDF | `3f229a580bc32dae06c9a8ebd1560747b1203940047786414fe79b9f628e4bcd` |

UI source commit: `c79668ac1ab834accf3016275bc2c39a8171261a`.
PNG SHA-256: `d76dd88ea0d653a560dbf5a76e13d27dc8def3a501e68b548096f913f520ec04`.
`assets/ui-provenance.json` содержит версии API, модельные хеши, хеши static и
настоящий demo-сценарий. Локальный адрес API не включён в публичный источник.

Интеграционная база исходников: `e3092125979ff1a75f99fe75af27cbb7bc2cb1c3` плюс
финальный presentation diff. `checks/final/handoff.json` фиксирует исходный
deck `fcea8f5` до точечной редакторской правки P3; её новые хеши приведены выше,
исходник и локальные проверки — `checks/final/integration-*`.
Резерв до редизайна сохранён в отдельном worktree `lct-deck-20260925/checks/baseline/`
(от корня репозитория: `05-presentation/checks/baseline/`). Его файлы не перезаписаны.

PPTX/PDF с контактами и checks исключены из Git. `render_review.py` доказал удаление
ровно 10 contact fields из review-копии; на worker-vm отправлена только она. Полный
PDF с контактами рендерился локально. Контакты не выводились в логи.

## Проверки

Все команды — из корня `lct-delivery-20260925`, каждый приведённый запуск завершён с exit 0.

| Проверка | Результат | Лог |
|---|---|---|
| `python3 05-presentation/build_presentation.py` | PASS; 16 slides, 18 illustrations, 184 text layout checks, 0 fit warnings | checks/final/build.log |
| `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 05-presentation/validate_presentation.py` | PASS; 57 template geometries, 18 pixel comparisons, 258 full / 181 technical text blocks, 16 full / 11 preview pages; 0 issues | checks/final/validate.log; checks/validation.json |
| `python3 05-presentation/test_submission_contract.py` | PASS; 3 passed / 0 failed / 0 skipped | checks/final/contract.log |
| Office schema/package validator, `--original` фактический шаблон | PASS; 0 reported issues | checks/final/office.log |
| `python3 05-presentation/render_review.py` | PASS; 16 slides, 10 contacts removed from review only | checks/final/redaction.log |
| LibreOffice + `pdftoppm -scale-to 1600 -png` на worker-vm | PASS; все 16 страниц; после последней подписи 13 повторены конвертация и только его PNG | checks/final/worker-render.log; worker-render-slide13.log |
| Визуальный просмотр автором | Просмотрены 1–16; русские тексты, шаблон, UI, native chart, кадры и новый layout 14 без обнаруженных обрезок/наложений | checks/final/slide-*.png; slides-report.md |
| Независимая визуальная приёмка 16 слайдов | PASS на `fcea8f5`; одно P3 по названию снижения задержки исправлено | `outputs/independent-review-20260925/deck/REPORT.md` |
| Точечная сборка после P3 | PASS: build/validate/3 contract tests, 16/11 страниц; новый slide14 визуально без обрезки. В notes округление уточнено до «примерно 27 %» | checks/final/integration-build.log; integration-validate.log; integration-contract.log; integration-slide-14.png |
| Независимый retest P3 | PASS; 26,96486 % соответствует подписи «на 27 %», full/preview совпадают | `outputs/independent-review-20260925/deck/P3-retest.md` |

Toolchain: Python 3.13.13, python-pptx 1.0.2, Pillow 12.3.0, lxml 5.3.2,
LibreOffice 25.2.7.2, Poppler 25.02.0; Montserrat из локальных шрифтов.
Снимок среды — checks/final/toolchain.json. Обезличенный рендер worker-vm:
`taskset -c 0,1`, `nice -n 10`, OMP/OPENBLAS=2; чужие процессы/VM не трогались.
Использован штатный renderer проекта; `load_workspace_dependencies` в среде отсутствует.

Harness усилен отдельно в e91c1cb: полный PDF и pdfinfo, без ослабления приёмки.
В этом финальном diff пороги, метрики, эталоны и обязательный шаблон не менялись.
Полный инференс ради дизайна не повторялся.

## Осталось за пределами этой авторской сборки

1. Пользовательская оценка дизайна готового deck.
2. Подтверждённый регламент в минутах: репетиционный script около 6 минут не назван официальным.
3. Отдельная репетиция выступления и резервного запуска; настоящий API screenshot её не заменяет.
4. Разрешённый deploy 27–28.09, внешний доступ жюри и фактическая загрузка материалов.
   Сейчас **DEPLOY PENDING**; презентация и сервис не объявлены опубликованными.
