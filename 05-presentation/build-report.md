# Финальный локальный кандидат — 25.09.2026

**PPTX и полный PDF собраны, машинные проверки пройдены, автор просмотрел все 16 страниц.**
Независимую финальную приёмку выполняет отдельный критик. Слайд 7 содержит настоящий
поиск нового UI через API, слайд 14 — результат обученного student. Его ускорение
не компенсировало потерю качества по заранее принятому критерию; релиз — d1_j48.

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
| Полный PPTX | `508c09c598b8e392f0eaf393557a874827de1abc78ef4e17df00f44e9126f6e0` |
| Полный PDF | `1d752ff582066c10cc5200a54e1cbfdb2bb9157de7c920e6c0a28c836879e74a` |
| Технический PDF | `4a3c2d451c2859766ba35f376d44febc9f3242afe2e6ac6198ce473ce7636e92` |

UI source commit: `c79668ac1ab834accf3016275bc2c39a8171261a`.
PNG SHA-256: `d76dd88ea0d653a560dbf5a76e13d27dc8def3a501e68b548096f913f520ec04`.
`assets/ui-provenance.json` содержит версии API, модельные хеши, хеши static и
настоящий demo-сценарий. Локальный адрес API не включён в публичный источник.

Интеграционная база исходников: `e3092125979ff1a75f99fe75af27cbb7bc2cb1c3` плюс
финальный presentation diff. Точный commit кандидата после фиксации и хеш diff —
`checks/final/handoff.json`; существенные входы — `checks/final/inputs.json`.
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
| Независимая финальная визуальная приёмка | PENDING: автор себя не принимает | Ведущий передаёт критикам |

Toolchain: Python 3.13.13, python-pptx 1.0.2, Pillow 12.3.0, lxml 5.3.2,
LibreOffice 25.2.7.2, Poppler 25.02.0; Montserrat из локальных шрифтов.
Снимок среды — checks/final/toolchain.json. Обезличенный рендер worker-vm:
`taskset -c 0,1`, `nice -n 10`, OMP/OPENBLAS=2; чужие процессы/VM не трогались.
Использован штатный renderer проекта; `load_workspace_dependencies` в среде отсутствует.

Harness усилен отдельно в e91c1cb: полный PDF и pdfinfo, без ослабления приёмки.
В этом финальном diff пороги, метрики, эталоны и обязательный шаблон не менялись.
Полный инференс ради дизайна не повторялся.

## Осталось за пределами этой авторской сборки

1. Независимая приёмка финального deck, затем пользовательская оценка дизайна.
2. Подтверждённый регламент в минутах: репетиционный script около 6 минут не назван официальным.
3. Отдельная репетиция выступления и резервного запуска; настоящий API screenshot её не заменяет.
4. Разрешённый deploy 27–28.09, внешний доступ жюри и фактическая загрузка материалов.
   Сейчас **DEPLOY PENDING**; презентация и сервис не объявлены опубликованными.
