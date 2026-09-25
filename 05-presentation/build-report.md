# Кандидат презентации — 25.09.2026

**Локальный кандидат собран и машинно проверен. Независимая визуальная приёмка ещё не выполнена.**
16 слайдов, обязательные страницы 7–11 шаблона сохранены. Слайд 7 содержит временное
описание рабочего пути до вставки финального screenshot UI. Новый ML-прототип
не приписан релизу: d1_j48 остаётся базой, новые результаты поступят от ведущего.

## Артефакты

- `ЛЦТ2026-задача7-ПРОСВЕТ.pptx`: редактируемый полный deck, 16 слайдов.
- `checks/ЛЦТ2026-задача7-ПРОСВЕТ.pdf`: полный PDF, **16 страниц**.
- `checks/content_preview.pptx` / `.pdf`: 11 технических страниц без контактов.
- `checks/review_redacted.pptx` / `.pdf`, `checks/slide-01.png`…`slide-16.png`: полный безопасный просмотр.
- `assets/pipeline.dot` / `.svg`: редактируемый источник pipeline. График и таблица в PPTX нативные.
- `slides.md`: источники каждого числа в notes. `защита/сценарий-выступления.md`: репетиционный script.

PPTX SHA-256: `50020f62475204d908ebddc2512c6ec01967f7b663236c6164f3d3fe0c97a4bc`.
Полный PDF SHA-256: `118a41fdbc290b16458d4b0c0bc0d7b0176076be726591afa22eb73fb126da04`.
Файлы с контактами и checks/ исключены из Git. `render_review.py` проверяет удаление
ровно 10 contact fields из review-копии, не редактируя оригинал. На worker-vm передан
только этот обезличенный deck. Полный PDF с контактами рендерится локально.

## Проверки текущей сборки

| Проверка | Результат | Лог |
|---|---|---|
| `python3 05-presentation/build_presentation.py` | PASS, exit 0; 16 slides, 17 illustrations, 185 text layout checks, 0 fit warnings | checks/design-build.log |
| `python3 05-presentation/validate_presentation.py` | PASS, exit 0; 57 template geometries, 17 pixel comparisons, 262 full PDF text blocks; 16 full / 11 preview pages, 0 issues | checks/design-validate.log; validation.json |
| `python3 05-presentation/test_submission_contract.py` | PASS, exit 0; 3 passed / 0 failed / 0 skipped | checks/design-contract.log |
| Office package/schema validator с исходным шаблоном | PASS, exit 0 | checks/office-validate.log |
| `python3 05-presentation/render_review.py` | PASS, exit 0; 16 slides / 10 contacts protected | checks/redaction.log |
| Визуальный просмотр автором | Просмотрены все 16; исправлены spacing шага7 и несовместимый rho в подписи8 | checks/slide-*.png |
| Независимая визуальная приёмка | PENDING, автор себя не принимает | Ведущий передаёт критику |

Источник baseline: `8e7a20580613f5a784c6ee47270a209cf1923e0e`, ветка `deck/20260925`.
Отдельное усиление harness: **e91c1cb**, полный PDF и pdfinfo без ослабления проверок.
Исходные хеши/ограничения: checks/baseline.json. Текущие существенные входы:
checks/candidate-inputs.json. Старые PPTX/PDF сохранены в checks/baseline/.
Финальное удаление пустой строки в конце slides.md не меняет notes: сборщик применяет strip().

Toolchain: Python 3.13.13, python-pptx 1.0.2, Pillow 12.3.0, lxml 5.3.2,
локальный LibreOffice 25.2.7.2 и Poppler 25.02.0. Montserrat установлен из проекта Fedora.
Полный снимок среды — checks/toolchain.json. Обезличенный рендер на worker-vm ограничен
двумя CPU (`taskset -c 0,1`), `nice -n 10`, OMP/OPENBLAS=2; чужие процессы не тронуты.
Использован штатный renderer проекта; инструмент `load_workspace_dependencies` в этой
среде отсутствует, bundled LibreOffice из skill недоступен.

## Открытые зависимости

1. Вставить final UI screenshot и actual API/demo provenance на слайд7, после приёмки UI.
2. По факту нового ML-опыта обновить слайды14/16, сохраняя независимое сравнение и release d1_j48 при отсутствии выигрыша.
3. Подтвердить длительность защиты: репетиционный script около 6 минут не объявлен регламентом организатора.
4. Получить независимую визуальную критику всего deck и исправить конкретные дефекты.
5. После разрешённого deploy 27–28.09 заменить DEPLOY PENDING фактами внешней проверки. Ничего не опубликовано и не отправлено жюри.
