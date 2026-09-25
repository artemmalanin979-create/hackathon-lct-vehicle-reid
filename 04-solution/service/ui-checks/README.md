# Проверка пользовательского пути

Клиент остаётся vanilla HTML/CSS/JS. Инструменты ниже не входят в runtime
образа и не требуют внешней сети. Нужны Playwright и Chrome; кроме
`resilience.cjs`, нужна локальная репетиция настоящего API с заполненной
галереей d1_j48.

Рабочая проверочная среда: Node 20.20.2, Playwright 1.63.0, Chrome
153.0.8010.52. Скрипты создают только собственные browser contexts и файлы в
явном каталоге вывода. Они не запускают/останавливают API, не пересоздают
галерею и не касаются чужих browser profiles.

```bash
export PLAYWRIGHT_MODULE=/путь/к/node_modules/playwright
node 04-solution/service/ui-checks/behavior.cjs \
  http://127.0.0.1:18070 /путь/к/data /путь/к/logs/behavior
node 04-solution/service/ui-checks/acceptance.cjs \
  http://127.0.0.1:18070 /путь/к/data /путь/к/logs/acceptance
node 04-solution/service/ui-checks/mutation.cjs \
  http://127.0.0.1:18070 /путь/к/data /путь/к/logs/mutation
node 04-solution/service/ui-checks/presentation-shots.cjs \
  http://127.0.0.1:18070 /путь/к/logs/deck-shots
node 04-solution/service/ui-checks/resilience.cjs \
  /путь/к/logs/resilience
```

- `behavior.cjs`: 8 проверок причинности запросов, ошибок, выгрузки и
  out-of-order объяснений. Нормальные и задержанные ответы приходят от
  настоящего API; задержка ставится **после разбора настоящего JSON**, поэтому
  отдельно проверяет revision guard, а не только AbortController.
- `acceptance.cjs`: 17 целевых сценариев. Реальные поиск, demo, сравнение,
  объяснение и экспорт; обе темы × 390/768/1440, mouse/touch, numeric bbox,
  200% масштаб с проверкой геометрии, loading/refusal/error, Unicode-имена,
  повтор готовности галереи, клавиатура. Сбои сети, пустая галерея и
  недоступное объяснение явно подменяются в browser context: производственные
  контейнеры ради них не ломаются. Снимки до завершения загрузки изображений
  не делаются.
- `bbox-geometry.cjs`: отдельный regression P2: обе темы × 100/200% ×
  mouse/touch; все x/y/w/h, неподвижность canvas во время drag, перенос, resize,
  пустой клик, настоящие POST-поля и JSON-экспорт. Аргументы BASE DATA OUT те же.
- `mutation.cjs`: loopback-only. Штатный GREEN → одна копия JS с удалённым
  guard → именно FAIL stale export → неизменность SHA исходника → GREEN.
  Подмена JS существует только внутри одноразового browser context;
  production static и build-cache не меняются.
- `presentation-shots.cjs`: 10 реальных viewport-снимков 1440 px: обложка,
  поиск, сравнение, объяснение, отказ в двух темах.
- `capture.cjs`: исходная матрица экранов, поддерживающая baseline-интерфейс.
- `resilience.cjs`: лёгкая изолированная браузерная проверка отказов клиента.
  Ответы `/api/ui/state`, `/api/search` и изображения подменены внутри одного
  browser context: 409, HTML 503 и сетевой отказ обновляют состояние; смена
  модели в обычной вкладке снимает старый экспорт, включая проверку прямо
  перед скачиванием; недоступный кадр пары получает понятное состояние и
  восстанавливается при выборе другого кандидата. Скрипт не запускает API и
  не измеряет качество модели; реальный путь проверяют `behavior.cjs` и
  `acceptance.cjs`.

`TEST_FILTER` ограничивает сценарии регулярным выражением. JSON результатов
содержит command, HEAD, branch, source/input SHA-256, toolchain, counts и
ссылки на снимки. Финальный отчёт — `REPORT-20260925.md`, evidence —
`evidence/20260925/`. Нулевое число внешних browser requests проверяется во
всех поведенческих и приёмочных прогонах.

Старые `shoot.py`, `api-before/`, `api-after/` сохранены как история 16.09.
Они используют фиксированные profile/port и прежнюю одну OSNet; это не
свидетельства текущего интерфейса или модели.
