/* Тонкий клиент сервиса Vehicle ReID.

   Без фреймворков и без единого обращения наружу: только методы этого же
   сервиса (/api/...). Файл разбит на части в порядке работы оператора:
   состояние сервиса -> кадр и рамка -> поиск -> результат/отказ -> выгрузка ->
   разбор пары.

   Оценки не пересчитываются в браузере: всё, что показано, пришло из API.  */
"use strict";

const $ = (id) => document.getElementById(id);
const POS = [0xe3, 0x49, 0x48];   // поддерживает совпадение
const NEG = [0x2a, 0x78, 0xd6];   // мешает / ниже порога
const A_MAX = 0.76;               // верхняя прозрачность плеча — как в render.py

const app = {
  revision: 0, busy: false, controllers: {}, frameRevision: 0,
  info: null,        // ответ /api/ui/state
  img: null,         // HTMLImageElement текущего кадра
  file: null,        // File — уходит в API как есть
  box: null,         // {x, y, w, h} в пикселях ИСХОДНОГО кадра
  fit: 1,            // масштаб показа кадра на канве
  last: null,        // снимок последнего результата (он же источник выгрузки)
  below: null,       // кандидаты ниже порога, если оператор их запросил
  selected: null,    // gallery_id, для которого показан разбор
};

const fmt6 = (v) => (v === null || v === undefined ? "—" : v.toFixed(6));
const fmtSigned = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(6);

/** Значение токена темы — чтобы канва не расходилась с CSS. */
const token = (name, fallback) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

/* ---------- 0. тема ---------- */

/* Начальное значение уже поставлено блокирующим скриптом в <head> — здесь
   только переключение и запоминание выбора. Палитра карты разбора темой не
   управляется: POS/NEG ниже зафиксированы. */
(function theme() {
  const btn = $("theme-btn");
  const label = () => {
    const dark = document.documentElement.dataset.theme === "dark";
    btn.title = btn.ariaLabel =
      "Сменить тему оформления (сейчас " + (dark ? "тёмная" : "светлая") + ")";
  };
  btn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("reid-theme", next); } catch (e) { /* приватный режим */ }
    label();
  });
  label();
})();

/* ---------- 1. состояние сервиса ---------- */

let bootRevision = 0, bootTimer;
async function boot() {
  clearTimeout(bootTimer);
  const generation = ++bootRevision;
  $("retry-state").disabled = true;
  const chip = $("state-chip");
  try {
    const response = await fetch("/api/ui/state");
    if (!response.ok) throw new Error("HTTP " + response.status);
    const info = await response.json();
    if (generation !== bootRevision) return;
    const previous = app.info;
    const configurationChanged = previous && ["service", "model", "score_scale",
      "default_threshold", "gallery_points", "storage_reachable"].some(key => previous[key] !== info[key]);
    const hadResult = !!app.last;
    if (configurationChanged && (hadResult || app.busy)) invalidateQuery();
    app.info = info;
    app.configurationChanged = configurationChanged && hadResult;
  } catch (error) {
    if (generation !== bootRevision) return;
    if (app.last || app.busy) invalidateQuery();
    app.ready = false;
    chip.className = "chip chip-bad"; chip.textContent = "Сервис недоступен";
    banner("Не удалось связаться с сервисом", "Проверьте соединение и нажмите «Проверить снова». Загруженный кадр останется на странице.");
    $("retry-state").disabled = false; syncRunButton(); return;
  }
  const info = app.info;
  $("foot-service").textContent = `Версия ${info.service} · Косинусный поиск · Порог ${fmt6(info.default_threshold)}`;
  $("foot-model").textContent = info.model.includes("d1_j48") ? "d1_j48 · Две модели + whitening · 512 признаков" : info.model;
  $("thr").placeholder = info.default_threshold.toFixed(6);
  $("thr-hint").textContent = `Пусто — калиброванный порог ${fmt6(info.default_threshold)}. Для обычного поиска менять его не нужно.`;
  $("banner").hidden = true;
  $("retry-state").disabled = false;
  app.ready = !!(info.storage_reachable && info.gallery_points);
  chip.className = "chip " + (app.ready ? "chip-ok" : "chip-bad");
  chip.textContent = app.ready ? `Галерея · ${info.gallery_points.toLocaleString("ru-RU")}` : "Галерея готовится";
  if (!info.storage_reachable) {
    chip.textContent = "Галерея недоступна";
    banner("Галерея временно недоступна", "Кадр можно подготовить сейчас. Поиск станет доступен после подключения галереи.", "Проверьте контейнер Qdrant и соединение с API.");
  } else if (!info.gallery_points) {
    banner("Подготавливаем галерею", "Кадр можно загрузить и выделить объект. Мы автоматически проверим готовность галереи.", "Загрузите галерею по инструкции запуска сервиса.");
    bootTimer = setTimeout(boot, 4000);
  } else if (app.configurationChanged) {
    banner("Условия поиска изменились", "Модель, галерея или порог обновились. Повторите поиск по выбранному кадру, чтобы получить актуальный результат.");
  } else if (!info.images_available) {
    banner("Кадры галереи недоступны", "Поиск работает. Пока можно сравнить идентификаторы и оценки; изображения кандидатов временно не отображаются.", "Проверьте монтирование каталога изображений (DATA_DIR).");
  }
  syncRunButton();
}
function banner(head, body, command) {
  $("banner").hidden = false;
  $("banner-head").textContent = head; $("banner-body").textContent = body;
  $("banner-details").hidden = !command;
  $("banner-cmd").hidden = !command;
  $("banner-cmd").textContent = command || "";
}
$("retry-state").addEventListener("click", boot);
function scrollToElement(element) {
  element.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
}
function localPath(value) { return typeof value === "string" && value.startsWith("/") && !value.startsWith("//") ? value : null; }
async function loadExamples() {
  try {
    const response = await fetch("/api/demo");
    if (!response.ok) throw new Error();
    const data = await response.json();
    app.examples = (data.examples || []).filter(example => localPath(example.image_url));
    if (!app.examples.length) throw new Error();
    const first = app.examples[0];
    $("hero-image").src = first.image_url;
    $("hero-image").hidden = false; $("hero-empty").hidden = true;
    $("demo-status").textContent = "Демонстрационный кадр проходит настоящий поиск по галерее.";
    $("try-demo").disabled = false;
    for (const example of app.examples) {
      const button = document.createElement("button");
      button.type = "button"; button.className = "demo-example";
      button.textContent = example.label; button.dataset.id = example.id;
      button.addEventListener("click", () => loadDemo(example));
      $("demo-examples").append(button);
    }
  } catch (error) {
    $("demo-status").textContent = "Демонстрационные кадры недоступны. Для поиска загрузите свой JPEG или PNG.";
    $("try-demo").disabled = true;
  }
}
async function loadDemo(example) {
  invalidateQuery();
  const generation = ++app.frameRevision;
  app.loadingFrame = true; syncRunButton();
  const controller = new AbortController();
  app.controllers.demo = controller;
  $("try-demo").disabled = true;
  $("demo-status").textContent = "Загружаем кадр…";
  try {
    const response = await fetch(example.image_url, {signal:controller.signal});
    if (!response.ok) throw new Error();
    const blob = await response.blob();
    if (generation !== app.frameRevision || controller.signal.aborted) return;
    const filename = (example.image_url.split("/").pop() || example.id).includes(".") ? example.image_url.split("/").pop() : `demo-${example.id}.jpg`;
    const loaded = await loadFrame(new File([blob], filename, {type:blob.type || "image/jpeg"}), example.bbox);
    if (!loaded) return;
    $("hero-image").src = example.image_url;
    for (const button of $("demo-examples").children) button.classList.toggle("active", button.dataset.id === example.id);
    $("demo-status").textContent = "Кадр готов. Проверьте рамку и нажмите «Найти кандидатов».";
  } catch (error) {
    if (error.name !== "AbortError") $("demo-status").textContent = "Кадр не загрузился. Попробуйте ещё раз или загрузите свой файл.";
  } finally {
    if (generation === app.frameRevision) { app.loadingFrame = false; syncRunButton(); }
    $("try-demo").disabled = !(app.examples && app.examples.length);
  }
}
$("try-demo").addEventListener("click", () => { if (app.examples && app.examples[0]) loadDemo(app.examples[0]); });
$("upload-start").addEventListener("click", () => $("file").click());
async function loadMaterials() {
  try {
    const response = await fetch("/api/materials");
    if (!response.ok) throw new Error();
    const data = await response.json();
    for (const item of data.items || []) {
      if (!localPath(item.url)) continue;
      const link = document.createElement("a"), label = document.createElement("span"), arrow = document.createElement("span");
      link.href = item.url; label.textContent = item.label; arrow.textContent = "↗"; arrow.setAttribute("aria-hidden", "true");
      if (item.bytes) { const note = document.createElement("small"); note.textContent = `${(item.bytes / 1024).toLocaleString("ru-RU", {maximumFractionDigits:0})} КБ · сохранённый комплект решения`; label.append(note); }
      link.append(label, arrow); $("material-links").append(link);
    }
    $("materials-status").textContent = "Файлы комплекта — сохранённые артефакты. Они не являются выгрузкой вашего текущего запроса.";
    if (data.deployment_status && data.deployment_status !== "DEPLOY PENDING") $("deployment-status").textContent = data.deployment_status;
  } catch (error) { $("materials-status").textContent = "Список файлов комплекта сейчас недоступен. Документация API открывается по ссылкам выше."; }
}

/* ---------- 2. кадр: выбор, перетаскивание ---------- */

const drop = $("drop"), fileInput = $("file");
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
$("change-file").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFrame(fileInput.files[0]);
});
for (const ev of ["dragenter", "dragover"]) {
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
}
for (const ev of ["dragleave", "drop"]) {
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
}
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) loadFrame(f);
});

function loadFrame(file, initialBox = null) {
  invalidateQuery(); hideError();
  const generation = ++app.frameRevision;
  app.loadingFrame = true;
  app.img = null; app.file = null; app.box = null;
  $("stage").hidden = true; $("qform").hidden = true; $("drop").hidden = false;
  $("query-step").textContent = "Выберите изображение";
  for (const id of ["bx", "by", "bw", "bh"]) $(id).value = "";
  if (app.objectURL) URL.revokeObjectURL(app.objectURL);
  app.objectURL = null;
  syncRunButton();
  if (!(["image/jpeg", "image/png"].includes(file.type) || (!file.type && /\.(jpe?g|png)$/i.test(file.name)))) {
    app.loadingFrame = false;
    showError("Нужен кадр JPEG или PNG. Выберите файл изображения.");
    scrollToElement($("workspace"));
    return Promise.resolve(false);
  }
  const url = URL.createObjectURL(file), image = new Image();
  return new Promise(resolve => {
    image.onload = () => {
      if (generation !== app.frameRevision) { URL.revokeObjectURL(url); resolve(false); return; }
      app.loadingFrame = false;
      app.objectURL = url; app.img = image; app.file = file;
      app.box = initialBox ? normalizeBox(initialBox) : null;
      $("file-name").textContent = file.name; $("file-name").title = file.name;
      $("file-size").textContent = image.naturalWidth + " × " + image.naturalHeight;
      $("drop").hidden = true; $("stage").hidden = false; $("qform").hidden = false;
      $("bx").max = image.naturalWidth - 1; $("by").max = image.naturalHeight - 1;
      $("bw").max = image.naturalWidth; $("bh").max = image.naturalHeight;
      layout(); pushBox(); scrollToElement($("workspace")); resolve(true);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      if (generation === app.frameRevision) { app.loadingFrame = false; syncRunButton(); showError("Изображение повреждено или не читается. Выберите другой JPEG или PNG."); scrollToElement($("workspace")); }
      resolve(false);
    };
    image.src = url;
  });
}
function normalizeBox(box) {
  const width = app.img.naturalWidth, height = app.img.naturalHeight;
  const x = Math.min(width - 1, Math.max(0, Math.round(Number(box.x) || 0)));
  const y = Math.min(height - 1, Math.max(0, Math.round(Number(box.y) || 0)));
  return {x, y, w:Math.min(width - x, Math.max(1, Math.round(Number(box.w) || 1))), h:Math.min(height - y, Math.max(1, Math.round(Number(box.h) || 1)))};
}

/* ---------- 3. рамка мышью ---------- */

const cv = $("canvas"), ctx = cv.getContext("2d");
let handleRadius = 10;
const HANDLE = 10;               // радиус захвата угла/стороны, в пикселях показа
let drag = null;

function layout() {
  if (!app.img) return;
  const wCss = cv.parentElement.clientWidth;
  app.fit = wCss / app.img.naturalWidth;
  const hCss = Math.round(app.img.naturalHeight * app.fit);
  const dpr = window.devicePixelRatio || 1;
  cv.style.height = hCss + "px";
  cv.width = Math.round(wCss * dpr);
  cv.height = Math.round(hCss * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
window.addEventListener("resize", layout);

function draw() {
  const w = cv.width / (window.devicePixelRatio || 1);
  const h = cv.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(app.img, 0, 0, w, h);
  if (!app.box) return;
  const b = toView(app.box);
  ctx.save();
  // Затемнение вне рамки: глаз сразу видит выбранный объект.
  ctx.fillStyle = "rgba(11,11,11,.42)";
  ctx.beginPath();
  ctx.rect(0, 0, w, h);
  ctx.rect(b.x, b.y, b.w, b.h);
  ctx.fill("evenodd");
  ctx.strokeStyle = "#f9fcfd";
  ctx.lineWidth = 2;
  ctx.strokeRect(b.x, b.y, b.w, b.h);
  ctx.fillStyle = "#fcfcfb";
  for (const [hx, hy] of corners(b)) { ctx.fillRect(hx - 4, hy - 4, 8, 8); ctx.strokeStyle = "#17252e"; ctx.lineWidth = 1; ctx.strokeRect(hx - 4, hy - 4, 8, 8); }
  ctx.restore();
}

const corners = (b) => [
  [b.x, b.y], [b.x + b.w / 2, b.y], [b.x + b.w, b.y],
  [b.x, b.y + b.h / 2], [b.x + b.w, b.y + b.h / 2],
  [b.x, b.y + b.h], [b.x + b.w / 2, b.y + b.h], [b.x + b.w, b.y + b.h],
];
const toView = (b) => ({ x: b.x * app.fit, y: b.y * app.fit,
                         w: b.w * app.fit, h: b.h * app.fit });

function pos(e) {
  const r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left) * app.img.naturalWidth / r.width,
           y: (e.clientY - r.top) * app.img.naturalHeight / r.height };
}

/** Что окажется под курсором: угол/сторона (индекс), тело рамки или пустое место. */
function hit(p) {
  if (!app.box) return null;
  const b = toView(app.box), v = { x: p.x * app.fit, y: p.y * app.fit };
  const cs = corners(b);
  for (let i = 0; i < cs.length; i++) {
    if (Math.abs(v.x - cs[i][0]) <= handleRadius && Math.abs(v.y - cs[i][1]) <= handleRadius) {
      return { kind: "handle", i };
    }
  }
  if (v.x > b.x && v.x < b.x + b.w && v.y > b.y && v.y < b.y + b.h) {
    return { kind: "move" };
  }
  return null;
}

const CURSORS = ["nwse-resize", "ns-resize", "nesw-resize", "ew-resize",
                 "ew-resize", "nesw-resize", "ns-resize", "nwse-resize"];

cv.addEventListener("pointermove", (e) => {
  if (drag) return;
  const h = hit(pos(e));
  cv.style.cursor = !h ? "crosshair"
    : h.kind === "move" ? "move" : CURSORS[h.i];
});

cv.addEventListener("pointerdown", (e) => {
  if (!app.img || !e.isPrimary || e.button !== 0) return;
  handleRadius = e.pointerType === "touch" ? 24 : HANDLE;
  cv.setPointerCapture(e.pointerId);
  const p = pos(e), h = hit(p);
  if (h && h.kind === "handle") {
    // Тянем угол/сторону: противоположная точка остаётся на месте.
    const b = app.box;
    const anchorX = [2, 4, 7].includes(h.i) ? b.x : [0, 3, 5].includes(h.i) ? b.x + b.w : null;
    const anchorY = [5, 6, 7].includes(h.i) ? b.y : [0, 1, 2].includes(h.i) ? b.y + b.h : null;
    drag = { mode: "resize", anchorX, anchorY, start: { ...b } };
  } else if (h && h.kind === "move") {
    drag = { mode: "move", dx: p.x - app.box.x, dy: p.y - app.box.y };
  } else {
    drag = { mode: "resize", anchorX: p.x, anchorY: p.y, start: null,
             prev: app.box, fresh: true };
    app.box = { x: Math.round(p.x), y: Math.round(p.y), w: 1, h: 1 };
  }
  onDrag(e);
});

cv.addEventListener("pointermove", onDrag);
cv.addEventListener("pointerup", () => {
  // Клик без протяжки — не рамка: возвращаем прежнюю, а не оставляем точку.
  if (drag && drag.fresh && app.box.w < 3 && app.box.h < 3) {
    app.box = drag.prev;
    pushBox();
  }
  drag = null;
  syncRunButton();
});
cv.addEventListener("pointercancel", () => { drag = null; syncRunButton(); });

function onDrag(e) {
  if (!drag) return;
  const p = pos(e), W = app.img.naturalWidth, H = app.img.naturalHeight;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  if (drag.mode === "move") {
    const b = app.box;
    app.box = { x: Math.round(clamp(p.x - drag.dx, 0, W - b.w)),
                y: Math.round(clamp(p.y - drag.dy, 0, H - b.h)), w: b.w, h: b.h };
  } else {
    const b = drag.start;
    const px = clamp(p.x, 0, W), py = clamp(p.y, 0, H);
    const ax = drag.anchorX === null ? null : drag.anchorX;
    const ay = drag.anchorY === null ? null : drag.anchorY;
    const x0 = ax === null ? b.x : Math.min(ax, px);
    const x1 = ax === null ? b.x + b.w : Math.max(ax, px);
    const y0 = ay === null ? b.y : Math.min(ay, py);
    const y1 = ay === null ? b.y + b.h : Math.max(ay, py);
    app.box = { x: Math.round(x0), y: Math.round(y0),
                w: Math.max(1, Math.round(x1 - x0)),
                h: Math.max(1, Math.round(y1 - y0)) };
  }
  pushBox();
}

/* поля x/y/w/h — второй, точный способ задать ту же рамку */

for (const id of ["bx", "by", "bw", "bh"]) {
  $(id).addEventListener("change", () => pushBox());
  $(id).addEventListener("input", () => {
    if (!app.img) return;
    const v = (k) => parseInt($(k).value, 10);
    const W = app.img.naturalWidth, H = app.img.naturalHeight;
    const b = { x: v("bx") || 0, y: v("by") || 0,
                w: Math.max(1, v("bw") || 1), h: Math.max(1, v("bh") || 1) };
    b.x = Math.min(Math.max(0, b.x), W - 1);
    b.y = Math.min(Math.max(0, b.y), H - 1);
    b.w = Math.min(b.w, W - b.x);
    b.h = Math.min(b.h, H - b.y);
    invalidateQuery();
    app.box = b;
    draw();
    syncRunButton();
  });
}

$("whole-frame").addEventListener("click", () => {
  if (!app.img) return;
  app.box = { x: 0, y: 0, w: app.img.naturalWidth, h: app.img.naturalHeight };
  pushBox();
});
$("clear-box").addEventListener("click", () => {
  invalidateQuery();
  app.box = null;
  for (const id of ["bx", "by", "bw", "bh"]) $(id).value = "";
  draw();
  syncRunButton();
});

function pushBox() {
  invalidateQuery();
  const b = app.box;
  if (b) {
    $("bx").value = b.x; $("by").value = b.y;
    $("bw").value = b.w; $("bh").value = b.h;
  } else {
    for (const id of ["bx", "by", "bw", "bh"]) $(id).value = "";
  }
  draw();
  syncRunButton();
}

function syncRunButton() {
  const ok = !app.loadingFrame && !app.busy && app.ready && app.img && app.box && app.box.w > 0 && app.box.h > 0;
  $("run").disabled = !ok;
  // Keep the canvas origin stable throughout a pointer gesture: this label can
  // wrap differently at narrow widths or 200% zoom and change the header height.
  if (!drag) $("query-step").textContent = app.img ? (app.box ? "Объект выделен" : "Выделите автомобиль") : "Выберите изображение";
}

/* ---------- 4. поиск ---------- */

$("run").addEventListener("click", () => runSearch());

/** Results belong to a frozen query, never to mutable input controls. */
function invalidateQuery() {
  app.revision += 1;
  for (const controller of Object.values(app.controllers)) controller.abort();
  app.controllers = {};
  app.busy = false;
  hideError();
  clearResults();
  $("run").textContent = "Найти кандидатов";
}
for (const id of ["topk", "thr"]) $(id).addEventListener("input", () => {
  invalidateQuery(); syncRunButton();
});
function snapshot() {
  return Object.freeze({ file: app.file, image: app.img,
    box: Object.freeze({ ...app.box }),
    top_k: Math.max(1, Math.min(100, parseInt($("topk").value, 10) || 10)),
    threshold: $("thr").value });
}
function queryForm(query, threshold) {
  const fd = new FormData();
  fd.append("file", query.file);
  for (const key of ["x", "y", "w", "h"]) fd.append(key, query.box[key]);
  fd.append("top_k", query.top_k);
  if (threshold !== undefined) fd.append("threshold", threshold);
  else if (query.threshold !== "") fd.append("threshold", query.threshold);
  return fd;
}
async function runSearch() {
  if (!app.ready || !app.file || !app.box || app.busy || app.loadingFrame) return;
  const query = snapshot();
  invalidateQuery();
  const revision = app.revision;
  const controller = new AbortController();
  app.controllers.search = controller;
  app.busy = true;
  $("run").disabled = true;
  $("run").textContent = "Идёт поиск…";
  $("search-loading").hidden = false; $("placeholder").hidden = true;
  document.querySelector(".results").setAttribute("aria-busy", "true");
  hideError();
  let refreshState = false;
  try {
    const r = await fetch("/api/search", { method: "POST", body: queryForm(query), signal: controller.signal });
    const data = await r.json();
    if (revision !== app.revision || controller.signal.aborted) return;
    if (!r.ok) {
      showError(apiError(r.status, data));
      refreshState = r.status === 409 || r.status === 503;
      return;
    }
    app.last = Object.freeze({ at: new Date(), snapshot: query, revision,
      query: Object.freeze({file: query.file.name, ...query.box, top_k: query.top_k}), resp: data });
    render();
    if (innerWidth < 851) scrollToElement(document.querySelector(".results"));
  } catch (e) {
    if (revision === app.revision && e.name !== "AbortError") showError("Не удалось обратиться к сервису. Проверьте соединение и повторите поиск.");
  } finally {
    if (revision === app.revision) {
      app.busy = false;
      $("search-loading").hidden = true;
      document.querySelector(".results").setAttribute("aria-busy", "false");
      if (!app.last) $("placeholder").hidden = false;
      $("run").textContent = "Найти кандидатов";
      syncRunButton();
      if (refreshState) void boot();
    }
  }
}

function apiError(status, data) {
  const detail = data && data.detail !== undefined
    ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))
    : "";
  if (status === 409) {
    return "Галерея не загружена (409). Поиску не с чем сравнивать: выполните "
      + "шаг загрузки галереи.\n" + detail;
  }
  if (status === 503) return "Хранилище недоступно (503).\n" + detail;
  if (status === 422) return "Запрос не принят (422): " + detail;
  return "Ошибка " + status + ": " + detail;
}

const showError = (t) => { $("error").hidden = false; $("error").textContent = t; };
const hideError = () => { $("error").hidden = true; $("error").textContent = ""; };

/* ---------- 5. показ результата ---------- */

function clearResults() {
  app.last = null; app.below = null; app.selected = null;
  $("placeholder").hidden = false;
  $("search-loading").hidden = true;
  document.querySelector(".results").setAttribute("aria-busy", "false");
  $("compare").hidden = true; app.comparison = null;
  $("summary").hidden = true;
  $("cards").hidden = true;
  $("cards").innerHTML = "";
  $("refusal").hidden = true;
  $("export").hidden = true;
  $("explain").hidden = true;
  $("ex-facts").replaceChildren();
  $("show-below").disabled = false;
  $("below-note").hidden = true;
}

function render() {
  const { resp, query } = app.last;
  const thr = resp.threshold;
  $("placeholder").hidden = true;
  $("export").hidden = false;
  $("explain").hidden = true;
  app.selected = null;

  // При отказе порог и лучшая близость уже разобраны в панели выше — в сводке
  // их не повторяем, чтобы одно и то же число не стояло на экране дважды.
  $("summary").hidden = false;
  $("summary").innerHTML =
    field("Принято", `<b>${resp.candidates.length}</b> из top-${query.top_k}`)
    + (resp.refusal ? ""
      : field("Порог", `<span class="num">${fmt6(thr)}</span>`)
        + field("Лучшая близость", `<span class="num">${fmt6(resp.best_confidence)}</span>`))
    + field("Запрос", `${esc(query.file)} · рамка ${query.x}, ${query.y}, ${query.w}×${query.h}`);

  $("refusal").hidden = !resp.refusal;
  if (resp.refusal) renderRefusal(resp);

  const rows = resp.candidates.map((c, i) => ({ ...c, rank: i + 1, below: false }));
  if (app.below) {
    const known = new Set(rows.map((r) => r.gallery_id));
    app.below.forEach((c, i) => {
      if (!known.has(c.gallery_id)) rows.push({ ...c, rank: i + 1, below: true });
    });
    rows.sort((a, b) => b.confidence - a.confidence);
    rows.forEach((r, i) => { r.rank = i + 1; });
    $("below-note").hidden = false;
  }
  renderCards(rows, thr);

  const hidden = Math.max(0, Math.min(query.top_k, app.info.gallery_points || query.top_k) - resp.candidates.length);
  $("show-below").hidden = !!app.below || hidden <= 0;
  $("show-below").textContent = `Показать ближайшие ниже порога (${hidden})`;
  if (!resp.refusal) {
    // Кнопку «ниже порога» держим и здесь: она относится к остатку top-k.
    const host = $("summary");
    if (hidden > 0 && !app.below) {
      host.insertAdjacentHTML("beforeend",
        `<span class="k">Остаток top-k <button type="button" class="link-btn"
         id="show-below-2">показать ${hidden} ниже порога</button></span>`);
      $("show-below-2").addEventListener("click", loadBelow);
    }
  }
}

const field = (k, v) => `<span><span class="k">${k}</span> ${v}</span>`;
const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function renderRefusal(resp) {
  const best = resp.best_confidence, thr = resp.threshold;
  // Шкала косинуса от 0 до 1: где остановилась лучшая близость и где порог.
  const pct = (v) => ((Math.min(1, Math.max(-1, v)) + 1) * 50).toFixed(2) + "%";
  const sc = $("scale");
  sc.style.setProperty("--best", pct(best === null ? 0 : best));
  sc.style.setProperty("--thr", pct(thr));
  sc.setAttribute("role", "img");
  sc.setAttribute("aria-label", `Косинусная шкала от минус одного до одного. Лучшая близость ${fmt6(best)}, порог ${fmt6(thr)}.`);
  sc.innerHTML = '<span class="mark best"></span><span class="mark thr"></span>'
    + '<span class="lab left" style="left:0">−1</span><span class="lab right" style="left:100%">1</span>'
    + '<span class="scale-caption"><span class="scale-best">Лучшая близость</span><span class="scale-thr">Порог</span></span>';
  $("refusal-facts").innerHTML =
    dt("Лучшая близость", fmt6(best))
    + dt("Порог отказа", fmt6(thr))
    + dt("До порога", best === null ? "—" : fmt6(Math.max(0, thr - best)))
    + dt("Просмотрено", (app.info.gallery_points || 0).toLocaleString("ru-RU")
        + " объектов галереи");
}

const dt = (k, v) => `<dt>${k}</dt><dd>${esc(v)}</dd>`;

function renderCards(rows, thr) {
  const list = $("cards");
  list.innerHTML = "";
  list.hidden = rows.length === 0;
  const images = app.info && app.info.images_available;
  for (const c of rows) {
    const li = document.createElement("li");
    li.className = "card" + (c.below ? " below" : "");
    li.dataset.gid = c.gallery_id;
    const width = ((Math.min(1, Math.max(-1, c.confidence)) + 1) * 50).toFixed(2);
    const thrPos = ((Math.min(1, Math.max(-1, thr)) + 1) * 50).toFixed(2);
    li.innerHTML = `
      <div class="card-img">
        <span class="rank">${c.rank}</span>
        ${images ? `<img alt="Кандидат ${esc(c.gallery_id)}" loading="lazy"
             src="/api/gallery/${encodeURIComponent(c.gallery_id)}/crop?size=560">`
          : '<span class="noimg">кадры галереи не смонтированы</span>'}
      </div>
      <div class="card-body">
        ${c.below ? '<span class="tag">ниже порога</span>' : ""}
        <span class="gid" title="${esc(c.gallery_id)}">${esc(c.gallery_id)}</span>
        <div class="conf">
          <span class="val">${c.confidence.toFixed(4)}</span>
          <span class="delta">${fmtSigned(c.confidence - thr)} к порогу</span>
        </div>
        <div class="meter"><i style="width:${width}%"></i><u style="left:${thrPos}%"></u></div>
        <div class="card-acts">
          ${images ? '<button type="button" class="link-btn act-compare">Сравнить</button><button type="button" class="link-btn act-frame">Кадр целиком</button>' : ""}
          ${images ? '<button type="button" class="link-btn act-explain">Разбор</button>' : ""}
        </div>
      </div>`;
    if (images) {
      const img = li.querySelector("img");
      img.addEventListener("error", () => {
        img.replaceWith(Object.assign(document.createElement("span"),
          { className: "noimg", textContent: "кадр недоступен" }));
      });
      li.querySelector(".act-compare").addEventListener("click", () => compare(c.gallery_id));
      let frame = false;
      li.querySelector(".act-frame").addEventListener("click", (e) => {
        frame = !frame;
        const el = li.querySelector("img");
        if (!el) return;
        el.src = `/api/gallery/${encodeURIComponent(c.gallery_id)}/crop?size=${
          frame ? 720 : 360}&view=${frame ? "frame" : "crop"}`;
        e.target.textContent = frame ? "только объект" : "кадр целиком";
      });
      li.querySelector(".act-explain").addEventListener("click",
        () => explain(c.gallery_id));
    }
    list.appendChild(li);
  }
}

$("show-below").addEventListener("click", loadBelow);

/** Кандидаты ниже порога — только по явному действию оператора.

    Тем же методом /api/search с порогом −1: сервис возвращает весь top-k.
    Отказ при этом не «отменяется» — он показан выше, а эти строки помечены
    как не прошедшие порог. */
async function loadBelow() {
  const last = app.last;
  if (!last || app.controllers.below) return;
  const controller = new AbortController();
  app.controllers.below = controller;
  $("show-below").disabled = true;
  try {
    const r = await fetch("/api/search", {method:"POST", body:queryForm(last.snapshot, -1), signal:controller.signal});
    const data = await r.json();
    if (app.last !== last || controller.signal.aborted) return;
    if (!r.ok) { showError(apiError(r.status, data)); return; }
    app.below = data.candidates;
    render();
  } catch (e) {
    if (app.last === last && e.name !== "AbortError") showError("Ближайшие объекты не загрузились. Повторите действие.");
  } finally {
    if (app.controllers.below === controller) delete app.controllers.below;
    if (app.last === last) $("show-below").disabled = false;
  }
}

/* ---------- 6. выгрузка ---------- */

$("exp-csv").addEventListener("click", () => download(buildCSV(), "csv", "text/csv"));
$("exp-json").addEventListener("click",
  () => download(buildJSON(), "json", "application/json"));

const csvCell = (v) => {
  const s = v === null || v === undefined ? "" : String(v);
  return /[",\n;]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
};

function buildCSV() {
  const { resp, query } = app.last;
  const head = ["query_image", "query_x", "query_y", "query_w", "query_h",
                "top_k", "threshold", "refusal", "best_confidence",
                "rank", "gallery_id", "confidence", "above_threshold"];
  const base = [query.file, query.x, query.y, query.w, query.h, query.top_k,
                resp.threshold, resp.refusal ? 1 : 0, resp.best_confidence];
  const lines = [head.join(",")];
  // Отказ — тоже результат: строка запроса остаётся, поля кандидата пусты.
  // Так выгрузка честно фиксирует «проверено, совпадения нет».
  if (!resp.candidates.length && !app.below) {
    lines.push(base.concat(["", "", "", ""]).map(csvCell).join(","));
  }
  resp.candidates.forEach((c, i) => {
    lines.push(base.concat([i + 1, c.gallery_id, c.confidence, 1]).map(csvCell).join(","));
  });
  if (app.below) {
    const known = new Set(resp.candidates.map((c) => c.gallery_id));
    app.below.forEach((c, i) => {
      if (!known.has(c.gallery_id)) {
        lines.push(base.concat([i + 1, c.gallery_id, c.confidence, 0]).map(csvCell).join(","));
      }
    });
  }
  lines.push("");
  return lines.join("\n");
}

function buildJSON() {
  const { resp, query, at } = app.last;
  const out = {
    generated_at: at.toISOString(),
    service: app.info.service,
    model: app.info.model,
    score_scale: app.info.score_scale,
    query: { image: query.file, x: query.x, y: query.y, w: query.w, h: query.h,
             top_k: query.top_k },
    threshold: resp.threshold,
    refusal: resp.refusal,
    best_confidence: resp.best_confidence,
    gallery_points: app.info.gallery_points,
    candidates: resp.candidates.map((c, i) => ({
      rank: i + 1, gallery_id: c.gallery_id, confidence: c.confidence,
      above_threshold: true })),
  };
  if (app.below) {
    const known = new Set(resp.candidates.map((c) => c.gallery_id));
    out.below_threshold = app.below
      .map((c, i) => ({ rank: i + 1, gallery_id: c.gallery_id,
                        confidence: c.confidence, above_threshold: false }))
      .filter((c) => !known.has(c.gallery_id));
  }
  return JSON.stringify(out, null, 2) + "\n";
}

function download(text, ext, mime) {
  const stamp = app.last.at.toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const stem = app.last.query.file.replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "_");
  const a = document.createElement("a");
  const url = URL.createObjectURL(new Blob([text], { type: mime + ";charset=utf-8" }));
  a.href = url;
  a.download = `reid-${stem}-${stamp}.${ext}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/* ---------- 7. разбор пары: на что смотрела модель ---------- */

async function explain(gid) {
  const last = app.last;
  if (!last) return;
  if (app.controllers.explain) app.controllers.explain.abort();
  const controller = new AbortController();
  app.controllers.explain = controller;
  const current = () => app.last === last && app.selected === gid && !controller.signal.aborted;
  const panel = $("explain");
  $("explain-body").hidden = true;
  panel.hidden = false;
  $("explain-lead").textContent = "Считается разложение…";
  scrollToElement(panel);
  for (const li of $("cards").children) li.classList.toggle("sel", li.dataset.gid === gid);
  app.selected = gid;

  $("ex-facts").replaceChildren();
  const fd = queryForm(last.snapshot);
  fd.append("gallery_id", gid);
  let data;
  try {
    const r = await fetch("/api/explain", { method: "POST", body: fd, signal: controller.signal });
    data = await r.json();
    if (!current()) return;
    if (!r.ok) { $("explain-lead").textContent = apiError(r.status, data); return; }
  } catch (e) {
    if (!current() || e.name === "AbortError") return;
    $("explain-lead").textContent = "Разбор недоступен. Результат поиска сохранён; повторите разбор.";
    return;
  }

  const scale = Math.max(
    ...data.query.flat().map(Math.abs), ...data.gallery.flat().map(Math.abs));
  $("explain-lead").textContent =
    "Объяснение первой модели OSNet-AIN (OMZ): сумма вкладов областей и свободного "
    + "члена равна её косинусу. Поиск использует d1_j48 — две модели и whitening, "
    + "поэтому его оценка может отличаться от числа ниже. Красным — то, что "
    + "поддерживает совпадение, синим — то, что ему мешает.";

  try {
    await Promise.all([
      drawTile($("ex-q"), () => drawQueryCrop(last.snapshot), data.query, scale, current),
      drawTile($("ex-g"), (context) => loadGalleryCrop(gid, context, current), data.gallery, scale, current),
    ]);
  } catch (error) {
    if (current()) $("explain-lead").textContent = "Кадр для разбора не загрузился. Результат поиска сохранён; повторите разбор.";
    return;
  }
  if (!current()) return;
  $("explain-body").hidden = false;
  drawBar($("ex-bar"), scale);
  $("ex-ticks").innerHTML = `<span>−${scale.toFixed(4)}</span><span>0</span>`
    + `<span>+${scale.toFixed(4)}</span>`;
  $("ex-q-note").textContent = "\u03a3 вкладов " + sumOf(data.query).toFixed(4);
  $("ex-g-note").textContent = "\u03a3 вкладов " + sumOf(data.gallery).toFixed(4);
  $("ex-facts").innerHTML =
    dt("Модель объяснения", "OSNet-AIN · OMZ")
    + dt("Косинус OSNet-AIN", fmt6(data.cos))
    + dt("Сетка признаков", data.grid.join(" × "))
    + dt("10 % ячеек дают", share(data.query, 0.1) + " вклада запроса")
    + dt("Невязка разложения", String(data.residual.query))
    + dt("Время расчёта", data.ms.toFixed(0) + " мс");
}

$("explain-close").addEventListener("click", () => {
  if (app.controllers.explain) app.controllers.explain.abort();
  app.selected = null;
  $("explain").hidden = true;
  for (const li of $("cards").children) li.classList.remove("sel");
});

const sumOf = (g) => g.flat().reduce((a, b) => a + b, 0);

/** Какая доля суммарного положительного вклада приходится на верхние p ячеек. */
function share(g, p) {
  const v = g.flat().filter((x) => x > 0).sort((a, b) => b - a);
  if (!v.length) return "—";
  const n = Math.max(1, Math.round(g.flat().length * p));
  const top = v.slice(0, n).reduce((a, b) => a + b, 0);
  const all = v.reduce((a, b) => a + b, 0);
  return (100 * top / all).toFixed(0) + " %";
}

const SIDE = 416;   // размер плитки разбора — ровно тот, в котором
                    // приходит кроп галереи (crop?size=416): без пересчёта

async function drawTile(canvas, paint, grid, scale, current = () => true) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = SIDE * dpr;
  canvas.height = SIDE * dpr;
  canvas.style.height = "auto";
  const c = canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.fillStyle = token("--card-2", "#f2f1ec");
  c.fillRect(0, 0, SIDE, SIDE);
  await paint(c);
  if (current()) heat(c, grid, scale);
}

/** Кроп запроса ровно так, как его видит модель: квадрат 208x208 из рамки. */
function drawQueryCrop(query) {
  const b = query.box;
  const c = $("ex-q").getContext("2d");
  c.imageSmoothingQuality = "high";
  c.drawImage(query.image, b.x, b.y, b.w, b.h, 0, 0, SIDE, SIDE);
}

function loadGalleryCrop(gid, c, current) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => { if (current()) c.drawImage(im, 0, 0, SIDE, SIDE); resolve(); };
    im.onerror = () => reject(new Error("Кадр галереи недоступен"));
    im.src = `/api/gallery/${encodeURIComponent(gid)}/crop?size=416`;
  });
}

/** Карта вкладов поверх кропа: тон — знак, прозрачность — величина. */
function heat(c, grid, scale) {
  const h = grid.length, w = grid[0].length;
  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  const octx = off.getContext("2d");
  const id = octx.createImageData(w, h);
  for (let r = 0; r < h; r++) {
    for (let k = 0; k < w; k++) {
      const v = grid[r][k];
      const col = v > 0 ? POS : NEG;
      const a = Math.min(1, Math.abs(v) / (scale || 1e-9)) * A_MAX;
      const i = (r * w + k) * 4;
      id.data[i] = col[0]; id.data[i + 1] = col[1]; id.data[i + 2] = col[2];
      id.data[i + 3] = Math.round(a * 255);
    }
  }
  octx.putImageData(id, 0, 0);
  c.imageSmoothingEnabled = true;
  c.imageSmoothingQuality = "high";
  c.drawImage(off, 0, 0, SIDE, SIDE);
}

function drawBar(canvas, scale) {
  const w = canvas.clientWidth || 240, h = 16;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const c = canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.fillStyle = "#fff";
  c.fillRect(0, 0, w, h);
  for (let px = 0; px < w; px++) {
    const v = (2 * px / (w - 1) - 1) * scale;
    const col = v > 0 ? POS : NEG;
    c.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${
      (Math.min(1, Math.abs(v) / (scale || 1e-9)) * A_MAX).toFixed(3)})`;
    c.fillRect(px, 0, 1, h);
  }
}

/* ---------- ordinary pair comparison: uses the immutable search input ---------- */
function compare(gid) {
  const last = app.last;
  if (!last) return;
  const candidate = [...last.resp.candidates, ...(app.below || [])].find(item => item.gallery_id === gid);
  if (!candidate) return;
  app.comparison = {last, gid, candidate, frame:false};
  $("compare").hidden = false;
  $("compare-query-name").textContent = last.query.file;
  $("compare-id").textContent = gid;
  $("compare-lead").textContent = `Косинусная близость ${fmt6(candidate.confidence)} · ${last.resp.candidates.some(item => item.gallery_id === gid) ? "Прошёл порог" : "Ниже порога — для ручной проверки"}. Оценка не является вероятностью совпадения.`;
  for (const card of $("cards").children) card.classList.toggle("sel", card.dataset.gid === gid);
  paintComparison(); scrollToElement($("compare"));
}
function paintComparison() {
  const comparison = app.comparison;
  if (!comparison || comparison.last !== app.last) return;
  const query = comparison.last.snapshot, canvas = $("compare-q");
  const box = comparison.frame ? {x:0,y:0,w:query.image.naturalWidth,h:query.image.naturalHeight} : query.box;
  const scale = Math.min(1, 1400 / Math.max(box.w, box.h));
  canvas.width = Math.max(1, Math.round(box.w * scale)); canvas.height = Math.max(1, Math.round(box.h * scale));
  const context = canvas.getContext("2d");
  context.drawImage(query.image, box.x, box.y, box.w, box.h, 0, 0, canvas.width, canvas.height);
  if (comparison.frame) { const b = query.box; context.strokeStyle = "#e34948"; context.lineWidth = Math.max(2,canvas.width/350); context.strokeRect(b.x*scale,b.y*scale,b.w*scale,b.h*scale); }
  const image = $("compare-g"), status = $("compare-image-status");
  const src = `/api/gallery/${encodeURIComponent(comparison.gid)}/crop?size=1200&view=${comparison.frame ? "frame" : "crop"}`;
  image.hidden = true;
  status.hidden = false;
  status.textContent = "Загружаем кадр кандидата…";
  image.onload = () => {
    if (app.comparison !== comparison || image.getAttribute("src") !== src) return;
    image.hidden = false;
    status.hidden = true;
  };
  image.onerror = () => {
    if (app.comparison !== comparison || image.getAttribute("src") !== src) return;
    image.hidden = true;
    status.textContent = "Кадр кандидата недоступен. Выберите другой объект для сравнения.";
  };
  image.src = src;
  image.alt = `Кандидат ${comparison.gid}${comparison.frame ? ", кадр целиком" : ", выбранный объект"}`;
  $("compare-view").textContent = comparison.frame ? "Только объекты" : "Кадры целиком";
}
$("compare-view").addEventListener("click", () => { if (app.comparison) { app.comparison.frame = !app.comparison.frame; paintComparison(); } });
$("compare-close").addEventListener("click", () => { $("compare").hidden = true; app.comparison = null; });
$("compare-explain").addEventListener("click", () => { if (app.comparison) explain(app.comparison.gid); });
$("hero-image").addEventListener("error", () => { $("hero-image").hidden = true; $("hero-empty").hidden = false; });
boot(); loadExamples(); loadMaterials();
