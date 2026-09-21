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

async function boot() {
  const chip = $("state-chip");
  try {
    const r = await fetch("/api/ui/state");
    if (!r.ok) throw new Error("HTTP " + r.status);
    app.info = await r.json();
  } catch (e) {
    chip.className = "chip chip-bad";
    chip.textContent = "сервис не отвечает";
    banner("Сервис не отвечает", "Страница загрузилась, но метод /api/ui/state "
      + "недоступен: " + e + ". Проверьте, что контейнер API запущен.");
    return;
  }
  const s = app.info;
  $("foot-service").textContent = "Сервис " + s.service + ", шкала оценки — "
    + (s.score_scale === "cosine" ? "косинусная близость" : s.score_scale)
    + ", порог отказа " + fmt6(s.default_threshold);
  $("foot-model").textContent = s.model;
  $("thr").placeholder = s.default_threshold.toFixed(6);
  $("thr-hint").textContent = "Пусто — порог сервиса, " + fmt6(s.default_threshold)
    + ". Порог обоснован калибровкой, менять его для обычного поиска не нужно.";

  if (!s.storage_reachable) {
    chip.className = "chip chip-bad";
    chip.textContent = "хранилище недоступно";
    banner("Хранилище галереи не отвечает",
      "Сервис работает, но векторная база (Qdrant) недоступна, поэтому поиск "
      + "выполнить нельзя. Поднимите её вместе с сервисом:",
      "podman-compose -f 04-solution/service/docker-compose.yml up -d");
    return;
  }
  if (!s.gallery_points) {
    chip.className = "chip chip-bad";
    chip.textContent = "галерея не загружена";
    banner("Галерея не загружена",
      "Хранилище поднято, но пустое: поиску не с чем сравнивать, API на запрос "
      + "ответит 409. Загрузка галереи — отдельный воспроизводимый шаг:",
      "podman-compose -f 04-solution/service/docker-compose.yml run --rm loader");
    return;
  }
  chip.className = "chip chip-ok";
  chip.textContent = "галерея: " + s.gallery_points.toLocaleString("ru-RU")
    + " объектов";
  if (!s.images_available) {
    banner("Кадры галереи не смонтированы",
      "Поиск работает, но показать кандидатов картинками нельзя: каталог кадров "
      + "недоступен сервису. Смонтируйте его в контейнер API (в compose — "
      + "переменная DATA_DIR); в списке останутся идентификаторы и оценки.");
  }
  enableForm();
}

function banner(head, body, cmd) {
  $("banner").hidden = false;
  $("banner-head").textContent = head;
  $("banner-body").textContent = body;
  $("banner-cmd").hidden = !cmd;
  if (cmd) $("banner-cmd").textContent = cmd;
}

function enableForm() {
  app.ready = true;
  syncRunButton();
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

function loadFrame(file) {
  const url = URL.createObjectURL(file);
  const im = new Image();
  im.onload = () => {
    app.img = im;
    app.file = file;
    app.box = null;
    $("file-name").textContent = file.name;
    $("file-size").textContent = im.naturalWidth + "×" + im.naturalHeight;
    $("drop").hidden = true;
    $("stage").hidden = false;
    $("qform").hidden = false;
    clearResults();
    layout();
    syncRunButton();
  };
  im.onerror = () => {
    showError("Файл не удалось прочитать как изображение. Нужен JPEG или PNG.");
    URL.revokeObjectURL(url);
  };
  im.src = url;
}

/* ---------- 3. рамка мышью ---------- */

const cv = $("canvas"), ctx = cv.getContext("2d");
const HANDLE = 9;               // радиус захвата угла/стороны, в пикселях показа
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
  ctx.strokeStyle = "#fcfcfb";
  ctx.lineWidth = 2;
  ctx.strokeRect(b.x, b.y, b.w, b.h);
  ctx.fillStyle = "#fcfcfb";
  for (const [hx, hy] of corners(b)) ctx.fillRect(hx - 3, hy - 3, 6, 6);
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
  return { x: (e.clientX - r.left) / app.fit, y: (e.clientY - r.top) / app.fit };
}

/** Что окажется под курсором: угол/сторона (индекс), тело рамки или пустое место. */
function hit(p) {
  if (!app.box) return null;
  const b = toView(app.box), v = { x: p.x * app.fit, y: p.y * app.fit };
  const cs = corners(b);
  for (let i = 0; i < cs.length; i++) {
    if (Math.abs(v.x - cs[i][0]) <= HANDLE && Math.abs(v.y - cs[i][1]) <= HANDLE) {
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
  if (!app.img) return;
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
});
cv.addEventListener("pointercancel", () => { drag = null; });

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
  app.box = null;
  for (const id of ["bx", "by", "bw", "bh"]) $(id).value = "";
  draw();
  syncRunButton();
});

function pushBox() {
  const b = app.box;
  if (b) {
    $("bx").value = b.x; $("by").value = b.y;
    $("bw").value = b.w; $("bh").value = b.h;
  }
  draw();
  syncRunButton();
}

function syncRunButton() {
  const ok = app.ready && app.img && app.box && app.box.w > 0 && app.box.h > 0;
  $("run").disabled = !ok;
}

/* ---------- 4. поиск ---------- */

$("run").addEventListener("click", () => runSearch());

function queryForm(threshold) {
  const fd = new FormData();
  fd.append("file", app.file);
  fd.append("x", app.box.x); fd.append("y", app.box.y);
  fd.append("w", app.box.w); fd.append("h", app.box.h);
  fd.append("top_k", Math.max(1, Math.min(100, parseInt($("topk").value, 10) || 10)));
  if (threshold !== undefined) fd.append("threshold", threshold);
  else if ($("thr").value !== "") fd.append("threshold", $("thr").value);
  return fd;
}

async function runSearch() {
  const btn = $("run");
  btn.disabled = true;
  btn.textContent = "Идёт поиск…";
  hideError();
  try {
    const r = await fetch("/api/search", { method: "POST", body: queryForm() });
    const data = await r.json();
    if (!r.ok) { showError(apiError(r.status, data)); clearResults(); return; }
    app.below = null;
    app.last = {
      at: new Date(),
      query: { file: app.file.name, ...app.box,
               top_k: parseInt($("topk").value, 10) || 10 },
      resp: data,
    };
    render();
  } catch (e) {
    showError("Не удалось обратиться к сервису: " + e);
  } finally {
    btn.textContent = "Найти кандидатов";
    syncRunButton();
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
  $("summary").hidden = true;
  $("cards").hidden = true;
  $("cards").innerHTML = "";
  $("refusal").hidden = true;
  $("export").hidden = true;
  $("explain").hidden = true;
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

  const hidden = query.top_k - resp.candidates.length;
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
  const pct = (v) => (Math.min(1, Math.max(0, v)) * 100).toFixed(2) + "%";
  const sc = $("scale");
  sc.style.setProperty("--best", pct(best === null ? 0 : best));
  sc.style.setProperty("--thr", pct(thr));
  sc.innerHTML =
    '<span class="mark best"></span><span class="mark thr"></span>'
    + '<span class="lab left" style="left:0">0</span>'
    + `<span class="lab" style="left:${pct(best === null ? 0 : best)}">лучшая ${fmt6(best)}</span>`
    + `<span class="lab" style="left:${pct(thr)};top:30px">порог ${fmt6(thr)}</span>`
    + '<span class="lab right" style="left:100%">1</span>';
  $("refusal-facts").innerHTML =
    dt("Лучшая близость", fmt6(best))
    + dt("Порог отказа", fmt6(thr))
    + dt("Не хватило", best === null ? "—" : fmtSigned(best - thr))
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
    const width = (Math.min(1, Math.max(0, c.confidence)) * 100).toFixed(2);
    const thrPos = (Math.min(1, Math.max(0, thr)) * 100).toFixed(2);
    li.innerHTML = `
      <div class="card-img">
        <span class="rank">${c.rank}</span>
        ${images ? `<img alt="Кандидат ${esc(c.gallery_id)}" loading="lazy"
             src="/api/gallery/${encodeURIComponent(c.gallery_id)}/crop?size=360">`
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
          ${images ? '<button type="button" class="link-btn act-frame">кадр целиком</button>' : ""}
          ${images ? '<button type="button" class="link-btn act-explain">разбор</button>' : ""}
        </div>
      </div>`;
    if (images) {
      const img = li.querySelector("img");
      img.addEventListener("error", () => {
        img.replaceWith(Object.assign(document.createElement("span"),
          { className: "noimg", textContent: "кадр недоступен" }));
      });
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
  const btn = $("show-below");
  btn.disabled = true;
  try {
    const r = await fetch("/api/search", { method: "POST", body: queryForm(-1) });
    const data = await r.json();
    if (!r.ok) { showError(apiError(r.status, data)); return; }
    app.below = data.candidates;
    render();
  } catch (e) {
    showError("Не удалось обратиться к сервису: " + e);
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
      .filter((c) => !known.has(c.gallery_id))
      .map((c, i) => ({ rank: i + 1, gallery_id: c.gallery_id,
                        confidence: c.confidence, above_threshold: false }));
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
  const panel = $("explain");
  panel.hidden = false;
  $("explain-lead").textContent = "Считается разложение…";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  for (const li of $("cards").children) li.classList.toggle("sel", li.dataset.gid === gid);
  app.selected = gid;

  const fd = new FormData();
  fd.append("file", app.file);
  fd.append("x", app.box.x); fd.append("y", app.box.y);
  fd.append("w", app.box.w); fd.append("h", app.box.h);
  fd.append("gallery_id", gid);
  let data;
  try {
    const r = await fetch("/api/explain", { method: "POST", body: fd });
    data = await r.json();
    if (!r.ok) { $("explain-lead").textContent = apiError(r.status, data); return; }
  } catch (e) {
    $("explain-lead").textContent = "Не удалось обратиться к сервису: " + e;
    return;
  }

  const scale = Math.max(
    ...data.query.flat().map(Math.abs), ...data.gallery.flat().map(Math.abs));
  $("explain-lead").textContent =
    "Объяснение первой модели OSNet-AIN (OMZ): сумма вкладов областей и свободного "
    + "члена равна её косинусу. Поиск использует d1_j48 — две модели и whitening, "
    + "поэтому его оценка может отличаться от числа ниже. Красным — то, что "
    + "поддерживает совпадение, синим — то, что ему мешает.";

  drawTile($("ex-q"), () => drawQueryCrop(), data.query, scale);
  drawTile($("ex-g"), (c) => loadGalleryCrop(gid, c), data.gallery, scale);
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

function drawTile(canvas, paint, grid, scale) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = SIDE * dpr;
  canvas.height = SIDE * dpr;
  canvas.style.height = "auto";
  const c = canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.fillStyle = token("--card-2", "#f2f1ec");
  c.fillRect(0, 0, SIDE, SIDE);
  const after = () => heat(c, grid, scale);
  const r = paint(c);
  if (r && typeof r.then === "function") r.then(after); else after();
}

/** Кроп запроса ровно так, как его видит модель: квадрат 208x208 из рамки. */
function drawQueryCrop() {
  const b = app.box;
  const c = $("ex-q").getContext("2d");
  c.imageSmoothingQuality = "high";
  c.drawImage(app.img, b.x, b.y, b.w, b.h, 0, 0, SIDE, SIDE);
}

function loadGalleryCrop(gid, c) {
  return new Promise((resolve) => {
    const im = new Image();
    im.onload = () => { c.drawImage(im, 0, 0, SIDE, SIDE); resolve(); };
    im.onerror = () => resolve();
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

boot();
