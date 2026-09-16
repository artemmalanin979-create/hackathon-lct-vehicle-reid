#!/usr/bin/env python3
"""Прогон интерфейса по всем состояниям в обеих темах + журнал сети.

Chrome ведётся напрямую по CDP (websocket-client), без фреймворков. Путь
оператора проходится настоящими действиями: файл кладётся в input, рамка
обводится мышью по канве, кнопки нажимаются.

Два состояния сервиса (галерея не загружена / хранилище недоступно) снимаются
подменой ответа /api/ui/state в самом браузере: реальный стек поднят и рушить
его ради снимка неправильно. Подмена — только клиентская, видна в коде ниже.
"""
import base64, json, os, shutil, subprocess, sys, time, urllib.request
import websocket

BASE = "http://localhost:8000/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "shots"
EXPORTS = sys.argv[2] if len(sys.argv) > 2 else "exports"
IMGDIR = "/home/artem/projects/hackathon-lct-vehicle-reid/data/images"
MATCH = ("a4f2a13bd2b54360a921c8ef7366e535", (849, 300, 736, 471))
REFUSE = ("5cfbbd42352245fb9ab4e93f0e17452a", (125, 368, 785, 457))
W, H, DPR = 1440, 900, 2

os.makedirs(OUT, exist_ok=True)
os.makedirs(EXPORTS, exist_ok=True)
net_log = []


# ---------- минимальный клиент CDP ----------

class CDP:
    def __init__(self, url):
        self.ws = websocket.create_connection(url, timeout=120)
        self.n = 0
        self.buf = []

    def send(self, method, params=None, sess=None):
        self.n += 1
        msg = {"id": self.n, "method": method, "params": params or {}}
        if sess:
            msg["sessionId"] = sess
        self.ws.send(json.dumps(msg))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})
            if "method" in m:
                self.buf.append(m)

    def drain(self):
        """Забрать накопленные события, не блокируясь."""
        self.ws.settimeout(0.05)
        try:
            while True:
                m = json.loads(self.ws.recv())
                if "method" in m:
                    self.buf.append(m)
        except Exception:
            pass
        finally:
            self.ws.settimeout(120)
        out, self.buf = self.buf, []
        return out


def launch():
    profile = "/tmp/reid-ui-shots-profile"
    shutil.rmtree(profile, ignore_errors=True)
    p = subprocess.Popen([
        "google-chrome", "--headless=new", "--remote-debugging-port=9333",
        "--user-data-dir=" + profile, "--no-first-run", "--no-default-browser-check",
        "--disable-gpu", "--hide-scrollbars", "--force-color-profile=srgb",
        "--remote-allow-origins=*",
        "--font-render-hinting=none", "--window-size=%d,%d" % (W, H), "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            with urllib.request.urlopen("http://127.0.0.1:9333/json/version", timeout=1) as r:
                return p, json.loads(r.read())["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("Chrome не поднялся")


# ---------- обёртки над страницей ----------

def js(c, sess, expr, await_promise=False):
    r = c.send("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                    "awaitPromise": await_promise}, sess)
    if "exceptionDetails" in r:
        raise RuntimeError("JS: " + json.dumps(r["exceptionDetails"])[:400])
    return r["result"].get("value")


def wait(c, sess, expr, note, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if js(c, sess, "!!(" + expr + ")"):
            return
        time.sleep(0.15)
    raise TimeoutError("не дождались: " + note)


def settle(c, sess):
    """Шрифты подгружены, картинки дорисованы, кадр отрисован."""
    js(c, sess, "document.fonts.ready", await_promise=True)
    wait(c, sess, "[...document.images].every(i => i.complete)", "картинки", 60)
    js(c, sess, "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))",
       await_promise=True)
    time.sleep(0.25)


def shot(c, sess, name):
    settle(c, sess)
    net_log.extend(c.drain())
    r = c.send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True}, sess)
    path = os.path.join(OUT, name + ".png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(r["data"]))
    print("   снимок", path, os.path.getsize(path) // 1024, "КБ")


def click(c, sess, selector, note=None):
    wait(c, sess, f"document.querySelector({selector!r})", note or selector)
    js(c, sess, f"document.querySelector({selector!r}).click()")


def put_file(c, sess, path):
    doc = c.send("DOM.getDocument", {}, sess)["root"]["nodeId"]
    node = c.send("DOM.querySelector", {"nodeId": doc, "selector": "#file"}, sess)["nodeId"]
    c.send("DOM.setFileInputFiles", {"files": [path], "nodeId": node}, sess)


def drag_box(c, sess, frac):
    """Рамка мышью прямо по канве: как её ведёт оператор."""
    r = js(c, sess, "(() => { const b = canvas.getBoundingClientRect();"
                    "return {x: b.x, y: b.y, w: b.width, h: b.height}; })()")
    x0, y0 = r["x"] + r["w"] * frac[0], r["y"] + r["h"] * frac[1]
    x1, y1 = r["x"] + r["w"] * frac[2], r["y"] + r["h"] * frac[3]
    ev = lambda t, x, y, btn: c.send("Input.dispatchMouseEvent", {
        "type": t, "x": x, "y": y, "button": "left", "clickCount": 1,
        "buttons": btn}, sess)
    ev("mousePressed", x0, y0, 1)
    for i in range(1, 9):
        ev("mouseMoved", x0 + (x1 - x0) * i / 8, y0 + (y1 - y0) * i / 8, 1)
    ev("mouseReleased", x1, y1, 0)


def set_box(c, sess, box):
    x, y, w, h = box
    js(c, sess, f"""(() => {{
      const v = {{bx: {x}, by: {y}, bw: {w}, bh: {h}}};
      for (const k in v) document.getElementById(k).value = v[k];
      document.getElementById('bh').dispatchEvent(new Event('input', {{bubbles: true}}));
    }})()""")


# ---------- сценарий ----------

STUB = """
(() => {
  const orig = window.fetch;
  window.fetch = function (u) {
    if (String(u).indexOf('/api/ui/state') >= 0) {
      return Promise.resolve(new Response(%s, {status: 200,
        headers: {'Content-Type': 'application/json'}}));
    }
    return orig.apply(this, arguments);
  };
})();
"""

NO_GALLERY = dict(service="0.1.0", model="osnet_ain_x1_0_vehicle_reid "
                  "(vehicle-reid-0001, OMZ 2022.1)", score_scale="cosine",
                  default_threshold=0.5495953464415451, storage_reachable=True,
                  gallery_points=0, images_available=True, explain_available=True)
NO_STORAGE = dict(NO_GALLERY, storage_reachable=False)


def page(c, theme, stub=None):
    """Новая вкладка с заданной темой (и, если нужно, с подменой состояния)."""
    tid = c.send("Target.createTarget", {"url": "about:blank"})["targetId"]
    sess = c.send("Target.attachToTarget", {"targetId": tid, "flatten": True})["sessionId"]
    for d in ("Page", "Runtime", "Network", "DOM"):
        c.send(d + ".enable", {}, sess)
    c.send("Emulation.setDeviceMetricsOverride",
           {"width": W, "height": H, "deviceScaleFactor": DPR, "mobile": False}, sess)
    pre = f"try {{ localStorage.setItem('reid-theme', '{theme}'); }} catch (e) {{}}"
    if stub:
        pre += STUB % json.dumps(json.dumps(stub))
    c.send("Page.addScriptToEvaluateOnNewDocument", {"source": pre}, sess)
    c.send("Page.navigate", {"url": BASE}, sess)
    wait(c, sess, "document.readyState === 'complete'", "загрузка страницы")
    return tid, sess


def close(c, tid):
    c.send("Target.closeTarget", {"targetId": tid})


def run_theme(c, theme):
    tag = "-" + theme
    print(" тема:", theme)

    # --- 01 пустое состояние -------------------------------------------------
    tid, sess = page(c, theme)
    c.send("Browser.setDownloadBehavior",
           {"behavior": "allow", "downloadPath": os.path.abspath(EXPORTS),
            "eventsEnabled": True})
    wait(c, sess, "document.getElementById('state-chip').classList.contains('chip-ok')",
         "состояние сервиса")
    assert js(c, sess, "document.documentElement.dataset.theme") == theme, "тема не встала"
    shot(c, sess, "01-start" + tag)

    # --- 02 кадр и рамка -----------------------------------------------------
    put_file(c, sess, os.path.join(IMGDIR, MATCH[0] + ".jpg"))
    wait(c, sess, "!document.getElementById('stage').hidden", "кадр загружен")
    drag_box(c, sess, (0.44, 0.27, 0.82, 0.45))          # рамка мышью
    set_box(c, sess, MATCH[1])                            # затем точные координаты
    wait(c, sess, "!document.getElementById('run').disabled", "кнопка поиска")
    shot(c, sess, "02-query" + tag)

    # --- 03 совпадение -------------------------------------------------------
    click(c, sess, "#run")
    wait(c, sess, "!document.getElementById('cards').hidden "
                  "&& document.querySelectorAll('#cards .card').length", "кандидаты")
    shot(c, sess, "03-match" + tag)

    # --- 04 кадр целиком -----------------------------------------------------
    click(c, sess, "#cards .card .act-frame")
    time.sleep(0.6)
    shot(c, sess, "04-frame-view" + tag)
    click(c, sess, "#cards .card .act-frame")             # вернуть кроп
    time.sleep(0.4)

    # --- 05 разбор -----------------------------------------------------------
    click(c, sess, "#cards .card .act-explain")
    wait(c, sess, "document.getElementById('ex-facts').children.length", "разбор", 180)
    js(c, sess, "window.scrollTo(0, 0)")
    shot(c, sess, "05-explain" + tag)
    click(c, sess, "#explain-close")

    # --- 06 выгрузка ---------------------------------------------------------
    click(c, sess, "#exp-csv")
    time.sleep(0.8)
    click(c, sess, "#exp-json")
    time.sleep(0.8)
    shot(c, sess, "06-export" + tag)

    # --- 07 отказ ------------------------------------------------------------
    put_file(c, sess, os.path.join(IMGDIR, REFUSE[0] + ".jpg"))
    wait(c, sess, "document.getElementById('file-name').textContent.indexOf('5cfbbd42') === 0",
         "второй кадр")
    drag_box(c, sess, (0.08, 0.35, 0.50, 0.78))
    set_box(c, sess, REFUSE[1])
    click(c, sess, "#run")
    wait(c, sess, "!document.getElementById('refusal').hidden", "отказ")
    shot(c, sess, "07-refusal" + tag)

    # --- 08 ниже порога ------------------------------------------------------
    click(c, sess, "#show-below")
    wait(c, sess, "document.querySelectorAll('#cards .card.below').length", "ниже порога")
    shot(c, sess, "08-below" + tag)
    close(c, tid)

    # --- 09/10 состояния сервиса --------------------------------------------
    tid, sess = page(c, theme, NO_GALLERY)
    wait(c, sess, "!document.getElementById('banner').hidden", "полоса «галерея не загружена»")
    shot(c, sess, "09-no-gallery" + tag)
    close(c, tid)

    tid, sess = page(c, theme, NO_STORAGE)
    wait(c, sess, "!document.getElementById('banner').hidden", "полоса «хранилище»")
    shot(c, sess, "10-no-storage" + tag)
    close(c, tid)


proc, ws_url = launch()
try:
    c = CDP(ws_url)
    c.send("Target.setDiscoverTargets", {"discover": True})
    for theme in ("light", "dark"):
        run_theme(c, theme)
finally:
    proc.terminate()

# ---------- журнал сети ----------

seen = []
for m in net_log:
    if m.get("method") == "Network.requestWillBeSent":
        p = m["params"]
        seen.append((p["request"]["method"], p["request"]["url"], p.get("type", "?")))
ext = [u for _, u, _ in seen
       if not (u.startswith("http://localhost:8000/") or u.startswith("blob:")
               or u.startswith("data:"))]
with open(os.path.join(os.path.dirname(OUT) or ".", "network-log.txt"), "w") as f:
    f.write("# Журнал сети браузера за полный прогон интерфейса (обе темы).\n"
            "# Снят по CDP: Network.requestWillBeSent, все запросы страницы.\n\n")
    for meth, u, t in seen:
        f.write(f"{meth:5} {t:12} {u}\n")
    f.write(f"\nвсего запросов: {len(seen)}\n")
    f.write(f"за пределы localhost:8000: {len(ext)}\n")
    for u in ext:
        f.write("  ВНЕШНИЙ: " + u + "\n")
print("\nвсего запросов:", len(seen), "| внешних:", len(ext))
print("выгруженные файлы:", sorted(os.listdir(EXPORTS)))
