#!/usr/bin/env python3
"""Сверка API до и после правок.

8002 — сборка из git HEAD (состояние до задания), 8000 — текущая.
Требование задания: существующие ответы не меняются. Проверяются и схема
(openapi.json), и живые ответы — побайтово.
"""
import json, sys, urllib.request

BASE, NEW = "http://localhost:8002", "http://localhost:8000"
IMG = "/home/artem/projects/hackathon-lct-vehicle-reid/data/images/%s.jpg"
ok = True


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def post_multipart(url, fields, filepath):
    boundary = "----reidcheck"
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
                 f"{v}\r\n").encode()
    with open(filepath, "rb") as f:
        data = f.read()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"q.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def post_json(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def report(name, a, b):
    global ok
    same = a == b
    ok &= same
    print(f"  {'совпадает' if same else 'РАСХОЖДЕНИЕ':>12}  {name}")
    if not same:
        print("    было:", a[:300])
        print("    стало:", b[:300])


print("== живые ответы (побайтово) ==")
for path in ("/api/health", "/api/version"):
    report(path, get(BASE + path), get(NEW + path))

QUERIES = [
    ("a4f2a13bd2b54360a921c8ef7366e535", dict(x=849, y=300, w=736, h=471, top_k=10)),
    ("5cfbbd42352245fb9ab4e93f0e17452a", dict(x=125, y=368, w=785, h=457, top_k=10)),
    ("486dd80d22f94a5496b3e670b19399a7", dict(x=1202, y=270, w=588, h=474, top_k=5)),
    ("06975950c6134fdfb4b0ed6a5cf0a50d", dict(x=318, y=321, w=1006, h=759, top_k=10)),
]
for qid, f in QUERIES:
    report(f"/api/search {qid[:8]}",
           post_multipart(BASE + "/api/search", f, IMG % qid),
           post_multipart(NEW + "/api/search", f, IMG % qid))

qid, f = QUERIES[0]
eb = post_multipart(BASE + "/api/embed", {k: v for k, v in f.items() if k != "top_k"}, IMG % qid)
en = post_multipart(NEW + "/api/embed", {k: v for k, v in f.items() if k != "top_k"}, IMG % qid)
report("/api/embed", eb, en)

vec = json.loads(eb)["embedding"]
report("/api/search/vector",
       post_json(BASE + "/api/search/vector", {"vector": vec, "top_k": 10}),
       post_json(NEW + "/api/search/vector", {"vector": vec, "top_k": 10}))
report("/api/search/vector (свой порог)",
       post_json(BASE + "/api/search/vector", {"vector": vec, "top_k": 3, "threshold": 0.9}),
       post_json(NEW + "/api/search/vector", {"vector": vec, "top_k": 3, "threshold": 0.9}))

print("\n== схема openapi.json: прежние элементы ==")
a = json.loads(get(BASE + "/openapi.json"))
b = json.loads(get(NEW + "/openapi.json"))
for p, spec in a["paths"].items():
    report(f"paths[{p}]", json.dumps(spec, sort_keys=True),
           json.dumps(b["paths"].get(p), sort_keys=True))
for s, spec in a.get("components", {}).get("schemas", {}).items():
    report(f"schemas[{s}]", json.dumps(spec, sort_keys=True),
           json.dumps(b.get("components", {}).get("schemas", {}).get(s), sort_keys=True))
report("info/openapi", json.dumps({k: a[k] for k in ("openapi", "info")}, sort_keys=True),
       json.dumps({k: b[k] for k in ("openapi", "info")}, sort_keys=True))

added = sorted(set(b["paths"]) - set(a["paths"]))
print("\n== добавлено (было пусто) ==")
for p in added:
    print("  +", p)

print("\nИТОГ:", "прежнее поведение API не изменилось" if ok else "ЕСТЬ РАСХОЖДЕНИЯ")
sys.exit(0 if ok else 1)
