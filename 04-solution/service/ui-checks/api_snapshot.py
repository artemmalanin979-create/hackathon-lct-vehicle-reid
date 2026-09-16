#!/usr/bin/env python3
"""Снимок ответов API: побайтово в каталог. Запускается до и после правок."""
import json, sys, os, urllib.request

NEW = "http://localhost:8000"
IMG = "/home/artem/projects/hackathon-lct-vehicle-reid/data/images/%s.jpg"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

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
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()

def post_json(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()

def save(name, data):
    with open(os.path.join(out, name), "wb") as f:
        f.write(data)
    print("  сохранено", name, len(data), "байт")

for path in ("/api/health", "/api/version", "/api/ui/state", "/openapi.json"):
    save(path.strip("/").replace("/", "_") + ".json", get(NEW + path))

QUERIES = [
    ("a4f2a13bd2b54360a921c8ef7366e535", dict(x=849, y=300, w=736, h=471, top_k=10)),
    ("5cfbbd42352245fb9ab4e93f0e17452a", dict(x=125, y=368, w=785, h=457, top_k=10)),
    ("486dd80d22f94a5496b3e670b19399a7", dict(x=1202, y=270, w=588, h=474, top_k=5)),
    ("06975950c6134fdfb4b0ed6a5cf0a50d", dict(x=318, y=321, w=1006, h=759, top_k=10)),
]
for qid, f in QUERIES:
    save(f"search_{qid[:8]}.json", post_multipart(NEW + "/api/search", f, IMG % qid))

qid, f = QUERIES[0]
eb = post_multipart(NEW + "/api/embed", {k: v for k, v in f.items() if k != "top_k"}, IMG % qid)
save("embed.json", eb)
vec = json.loads(eb)["embedding"]
save("search_vector.json", post_json(NEW + "/api/search/vector", {"vector": vec, "top_k": 10}))
save("search_vector_thr.json",
     post_json(NEW + "/api/search/vector", {"vector": vec, "top_k": 3, "threshold": 0.9}))
# разбор пары — он тоже часть API интерфейса
save("explain.json", post_multipart(NEW + "/api/explain",
     dict(x=849, y=300, w=736, h=471, gallery_id="db620271348d404a9a0b978d96c63f29"),
     IMG % "a4f2a13bd2b54360a921c8ef7366e535"))
save("crop.bin", get(NEW + "/api/gallery/db620271348d404a9a0b978d96c63f29/crop?size=360"))
print("готово ->", out)
