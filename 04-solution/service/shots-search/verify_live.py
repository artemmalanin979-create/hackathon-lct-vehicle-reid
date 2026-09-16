#!/usr/bin/env python3
"""Сверка: то ли выдаёт живой сервис, что посчитано на векторах baseline."""
import json, subprocess, sys
IMGS = "/home/artem/projects/hackathon-lct-vehicle-reid/data/images"
recs = {r["qi"]: r for r in json.load(open("picks_all.json"))}
QIS = [int(a) for a in sys.argv[1:]] or [405, 67, 302, 886, 656, 577, 515, 1011, 434, 721, 1083]
bad = 0
for qi in QIS:
    r = recs[qi]
    x, y, w, h = r["box"]
    out = subprocess.run(["curl", "-s", "-X", "POST", "http://localhost:8000/api/search",
        "-F", f"file=@{IMGS}/{r['img']}.jpg", "-F", f"x={x}", "-F", f"y={y}",
        "-F", f"w={w}", "-F", f"h={h}", "-F", "top_k=10", "-F", "threshold=-1"],
        capture_output=True, text=True).stdout
    got = json.loads(out)["candidates"]
    exp = r["top10"]
    ok = all(g["gallery_id"] == e[0] and abs(g["confidence"] - e[3]) < 6e-4
             for g, e in zip(got, exp))
    bad += not ok
    print(f"qi={qi:4d} {'совпало' if ok else 'РАСХОЖДЕНИЕ'}  "
          + " ".join(f"{g['confidence']:.4f}" for g in got[:4])
          + " | ожидалось " + " ".join(f"{e[3]:.4f}" for e in exp[:4]))
print("расхождений:", bad)
