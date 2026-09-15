"""Проверки соответствия ТЗ (разделы 8–9) и критериям приёмки, на оригинале."""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "run")
import numpy as np
import reid_metrics as metrics

RUN = dict(threshold=0.5, camera_policy="market", refusal_mode="top1")

# --- T1: полная галерея vs первые 10 кандидатов (ТЗ §8 submission.csv) ----
ng = 12
gids = [1] + list(range(100, 100 + ng - 1))   # единственное совпадение
scores = [[0.1] + [round(0.9 - 0.05 * i, 4) for i in range(ng - 1)]]
# совпадение (столбец 0, оценка 0.1) оказывается на 12-м месте
r = metrics.evaluate(scores, [1], gids, [0], [1] * ng, [False], **RUN)
q = r["per_query"][0]
print("T1: полная галерея: mAP=%.6f Rank-1=%s mINP=%.6f ранг совпадения=%s"
      % (r["ranking"]["mAP"], r["ranking"]["Rank-1"], r["ranking"]["mINP"],
         q["positive_ranks"]))
print("    submission.csv по ТЗ содержит только топ-10 -> совпадение в сдаваемый"
      " артефакт не попадает; AP по сданному списку = 0, mINP не определён.")
print("    Код не умеет считать метрику усечённого списка (ranking_scope=%r),"
      % r["protocol"]["ranking_scope"])
print("    параметра усечения нет.")

# --- T2: ничьи и «лестность» при вырожденной модели (критерий 5) ----------
nq, ngal, m = 40, 40, 2
gids2 = []
for i in range(nq):
    gids2 += [i, i]  # галерея сгруппирована по идентичности: [0,0,1,1,2,2...]
gids2 = gids2[:ngal * 2]
ng2 = len(gids2)
const = [[1.0] * ng2 for _ in range(nq)]
qids2 = list(range(nq))
sorted_g = metrics.evaluate(const, qids2, gids2, [0] * nq, [1] * ng2,
                            [False] * nq, **RUN)
rng = np.random.default_rng(7)
perm = rng.permutation(ng2)
shuf = metrics.evaluate(np.asarray(const)[:, perm], qids2,
                        [gids2[j] for j in perm], [0] * nq, [1] * ng2,
                        [False] * nq, **RUN)
print("\nT2: вырожденная модель (все оценки равны), %d запросов, %d объектов:"
      % (nq, ng2))
print("    галерея, отсортированная по vehicle_id:  mAP=%.4f Rank-1=%.4f"
      % (sorted_g["ranking"]["mAP"], sorted_g["ranking"]["Rank-1"]))
print("    та же галерея в случайном порядке:       mAP=%.4f Rank-1=%.4f"
      % (shuf["ranking"]["mAP"], shuf["ranking"]["Rank-1"]))

# --- T3: независимая проверка случайного уровня (критерий 5) --------------
rng = np.random.default_rng(424242)
nq3, n3, m3 = 2000, 60, 3
gids3 = list(range(1000, 1000 + n3)); gids3[-m3:] = [1] * m3  # позитивы в конце
res = metrics.evaluate(rng.random((nq3, n3)), [1] * nq3, gids3, [0] * nq3,
                       [1] * n3, [False] * nq3, **RUN)
h = sum(1 / r_ for r_ in range(1, n3 + 1))
exp_map = h / n3 + (m3 - 1) * (n3 - h) / (n3 * (n3 - 1))
print("\nT3: случайные оценки (свой seed, позитивы в конце галереи):"
      " mAP=%.4f ожидание=%.4f Rank-1=%.4f ожидание=%.4f"
      % (res["ranking"]["mAP"], exp_map, res["ranking"]["Rank-1"], m3 / n3))

# --- T4: CLI со всеми опциональными массивами (round-trip npz) ------------
with tempfile.TemporaryDirectory() as td:
    npz = Path(td) / "in.npz"
    out = Path(td) / "out.json"
    np.savez(npz,
             scores=np.array([[0.9, 0.8, 0.7], [0.2, 0.6, 0.1]]),
             query_ids=np.array(["a", "z"]), gallery_ids=np.array(["b", "a", "a"]),
             query_cameras=np.array([0, 0]), gallery_cameras=np.array([1, 0, 1]),
             known_absent=np.array([False, True]),
             gallery_keys=np.array(["k2", "k0", "k1"]),
             gallery_junk=np.array([False, False, False]),
             exclude_mask=np.zeros((2, 3), dtype=bool),
             query_frames=np.array([1, 2]), gallery_frames=np.array([1, 3, 4]))
    p = subprocess.run([sys.executable, "-B", "run/reid_metrics.py", str(npz),
                        "--threshold", "0.5", "--camera-policy", "market",
                        "--refusal-mode", "top1", "--output", str(out)],
                       capture_output=True, text=True)
    import json
    cli = json.loads(out.read_text()) if p.returncode == 0 else None
    direct = metrics.evaluate([[0.9, 0.8, 0.7], [0.2, 0.6, 0.1]], ["a", "z"],
                              ["b", "a", "a"], [0, 0], [1, 0, 1], [False, True],
                              gallery_keys=["k2", "k0", "k1"],
                              gallery_junk=[False] * 3,
                              exclude_mask=[[False] * 3] * 2,
                              query_frames=[1, 2], gallery_frames=[1, 3, 4],
                              include_rankings=True, **RUN)
    same = cli is not None and \
        cli["ranking"] == {k: direct["ranking"][k] for k in cli["ranking"]} and \
        cli["per_query"][0]["ranking"] == direct["per_query"][0]["ranking"]
    print("\nT4: CLI со всеми опциональными массивами: код возврата=%s,"
          " совпадает с прямым вызовом: %s" % (p.returncode, same))
    if p.returncode:
        print(p.stderr[-500:])

# --- T5: пример «mAP по всем запросам» (§9) при наличии distractor-запросов
r5 = metrics.evaluate([[0.9, 0.1], [0.8, 0.7]], [1, 9], [1, 2], [0, 0], [1, 1],
                      [False, True], **RUN)
print("\nT5: 2 запроса (1 known с AP=1, 1 unknown): код mAP=%s"
      " (знаменатель=%s valid); прочтение «по всем запросам» дало бы 0.5"
      % (r5["ranking"]["mAP"], r5["ranking"]["num_valid_queries"]))
