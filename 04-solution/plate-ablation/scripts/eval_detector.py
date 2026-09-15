"""Честная оценка локализации: детектор против ручной разметки 36 кропов val."""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
FONT = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 14)
sys.path.insert(0, 'work')
from plate_detect import detect
DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
labels = json.load(open("work/labels.json"))

def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    if x1 <= x0 or y1 <= y0: return 0.0, 0.0
    inter = (x1-x0)*(y1-y0)
    return inter/(a[2]*a[3]+b[2]*b[3]-inter), inter/(b[2]*b[3])

rows, TW, TH, cols = [], 330, 270, 4
sheets = [Image.new("RGB", (TW*cols, TH*3), (15,15,15)) for _ in range(3)]
draws = [ImageDraw.Draw(s) for s in sheets]
t0 = time.perf_counter()
for r in labels:
    with Image.open(DATA/"images"/f"{r['image_id']}.jpg") as im:
        crop = im.convert("RGB").crop((r["x"], r["y"], r["x"]+r["w"], r["y"]+r["h"]))
    det = detect(np.asarray(crop), topn=1)
    box = None if not det else [det[0][1], det[0][2], det[0][3], det[0][4]]
    gt = r["plate"]
    rec = dict(idx=r["idx"], gt=gt, det=box)
    if gt and box:
        i, cov = iou(box, gt)
        cx, cy = box[0]+box[2]/2, box[1]+box[3]/2
        rec.update(iou=i, cover=cov, center_in=bool(gt[0] <= cx <= gt[0]+gt[2] and gt[1] <= cy <= gt[1]+gt[3]),
                   area_ratio=(box[2]*box[3])/(gt[2]*gt[3]))
    rows.append(rec)
    vis = crop.copy(); d = ImageDraw.Draw(vis); lw = max(2, r["w"]//250)
    if gt: d.rectangle([gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]], outline=(0,255,0), width=lw)
    if box: d.rectangle([box[0], box[1], box[0]+box[2], box[1]+box[3]], outline=(255,0,0), width=lw)
    sc = min(TW/vis.width, (TH-16)/vis.height)
    vis = vis.resize((int(vis.width*sc), int(vis.height*sc)), Image.BILINEAR)
    sh, k = r["idx"]//12, r["idx"] % 12
    ox, oy = (k % cols)*TW, (k//cols)*TH
    sheets[sh].paste(vis, (ox, oy+16))
    lab = f"#{r['idx']}"
    if gt and box: lab += f" IoU={rec['iou']:.2f} покр={rec['cover']:.2f}"
    elif not gt: lab += " разметка: пластины нет"
    elif not box: lab += " детектор: пусто"
    draws[sh].text((ox+3, oy+3), lab, fill=(255,255,0), font=FONT)
for i, s in enumerate(sheets): s.save(f"pics/detector_vs_labels_{i}.png")

vis_gt = [r for r in rows if r["gt"]]
det_any = [r for r in vis_gt if r["det"]]
ious = np.array([r["iou"] for r in det_any]); cov = np.array([r["cover"] for r in det_any])
hit = np.array([r["center_in"] for r in det_any])
summary = {
  "кропов размечено": len(rows),
  "пластина видна": len(vis_gt),
  "детектор выдал бокс": len(det_any),
  "центр детекции внутри пластины": int(hit.sum()),
  "hit_rate": round(float(hit.mean()), 3),
  "покрытие пластины >=0.5": int((cov >= 0.5).sum()),
  "покрытие пластины >=0.8": int((cov >= 0.8).sum()),
  "покрытие: медиана": round(float(np.median(cov)), 3),
  "покрытие: среднее": round(float(cov.mean()), 3),
  "IoU: медиана": round(float(np.median(ious)), 3),
  "IoU>=0.3": int((ious >= 0.3).sum()), "IoU>=0.5": int((ious >= 0.5).sum()),
  "площадь детекции / площадь пластины (медиана)": round(float(np.median([r["area_ratio"] for r in det_any])), 2),
  "на кропе без видимой пластины детектор": ("выдал бокс" if [r for r in rows if not r["gt"]][0]["det"] else "пусто"),
  "время на кроп, с": round((time.perf_counter()-t0)/len(rows), 3),
}
json.dump({"summary": summary, "per_crop": rows}, open("out/detector_eval.json","w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("\nпромахи (центр вне пластины):", [r["idx"] for r in det_any if not r["center_in"]])
