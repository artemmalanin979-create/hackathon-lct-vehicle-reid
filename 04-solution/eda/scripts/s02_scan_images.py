#!/usr/bin/env python3
"""s02: один проход по всем файлам images/: md5, размер, формат, EXIF/метаданные,
таблицы квантования JPEG, миниатюра 32x32 (grayscale) как кэш признаков.
Выход: out_s02_meta.json (сводка), scan_files.json (список файлов в порядке скана),
scan_meta.jsonl (по-файлово), thumbs32.npy (uint8 N x 1024).
Запуск: python3 s02_scan_images.py"""
from pathlib import Path
import hashlib, json, os
from multiprocessing import Pool
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = str(Path(os.environ.get("REID_DATA_DIR", REPO / "data")) / "images")
HERE = os.path.dirname(os.path.abspath(__file__))

def one(fname):
    path = os.path.join(DATA, fname)
    with open(path, "rb") as f:
        data = f.read()
    md5 = hashlib.md5(data).hexdigest()
    rec = {"f": fname, "bytes": len(data), "md5": md5}
    try:
        im = Image.open(path)
        rec.update(fmt=im.format, size=list(im.size), mode=im.mode,
                   info_keys=sorted(k for k in im.info if k != "exif"),
                   progressive=bool(im.info.get("progressive", 0)))
        ex = dict(im.getexif())
        rec["exif_n"] = len(ex)
        if ex:
            rec["exif"] = {str(k): str(v)[:80] for k, v in ex.items()}
        q = getattr(im, "quantization", None)
        if q:
            blob = json.dumps({str(k): list(v) for k, v in sorted(q.items())})
            rec["qhash"] = hashlib.md5(blob.encode()).hexdigest()[:12]
            rec["qtables"] = blob
        # быстрый черновой декод для миниатюры
        im.draft("L", (160, 160))
        g = im.convert("L").resize((32, 32), Image.BILINEAR)
        rec["thumb"] = np.asarray(g, dtype=np.uint8).tobytes().hex()
    except Exception as e:
        rec["error"] = repr(e)
    return rec

def main():
    files = sorted(os.listdir(DATA))
    with Pool(10) as p:
        recs = p.map(one, files, chunksize=64)
    thumbs = np.zeros((len(recs), 1024), dtype=np.uint8)
    qtables_uniq = {}
    with open(os.path.join(HERE, "scan_meta.jsonl"), "w") as f:
        for i, r in enumerate(recs):
            t = r.pop("thumb", None)
            if t:
                thumbs[i] = np.frombuffer(bytes.fromhex(t), dtype=np.uint8)
            qt = r.pop("qtables", None)
            if qt is not None and r.get("qhash") not in qtables_uniq:
                qtables_uniq[r["qhash"]] = qt
            f.write(json.dumps(r) + "\n")
    np.save(os.path.join(HERE, "thumbs32.npy"), thumbs)
    with open(os.path.join(HERE, "scan_files.json"), "w") as f:
        json.dump(files, f)

    from collections import Counter
    sizes = Counter(tuple(r["size"]) for r in recs if "size" in r)
    fmts = Counter(r.get("fmt") for r in recs)
    modes = Counter(r.get("mode") for r in recs)
    qh = Counter(r.get("qhash") for r in recs)
    exif_n = Counter(r.get("exif_n", -1) for r in recs)
    info = Counter(tuple(r.get("info_keys", [])) for r in recs)
    prog = Counter(r.get("progressive") for r in recs)
    errors = [r for r in recs if "error" in r]
    md5s = Counter(r["md5"] for r in recs)
    dups = {h: c for h, c in md5s.items() if c > 1}
    out = {
        "n_files": len(files),
        "formats": dict(fmts), "modes": dict(modes),
        "resolutions": {f"{w}x{h}": c for (w, h), c in sizes.most_common()},
        "n_resolutions": len(sizes),
        "qhash_counts": dict(qh.most_common()),
        "qtables_by_hash": qtables_uniq,
        "exif_tag_counts": {str(k): v for k, v in exif_n.items()},
        "info_keys_variants": {" ".join(k): v for k, v in info.most_common()},
        "progressive": {str(k): v for k, v in prog.items()},
        "byte_dup_md5_groups": len(dups),
        "byte_dup_files_total": sum(dups.values()) if dups else 0,
        "errors": errors[:20],
    }
    with open(os.path.join(HERE, "out_s02_meta.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    o2 = dict(out); o2.pop("qtables_by_hash", None)
    print(json.dumps(o2, ensure_ascii=False, indent=1)[:4000])

if __name__ == "__main__":
    main()
