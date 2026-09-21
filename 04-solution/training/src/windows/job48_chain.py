"""job_48: оркестратор ОДНОЙ попытки дообучения на объединённом наборе.

Запускается на worker через Invoke-WmiMethod Win32_Process Create (переживает обрыв SSH):
    D:\\lct-reid\\py312\\python.exe D:\\lct-reid\\job48_chain.py

Зашитое ДО старта расписание (LP-FT, recipe 7.6, параметры ain_v2, проверенные на наших данных):
  этап 1 (LP):        5 эпох, --freeze-fc (только classifier), lr 3.5e-4
  этап 2 (частичный): 15 эпох, conv4+conv5+fc, lr 3.5e-5, cls-lr 3.5e-4
  этап 3 (полный):    6 эпох, всё, lr 1.5e-5, cls-lr 3.5e-4
                      ВОРОТА: только если лучшая dev mAP этапа 2 > baseline (dev mAP
                      необученной модели, эпоха 0) И CH не растёт два замера подряд к концу
                      этапа 2 (kill-rule 8.1: иначе прогон убивается, этап 3 не делается).
  выбор модели: этап с большей dev mAP на конец этапа (s2 vs s3); ничья -> меньший CH.
                      Выбранный этап экспортируется в ONNX (export_onnx2.py).

Статус и весь вывод дочерних процессов — в runs/combined_v1/chain.log / chain_status.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(r"D:\lct-reid")
PY = str(ROOT / "py312" / "python.exe")
TRAIN = str(ROOT / "reid_train5.py")
EXPORT = str(ROOT / "export_onnx2.py")
RUN = "combined_v1"
RUN_DIR = ROOT / "runs" / RUN
STATUS = RUN_DIR / "chain_status.json"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUN_DIR / "chain.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def set_status(**kw):
    d = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    d.update(kw)
    d["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(d, indent=1))


def run_stage(stage, epochs, extra):
    cmd = [PY, TRAIN, "--run", RUN, "--stage", str(stage), "--epochs", str(epochs)] + extra
    log(f"START stage{stage}: {' '.join(cmd[2:])}")
    t0 = time.perf_counter()
    with open(RUN_DIR / f"chain_s{stage}.out.txt", "ab") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    el = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"stage{stage} вернул код {r.returncode}")
    log(f"DONE stage{stage} за {el/60:.1f} мин")
    return read_log(stage)


def read_log(stage):
    recs = []
    p = RUN_DIR / f"stage{stage}_log.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def dev_recs(recs):
    return [r for r in recs if "dev" in r]


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    set_status(state="running", run=RUN)
    log("=== job_48 chain старт ===")
    try:
        # --- этап 1: LP (только классификатор) ---
        recs1 = run_stage(1, 5, ["--freeze-fc", "--lr", "3.5e-4"])
        base = dev_recs(recs1)[0]["dev"]["mAP"]     # необученная модель на dev
        log(f"baseline dev mAP (эпоха 0): {base:.5f}")

        # --- этап 2: conv4+conv5 ---
        recs2 = run_stage(2, 15, ["--resume-from", "stage1.pt",
                                  "--lr", "3.5e-5", "--cls-lr", "3.5e-4"])
        d2 = dev_recs(recs2)
        best2 = max(d2[1:], key=lambda r: r["dev"]["mAP"])
        log(f"stage2: лучшая dev mAP {best2['dev']['mAP']:.5f} @ эпоха {best2['epoch']}; "
            f"последние CH: {[r['dev']['CH'] for r in d2[-3:]]}")

        # --- ворота этапа 3 ---
        ch_tail = [r["dev"]["CH"] for r in d2[-3:]]
        ch_rising = len(ch_tail) >= 3 and ch_tail[0] < ch_tail[1] < ch_tail[2]
        gate_map = best2["dev"]["mAP"] > base
        log(f"ворота этапа 3: mAP {best2['dev']['mAP']:.5f} > base {base:.5f} = {gate_map}; "
            f"CH rising tail = {ch_rising}")

        winner = 2
        recs3 = None
        if gate_map and not ch_rising:
            recs3 = run_stage(3, 6, ["--resume-from", "stage2.pt",
                                     "--lr", "1.5e-5", "--cls-lr", "3.5e-4"])
            d3 = dev_recs(recs3)
            best3 = max(d3[1:], key=lambda r: r["dev"]["mAP"])
            log(f"stage3: лучшая dev mAP {best3['dev']['mAP']:.5f} @ эпоха {best3['epoch']}")
            if best3["dev"]["mAP"] > best2["dev"]["mAP"]:
                winner = 3
            elif best3["dev"]["mAP"] == best2["dev"]["mAP"] and \
                    d3[-1]["dev"]["CH"] < d2[-1]["dev"]["CH"]:
                winner = 3
        else:
            log("этап 3 пропущен по воротам")

        # --- экспорт победителя ---
        out_onnx = ROOT / f"job48_combined_s{winner}.onnx"
        r = subprocess.run([PY, EXPORT, "--weights", str(RUN_DIR / f"stage{winner}.pt"),
                            "--out", str(out_onnx)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log(f"export stage{winner} -> {out_onnx}: rc={r.returncode}\n{r.stdout[-800:]}")
        if r.returncode != 0:
            raise RuntimeError("export провален")

        set_status(state="done", winner=winner, baseline_dev_mAP=round(base, 5),
                   best_s2_dev_mAP=round(best2["dev"]["mAP"], 5),
                   onnx=str(out_onnx))
        log(f"=== ГОТОВО: победитель этап {winner} ===")
    except Exception:
        err = traceback.format_exc()
        log(f"ОШИБКА:\n{err}")
        set_status(state="error", error=err[-2000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
