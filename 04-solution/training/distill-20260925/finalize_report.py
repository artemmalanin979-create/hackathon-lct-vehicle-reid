#!/usr/bin/env python3
"""Assemble evidence, then update the owned Obsidian note and its mirror."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument("--note",type=Path,required=True);a=p.parse_args()
    run=HERE/"artifacts/run"; evidence=HERE/"results"
    for f in run.glob("*.json"):
        (evidence/f.name).write_bytes(f.read_bytes())
    (evidence/"export.json").write_bytes((run/"export/export.json").read_bytes())
    ev=json.loads((run/"evaluation.json").read_text());bench=json.loads((run/"benchmark.json").read_text())
    training=json.loads((run/"training.json").read_text());sel=json.loads((run/"selection.json").read_text());env=json.loads((run/"environment.json").read_text())
    parity=json.loads((run/"export_ranking_check.json").read_text());ex=json.loads((run/"export/export.json").read_text())
    speed=1-bench["measurements"]["student"]["batch1_p95_ms"]/bench["measurements"]["baseline"]["batch1_p95_ms"]
    overhead=bench["measurements"]["fusion"]["batch1_p95_ms"]/bench["measurements"]["baseline"]["batch1_p95_ms"]-1
    stud_quality=all(ev["comparisons"]["student"][m]["delta"]>=0 and ev["comparisons"]["student"][m]["ci95"][0]>=0 for m in ["cosine","KR"])
    fusion_quality=all(ev["comparisons"]["fusion"][m]["delta"]>=.005 and ev["comparisons"]["fusion"][m]["ci95"][0]>0 for m in ["cosine","KR"])
    decision={"student_quality_pass":stud_quality,"student_p95_speedup":speed,"student_speed_pass":speed>=.25,
              "fusion_quality_pass":fusion_quality,"fusion_p95_overhead":overhead,"fusion_speed_pass":overhead<=.1,
              "student_promoted":False,"fusion_promoted":False,"release":"d1_j48 unchanged",
              "reason":"both trained student and fusion fail frozen quality gates","independent_review":"PENDING"}
    if stud_quality or fusion_quality:raise AssertionError("manual acceptance review required")
    (evidence/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    artifacts={str(f.relative_to(HERE)):{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"bytes":f.stat().st_size}
               for f in run.rglob("*") if f.is_file()}
    (evidence/"local_artifact_manifest.json").write_text(json.dumps(artifacts,indent=2)+"\n")
    records=[("contracts",0,5,0,0,"logs/contracts-green.txt"),("guard mutations",0,4,0,0,"results/mutations.json"),
             ("canonical evaluator tests",0,60,0,0,"logs/evaluator-tests.txt"),("smoke training+export",0,2,0,0,"results/smoke-export.json"),
             ("main+ablation training",0,60,0,0,"logs/train.txt"),("quality evaluation",0,8,0,0,"logs/evaluate.txt"),
             ("ONNX export",0,1860,0,0,"logs/export.txt"),("ONNX ranking/refusal",0,2220,0,0,"results/export_ranking_check.json"),
             ("CPU benchmark",0,120,0,0,"results/benchmark.json"),
             ("real image+bbox inference CLI",0,1,0,0,"results/infer_smoke.json")]
    manifest={"code_commit":"53f7261","training_source_sha256":env["experiment_sha256"],"protocol_sha256":env["protocol_sha256"],
              "baseline_sha":"8e7a20580613f5a784c6ee47270a209cf1923e0e","records":[dict(zip(["check","exit_code","cases","failed","skipped","log"],r)) for r in records],
              "counts_note":"cases refer to tests, epochs, vectors, query-mode pairs, or measured descriptor calls as named; not one summed test count",
              "clock_note":"worker wall clock lags workstation by about2.6h; protocol/code hashes and sequential orchestration/monotonic durations establish provenance, not mtimes",
              "not_run":["release integration","public deployment","full CNN retraining","fresh untouched holdout evaluation","end-to-end API latency","independent visual view annotation"],
              "initial_red":"missing contracts module import, NOT a killed mutation; separate four semantic mutations establish guard sensitivity"}
    (evidence/"verification_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    lines=["# Дистилляция d1_j48: прототип обучен, в релиз не принят", "",
           "Student и fusion не прошли заранее объявленный критерий качества. Сдаваемой остаётся **d1_j48**; release-код, модели, пороги и gallery не менялись.","",
           "## Что реально выполнено", "",
           f"Замороженная combined_v1 + residual MLP512→256→512; один seed20260925, main30 эпох и одна ablation30 без relational-loss. Обучение заняло {env['total_seconds']:.1f}с, peakRSS {env['peak_rss_mib']:.1f}MiB. Main выбран на эпохе{training['main']['best_epoch']} по dev embedding loss, ablation на эпохе{training['ablation']['best_epoch']}. Внешний validation не участвовал в выборе.","",
           f"Fit6639 кадров/1071ID; исторический dev100 —609 кадров. Refusal dev: {sel['dev_query']} query ×{sel['dev_gallery']} gallery,25 ID без gallery-пары. Teacher whitening обучен только на fit. Fusion: вес student{sel['fusion_student_weight']}, выбран на dev; размерность1024 — исключительно исследовательский формат.","",
           "## Качество на многократно использованном validation", "",
           "1110query×750gallery,832 query с межкамерной парой. Market/presence, full-gallery AP, KR(6,3,0.3). F1/TNR ниже используют отдельные **dev-пороги**, в том числе для baseline; официальные релизные пороги показаны отдельно.","",
           "| Модель | Режим | mAP | Rank-1 | Rank-5 | mINP | F1 dev-порог | TNR dev-порог | Camera gap mAP |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    csvrows=[]
    for name in ["baseline","student","fusion","ablation"]:
        for mode in ["cosine","KR"]:
            z=ev["metrics"][name][mode]
            values=[z[k] for k in ["mAP","Rank-1","Rank-5","mINP","F1","TNR","camera_gap_mAP"]]
            lines.append(f"| {name} | {mode} | "+" | ".join(f"{n:.6f}" for n in values)+" |")
            csvrows.append({"model":name,"mode":mode,**{k:v for k,v in z.items() if k not in ["counts","by_camera","official_threshold_result"]}})
    with (evidence/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    lines.extend(["", "Официальный baseline при сохранённых релизных порогах:", "", "| Режим | Порог | F1 | TNR |", "|---|---:|---:|---:|"])
    for mode in ["cosine","KR"]:
        z=ev["metrics"]["baseline"][mode]["official_threshold_result"]
        lines.append(f"| {mode} | {z['threshold']:.16f} | {z['F1']:.6f} | {z['TNR']:.6f} |")
    lines.extend(["", "Максимизация F1 на маленьком dev дала слабый перенос отказа на cosine: TNR низкий. Эти экспериментальные пороги не пригодны для переноса в сервис. После просмотра validation пороги не менялись.","",
                  "## Неопределённость и ошибки","","Парный bootstrap4000 по vehicle_id, seed20260925; обычные CI95 и Holm по четырём сравнениям student/fusion×cosine/KR. Абляция описательная; выигрышный seed не выбирался.","",
                  "| Сравнение с d1 | Режим | ΔmAP | CI95 | p Holm | Исправлено / испорчено top-1 |","|---|---|---:|---|---:|---:|"])
    for name in ["student","fusion","ablation"]:
        for mode in ["cosine","KR"]:
            z=ev["comparisons"][name][mode]
            ph=f"{z['p_Holm']:.6f}" if 'p_Holm' in z else "описательно"
            lines.append(f"| {name} | {mode} | {z['delta']:+.6f} | [{z['ci95'][0]:+.6f};{z['ci95'][1]:+.6f}] | {ph} | {len(z['fixed_query_ids'])} / {len(z['broken_query_ids'])} |")
    lines.extend(["", "Перечни исправленных/испорченных query, корреляции ошибок и разбиение по камерам: [evaluation.json](results/evaluation.json). Независимой view-разметки нет: view-strata **NOT MEASURED**. Нельзя приписывать падение исключительно MLP или relational-loss: teacher whitening fit-only отличается от релизного, а отдельная абляция этого фактора не предусматривалась.","",
                  "## Измеренная стоимость", "", "worker-vm CPU, ORT1.30.0,2threads,batch1,warmup5,40кадров; порядок трёх моделей чередуется. Измеряется готовый тензор→нормированный признак; decode/API/Qdrant/KR сюда не входят. Узел общий, фоновая нагрузка не устранялась.","",
                  "| Модель | p50 мс | p95 мс | FPS batch1 | Cold start с | PeakRSS cold MiB | Веса МБ | Размерность / raw gallery750 |", "|---|---:|---:|---:|---:|---:|---:|---|"])
    for name,z in bench["measurements"].items():
        lines.append(f"| {name} | {z['batch1_p50_ms']:.2f} | {z['batch1_p95_ms']:.2f} | {z['FPS_batch1']:.2f} | {z['cold_start_seconds']:.3f} | {z['cold_process_peak_rss_mib']:.1f} | {z['weights_bytes']/1e6:.3f} | {z['embedding_dimensions']} / {z['gallery750_raw_index_bytes']}Б |")
    lines.extend(["", f"Ускорение student по p95:{speed:.1%}; overhead fusion:{overhead:+.1%}. Даже выигрыш скорости не разрешает потерю качества: критерий student — lowerCI≥0 и pointΔ≥0 в обеих метриках. Он не выполнен. Fusion также не достигΔmAP≥0.005/lowerCI>0. VRAM — неприменимо(CPU); размер индекса — только raw векторы, без накладных расходов Qdrant. PeakRSS по модели измерен в отдельном холодном процессе, не является отдельным продолжительным нагрузочным тестом.","",
                  "## Экспорт, проверки и артефакты", "",
                  f"Head ONNX на1860 признаках: maxabs {ex['max_abs']:.3g}. Полный image→student ONNX на40 реальных cached-crop входах: maxabs {bench['image_export_max_abs_vs_cached_torch_head']:.3g}. Cosine/KR:0 изменений top-1 и0 изменений отказа на1110query каждого режима; mAP совпал. [Проверка](results/export_ranking_check.json).",
                  "", "5 новых behavioral tests PASS;4/4 содержательных guard-removal mutations убиты в одноразовых копиях; исходный hash неизменён;60 штатных тестов evaluator PASS. Первоначальный RED был отсутствующим модулем, он **не считается** убитой мутацией. [Полный manifest](results/verification_manifest.json).", "",
                  "| Артефакт | SHA-256 |", "|---|---|"])
    for relative in ["artifacts/run/main.npz","artifacts/run/export/head.onnx","artifacts/run/export/student_combined_v1.onnx"]:
        lines.append(f"| `{relative}` | `{artifacts[relative]['sha256']}` |")
    lines.extend(["", "Артефакты не входят в Git; сохранены локально в этом worktree и на `worker-vm:~/lct-reid/jobs/distill_20260925/results/`. Перед удалением worktree скопировать **весь** `artifacts/run/` в каталог принятого handoff и проверить [local_artifact_manifest.json](results/local_artifact_manifest.json). Исходники/config/метрики/логи входят в Git. Никаких новых внешних весов или датасетов не скачивалось; исходный provenance/licensing OSNet и combined_v1 остаётся прежним.","",
                  "## Запуск готового ядра", "", "`infer.py` принимает настоящий JPEG/PNG и bbox, проверяет SHA полного ONNX и выдаёт нормированный float32-вектор512. Проверен реальным validation JPEG и исходным bbox: результат совпал с независимым training-runtime в пределах1e-5. Точная выполненная команда и погрешность — [infer_smoke.json](results/infer_smoke.json). Имена модели и результата явно экспериментальные; endpoint релизного сервиса не подменяется.","",
                  "```bash", "/home/artem/projects/hackathon-lct-vehicle-reid/.venv/bin/python \\",
                  "  04-solution/training/distill-20260925/infer.py \\",
                  "  --image /path/to/frame.jpg --bbox 10 20 200 100 \\",
                  "  --out ./student-vector.npy", "```", "",
                  "Путь/рамка в этом примере заменяются своими. Модель по умолчанию берётся из `artifacts/run/export/student_combined_v1.onnx`; после интеграции копируется весь `artifacts/run/`. Повторную запись существующего результата CLI отклоняет. Для собственного окружения нужны NumPy, Pillow и ONNX Runtime; версии проверенного окружения сохранены в export/benchmark manifests.","",
                  "## Воспроизведение обучения", "", "Команды на разрешённом worker-vm; `--repo` должен указывать на checkout/snapshot с SHA из protocol.json. Входные кэши уже находятся по проверяемым абсолютным путям manifest. Для другой машины сначала подготовить те же входы и явный новый path mapping; скрытого download нет. Новый запуск — только в **новый** OUT, существующие main.npz/training.json защищены от перезаписи.","",
                  "```bash", "cd ~/lct-reid/jobs/distill_20260925", "export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2",
                  "systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 python3 code/experiment.py train --repo repo --out fresh-run --smoke-only",
                  "# Export/check smoke locally with export_head.py before the training continuation.",
                  "systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 python3 code/experiment.py train --repo repo --out fresh-run",
                  "systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 python3 code/experiment.py evaluate --repo repo --out fresh-run",
                  "# Export main.npz + export_validation.npz using export_head.py --backbone ... --out fresh-run/export.",
                  "systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 python3 code/verify_export.py --repo repo --out fresh-run",
                  "systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 ../job_45/venv/bin/python code/benchmark.py --repo repo --out fresh-run", "```", "",
                  "Окружение обучения: Python3.14.3/torch2.9.1/NumPy2.4.6. Export: ONNX1.22.0/ORT1.30.0 в существующей локальной venv. Benchmark: существующая job45 venv, ORT1.30.0. Архивная learn_lw вызывается из её точного AST без запуска жёстко заданных исторических entry point. Нынешний service KR скопирован в изолированный snapshot, старый worker repo не изменён.", "",
                  "Worker clock отстаёт от workstation примерно на2.6ч. Связь результатов подтверждается protocol/source SHA и monotonic durations; mtime не используется как доказательство порядка. Код обучения: `53f7261`, baseline:`8e7a205`, frozen protocol:`e1557af`.","",
                  "Вывод ограничен данным рецептом и бюджетом: он не доказывает невозможности дистилляции вообще. Дополнительных обучений после отрицательной оценки не запускалось.","",
                  "Независимая приёмка: **PENDING**. Следующее действие — критик проверяет diff, входы, канонический baseline и отрицательный вывод; после приёмки интегрируются исходники и отчёт, default d1_j48 остаётся."])
    text="\n".join(lines)+"\n"
    # This exact note is assigned to this task; source is edited first.
    a.note.write_text(text)
    (HERE/"README.md").write_text(a.note.read_text())
    print(json.dumps(decision))


if __name__ == "__main__":main()
