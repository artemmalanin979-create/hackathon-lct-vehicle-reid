#!/usr/bin/env python3
"""Пакетный прогон: произвольный тестовый набор -> три сдаваемых файла.

Одна команда, все пути — аргументы, ничего не зашито. Хранилище и сеть не
нужны: прогон читает изображения и CSV с диска и пишет в указанный каталог:

    python -m app.batch \
        --images-dir /data/images \
        --query /data/test_query.csv \
        --gallery /data/test_gallery.csv \
        --out-dir /out

Выход: submission.csv, embeddings.npy, candidates.csv (схемы — README датасета)
и run_info.json со счётчиками и версиями (для протокола, к сдаче не требуется).

По умолчанию кандидаты упорядочиваются переранжированием (k-reciprocal); флаг
--no-rerank возвращает прежнее упорядочивание по косинусу. `embeddings.npy` в
обоих режимах один и тот же: переранжирование работает после извлечения векторов
и на них не влияет.

Зависимости — только numpy/pillow/onnxruntime; FastAPI и Qdrant не импортируются.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from .core import config
from .input_checks import fail_inputs, require_dataset, require_files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="каталог изображений (плоский, JPEG/PNG)")
    ap.add_argument("--query", type=Path, required=True,
                    help="CSV запросов: image_id,x,y,w,h")
    ap.add_argument("--gallery", type=Path, required=True,
                    help="CSV галереи: image_id,x,y,w,h")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="каталог для сдаваемых файлов (создаётся)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="порог режима отказа (по умолчанию — обоснованный в README, "
                         "свой для каждой шкалы)")
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=config.RERANK_DEFAULT,
                    help="переранжирование кандидатов (по умолчанию включено)")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false",
                    help="прежнее упорядочивание по косинусу, порог на шкале косинуса")
    ap.add_argument("--batch", type=int, default=32, help="размер батча инференса")
    ap.add_argument("--threads", type=int, default=0,
                    help="intra-op потоки onnxruntime (0 = по умолчанию)")
    args = ap.parse_args()
    require_dataset([args.query, args.gallery], args.images_dir)
    require_files([config.MODEL_PATH, config.MODEL2_PATH, config.WHITENING_PATH],
                  hint="Восстановите веса из Git; см. service/model/fetch_model.sh.")

    import numpy as np
    from .core.model import Embedder
    from .core.preprocess import CropInputError, crop_problems, read_rows
    from .core.ranking import cosine_scores, validate_scores
    from .core.rerank import rerank_scores
    from .core.submission import save_embeddings, write_candidates, write_submission

    if args.threshold is None:
        args.threshold = (config.DEFAULT_THRESHOLD_RERANK if args.rerank
                          else config.DEFAULT_THRESHOLD)
    try:
        validate_scores([], args.threshold)
    except ValueError as exc:
        ap.error(str(exc))

    t0 = time.perf_counter()
    q_rows = read_rows(args.query)
    g_rows = read_rows(args.gallery)
    if not q_rows or not g_rows:
        raise SystemExit("пустой query или gallery CSV — прогон не имеет смысла")

    problems = (crop_problems(args.images_dir, q_rows, args.query)
                + crop_problems(args.images_dir, g_rows, args.gallery))
    if problems:
        fail_inputs(problems, "Исправьте изображения/bbox и повторите запуск; "
                    "частичный комплект не создаётся, строки не пропускаются.")

    embedder = Embedder(threads=args.threads)  # sha256 весов проверяется здесь
    try:
        q_emb = embedder.embed_rows(args.images_dir, q_rows, args.batch)
        g_emb = embedder.embed_rows(args.images_dir, g_rows, args.batch)
    except CropInputError as exc:
        # Например, файл был изменён между проверкой и чтением. Не подменять
        # его кропом и не выдавать частичный комплект за успешный прогон.
        print(f"Ошибка входного кадра после проверки: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    t_embed = time.perf_counter() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Сначала векторы: они не зависят от способа упорядочивания кандидатов.
    emb = save_embeddings(args.out_dir / "embeddings.npy", q_emb, g_emb)

    t_rank = time.perf_counter()
    if args.rerank:
        scores = rerank_scores(q_emb, g_emb, config.RERANK_K1, config.RERANK_K2,
                               config.RERANK_LAMBDA)
    else:
        scores = cosine_scores(q_emb, g_emb)
    t_rank = time.perf_counter() - t_rank
    q_ids = [r.image_id for r in q_rows]
    g_ids = [r.image_id for r in g_rows]
    write_submission(args.out_dir / "submission.csv", q_ids, g_ids, scores)
    counts = write_candidates(args.out_dir / "candidates.csv", q_ids, g_ids,
                              scores, args.threshold)

    info = {
        **counts,
        "rerank": bool(args.rerank),
        "score_scale": "rerank_confidence_1_minus_distance" if args.rerank else "cosine",
        "rerank_params": ([config.RERANK_K1, config.RERANK_K2, config.RERANK_LAMBDA]
                          if args.rerank else None),
        "queries": len(q_rows),
        "gallery": len(g_rows),
        "embeddings_shape": [int(x) for x in emb.shape],
        "embed_elapsed_s": round(t_embed, 3),
        "rank_elapsed_s": round(t_rank, 3),
        "total_elapsed_s": round(time.perf_counter() - t0, 3),
        "model": config.MODEL_NAME,
        "model_sha256": config.MODEL_SHA256,
        "model2": config.MODEL2_NAME,
        "model2_sha256": config.MODEL2_SHA256,
        "whitening_sha256": config.WHITENING_SHA256,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": __import__("onnxruntime").__version__,
        },
    }
    (args.out_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
