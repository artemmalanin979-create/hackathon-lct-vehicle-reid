#!/usr/bin/env python3
"""Прекэш миниатюр (train + test) и сверка рецепта «серий» с EDA:
на выданном тесте должно получиться 953 серии в query и 748 в gallery."""
import json
import os

import common as C


def main():
    train = C.load_train()
    train_ids = [r["image_id"] for r in train]
    q_ids = [r["image_id"] for r in C.read_csv("test_query.csv")]
    g_ids = [r["image_id"] for r in C.read_csv("test_gallery.csv")]

    Tt = C.thumbs_for(train_ids, "thumbs_train")
    Tq = C.thumbs_for(q_ids, "thumbs_test_query")
    Tg = C.thumbs_for(g_ids, "thumbs_test_gallery")

    res = {
        "test_query": C.series_stats(C.normed(Tq)),
        "test_gallery": C.series_stats(C.normed(Tg)),
        "expected_from_eda": {"test_query_series": 953, "test_gallery_series": 748},
    }
    res["recipe_reproduced"] = (res["test_query"]["series"] == 953
                                and res["test_gallery"]["series"] == 748)
    with open(os.path.join(C.OUT, "s00_recipe_check.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    assert Tt.shape == (9556, 1024)


if __name__ == "__main__":
    main()
