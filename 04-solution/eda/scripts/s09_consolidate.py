#!/usr/bin/env python3
"""s09: сводка ключевых чисел отчёта из out_s01..s08 (единая точка воспроизводимости).
Запуск: python3 s09_consolidate.py -> out_s09_summary.json"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda n: json.load(open(os.path.join(HERE, n)))
s01, s02, s03 = L("out_s01.json"), L("out_s02_meta.json"), L("out_s03.json")
s04, s05, s05b = L("out_s04.json"), L("out_s05.json"), L("out_s05b.json")
s06, s06b, s06d = L("out_s06.json"), L("out_s06b.json"), L("out_s06d.json")
s07, s08 = L("out_s07.json"), L("out_s08.json")

sm = {
 "rows": {"train": s01["train.csv"]["n_rows"], "query": s01["test_query.csv"]["n_rows"],
          "gallery": s01["test_gallery.csv"]["n_rows"]},
 "files": s02["n_files"],
 "identities_train": s01["train_identities"]["n_identities"],
 "cameras_train": s01["train_cameras"]["n_cameras"],
 "resolutions": s02["resolutions"], "exif": s02["exif_tag_counts"],
 "qtables": list(s02["qhash_counts"].values()),
 "csv_file_mismatch": [s03["csv_ids_without_file"]["n"], s03["files_not_in_any_csv"]["n"]],
 "bbox_invalid": s03["bbox_out_total"],
 "byte_dups": {"groups": s02["byte_dup_md5_groups"],
               "pattern": s03["byte_dup_split_pattern"]},
 "neardup_0.90": s05["neardup_pair_counts"]["0.9"]["by_split_pair"],
 "queries_with_gallery_neardup_0.90": s05["queries_with_gallery_neardup"]["0.9"],
 "qg_same_vehicle_same_point": {
    "A_same_moment_queries": s05b["A_same_moment_pairs"]["n_queries_with_same_vehicle_like"],
    "B_parked_queries": s05b["B_parked_pairs_iou0.8_cos_lt_0.9"]["n_queries_with_same_vehicle_like"],
    "union_queries": s05b["queries_with_any_same_vehicle_same_place_candidate"],
    "of_queries": s01["test_query.csv"]["n_rows"]},
 "camera_surrogate": {
    "global_clustering_grad96_knn_t0.4": s06d["mutual_knn_grad96_train"]["t0.4"],
    "pairlevel_grad96_t0.4": s06d["pairlevel_grad96"]["0.4"],
    "targeted_instrument": s06d["instrument_full_estimate"],
    "same_vid_pairs_eval_base": s06d["same_vid_pairs"]},
 "pass_groups_ff0.90": s06["pass_groups_fullframe_0.90"],
 "camera_id_sanity": {
    "intra_scene_cos_mean": s04["intra_camera_cosine"]["mean_over_cameras"],
    "inter_scene_cos": s04["inter_camera_cosine_mean"],
    "cam90_intra": s04["intra_camera_cosine"]["cam90"],
    "nearest_centroid_acc": s04["nearest_centroid_camera_accuracy"]["2fold_mean"],
    "clusters_inside_cam90": s06["scene_clusters_inside_camera"]["90"],
    "clusters_inside_cam89": s06["scene_clusters_inside_camera"]["89"]},
 "plates": {"sampled": s07["n_sampled"],
            "grad_ratio_in_box": s07["grad_ratio_in_smooth_box"]["all"]},
 "complexity": {k: {"area_pct_p50": v["bbox_area_share_of_frame_pct"]["p50"],
                    "dark_lt_60": v["share_dark_lt_60"],
                    "touch_edge": v["share_touch_frame_edge"]} for k, v in s08.items()},
}
json.dump(sm, open(os.path.join(HERE, "out_s09_summary.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(sm, ensure_ascii=False, indent=1))
