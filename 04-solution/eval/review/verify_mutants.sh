#!/bin/bash
for m in M1_keys_filter_mask M2_pr_thresholds_sign M3_top_score_sign M4_eligible_count M5_relevant_pairs M6_positive_ranks_0based M7_num_relevant M8_float32 M9_accepted_strict; do
  rm -rf vtmp; cp -r run vtmp; rm -rf vtmp/mutants vtmp/*.log vtmp/*.json 2>/dev/null
  cp sources_manifest_backup 2>/dev/null vtmp/ 2>/dev/null
  cp "mutant_${m}.py" vtmp/reid_metrics.py
  out=$(cd vtmp && timeout 400 python3 -B verify.py 2>&1)
  code=$?
  if [ $code -eq 0 ]; then
    echo "$m: verify.py PASSED ENTIRELY"
  else
    stage=$(echo "$out" | grep -oE 'in (differential_ranking|differential_refusal|random_level|mutation_check|compatibility_edges|cli_and_comparator|main)' | tail -1)
    lastline=$(echo "$out" | tail -1)
    echo "$m: verify.py FAILED ($stage) :: $lastline"
  fi
done
rm -rf vtmp
