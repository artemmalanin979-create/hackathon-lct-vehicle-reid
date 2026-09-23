"""Compare all four untouched output files and retain raw SHA-256 evidence."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True, help='validation workspace with artifacts/ and evidence/')
args = parser.parse_args()
root = args.root.resolve()
files = ('embeddings.npy', 'submission.csv', 'candidates.csv', 'run_info.json')
records = []
for name in files:
    before = (root / 'artifacts/j81-full-before' / name).read_bytes()
    after = (root / 'artifacts/j81-full-after' / name).read_bytes()
    record = {'file': name, 'before_bytes': len(before), 'after_bytes': len(after),
              'before_sha256': hashlib.sha256(before).hexdigest(),
              'after_sha256': hashlib.sha256(after).hexdigest(), 'byte_equal': before == after}
    records.append(record)
    assert record['byte_equal'], record
results = {side: json.loads((root / 'evidence' / f'j81-full-{side}.validation.json').read_text())
           for side in ('before', 'after')}
for side, result in results.items():
    assert result['metrics']['mAP'] == 0.7740915539438481, result
    assert result['metrics']['Rank-1'] == 0.7307692307692307, result
    for record in records:
        assert result['files'][record['file']]['sha256'] == record[f'{side}_sha256']
assert results['before']['reported_clock'] == results['after']['reported_clock']
summary = {'all_four_byte_equal': True, 'files': records,
           'metrics': {s: r['metrics'] for s, r in results.items()},
           'reporting_clock': 'baseline actual readings replayed in validation wrapper only',
           'production_clock_unchanged': True,
           'actual_elapsed_s': {s: r['actual_elapsed_s'] for s, r in results.items()},
           'wall_s': {s: r['batch_wall_s'] for s, r in results.items()},
           'rss_peak_bytes': {s: r['rss_peak_bytes'] for s, r in results.items()}}
(root / 'evidence/bitwise-comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
