"""Hash the 1860 original images, CSVs and both source snapshots on the worker."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
expected = json.loads((root / 'evidence/input-expectations.json').read_text())
data = Path('/home/fedora/lct-reid/jobs/job_73-OQZJggyo/data-val-only')


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


checked = []
for entry in expected['images']:
    path = data / entry['path']
    actual = {'path': entry['path'], 'bytes': path.stat().st_size, 'sha256': checksum(path)}
    assert actual == entry, actual
    checked.append(actual)
sources = {}
for version in ['before', 'after']:
    snapshot = root / 'snapshots' / version
    for entry in expected['repo_files']:
        path = snapshot / entry['path']
        assert path.stat().st_size == entry['bytes'] and checksum(path) == entry['sha256']
    sources[version] = {str(path.relative_to(snapshot)): checksum(path)
                        for path in sorted(snapshot.rglob('*')) if path.is_file()}
protected = [p for p in sources['before'] if '/app/core/' in p or '/model/' in p or '/split/' in p or '/eval/' in p]
assert all(sources['before'][p] == sources['after'][p] for p in protected)
result = {'image_count': len(checked), 'image_bytes': sum(x['bytes'] for x in checked),
          'all_image_hashes_match_manifest': True, 'all_csv_hashes_match_manifest': True,
          'unchanged_protected_files': len(protected), 'snapshots': sources, 'images': checked}
(root / 'evidence/input-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ['snapshots', 'images']}))
