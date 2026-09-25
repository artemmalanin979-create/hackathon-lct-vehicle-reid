#!/usr/bin/env python3
"""Verify every file in a locally assembled release; no network or mutations."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / 'release.json').read_text())
    failures = []
    for name, spec in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(name)
        elif path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            failures.append(name)
    expected = set(manifest['files']) | {'release.json'}
    unexpected = sorted(str(path.relative_to(root)) for path in root.rglob('*')
                        if path.is_file() and str(path.relative_to(root)) not in expected)
    failures.extend(unexpected)
    print(json.dumps({'status': 'FAIL' if failures else 'PASS', 'source_sha': manifest['source_sha'],
                      'checked': len(manifest['files']), 'failures': failures}, indent=2))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
