"""Actual ENOSPC, EBUSY and EXDEV in a private mount namespace (<8 MiB)."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'snapshots/after/04-solution/service'))
from app import batch_safety as safety

FILES = ('embeddings.npy', 'submission.csv', 'candidates.csv', 'run_info.json')
BASE = ROOT / 'filesystem-cases'
BASE.mkdir(exist_ok=False)
mnt = BASE / 'tmpfs'
mnt.mkdir()
subprocess.run(['mount', '-t', 'tmpfs', '-o', 'size=2m,mode=1777', 'job81-enospc', str(mnt)], check=True)


def fill(path, value):
    path.mkdir(exist_ok=True)
    for name in FILES:
        (path / name).write_text(value)


def contents(path):
    return {p.name: p.read_text() for p in sorted(path.iterdir())} if path.exists() else None


records = []
for previous in [False, True]:
    out = mnt / f'out-{previous}'
    if previous:
        fill(out, 'old')
    before = contents(out)
    error = None
    try:
        with safety.atomic_output(out) as stage:
            with (stage / 'embeddings.npy').open('wb') as stream:
                for _ in range(64):
                    stream.write(b'x' * 65536)
    except OSError as exc:
        error = {'type': type(exc).__name__, 'errno': exc.errno, 'message': str(exc)}
    assert error and error['errno'] == errno.ENOSPC
    record = {'case': f'ENOSPC_previous_{previous}', 'before': before, 'final': contents(out),
              'error': error, 'staging': [p.name for p in mnt.glob(f'.{out.name}.incomplete-*')]}
    assert record['final'] == before and not record['staging']
    record['passed'] = True
    records.append(record)

out = BASE / 'bind-output'
out.mkdir()
mount_source = mnt / 'bound-source'
fill(mount_source, 'old')
subprocess.run(['mount', '--bind', str(mount_source), str(out)], check=True)
before = contents(out)
error = None
try:
    with safety.atomic_output(out) as stage:
        fill(stage, 'new')
except SystemExit as exc:
    error = {'type': type(exc).__name__, 'message': str(exc),
             'cause_errno': getattr(exc.__cause__, 'errno', None)}
assert error and error['cause_errno'] == errno.EBUSY
assert contents(out) == before
records.append({'case': 'bind_output_EBUSY', 'error': error, 'before': before,
                'final': contents(out), 'passed': True, 'limitation': 'publish a child of the mounted directory'})

out = BASE / 'forced-cross-filesystem'
fill(out, 'old')
stage = mnt / 'forced-stage'
stage.mkdir()
before = contents(out)
error = None
try:
    with patch.object(safety.tempfile, 'mkdtemp', return_value=str(stage)):
        with safety.atomic_output(out) as actual:
            fill(actual, 'new')
except SystemExit as exc:
    error = {'type': type(exc).__name__, 'message': str(exc),
             'cause_errno': getattr(exc.__cause__, 'errno', None)}
assert error and error['cause_errno'] == errno.EXDEV
assert contents(out) == before and not stage.exists()
records.append({'case': 'forced_cross_filesystem_EXDEV', 'error': error,
                'injection': 'mkdtemp location only; actual renameat2 across devices',
                'before': before, 'final': contents(out), 'passed': True})
(ROOT / 'evidence/filesystem.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records, indent=2))
