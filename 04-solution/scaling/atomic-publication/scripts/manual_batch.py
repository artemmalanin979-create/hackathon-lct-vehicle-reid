"""Eight real SIGKILL experiments through app.batch with two actual ONNX models.

Only I/O observation, pause points and reporting clocks are instrumented. The
image reader, models, preprocessing, ranking and serializers are production code.
"""
import argparse
import builtins
import csv
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/work')
BASE = ROOT / 'manual-batch'
FILES = ('embeddings.npy', 'submission.csv', 'candidates.csv', 'run_info.json')


def hashes(path):
    if path.is_symlink():
        return {'symlink': os.readlink(path)}
    if not path.exists():
        return None
    return {p.name: {'size': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(path.iterdir()) if p.is_file()}


def child(phase, out, old=False):
    from app import batch, batch_safety as safety

    def pause():
        print('READY:' + phase, flush=True)
        signal.pause()

    model = batch.Embedder

    def observe_model(*args, **kwargs):
        print('EMBEDDER_CREATED', flush=True)
        return model(*args, **kwargs)

    batch.Embedder = observe_model
    clock = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    batch.time = SimpleNamespace(perf_counter=lambda: next(clock))
    save = batch.save_embeddings

    def save_then_pause(*args, **kwargs):
        result = save(*args, **kwargs)
        if phase == 'after_first':
            pause()
        return result

    batch.save_embeddings = save_then_pause
    publish = safety._publish

    def observed_publish(stage, destination):
        if phase == 'before_publish':
            pause()
        publish(stage, destination)
        if phase == 'after_publish':
            pause()

    safety._publish = observed_publish
    original_open = builtins.open

    class PartialWriter:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, data):
            n = self.stream.write(data[:max(1, len(data) // 2)])
            self.stream.flush()
            os.fsync(self.stream.fileno())
            pause()  # the file is still open, with only half its first write
            return n

    def observed_open(file, mode='r', *args, **kwargs):
        stream = original_open(file, mode, *args, **kwargs)
        if phase == 'mid_write' and mode == 'wb' and str(file).endswith('/embeddings.npy'):
            return PartialWriter(stream)
        return stream

    builtins.open = observed_open
    prefix = 'old' if old else 'new'
    sys.argv = ['app.batch', '--images-dir', '/data/images',
                '--query', str(BASE / f'{prefix}-query.csv'),
                '--gallery', str(BASE / f'{prefix}-gallery.csv'),
                '--out-dir', str(out), '--threads', '1', '--batch', '32']
    batch.main()


def command(phase, out, old=False):
    args = [sys.executable, '-B', '-u', __file__, '--child', phase, '--out', str(out)]
    return args + (['--old'] if old else [])


def main():
    BASE.mkdir(exist_ok=False)
    split = Path('/snapshot/04-solution/split/files')
    for name, nq, ng in [('old', 2, 1), ('new', 1, 2)]:
        for kind, count in [('query', nq), ('gallery', ng)]:
            with (split / f'val_{kind}.csv').open() as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)[:count]
            with (BASE / f'{name}-{kind}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    expected = {}
    for name in ['old', 'new']:
        out = BASE / f'control-{name}'
        result = subprocess.run(command('normal', out, old=name == 'old'), capture_output=True, text=True, timeout=90)
        (ROOT / 'logs' / f'manual-control-{name}.txt').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr
        expected[name] = hashes(out)
        assert set(expected[name]) == set(FILES)
    assert expected['old'] != expected['new']
    records = []
    for phase in ['mid_write', 'after_first', 'before_publish', 'after_publish']:
        for previous in [False, True]:
            case = f'{phase}-previous-{previous}'
            root = BASE / case
            root.mkdir()
            out = root / 'out'
            if previous:
                shutil.copytree(BASE / 'control-old', out)
            proc = subprocess.Popen(command(phase, out), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, bufsize=1)
            lines = []
            try:
                deadline = time.monotonic() + 90
                while True:
                    remaining = deadline - time.monotonic()
                    assert remaining > 0 and select.select([proc.stdout], [], [], remaining)[0], case
                    line = proc.stdout.readline()
                    assert line, (case, proc.poll(), proc.stderr.read())
                    lines.append(line)
                    if line.strip() == 'READY:' + phase:
                        break
                os.kill(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate(timeout=10)
                final = hashes(out)
                wanted = expected['new'] if phase == 'after_publish' else (expected['old'] if previous else None)
                record = {'case': case, 'pid': proc.pid, 'returncode': proc.returncode,
                          'final': final, 'expected_final': wanted,
                          'staging': {p.name: hashes(p) for p in root.glob('.out.incomplete-*')},
                          'passed': proc.returncode == -signal.SIGKILL and final == wanted}
                (ROOT / 'logs' / f'manual-{case}.txt').write_text(''.join(lines) + stdout + stderr)
                records.append(record)
                assert record['passed'], record
                print(json.dumps({'case': case, 'passed': True}), flush=True)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=10)

    for kind in ['new', 'empty', 'existing', 'foreign', 'destination_symlink', 'file_symlink']:
        root = BASE / kind
        root.mkdir()
        out = root / 'out'
        if kind in ['existing', 'foreign', 'file_symlink']:
            shutil.copytree(BASE / 'control-old', out)
        if kind == 'empty':
            out.mkdir()
        if kind == 'foreign':
            (out / 'notes.txt').write_text('user data must survive')
        if kind == 'destination_symlink':
            shutil.copytree(BASE / 'control-old', root / 'target')
            out.symlink_to(root / 'target', target_is_directory=True)
        if kind == 'file_symlink':
            (root / 'target').write_text('user data must survive')
            (out / 'embeddings.npy').unlink()
            (out / 'embeddings.npy').symlink_to(root / 'target')
        before = hashes(out)
        target = root / 'target'
        target_before = (hashes(target) if target.is_dir() else target.read_text()) if target.exists() else None
        result = subprocess.run(command('normal', out), capture_output=True, text=True, timeout=90)
        refused = kind in ['foreign', 'destination_symlink', 'file_symlink']
        final = hashes(out)
        passed = (result.returncode != 0 and 'EMBEDDER_CREATED' not in result.stdout and final == before
                  if refused else result.returncode == 0 and final == expected['new'])
        if target_before is not None:
            assert (hashes(target) if target.is_dir() else target.read_text()) == target_before
        record = {'case': kind, 'returncode': result.returncode, 'before': before, 'final': final,
                  'rejected_before_model': refused and 'EMBEDDER_CREATED' not in result.stdout,
                  'target_preserved': target_before is not None,
                  'staging': [p.name for p in root.glob('.out.incomplete-*')], 'passed': passed}
        (ROOT / 'logs' / f'manual-{kind}.txt').write_text(result.stdout + result.stderr)
        records.append(record)
        assert passed and not record['staging'], record
        print(json.dumps({'case': kind, 'passed': passed}), flush=True)
    # /limited is a private, size-limited tmpfs provided by the container.
    # Retain only one page free so the genuine numpy serializer runs out of
    # space after opening the first output. No fabricated OSError is involved.
    limited = Path('/limited')
    for previous in [False, True]:
        root = limited / str(previous)
        root.mkdir()
        out = root / 'out'
        if previous:
            shutil.copytree(BASE / 'control-old', out)
        before = hashes(out)
        st = os.statvfs(limited)
        ballast = root / 'ballast'
        ballast.write_bytes(b'x' * (st.f_bavail * st.f_frsize - st.f_frsize))
        result = subprocess.run(command('normal', out), capture_output=True, text=True, timeout=90)
        final = hashes(out)
        staging = list(root.glob('.out.incomplete-*'))
        record = {'case': f'app_batch_ENOSPC_previous_{previous}', 'returncode': result.returncode,
                  'filesystem': 'private 2 MiB tmpfs, one page free before batch',
                  'before': before, 'final': final, 'staging': [p.name for p in staging],
                  'stderr': result.stderr,
                  'passed': result.returncode != 0 and final == before and not staging}
        records.append(record)
        (ROOT / 'logs' / f'manual-enospc-{previous}.txt').write_text(result.stdout + result.stderr)
        assert record['passed'], record
        # This ballast belongs only to this experiment; release it for the
        # second case. Nothing outside this private mount is removed.
        ballast.unlink()
        print(json.dumps({'case': record['case'], 'passed': True}), flush=True)
    (ROOT / 'evidence/manual-batch.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--old', action='store_true')
    args = parser.parse_args()
    if args.child:
        child(args.child, args.out, args.old)
    else:
        main()
