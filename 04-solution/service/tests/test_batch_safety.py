"""All-or-nothing output, using only the standard library."""
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import batch_safety as safety

# The output contract is independent of the implementation under test.
EXPECTED_FILES = {'embeddings.npy', 'submission.csv', 'candidates.csv', 'run_info.json'}


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'out'

    def fill(self, directory, value):
        directory.mkdir(exist_ok=True)
        for name in EXPECTED_FILES:
            (directory / name).write_text(value)

    def assert_complete(self, value):
        self.assertEqual({p.name for p in self.out.iterdir()}, EXPECTED_FILES)
        self.assertTrue(all(p.read_text() == value for p in self.out.iterdir()))

    def test_success_is_invisible_until_all_files_are_ready(self):
        with safety.atomic_output(self.out) as stage:
            self.fill(stage, 'new')
            self.assertFalse(self.out.exists())
        self.assert_complete('new')
        self.assertEqual(list(self.root.iterdir()), [self.out])

    def test_replace_nonempty_result_in_one_exchange(self):
        self.fill(self.out, 'old')
        with safety.atomic_output(self.out) as stage:
            self.fill(stage, 'new')
            self.assert_complete('old')
        self.assert_complete('new')
        self.assertEqual(list(self.root.iterdir()), [self.out])

    def test_replace_precreated_empty_directory(self):
        self.out.mkdir()
        with safety.atomic_output(self.out) as stage:
            self.fill(stage, 'new')
        self.assert_complete('new')

    def test_exception_does_not_damage_old_result(self):
        self.fill(self.out, 'old')
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            with safety.atomic_output(self.out) as stage:
                (stage / 'embeddings.npy').write_text('partial')
                raise RuntimeError('inference failed')
        self.assert_complete('old')
        self.assertEqual(list(self.root.iterdir()), [self.out])

    def test_incomplete_set_is_not_published(self):
        with self.assertRaisesRegex(RuntimeError, 'Неполный'):
            with safety.atomic_output(self.out) as stage:
                (stage / 'embeddings.npy').write_text('partial')
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_rename_preserves_previous_complete_result(self):
        self.fill(self.out, 'old')
        with patch.object(safety, '_publish', side_effect=OSError('simulated EBUSY')):
            with self.assertRaisesRegex(SystemExit, 'Не удалось опубликовать'):
                with safety.atomic_output(self.out) as stage:
                    self.fill(stage, 'new')
        self.assert_complete('old')
        self.assertEqual(list(self.root.iterdir()), [self.out])

    def test_unrelated_files_are_not_removed(self):
        self.out.mkdir()
        (self.out / 'keep.txt').write_text('user data')
        with self.assertRaisesRegex(SystemExit, 'посторонние'):
            with safety.atomic_output(self.out):
                self.fail('must reject before yielding')
        self.assertEqual((self.out / 'keep.txt').read_text(), 'user data')

    def test_symlink_is_rejected_without_following_it(self):
        real = self.root / 'real'
        real.mkdir()
        self.out.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(SystemExit, 'symlink'):
            with safety.atomic_output(self.out):
                self.fail('must reject')
        self.assertEqual(list(real.iterdir()), [])

    def test_destination_is_rechecked_before_publication(self):
        self.fill(self.out, 'old')
        with self.assertRaisesRegex(SystemExit, 'посторонние'):
            with safety.atomic_output(self.out) as stage:
                self.fill(stage, 'new')
                (self.out / 'notes.txt').write_text('concurrent user data')
        self.assertEqual((self.out / 'notes.txt').read_text(), 'concurrent user data')
        for name in EXPECTED_FILES:
            self.assertEqual((self.out / name).read_text(), 'old')
        self.assertEqual(list(self.root.glob('.out.incomplete-*')), [])

    def test_symlink_inside_previous_result_is_rejected(self):
        self.fill(self.out, 'old')
        target = self.root / 'user-data'
        target.write_text('keep me')
        linked = self.out / 'embeddings.npy'
        linked.unlink()
        linked.symlink_to(target)
        with self.assertRaisesRegex(SystemExit, 'посторонние'):
            with safety.atomic_output(self.out):
                self.fail('must reject before writing a new result')
        self.assertTrue(linked.is_symlink())
        self.assertEqual(target.read_text(), 'keep me')
        for name in EXPECTED_FILES - {'embeddings.npy'}:
            self.assertEqual((self.out / name).read_text(), 'old')
        self.assertEqual(list(self.root.glob('.out.incomplete-*')), [])

    def test_candidates_are_required_even_when_other_three_files_exist(self):
        with self.assertRaisesRegex(RuntimeError, 'Неполный'):
            with safety.atomic_output(self.out) as stage:
                for name in EXPECTED_FILES - {'candidates.csv'}:
                    (stage / name).write_text('new')
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob('.out.incomplete-*')), [])

    def test_sigkill_during_publication_keeps_a_complete_generation(self):
        # A child stops at the first filesystem rename, or just after the
        # exchange. The parent inspects the public path and sends real SIGKILL.
        # A remove-old / move-files-one-by-one implementation exposes an empty
        # or mixed generation at this boundary and must fail this test.
        code = '''import os, signal, sys
from pathlib import Path
from app import batch_safety as safety
out = Path(sys.argv[1])
names = ('embeddings.npy', 'submission.csv', 'candidates.csv', 'run_info.json')
armed = False
def stop():
    print('publication boundary', flush=True)
    os.kill(os.getpid(), signal.SIGSTOP)
def audit(event, args):
    if armed and event == 'os.rename':
        stop()
sys.addaudithook(audit)
publish = safety._publish
def observed_publish(stage, destination):
    global armed
    armed = True
    publish(stage, destination)
    stop()
safety._publish = observed_publish
with safety.atomic_output(out) as stage:
    for name in names:
        (stage / name).write_text('new')
'''
        for previous in [False, True]:
            with self.subTest(previous=previous):
                out = self.root / ('existing' if previous else 'new')
                if previous:
                    self.fill(out, 'old')
                child = subprocess.Popen(
                    [sys.executable, '-B', '-c', code, str(out)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    env={**os.environ, 'PYTHONPATH': str(Path(safety.__file__).parents[1])},
                )
                try:
                    self.assertTrue(select.select([child.stdout], [], [], 10)[0],
                                    'child did not reach publication')
                    self.assertEqual(child.stdout.readline().strip(), 'publication boundary')
                    child.kill()
                    child.communicate(timeout=10)
                    self.assertEqual(child.returncode, -signal.SIGKILL)
                    if out.exists():
                        self.assertEqual({p.name for p in out.iterdir()}, EXPECTED_FILES)
                        values = {p.read_text() for p in out.iterdir()}
                        self.assertIn(values, [{'old'}, {'new'}] if previous else [{'new'}])
                    else:
                        self.assertFalse(previous, 'previous complete result disappeared')
                finally:
                    if child.poll() is None:
                        child.kill()
                    child.communicate(timeout=10)

    def test_abrupt_death_leaves_only_explicitly_incomplete_staging(self):
        for previous in [False, True]:
            with self.subTest(previous=previous):
                if previous:
                    self.fill(self.out, 'old')
                code = '''import os,sys
from pathlib import Path
from app.batch_safety import atomic_output
with atomic_output(Path(sys.argv[1])) as stage:
    (stage/'embeddings.npy').write_text('partial')
    os._exit(137)
'''
                result = subprocess.run([sys.executable, '-B', '-c', code, str(self.out)],
                                        env={**os.environ, 'PYTHONPATH': str(Path(safety.__file__).parents[1])})
                self.assertEqual(result.returncode, 137)
                if previous:
                    self.assert_complete('old')
                else:
                    self.assertFalse(self.out.exists())
                self.assertTrue(list(self.root.glob('.out.incomplete-*')))


if __name__ == '__main__':
    unittest.main()
