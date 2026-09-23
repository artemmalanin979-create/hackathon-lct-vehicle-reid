"""Replay the critic's seven publication mutations in isolated local copies."""
import argparse
import difflib
import json
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--work-dir', type=Path, required=True, help='fresh evidence and mutant directory')
args = parser.parse_args()
ROOT = args.work_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / 'evidence').mkdir(exist_ok=True)
REPO = args.repo.resolve()
SERVICE = REPO / '04-solution/service'
source = (SERVICE / 'app/batch_safety.py').read_text()
current_tests = (SERVICE / 'tests/test_batch_safety.py').read_text()
old_tests = subprocess.check_output(
    ['git', '-C', str(REPO), 'show', 'guard/memory:04-solution/service/tests/test_batch_safety.py'], text=True)
old_tests = old_tests[:old_tests.index('class MemoryTests')] + old_tests[old_tests.index('class PublicationTests'):]
critic = json.loads((PACKAGE / 'evidence/critic-publication-mutations.json').read_text())
start = source.index('def _publish(')
end = source.index('\n\n@contextmanager', start)
mutations = [(row['id'], row['description'], row.get('old', source[start:end]), row['new'])
             for row in critic['mutations']]
new_tests = {
    'M14': 'test_destination_is_rechecked_before_publication',
    'M15': 'test_symlink_inside_previous_result_is_rejected',
    'M16': 'test_candidates_are_required_even_when_other_three_files_exist',
    'M18': 'test_sigkill_during_publication_keeps_a_complete_generation',
}
runner = '''import importlib.util,json,sys,unittest
spec=importlib.util.spec_from_file_location('test_batch_safety',sys.argv[1])
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
suite=(unittest.defaultTestLoader.loadTestsFromName('PublicationTests.'+sys.argv[2],mod)
       if len(sys.argv)>2 else unittest.defaultTestLoader.loadTestsFromModule(mod))
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('RESULT_JSON='+json.dumps({'tests':result.testsRun,'failed':[str(t) for t,_ in result.failures],
 'errors':[str(t) for t,_ in result.errors],'success':result.wasSuccessful()}))
sys.exit(0 if result.wasSuccessful() else 1)
'''
records = []
for key, description, old, new in mutations:
    assert source.count(old) == 1, (key, source.count(old))
    mutated = source.replace(old, new)
    record = {'id': key, 'description': description, 'old': old, 'new': new}
    cases = [('original', old_tests, None), ('extended', current_tests, None)]
    if key[:3] in new_tests:
        cases.append(('dedicated', current_tests, new_tests[key[:3]]))
    for label, tests, selected in cases:
        directory = ROOT / 'mutations' / key / label
        (directory / 'app').mkdir(parents=True, exist_ok=False)
        (directory / 'tests').mkdir()
        (directory / 'app/__init__.py').write_text('')
        (directory / 'app/batch_safety.py').write_text(mutated)
        test_path = directory / 'tests/test_batch_safety.py'
        test_path.write_text(tests)
        command = [sys.executable, '-B', '-c', runner, str(test_path)]
        if selected:
            command.append(selected)
        cp = subprocess.run(command, capture_output=True, text=True, timeout=40)
        log = ROOT / 'evidence' / 'mutations' / f'{key}-{label}.txt'
        log.parent.mkdir(exist_ok=True)
        # Normalize terminal trailing spaces in the checked-in transcript.
        log.write_text('\n'.join(line.rstrip() for line in (cp.stdout + cp.stderr).splitlines()) + '\n')
        result_line = next(line for line in cp.stdout.splitlines() if line.startswith('RESULT_JSON='))
        record[label] = {**json.loads(result_line.split('=', 1)[1]), 'returncode': cp.returncode,
                         'log': str(log.relative_to(ROOT))}
    record['new_test'] = new_tests.get(key[:3])
    assert not record['extended']['success'], record
    if record['new_test']:
        assert record['original']['success'], record
        assert record['dedicated']['failed'] and not record['dedicated']['errors'], record
    patch_path = ROOT / 'evidence' / 'mutations' / f'{key}.patch'
    patch_path.write_text(''.join(difflib.unified_diff(source.splitlines(True), mutated.splitlines(True),
                                                    fromfile='original', tofile=key, n=0)))
    records.append(record)
    print(key, 'KILLED', 'dedicated=' + str(record['new_test']), flush=True)
(ROOT / 'evidence/mutations.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
