"""Sequential full before/after runs on one worker CPU, with private cgroups."""
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[1]
for directory in ('logs', 'evidence', 'artifacts'):
    (root / directory).mkdir(exist_ok=True)
(root / 'evidence' / 'validation.started').open('x').close()

for source in ('before', 'after'):
    name = f'j81-full-{source}'
    cmd = ['sudo', '-n', 'podman', 'run', '--name', name, '--network', 'none',
           '--cpuset-cpus', '0', '--cpus', '1', '--memory', '4g', '--memory-swap', '4g',
           '--pids-limit', '128', '--user', '1000:1000', '--cgroupns', 'private',
           '--security-opt', 'label=disable', '--workdir', '/work',
           '-v', f'{root}:/work:rw', '-v', f'{root}/snapshots/{source}:/snapshot:ro',
           '-v', '/home/fedora/lct-reid/jobs/job_73-OQZJggyo/data-val-only:/data:ro',
           '-e', 'PYTHONPATH=/snapshot/04-solution/service', '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
           '-e', 'MODEL_PATH=/snapshot/04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx',
           '-e', 'MODEL2_PATH=/snapshot/04-solution/service/model/osnet_ain_combined_v1.onnx',
           '-e', 'WHITENING_PATH=/snapshot/04-solution/service/model/lw_ens_j48_rho0.5.npz',
           'localhost/job72-jury:6179b69', 'python', '-B', '-u', '/work/scripts/full_validation.py', name]
    if source == 'after':
        cmd += ['--replay', 'j81-full-before.validation.json']
    start = time.monotonic()
    with (root / 'logs' / f'{name}.txt').open('w') as log:
        cp = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    inspect = subprocess.run(['sudo', '-n', 'podman', 'inspect', name], capture_output=True, text=True)
    result = {'command': cmd, 'returncode': cp.returncode, 'wall_s': time.monotonic() - start,
              'inspect': json.loads(inspect.stdout) if inspect.returncode == 0 else inspect.stderr}
    (root / 'evidence' / f'{name}.controller.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'name': name, 'returncode': cp.returncode, 'wall_s': result['wall_s']}), flush=True)
    if cp.returncode:
        raise SystemExit(cp.returncode)
