"""Real ONNX inference; record/replay batch reporting clocks, retain actual time."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np

from app import batch
from app.core import config

sys.path.insert(0, '/snapshot/04-solution/eval')
from reid_metrics import evaluate

p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--replay')
a = p.parse_args()
root = Path('/work')
split = Path('/snapshot/04-solution/split/files')
out = root / 'artifacts' / a.name
clock = []
replay = iter(json.loads((root / 'evidence' / a.replay).read_text())['reported_clock']) if a.replay else None


def reporting_clock():
    actual = time.perf_counter()
    reported = next(replay) if replay is not None else actual
    clock.append({'actual': actual, 'reported': reported})
    return reported


# Replace the module reference only. ONNX, numpy and other modules keep their
# clocks. Production sources are never changed by the validation harness.
batch.time = SimpleNamespace(perf_counter=reporting_clock)
sys.argv = ['app.batch', '--images-dir', '/data/images',
            '--query', str(split / 'val_query.csv'),
            '--gallery', str(split / 'val_gallery.csv'),
            '--out-dir', str(out), '--threads', '1', '--batch', '32']
start = time.monotonic()
print(json.dumps({'stage': 'start', 'name': a.name, 'replay': a.replay}), flush=True)
batch.main()
batch_wall = time.monotonic() - start
assert len(clock) == 5, clock
if replay is not None:
    assert next(replay, None) is None, 'unused reporting clock values'

qm = list(csv.DictReader((split / 'val_query.csv').open()))
gm = list(csv.DictReader((split / 'val_gallery.csv').open()))
assert (len(qm), len(gm)) == (1110, 750)
vectors = np.load(out / 'embeddings.npy', allow_pickle=False)
assert vectors.shape == (1860, 512) and vectors.dtype == np.float32
scores = batch.rerank_scores(vectors[:1110], vectors[1110:],
                             config.RERANK_K1, config.RERANK_K2, config.RERANK_LAMBDA)
metrics = evaluate(
    scores, query_ids=[r['vehicle_id'] for r in qm], gallery_ids=[r['vehicle_id'] for r in gm],
    query_cameras=[r['camera_id'] for r in qm], gallery_cameras=[r['camera_id'] for r in gm],
    known_absent=np.array([r['has_mate'] == '0' for r in qm]), camera_policy='market',
    refusal_mode='presence', threshold=config.DEFAULT_THRESHOLD_RERANK)['ranking_full_gallery']
assert metrics['mAP'] == 0.7740915539438481, metrics
assert metrics['Rank-1'] == 0.7307692307692307, metrics
actual = [v['actual'] for v in clock]
result = {
    'name': a.name, 'batch_wall_s': batch_wall, 'wall_with_metrics_s': time.monotonic() - start,
    'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    'reported_clock': [v['reported'] for v in clock], 'actual_clock': actual,
    'reporting_clock_replayed_from': a.replay,
    'actual_elapsed_s': {'embed_elapsed_s': actual[1] - actual[0],
                         'rank_elapsed_s': actual[3] - actual[2],
                         'total_elapsed_s': actual[4] - actual[0]},
    'metrics': metrics,
    'files': {f.name: {'bytes': f.stat().st_size, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
              for f in sorted(out.iterdir())},
    'run_info': json.loads((out / 'run_info.json').read_text()),
    'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                 'onnxruntime': __import__('onnxruntime').__version__},
}
(root / 'evidence' / f'{a.name}.validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
