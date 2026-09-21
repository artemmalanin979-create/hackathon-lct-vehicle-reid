#!/usr/bin/env python3
"""Snapshot canonical inputs and relax only combined_v1 ONNX input metadata."""
from pathlib import Path
import csv, hashlib, json, shutil
import numpy as np
import onnx

JOB = Path(__file__).resolve().parent
ROOT = Path.home() / 'lct-reid'
REPO = ROOT / 'clean-tree'
PAYLOAD = JOB / 'payload'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    manifest = {'files': {}, 'images': [], 'source_repo': str(REPO)}
    def copy(src, rel):
        dst = PAYLOAD / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest['files'][rel] = {'sha256': sha(dst), 'bytes': dst.stat().st_size,
                                  'source': str(src)}
    files = ['service/app/__init__.py', 'service/app/core/__init__.py',
             'service/app/core/config.py', 'service/app/core/preprocess.py',
             'service/app/core/model.py', 'service/app/core/rerank.py',
             'eval/reid_metrics.py', 'eval/scope_metrics.py',
             'eval/test_metrics.py', 'eval/test_protocols.py',
             'plate-ablation/scripts/extract_variants.py',
             'plate-ablation/scripts/mask_ops.py', 'cloud/prices.json',
             'split/files/val_query.csv', 'split/files/val_gallery.csv',
             'service/model/osnet_ain_x1_0_vehicle_reid.onnx',
             'service/model/osnet_ain_combined_v1.onnx',
             'service/model/lw_ens_j48_rho0.5.npz']
    for rel in files: copy(REPO / '04-solution' / rel, '04-solution/' + rel)
    boxfiles = sorted((ROOT / 'jobs/job_45b/mirror').glob('*/04-solution/plate-ablation/work/boxes.json'))
    assert len(boxfiles) == 2 and len({sha(p) for p in boxfiles}) == 1
    copy(boxfiles[0], 'boxes.json')
    for sp in ['val_query', 'val_gallery']:
        rows = list(csv.DictReader(open(REPO / f'04-solution/split/files/{sp}.csv')))
        ids = [r['image_id'] for r in rows]
        sources = {'osnet': (REPO / f'04-solution/training/attempt-2/out/{sp}_osnet.npy',
                              REPO / f'04-solution/training/attempt-2/out/{sp}.ids'),
                   'combined_v1': (ROOT / f'jobs/job_48/out/{sp}_j48.npy',
                                   ROOT / f'jobs/job_48/out/{sp}_j48.ids')}
        for name, (vec, side) in sources.items():
            assert side.read_text().split() == ids, f'Wrong reference order: {side}'
            assert np.load(vec, allow_pickle=False).shape == (len(ids), 512)
            copy(vec, f'reference/{name}_{sp}.npy')
            copy(side, f'reference/{name}_{sp}.ids')
        for r in rows:
            p = ROOT / 'data/images' / (r['image_id'] + '.jpg')
            assert p.is_file()
            manifest['images'].append({'split': sp, 'image_id': r['image_id'],
                'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    model = PAYLOAD / '04-solution/service/model/osnet_ain_combined_v1.onnx'
    dynamic = model.with_name('osnet_ain_combined_v1_dynamic.onnx')
    m = onnx.load(model)
    nodes = [n.SerializeToString() for n in m.graph.node]
    weights = [n.SerializeToString() for n in m.graph.initializer]
    shape_before = str(m.graph.input[0])
    for i, name in [(2, 'height'), (3, 'width')]:
        d = m.graph.input[0].type.tensor_type.shape.dim[i]
        d.ClearField('dim_value'); d.dim_param = name
    onnx.checker.check_model(m)
    onnx.save(m, dynamic)
    assert nodes == [n.SerializeToString() for n in m.graph.node]
    assert weights == [n.SerializeToString() for n in m.graph.initializer]
    manifest['files'][str(dynamic.relative_to(PAYLOAD))] = {
        'sha256': sha(dynamic), 'bytes': dynamic.stat().st_size,
        'source': 'Derived: input H/W metadata only, graph nodes and initializers unchanged'}
    manifest['dynamic_edit'] = {'source_sha256': sha(model), 'derived_sha256': sha(dynamic),
        'nodes_identical': True, 'initializers_identical': True,
        'before': shape_before, 'after': str(m.graph.input[0]),
        'node_count': len(nodes), 'initializer_count': len(weights)}
    manifest['boxes_sources'] = [str(p) for p in boxfiles]
    manifest['counts'] = {'images': len(manifest['images']),
                          'image_bytes': sum(r['bytes'] for r in manifest['images'])}
    (PAYLOAD / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'payload': str(PAYLOAD), 'counts': manifest['counts'],
                      'dynamic_edit': manifest['dynamic_edit']}, indent=2))

if __name__ == '__main__': main()
