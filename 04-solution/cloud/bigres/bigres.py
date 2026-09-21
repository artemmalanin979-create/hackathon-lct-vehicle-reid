#!/usr/bin/env python3
"""Big-resolution inference, canonical evaluation and mandatory plate ablation.

No training, no cloud API, no external services. All metrics originate in eval/.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).resolve().parent
SIZES = (208, 256, 320, 384)
MODELS = ('osnet', 'combined_v1')
CONFIGS = (*MODELS, 'd1_j48')
PAIRS = [('plate_ring', 'shift_ring'), ('platepad_ring', 'shiftpad_ring'),
         ('plate_gray127', 'shift_gray127')]
VARIANTS = ['base'] + [v for pair in PAIRS for v in pair]
B, SEED = 4000, 20260921


def stamp(): return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def save_array(path, arr):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f: np.save(f, arr, allow_pickle=False)
    tmp.replace(path)


def paired_bootstrap(a, b, groups=None):
    """Resample aligned per-query outputs of the canonical evaluator."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError('Invalid paired bootstrap inputs')
    delta = a - b
    rng = np.random.default_rng(SEED)
    samples = np.empty(B)
    if groups is None:
        for start in range(0, B, 100):
            idx = rng.integers(0, len(a), (min(100, B - start), len(a)))
            samples[start:start + len(idx)] = delta[idx].mean(axis=1)
    else:
        groups = np.asarray(groups)
        if groups.shape != a.shape: raise ValueError('Mismatched vehicle IDs')
        members = [np.flatnonzero(groups == u) for u in np.unique(groups)]
        sums = np.array([delta[m].sum() for m in members])
        counts = np.array([len(m) for m in members])
        idx = rng.integers(0, len(members), (B, len(members)))
        samples[:] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    # Add-one Monte Carlo correction; never claim p=0 or p>1.
    p = min(1., 2 * min((1 + np.count_nonzero(samples <= 0)) / (B + 1),
                        (1 + np.count_nonzero(samples >= 0)) / (B + 1)))
    return {'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'delta': float(delta.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'p_two_sided': p, 'B': B, 'seed': SEED, 'n_queries': len(a),
            'grouped_by_vehicle': groups is not None}


class Experiment:
    def __init__(self, args):
        self.args = args
        self.payload = args.payload.resolve()
        self.solution = self.payload / '04-solution'
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((self.payload / 'manifest.json').read_text())
        for rel, info in self.manifest['files'].items():
            if sha(self.payload / rel) != info['sha256']:
                raise RuntimeError(f'Payload hash mismatch: {rel}')
        sys.path[:0] = [str(self.solution / 'service'), str(self.solution / 'eval'),
                        str(self.solution / 'plate-ablation/scripts')]
        from app.core import preprocess, model, rerank
        import reid_metrics
        import extract_variants
        import mask_ops
        self.pp, self.model, self.rr = preprocess, model, rerank
        self.ev, self.plate, self.mask = reid_metrics, extract_variants, mask_ops
        self.boxes = json.loads((self.payload / 'boxes.json').read_text())
        self.meta = {}; self.rows = {}
        for sp in ['val_query', 'val_gallery']:
            path = self.solution / f'split/files/{sp}.csv'
            with open(path, newline='') as f: self.meta[sp] = list(csv.DictReader(f))
            self.rows[sp] = self.pp.read_rows(path)
            if len(self.rows[sp]) != len(self.boxes[sp]): raise RuntimeError('Box row mismatch')
        self.P, self.m = self.model._load_whitening(
            self.solution / 'service/model/lw_ens_j48_rho0.5.npz')
        self.paths = {'osnet': self.solution / 'service/model/osnet_ain_x1_0_vehicle_reid.onnx',
                      'combined_v1': self.solution / 'service/model/osnet_ain_combined_v1_dynamic.onnx'}
        self.image_index = {r['image_id']: r for r in self.manifest['images']}
        self.sessions = {}
        self.timings = []
        self.fingerprint = sha(self.payload / 'manifest.json')
        self.code_hash = sha(Path(__file__))
        self._cuda_preloaded = False

    def session(self, name, device=None, original=False):
        device = device or self.args.device
        key = (name, device, original)
        if key in self.sessions: return self.sessions[key]
        if device == 'cuda':
            if not self._cuda_preloaded and hasattr(ort, 'preload_dlls'):
                ort.preload_dlls(directory='')
                self._cuda_preloaded = True
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDAExecutionProvider unavailable; install requirements-gpu.txt. CPU fallback is forbidden.')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.args.threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        if device == 'cuda':
            # Host shape operations may use CPU; inference must actually include CUDA kernels.
            opts.enable_profiling = True
            opts.profile_file_prefix = str(self.out / f'ort_{name}')
        providers = ([('CUDAExecutionProvider', {'device_id': self.args.gpu_id,
                      'cudnn_conv_algo_search': 'HEURISTIC', 'use_tf32': '0'}),
                      'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider'])
        path = self.paths[name]
        if original and name == 'combined_v1':
            path = path.with_name('osnet_ain_combined_v1.onnx')
        s = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        if device == 'cuda' and s.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA session fell back to CPU; stopping before extraction')
        s.disable_fallback()
        if device == 'cuda':
            x = np.zeros((1, 3, 208, 208), dtype=np.float32)
            s.run(None, {s.get_inputs()[0].name: x})
            profile = Path(s.end_profiling())
            events = json.loads(profile.read_text())
            counts = Counter(e.get('args', {}).get('provider', '') for e in events
                             if e.get('args', {}).get('provider'))
            if counts.get('CUDAExecutionProvider', 0) == 0:
                raise RuntimeError('No CUDA kernels were executed')
            write_json(self.out / f'cuda_{name}.json', {'providers': s.get_providers(),
                        'profile': profile.name, 'kernel_counts': dict(counts)})
        self.sessions[key] = s
        return s

    @contextmanager
    def input_size(self, size):
        # Single-threaded scoped configuration of the canonical preprocessing module.
        old = self.pp.INPUT_SIZE
        self.pp.INPUT_SIZE = size
        try: yield
        finally: self.pp.INPUT_SIZE = old

    def control_box(self, box, row, i):
        choices = self.mask.controls(box, (row.h, row.w), 1000 + i)
        for name in ['shift', 'mirror', 'side', 'rand1', 'rand2']:
            candidate = choices[name]
            if (candidate[2:] == tuple(box[2:]) and
                self.mask._fits(candidate, row.h, row.w) and
                not self.mask._overlap(candidate, box)):
                return candidate, name
        raise RuntimeError(f'No disjoint equal-area control: {row.image_id}')

    def tensor(self, sp, i, size, variant='base'):
        row = self.rows[sp][i]
        with self.input_size(size):
            if variant == 'base':
                return self.pp.load_crop(self.args.images, row)
            with Image.open(self.pp.resolve_image_path(self.args.images, row.image_id)) as im:
                arr = np.asarray(im.convert('RGB').crop((row.x, row.y, row.x + row.w, row.y + row.h)))
            b = self.boxes[sp][i]
            box = tuple(b[:4]) if b else None
            variants = self.plate.make_variants(arr, box, seed=1000 + i)
            if box and variant in ['shift_ring', 'shiftpad_ring', 'shift_gray127']:
                target = self.plate.pad_box(box, (row.h, row.w)) if variant == 'shiftpad_ring' else box
                ctl, source = self.control_box(target, row, i)
                if source != 'shift':
                    variants[variant] = self.mask.fill(arr, ctl, 'gray127' if variant == 'shift_gray127' else 'ring')
            arr = variants[variant]
            # Mask BEFORE resize. The canonical function owns resize, RGB, dtype and NCHW.
            return self.pp.crop_to_input(Image.fromarray(arr), 0, 0, row.w, row.h)

    def tensor_batch(self, items, size, variant='base'):
        return np.ascontiguousarray(np.stack([self.tensor(sp, i, size, variant) for sp, i in items]))

    def selected(self, small=False):
        if not small:
            return {sp: list(range(len(rows))) for sp, rows in self.rows.items()}
        qm, gm = self.meta['val_query'], self.meta['val_gallery']
        qs, gs = [], set()
        for i, q in enumerate(qm):
            if q.get('has_mate', '1') == '0': continue
            pos = [j for j, g in enumerate(gm) if g['vehicle_id'] == q['vehicle_id']
                   and g['camera_id'] != q['camera_id']]
            if pos and len(gs | set(pos)) <= 32:
                qs.append(i); gs.update(pos)
            if len(qs) == 24: break
        qs += [i for i, q in enumerate(qm) if q.get('has_mate') == '0'][:32 - len(qs)]
        for i in range(len(gm)):
            if len(gs) == 32: break
            gs.add(i)
        if len(qs) != 32 or len(gs) != 32: raise RuntimeError('Could not form 64-crop dry sample')
        return {'val_query': sorted(qs), 'val_gallery': sorted(gs)}

    def audit_geometry(self):
        counts = {}; examples = []; fallbacks = []
        for sp, rows in self.rows.items():
            detected = 0
            for i, row in enumerate(rows):
                b = self.boxes[sp][i]
                if not b: continue
                detected += 1
                box = tuple(b[:4]); x, y, w, h = box
                if not all(isinstance(v, int) for v in box): raise RuntimeError('Non-integral box')
                if not (w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= row.w and y + h <= row.h):
                    raise RuntimeError(f'Box outside native crop: {sp}/{i}/{box}/{row}')
                for b0 in [box, self.plate.pad_box(box, (row.h, row.w))]:
                    ctl, origin = self.control_box(b0, row, i)
                    if ctl[2:] != b0[2:] or not self.mask._fits(ctl, row.h, row.w):
                        raise RuntimeError('Control area or bounds mismatch')
                    if origin != 'shift': fallbacks.append({'split': sp, 'index': i, 'box': list(b0), 'control': list(ctl), 'source': origin})
                for size in SIZES:
                    scaled = np.array([x / row.w, y / row.h, w / row.w, h / row.h]) * size
                    if not np.allclose(scaled / size, [x / row.w, y / row.h, w / row.w, h / row.h], atol=1e-14):
                        raise RuntimeError('Mask geometry mismatch')
                if len(examples) < 4:
                    examples.append({'split': sp, 'index': i, 'crop_wh': [row.w, row.h],
                        'native_box_xywh': box, 'scaled_box_xywh': {str(s):
                          [x * s / row.w, y * s / row.h, w * s / row.w, h * s / row.h] for s in SIZES}})
            counts[sp] = {'rows': len(rows), 'detected': detected, 'missing': len(rows) - detected}
        result = {'status': 'passed', 'all_native_boxes_in_bounds': True,
                  'equal_area_controls_disjoint': True, 'mask_applied_before_resize': True,
                  'counts': counts, 'examples': examples, 'control_fallbacks': fallbacks,
                  'note': 'No integer coordinate rescaling in inference: native crop is masked, then canonical PIL bilinear resize.'}
        write_json(self.out / 'geometry.json', result)
        return result

    def ablation_selection(self, selection):
        # CPU smoke: 4 queries and 4 gallery crops, retaining real cross-camera mates.
        qm, gm = self.meta['val_query'], self.meta['val_gallery']
        qs, gs = [], set()
        for i in selection['val_query']:
            mates = [j for j in selection['val_gallery'] if
                     gm[j]['vehicle_id'] == qm[i]['vehicle_id'] and gm[j]['camera_id'] != qm[i]['camera_id']]
            if mates and len(gs | {mates[0]}) <= 4:
                qs.append(i); gs.add(mates[0])
            if len(qs) == 4: break
        for j in selection['val_gallery']:
            if len(gs) == 4: break
            gs.add(j)
        if len(qs) != 4 or len(gs) != 4: raise RuntimeError('Cannot form 8-crop ablation smoke sample')
        return {'val_query': sorted(qs), 'val_gallery': sorted(gs)}

    def audit_images(self, selection):
        for sp, indices in selection.items():
            for i in indices:
                iid = self.rows[sp][i].image_id
                p = self.pp.resolve_image_path(self.args.images, iid)
                if sha(p) != self.image_index[iid]['sha256']:
                    raise RuntimeError(f'Image checksum mismatch: {iid}')

    def features(self, name, size, variant, selection):
        result = {}
        for sp, indices in selection.items():
            directory = self.out / 'features' / f'{name}_{size}_{variant}'
            path = directory / f'{sp}.npy'
            info_path = directory / f'{sp}.json'
            key = {'payload_sha256': self.fingerprint, 'code_sha256': self.code_hash,
                   'device': self.args.device, 'ort_version': ort.__version__,
                   'numpy_version': np.__version__, 'size': size, 'model': name,
                   'variant': variant, 'indices': indices,
                   'ids': [self.rows[sp][i].image_id for i in indices],
                   'batch_size': self.args.batch_size, 'threads': self.args.threads}
            if path.is_file() and info_path.is_file():
                old = json.loads(info_path.read_text())
                if old['key'] != key: raise RuntimeError(f'Incompatible cache: use a new --out: {path}')
                if sha(path) != old['sha256']: raise RuntimeError(f'Corrupt feature cache: {path}')
                arr = np.load(path, allow_pickle=False)
                if arr.shape != (len(indices), 512) or not np.isfinite(arr).all():
                    raise RuntimeError('Invalid cached feature matrix')
                result[sp] = arr
                continue
            sess = self.session(name)
            inp = sess.get_inputs()[0].name
            acc = np.empty((len(indices), 512), np.float32)
            prep_s = infer_s = 0.
            t0 = time.perf_counter()
            for start in range(0, len(indices), self.args.batch_size):
                ids = indices[start:start + self.args.batch_size]
                t = time.perf_counter(); x = self.tensor_batch([(sp, i) for i in ids], size, variant)
                prep_s += time.perf_counter() - t
                t = time.perf_counter(); y = sess.run(None, {inp: x})[0]
                infer_s += time.perf_counter() - t
                if y.shape != (len(ids), 512) or not np.isfinite(y).all():
                    raise RuntimeError('Unexpected or non-finite ONNX output')
                acc[start:start + len(ids)] = self.model.l2norm(y)
                if start % (self.args.batch_size * 20) == 0:
                    print(f'{stamp()} {name} {size} {variant} {sp} {start + len(ids)}/{len(indices)}', flush=True)
            save_array(path, acc)
            rec = {'key': key, 'sha256': sha(path), 'seconds': time.perf_counter() - t0,
                   'preprocess_seconds': prep_s, 'inference_seconds': infer_s}
            write_json(info_path, rec); self.timings.append(rec)
            result[sp] = acc
        return result

    def fuse(self, a, b):
        out = {}
        for sp in a:
            x = self.model.l2norm(self.model.l2norm(a[sp]) + self.model.l2norm(b[sp]))
            out[sp] = self.model.l2norm((x - self.m) @ self.P.T)
        return out

    def reference(self, selection):
        vec = {name: {sp: np.load(self.payload / f'reference/{name}_{sp}.npy', allow_pickle=False)[idx]
                     for sp, idx in selection.items()} for name in MODELS}
        return self.fuse(vec['osnet'], vec['combined_v1'])

    def evaluate(self, vectors, selection, tag):
        qm = [self.meta['val_query'][i] for i in selection['val_query']]
        gm = [self.meta['val_gallery'][i] for i in selection['val_gallery']]
        q, g = vectors['val_query'], vectors['val_gallery']
        summary = {}; arrays = {}
        for mode in ['cos', 'kr']:
            t = time.perf_counter()
            scores = (self.ev.scores_from_embeddings(q, g, metric='cosine') if mode == 'cos'
                      else -self.rr.rerank_distances(q, g, 6, 3, .3))
            res = self.ev.evaluate(scores, query_ids=[r['vehicle_id'] for r in qm],
                    gallery_ids=[r['vehicle_id'] for r in gm],
                    query_cameras=[r['camera_id'] for r in qm], gallery_cameras=[r['camera_id'] for r in gm],
                    known_absent=np.array([r.get('has_mate', '1') == '0' for r in qm]),
                    threshold=0., camera_policy='market', refusal_mode='presence')
            known = [r for r in res['per_query'] if r['status'] == 'known']
            ap = np.array([r['ap'] for r in known]); r1 = np.array([r['rank1'] for r in known], dtype=float)
            idx = np.array([selection['val_query'][r['query_index']] for r in known])
            groups = np.array([self.meta['val_query'][i]['vehicle_id'] for i in idx])
            full = res['ranking_full_gallery']
            if not len(ap) or not np.isclose(ap.mean(), full['mAP'], atol=1e-12):
                raise RuntimeError('No valid queries or metric inconsistency')
            summary[mode] = {k: full[k] for k in ['mAP', 'Rank-1', 'Rank-5', 'mINP', 'num_valid_queries']}
            summary[mode]['seconds'] = time.perf_counter() - t
            arrays[mode] = {'ap': ap, 'rank1': r1, 'indices': idx, 'groups': groups}
            write_json(self.out / 'eval' / f'{tag}_{mode}.json', res)
        return summary, arrays

    def compare(self, values, baseline):
        if not np.array_equal(values['indices'], baseline['indices']):
            raise RuntimeError('Paired bootstrap query ordering mismatch')
        return {'mAP': paired_bootstrap(values['ap'], baseline['ap']),
                'Rank-1': paired_bootstrap(values['rank1'], baseline['rank1']),
                'mAP_grouped_by_vehicle': paired_bootstrap(values['ap'], baseline['ap'], values['groups'])}

    def run(self, small=False):
        selection = self.selected(small)
        self.audit_images(selection); self.audit_geometry()
        write_json(self.out / 'selection.json', {sp: {'indices': ids,
            'image_ids': [self.rows[sp][i].image_id for i in ids]} for sp, ids in selection.items()})
        # Reproduce the actual 0.7741 reference from per-query scores, never bootstrap a scalar.
        full = self.selected(False)
        full_summary, _ = self.evaluate(self.reference(full), full, 'reference_full_208')
        if (f"{full_summary['kr']['mAP']:.4f}" != '0.7741' or
            f"{full_summary['kr']['Rank-1']:.4f}" != '0.7308'):
            raise RuntimeError(f'Reference does not reproduce required baseline: {full_summary}')
        baseline_summary, baseline = self.evaluate(self.reference(selection), selection, 'reference_selected_208')
        result = {'status': 'running', 'started': stamp(), 'dry_run': small,
            'counts': {sp: len(ids) for sp, ids in selection.items()},
            'note': ('Small sample checks only; its metrics are NOT full-validation results.' if small
                     else 'Full validation; compare to frozen d1_j48 208 KR per-query reference.'),
            'baseline_full': full_summary, 'baseline_selected': baseline_summary,
            'payload_sha256': self.fingerprint, 'code_sha256': self.code_hash,
            'whitening': 'Frozen service train_fit rho=0.5; float32, never refit on validation or new resolution.',
            'device': self.args.device, 'batch_size': self.args.batch_size, 'threads': self.args.threads,
            'environment': {'python': sys.version, 'numpy': np.__version__, 'onnxruntime': ort.__version__,
                            'platform': platform.platform(), 'available_providers': ort.get_available_providers()},
            'configs': {}, 'ablation': {}, 'checks_208': {},
            'inference_calls_per_crop_base': 8, 'kr': [6, 3, .3], 'bootstrap_replicates': B}
        write_json(self.out / 'results.json', result)
        for size in SIZES:
            feat = {name: self.features(name, size, 'base', selection) for name in MODELS}
            feat['d1_j48'] = self.fuse(feat['osnet'], feat['combined_v1'])
            active = []
            for name in CONFIGS:
                tag = f'{name}_{size}'
                summary, values = self.evaluate(feat[name], selection, tag + '_base')
                compared = {mode: self.compare(values[mode], baseline['kr']) for mode in ['cos', 'kr']}
                result['configs'][tag] = {'metrics': summary, 'versus_frozen_208_KR': compared,
                    'exploratory_multiple_comparisons': True}
                # Also require ablation for a gain over the same configuration at 208.
                if size == 208:
                    gain_over_own = False
                    if name == 'd1_j48':
                        reference = self.reference(selection)
                        diff = max(float(np.max(np.abs(feat[name][sp] - reference[sp]))) for sp in selection)
                        result['checks_208']['max_feature_abs_diff_to_reference'] = diff
                        if diff > .001: raise RuntimeError(f'208 reference feature mismatch: {diff}')
                        if not small and abs(summary['kr']['mAP'] - full_summary['kr']['mAP']) > .0005:
                            raise RuntimeError('Fresh 208 baseline drift is too large')
                else:
                    own = result['configs'][f'{name}_208']['metrics']
                    gain_over_own = any(summary[m]['mAP'] > own[m]['mAP'] + 1e-12 for m in ['cos', 'kr'])
                gain_vs_baseline = any(v['mAP']['delta'] > 1e-12 for v in compared.values())
                required = small or gain_vs_baseline or gain_over_own
                result['configs'][tag]['plate_ablation_required'] = required
                result['configs'][tag]['plate_ablation_reason'] = ('forced_dry_run' if small else
                    'gain_vs_baseline_or_own_208' if required else 'no_gain')
                if required: active.append((name, values))
            write_json(self.out / 'results.json', result)
            if active:
                ab_selection = self.ablation_selection(selection) if small else selection
                if small:
                    smoke_active = []
                    for name, _ in active:
                        subset = {sp: feat[name][sp][[selection[sp].index(i) for i in ids]]
                                  for sp, ids in ab_selection.items()}
                        _, bv = self.evaluate(subset, ab_selection, f'{name}_{size}_ablation_base')
                        smoke_active.append((name, bv))
                    active = smoke_active
                needed = set()
                for name, _ in active: needed.update(MODELS if name == 'd1_j48' else [name])
                ablation_values = {}
                for variant in VARIANTS[1:]:
                    vf = {name: self.features(name, size, variant, ab_selection) for name in sorted(needed)}
                    if 'd1_j48' in [name for name, _ in active]: vf['d1_j48'] = self.fuse(vf['osnet'], vf['combined_v1'])
                    for name, _ in active:
                        tag = f'{name}_{size}'
                        summary, vals = self.evaluate(vf[name], ab_selection, tag + '_' + variant)
                        result['ablation'].setdefault(tag, {'metrics': {}, 'contrasts': {}, 'counts': {sp: len(ids) for sp, ids in ab_selection.items()}, 'dry_run_smoke_only': small})['metrics'][variant] = summary
                        ablation_values[(name, variant)] = vals
                for name, basevalues in active:
                    tag = f'{name}_{size}'
                    warning = False
                    for plate, control in PAIRS:
                        contrasts = {}
                        for mode in ['cos', 'kr']:
                            pv = ablation_values[(name, plate)][mode]
                            cv = ablation_values[(name, control)][mode]
                            contrasts[mode] = self.compare(cv, pv)  # positive = specific plate penalty
                            contrasts[mode]['plate_minus_unmasked'] = self.compare(pv, basevalues[mode])
                            warning |= contrasts[mode]['mAP_grouped_by_vehicle']['ci95'][0] > 0
                        result['ablation'][tag]['contrasts'][plate + '_vs_' + control] = contrasts
                    result['ablation'][tag]['status'] = ('plate_specific_degradation_detected' if warning
                        else 'no_significant_plate_specific_degradation_detected')
                    result['ablation'][tag]['interpretation'] = 'Non-significance does not prove equivalence; inspect detection coverage and confidence intervals.'
                write_json(self.out / 'results.json', result)
            write_json(self.out / 'timings.json', self.timings)
        # Every positive result has all three mask/control contrasts before success.
        for tag, rec in result['configs'].items():
            if rec['plate_ablation_required'] and len(result['ablation'].get(tag, {}).get('contrasts', {})) != 3:
                raise RuntimeError(f'Missing required ablation: {tag}')
        result['status'] = 'complete'; result['finished'] = stamp()
        write_json(self.out / 'results.json', result)
        print(f'{stamp()} COMPLETE: {self.out / "results.json"}', flush=True)
        return result

    def benchmark(self):
        sample = self.selected(True)
        items = [(sp, i) for sp, ids in sample.items() for i in ids][:50]
        self.audit_images({sp: [i for s, i in items if s == sp] for sp in sample})
        result = {'status': 'running', 'device': self.args.device, 'crops_per_size_per_model': 50,
                  'threads': self.args.threads, 'batch_size': self.args.batch_size,
                  'platform': platform.platform(), 'python': sys.version,
                  'numpy': np.__version__, 'onnxruntime': ort.__version__,
                  'warmup': 'One full batch per model and input before timing; timing excludes model loading.',
                  'samples': [{'split': sp, 'index': i, 'image_id': self.rows[sp][i].image_id} for sp, i in items],
                  'sizes': {}}
        for size in SIZES:
            t = time.perf_counter()
            chunks = [self.tensor_batch(items[s:s + self.args.batch_size], size)
                      for s in range(0, len(items), self.args.batch_size)]
            prep = time.perf_counter() - t
            rec = {'preprocess_seconds_50': prep, 'models': {}}
            for name in MODELS:
                s = self.session(name); inp = s.get_inputs()[0].name
                s.run(None, {inp: chunks[0]})
                times = []
                for x in chunks:
                    t = time.perf_counter(); y = s.run(None, {inp: x})[0]; times.append(time.perf_counter() - t)
                    if y.shape != (len(x), 512) or not np.isfinite(y).all(): raise RuntimeError('Invalid benchmark output')
                sec = sum(times)
                rec['models'][name] = {'inference_seconds_50': sec, 'inference_ms_per_crop': sec * 20,
                    'crops_per_second': 50 / sec, 'batch_seconds': times}
                print(f'{stamp()} bench {name} {size}: {sec:.3f}s/50', flush=True)
            rec['ensemble_inference_seconds_50'] = sum(r['inference_seconds_50'] for r in rec['models'].values())
            rec['ensemble_end_to_end_seconds_50'] = prep + rec['ensemble_inference_seconds_50']
            result['sizes'][str(size)] = rec
            write_json(self.out / 'benchmark.json', result)
        result['status'] = 'complete'; write_json(self.out / 'benchmark.json', result)
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['run', 'dry-run', 'benchmark', 'geometry'])
    p.add_argument('--payload', type=Path, default=HERE / 'payload')
    p.add_argument('--images', type=Path, default=Path(os.environ.get('BIGRES_IMAGES',
                    str(HERE / 'data/images' if (HERE / 'data/images').is_dir()
                        else Path.home() / 'lct-reid/data/images'))))
    p.add_argument('--out', type=Path)
    p.add_argument('--device', choices=['cpu', 'cuda'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--gpu-id', type=int, default=0)
    return p


def main():
    args = parser().parse_args()
    if args.batch_size < 1 or args.threads < 1: raise SystemExit('Positive batch-size and threads required')
    args.device = args.device or ('cuda' if args.command == 'run' else 'cpu')
    if args.command == 'dry-run' and args.device != 'cpu': raise SystemExit('dry-run requires CPU')
    args.out = args.out or HERE / 'out' / (args.command + '_' + args.device)
    t = time.perf_counter()
    try:
        exp = Experiment(args)
        if args.command == 'geometry': exp.audit_geometry()
        elif args.command == 'benchmark': exp.benchmark()
        else: exp.run(small=args.command == 'dry-run')
    except Exception as e:
        write_json(args.out / 'failure.json', {'error': str(e), 'type': type(e).__name__, 'time': stamp()})
        raise
    print(f'Total elapsed: {time.perf_counter() - t:.2f}s', flush=True)

if __name__ == '__main__': main()
