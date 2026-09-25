#!/usr/bin/env python3
"""Assemble an offline candidate locally. Never upload, pay, or deploy remotely.

Requires a clean tracked Git tree and an image built with the exact source SHA
in org.opencontainers.image.revision. The supplied slides must be the reviewed
technical PDF without the mandatory contacts page; full deck is delivered aside.
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from verify_release import digest


def preflight(repo: Path, data: Path, public_slides: Path) -> list[dict]:
    """Reject stale or mismatched inputs before creating a release directory."""
    subprocess.run(['python3', str(repo / '06-documentation/build_pdf.py'), '--check'],
                   cwd=repo, check=True, capture_output=True)
    artifact_dir = repo / '04-solution/service/artifacts-final'
    manifest = json.loads((artifact_dir / 'manifest.json').read_text())
    for name, spec in manifest['files'].items():
        path = artifact_dir / name
        if not path.is_file() or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise SystemExit(f'Canonical artifact mismatch: {name}')
    for spec in manifest['input_csvs']:
        path = data / spec['path']
        if not path.is_file() or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise SystemExit(f'Canonical input CSV mismatch: {spec["path"]}')
    model_dir = repo / '04-solution/service/model'
    for name, expected in (
        ('osnet_ain_x1_0_vehicle_reid.onnx', manifest['run_info']['model_sha256']),
        ('osnet_ain_combined_v1.onnx', manifest['run_info']['model2_sha256']),
        ('lw_ens_j48_rho0.5.npz', manifest['run_info']['whitening_sha256']),
    ):
        path = model_dir / name
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f'Canonical model mismatch: {name}')
    info = subprocess.check_output(['pdfinfo', str(public_slides)], text=True)
    pages = re.search(r'^Pages:\s*(\d+)\s*$', info, re.MULTILINE)
    if pages is None or int(pages.group(1)) != 11:
        raise SystemExit('Public technical PDF must have exactly 11 pages')
    private_team = repo / '05-presentation/team-data.md'
    if not private_team.is_file():
        raise SystemExit('Local contact source is needed to screen the public PDF')
    phone_numbers = set(re.findall(r'\+7\d{10}', private_team.read_text()))
    if not phone_numbers:
        raise SystemExit('No local contact numbers found for public PDF screening')
    slide_text = subprocess.check_output(['pdftotext', '-layout', str(public_slides), '-'], text=True)
    slide_digits = re.sub(r'\D', '', slide_text)
    if any(re.sub(r'\D', '', number) in slide_digits for number in phone_numbers):
        raise SystemExit('Public technical PDF contains a private contact number')
    inputs = json.loads((repo / '04-solution/reproduce/inputs-manifest.json').read_text())
    for spec in inputs['data']['test']:
        path = (data / spec['path']).resolve()
        if not path.is_relative_to(data.resolve()) or not path.is_file() or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise SystemExit(f'Input mismatch: {spec["path"]}')
    return inputs['data']['test']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--public-slides', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--engine', choices=('podman', 'docker'), default='podman')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    service = repo / '04-solution/service'
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=repo, text=True)
    if status:
        raise SystemExit('Commit and review all source changes before assembling a release')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    info = json.loads(subprocess.check_output([args.engine, 'image', 'inspect', args.image]))[0]
    if info.get('Config', {}).get('Labels', {}).get('org.opencontainers.image.revision') != sha:
        raise SystemExit('Image revision label must match clean HEAD')
    if not args.public_slides.is_file():
        raise SystemExit('Reviewed technical slides are required')
    inputs = preflight(repo, args.data, args.public_slides)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'runtime').mkdir()
    (out / 'data').mkdir()
    shutil.copytree(service / 'deploy', out / 'deploy', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(service / 'deploy/compose.yaml', out / 'compose.yaml')
    shutil.copy2(service / 'deploy/selinux.yaml', out / 'selinux.yaml')
    shutil.copytree(service / 'artifacts-final', out / 'artifacts')
    materials = out / 'materials'; materials.mkdir()
    shutil.copy2(repo / '06-documentation/SOLUTION.pdf', materials / 'solution.pdf')
    shutil.copy2(args.public_slides, materials / 'presentation.pdf')
    for name in ('manifest.json', 'submission.csv', 'candidates.csv', 'embeddings.npy', 'run_info.json'):
        shutil.copy2(service / 'artifacts-final' / name, materials / name)
    for spec in inputs:
        name = spec['path']; source = (args.data / name).resolve()
        target = out / 'data' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    subprocess.run(['git', 'archive', '--format=tar.gz', f'--output={out / "source.tar.gz"}', sha], cwd=repo, check=True)
    save = [args.engine, 'save', '--output', str(out / 'runtime/service.tar')]
    if args.engine == 'podman':
        save += ['--format', 'docker-archive']
    subprocess.run(save + [args.image], check=True)
    shutil.copy2(service / 'offline/qdrant-v1.15.5.tar.gz', out / 'runtime/qdrant-v1.15.5.tar.gz')
    manifest_digest = digest(out / 'artifacts/manifest.json')
    (out / 'release.env').write_text(f'SERVICE_IMAGE={args.image}\nQDRANT_COLLECTION=lct_{manifest_digest[:16]}\n'
                                     'API_PORT=18071\nDEPLOYMENT_STATUS=DEPLOY PENDING\n')
    files = {str(path.relative_to(out)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
             for path in sorted(out.rglob('*')) if path.is_file()}
    result = {'source_sha': sha, 'image_id': info.get('Id'), 'model': 'd1_j48',
              'deployment_status': 'DEPLOY PENDING', 'files': files}
    (out / 'release.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'source_sha': sha, 'files': len(files), 'bytes': sum(f['bytes'] for f in files.values()),
                      'directory': str(out)}, indent=2))


if __name__ == '__main__':
    main()
