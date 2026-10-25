"""Integrity checks for the explicitly unqualified audit-fix candidate."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_runtime(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'candidate_manifest.json').read_text('utf-8'))
    if manifest.get('name') != 'FORMAT-CONSISTENCY-20261025':
        raise ValueError('Unknown candidate')
    files = manifest['files']
    required = {'run.py', 'tools/run_final_submission.py', 'tools/verify_candidate.py',
                'tools/output_transaction.py', 'configs/phase_i/final_i.json',
                'models/detector/best.pt', 'models/ocr/best.pt'}
    if not required <= files.keys():
        raise ValueError('Incomplete candidate manifest')
    for name, info in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing/unsafe candidate file: ' + name)
        if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError('Candidate file changed: ' + name)
    config = root / 'configs/phase_i/final_i.json'
    settings = json.loads(config.read_text('utf-8'))
    if settings['name'] != manifest['name']:
        raise ValueError('Candidate configuration name mismatch')
    for key in ('detector', 'ocr'):
        relative = f'models/{key}/best.pt'
        if settings[key] != relative or settings[key + '_sha256'] != files[relative]['sha256']:
            raise ValueError('Candidate model binding mismatch')
    reference = 'models/detector/ordinary_reference.pt'
    if reference not in files or settings.get('ordinary_detector') != reference or settings.get('ordinary_detector_sha256') != files[reference]['sha256']:
        raise ValueError('Ordinary reference model binding mismatch')
    return config


if __name__ == '__main__':
    print(verify_runtime(Path(__file__).resolve().parents[1]))
