"""Portable FINAL-I entry point. Uses only the exported, verified runtime."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_runtime(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'bundle_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('name') != 'FINAL-I' or manifest.get('default_profile') != 'russian_v1':
        raise ValueError('Expected a frozen FINAL-I runtime with russian_v1')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Runtime file manifest is empty')
    required = {'tools/final_i_runtime.py', 'configs/phase_i/final_i.json',
                'provenance/final_i.original.json', 'models/detector/best.pt', 'models/ocr/best.pt',
                'tools/predict_images_g.py', 'src/inference_io.py'}
    if not required.issubset(files):
        raise ValueError('Runtime manifest is incomplete')
    for name, entry in files.items():
        relative = PurePosixPath(name)
        path = (root / name).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
            raise ValueError('Manifest path escapes runtime')
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Runtime file missing or modified: ' + name)
    config_path = root / 'configs/phase_i/final_i.json'
    settings = json.loads(config_path.read_text(encoding='utf-8'))
    original = json.loads((root / 'provenance/final_i.original.json').read_text(encoding='utf-8'))
    if digest(root / 'provenance/final_i.original.json') != manifest['source_config_sha256']:
        raise ValueError('Frozen config provenance mismatch')
    for key in ('detector', 'ocr'):
        if settings.get(key) != f'models/{key}/best.pt':
            raise ValueError('Runtime weights must be inside the exported package')
        if digest(root / settings[key]) != settings[key + '_sha256']:
            raise ValueError('Frozen checkpoint mismatch: ' + key)
    semantic = lambda value: {k: v for k, v in value.items() if k not in ('detector', 'ocr')}
    if semantic(settings) != semantic(original) or settings.get('name') != 'FINAL-I':
        raise ValueError('Export changed frozen recognition settings')
    if settings.get('recognition_profile') != 'russian_v1':
        raise ValueError('Frozen recognition profile changed')
    return config_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    if not args.verify_only and (args.input is None or args.output is None):
        parser.error('--input and --output are required for recognition')
    try:
        config = verify_runtime()
        if args.verify_only:
            print('FINAL-I runtime verified; frozen detector/OCR/settings; russian_v1')
            return 0
        from tools.predict_images_g import main as predict
        predict(['--input', str(args.input), '--output', str(args.output),
                 '--device', args.device, '--config', str(config),
                 '--recognition-profile', 'russian_v1'])
        verify_runtime()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
