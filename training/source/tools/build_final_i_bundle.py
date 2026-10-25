"""Export FINAL-I without changing the frozen project, models, or test state."""
from pathlib import Path
import argparse
import json
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference_io import file_sha256
from tools.build_inference_bundle import build_bundle
from tools.final_i_runtime import verify_runtime

EVIDENCE = (
    'reports/phase_i/final_val.json', 'reports/phase_i/final_test.json',
    'reports/phase_i/detector_comparison.csv', 'reports/phase_i/selection.json',
    'reports/phase_i/balanced_20261019/selection_policy.json',
)


def verify_freeze(root):
    path = root / 'configs/phase_i/final_i_freeze.json'
    freeze = json.loads(path.read_text(encoding='utf-8'))
    if freeze.get('status') != 'FINAL-I-FROZEN':
        raise ValueError('FINAL-I has not been frozen')
    config = root / 'configs/phase_i/final_i.json'
    if freeze['hashes'].get(str(config)) != file_sha256(config):
        raise ValueError('FINAL-I config is not bound to the freeze')
    for name, expected in freeze['hashes'].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError('Frozen file changed: ' + name)
    return path


def build_final_i_bundle(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; previous snapshots are preserved')
    if output == root or root.is_relative_to(output):
        raise ValueError('Output overlaps the source project')
    freeze_path = verify_freeze(root)
    config = root / 'configs/phase_i/final_i.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='final-i-export-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'runtime'
        manifest = build_bundle(stage, root=root, config=config)
        # The existing builder exports the shared G engine; only path fields relocate.
        legacy_config = stage / 'configs/phase_g/best_g.json'
        portable_config = stage / 'configs/phase_i/final_i.json'
        portable_config.parent.mkdir(parents=True)
        legacy_config.rename(portable_config)
        for name in ('tools/final_i_runtime.py', 'docs/final_i_delivery.md', 'docs/final_i_results.md'):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        provenance = stage / 'provenance'
        provenance.mkdir()
        shutil.copy2(config, provenance / 'final_i.original.json')
        shutil.copy2(freeze_path, provenance / 'final_i_freeze.original.json')
        for name in EVIDENCE:
            shutil.copy2(root / name, provenance / Path(name).name)
        (stage / 'README.md').write_bytes((root / 'docs/final_i_delivery.md').read_bytes())
        manifest.update(name='FINAL-I', scope='Portable frozen inference and saved evidence; no training or test data',
            default_profile='russian_v1', source_freeze_sha256=file_sha256(freeze_path),
            model_selection_uses_test=False, config_path_fields_relocated=True)
        manifest['files'] = {p.relative_to(stage).as_posix(): {'sha256': file_sha256(p), 'bytes': p.stat().st_size}
                             for p in sorted(stage.rglob('*')) if p.is_file() and p.name != 'bundle_manifest.json'}
        (stage / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        verify_runtime(stage)
        verify_freeze(root)
        stage.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_final_i_bundle(args.output)
    print('Exported FINAL-I:', args.output, '; verified files:', len(manifest['files']))


if __name__ == '__main__':
    main()
