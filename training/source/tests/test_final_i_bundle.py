"""A portable FINAL-I must preserve recognition semantics and reject damaged exports."""
from pathlib import Path
import json
import shutil

import pytest

from src.inference_io import file_sha256, load_runtime_config
from tools.build_inference_bundle import RUNTIME_FILES
from tools.build_final_i_bundle import build_final_i_bundle, EVIDENCE
from tools.final_i_runtime import verify_runtime


def source(root):
    root.mkdir()
    for name in (*RUNTIME_FILES, *EVIDENCE, 'tools/final_i_runtime.py',
                 'docs/final_i_delivery.md', 'docs/final_i_results.md'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('owned fixture', encoding='utf-8')
    settings = dict(name='FINAL-I', imgsz=640, confidence=.5, iou=.5,
                    recognition_profile='russian_v1', split={'method': 'B_projection'},
                    multi=True, postprocess='current', expansion={})
    for key in ('detector', 'ocr'):
        path = root / f'models/phase_i/{key}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((key + '-frozen-checkpoint').encode())
        settings[key] = str(path)
        settings[key + '_sha256'] = file_sha256(path)
    config = root / 'configs/phase_i/final_i.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(settings), encoding='utf-8')
    protected = [config, root / 'src/image_anpr_g.py',
                 *(Path(settings[k]) for k in ('detector', 'ocr'))]
    freeze = {'status': 'FINAL-I-FROZEN', 'hashes': {str(p): file_sha256(p) for p in protected}}
    (config.parent / 'final_i_freeze.json').write_text(json.dumps(freeze), encoding='utf-8')
    return config, settings


def test_export_preserves_frozen_config_and_runtime_settings_after_relocation(tmp_path):
    root = tmp_path / 'source'
    config, original = source(root)
    before = config.read_bytes()
    export = tmp_path / 'runtime'
    manifest = build_final_i_bundle(export, root=root)
    assert config.read_bytes() == before
    assert manifest['name'] == 'FINAL-I' and manifest['default_profile'] == 'russian_v1'
    assert not (export / 'configs/phase_g/best_g.json').exists()
    moved = tmp_path / 'new folder' / 'runtime'
    moved.parent.mkdir()
    shutil.copytree(export, moved)
    path = verify_runtime(moved)
    settings = load_runtime_config(path, moved)
    assert all(Path(settings[k]).is_relative_to(moved) for k in ('detector', 'ocr'))
    assert settings['confidence'] == original['confidence']
    assert settings['recognition_profile'] == 'russian_v1'
    assert (moved / 'provenance/final_i.original.json').read_bytes() == before
    with pytest.raises(ValueError, match='preserved'):
        build_final_i_bundle(export, root=root)


def test_freeze_change_blocks_export_before_any_publication(tmp_path):
    root = tmp_path / 'source'
    source(root)
    (root / 'src/image_anpr_g.py').write_text('changed algorithm')
    with pytest.raises(ValueError, match='Frozen file changed'):
        build_final_i_bundle(tmp_path / 'runtime', root=root)
    assert not (tmp_path / 'runtime').exists()


@pytest.mark.parametrize('changed', ['models/detector/best.pt', 'src/image_anpr_g.py'])
def test_runtime_detects_damaged_weights_or_pipeline(tmp_path, changed):
    root = tmp_path / 'source'
    source(root)
    export = tmp_path / 'runtime'
    build_final_i_bundle(export, root=root)
    (export / changed).write_bytes(b'changed')
    with pytest.raises(ValueError, match='missing or modified'):
        verify_runtime(export)


def test_export_cannot_change_threshold_even_if_manifest_is_recalculated(tmp_path):
    root = tmp_path / 'source'
    source(root)
    export = tmp_path / 'runtime'
    manifest = build_final_i_bundle(export, root=root)
    config = export / 'configs/phase_i/final_i.json'
    settings = json.loads(config.read_text())
    settings['confidence'] = .1
    config.write_text(json.dumps(settings), encoding='utf-8')
    manifest['files']['configs/phase_i/final_i.json'] = {'bytes': config.stat().st_size, 'sha256': file_sha256(config)}
    (export / 'bundle_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='recognition settings'):
        verify_runtime(export)


def test_runtime_manifest_cannot_resolve_external_paths(tmp_path):
    root = tmp_path / 'source'
    source(root)
    export = tmp_path / 'runtime'
    manifest = build_final_i_bundle(export, root=root)
    manifest['files']['../outside'] = {'bytes': 0, 'sha256': '0' * 64}
    (export / 'bundle_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='escapes'):
        verify_runtime(export)


def test_runtime_cli_requires_input_and_does_not_expose_profile_override():
    from tools.final_i_runtime import main
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(['--verify-only', '--recognition-profile', 'frozen_g'])
    assert error.value.code == 2
