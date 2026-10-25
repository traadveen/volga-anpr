"""Submission I/O correctness; models are isolated fixtures, never real test data."""
import csv
import json
from pathlib import Path
import socket

import pytest
from PIL import Image

from src.inference_io import file_sha256
from tools import run_final_submission as runner
from tools.validate_output import COLUMNS, validate_file, validate_rows


def settings(tmp_path, monkeypatch):
    config = dict(name='FINAL-I', imgsz=640, confidence=.5, iou=.5, split={},
                  multi=True, postprocess='current', expansion={}, recognition_profile='russian_v1')
    for key in ('detector', 'ocr'):
        path = tmp_path / (key + '.pt')
        path.write_bytes(key.encode())
        config[key] = str(path)
        config[key + '_sha256'] = file_sha256(path)
    path = tmp_path / 'final.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr(runner, 'runtime_config', lambda root: path)
    return config


def models(monkeypatch, predictions=()):
    from src import image_anpr_f, image_anpr_g, plate_formats
    seen = []
    class Pipeline:
        def __init__(self, *args, **kwargs):
            with pytest.raises(RuntimeError, match='Network access'):
                socket.getaddrinfo('not-used.invalid', 443)
        def process(self, frame, image_name):
            seen.append((image_name, frame.shape))
            return dict(image=image_name, predictions=[dict(p, image=image_name) for p in predictions],
                        detections=[], cached_detections=False)
    monkeypatch.setattr(image_anpr_f, 'DetectorF', lambda *args: object())
    monkeypatch.setattr(image_anpr_g, 'OCRG', lambda *args: object())
    monkeypatch.setattr(image_anpr_g, 'ImagePipelineG', Pipeline)
    monkeypatch.setattr(plate_formats, 'apply_russian_profile', lambda r: r)
    return seen


def test_mixed_folder_preserves_good_predictions_and_records_bad_image(tmp_path, monkeypatch):
    settings(tmp_path, monkeypatch)
    predictions = [dict(plate_num='A123BC7#7', plate_type='type1', confidence=.8),
                   dict(plate_num='B456CT77', plate_type='type1a', confidence=.9),
                   dict(plate_num='AA22577', plate_type='type1b', confidence=.7)]
    seen = models(monkeypatch, predictions)
    images = tmp_path / 'input'; images.mkdir()
    Image.new('RGB', (40, 30)).save(images / 'кадр.PNG')
    (images / 'broken.jpg').write_bytes(b'not an image')
    output = tmp_path / 'result.csv'
    assert runner.main(['--input', str(images), '--output', str(output)]) == 0
    assert len(seen) == 1 and seen[0][0] == 'кадр.PNG'
    assert validate_file(output, {'кадр.PNG'})['valid']
    with output.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter=';'))
    assert len(rows) == 3 and {r['plate_type'] for r in rows} == {'type1', 'type1a', 'type1b'}
    assert rows[0]['plate_num'] == 'A123BC7#7'
    manifest = json.loads(output.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    assert (manifest['input_images'], manifest['processed_images'], manifest['skipped_images']) == (2, 1, 1)
    assert not manifest['ground_truth_input']
    errors = json.loads(output.with_suffix('.errors.json').read_text())['errors']
    assert errors[0]['image'] == 'broken.jpg'


def test_empty_folder_publishes_header_without_loading_models(tmp_path, monkeypatch):
    settings(tmp_path, monkeypatch)
    empty = tmp_path / 'empty'; empty.mkdir()
    monkeypatch.setattr(runner, 'recognize', lambda *a: pytest.fail('Empty directory loaded models'))
    output = tmp_path / 'empty.csv'
    assert runner.main(['--input', str(empty), '--output', str(output)]) == 0
    assert output.read_bytes() == b'image;plate_num;plate_type;confidence\n'
    assert validate_file(output)['valid']


def test_all_bad_preserves_existing_csv(tmp_path, monkeypatch):
    settings(tmp_path, monkeypatch)
    seen = models(monkeypatch)
    image = tmp_path / 'broken.jpg'; image.write_bytes(b'bad')
    output = tmp_path / 'result.csv'; output.write_bytes(b'previous success')
    with pytest.raises(SystemExit) as error:
        runner.main(['--input', str(image), '--output', str(output)])
    assert error.value.code == 2 and output.read_bytes() == b'previous success'
    assert not seen and output.with_suffix('.errors.json').exists()


@pytest.mark.parametrize('shape', [(1, 1), (4000, 3000)])
def test_tiny_and_large_image_single_input(tmp_path, monkeypatch, shape):
    settings(tmp_path, monkeypatch)
    seen = models(monkeypatch)
    image = tmp_path / 'image.png'; Image.new('RGB', shape).save(image)
    output = tmp_path / 'result.csv'
    assert runner.main(['--input', str(image), '--output', str(output)]) == 0
    assert seen[0][1][:2] == (shape[1], shape[0])
    assert validate_file(output)['rows'] == 0


@pytest.mark.parametrize('value', ['nan', 'inf', '-0.1', '1.1', True])
def test_invalid_confidence_is_rejected(value):
    row = dict(zip(COLUMNS, ['x.jpg', 'A123BC77', 'type1', value]))
    assert not validate_rows([row])['valid']


@pytest.mark.parametrize('text,kind', [('AA22577', 'type1b'), ('A123BC7#7', 'type1a'), ('', 'other')])
def test_type_specific_output_masks_and_unknowns(text, kind):
    row = dict(zip(COLUMNS, ['x.jpg', text, kind, .8]))
    assert validate_rows([row])['valid']


def test_wrong_separator_extra_columns_paths_and_lowercase_fail(tmp_path):
    output = tmp_path / 'result.csv'
    output.write_text('image,plate_num,plate_type,confidence\nx.jpg,A123BC77,type1,.9\n')
    assert not validate_file(output)['valid']
    row = dict(zip(COLUMNS, ['folder/x.jpg', 'a123BC77', 'type1', .9]))
    assert len(validate_rows([row])['errors']) == 2
    output.write_text('image;plate_num;plate_type;confidence\nx.jpg;A123BC77;type1;.9;extra\n')
    assert not validate_file(output)['valid']


def test_exact_duplicate_rows_are_reported_and_not_removed():
    row = dict(zip(COLUMNS, ['x.jpg', 'A123BC77', 'type1', .9]))
    result = validate_rows([row, row])
    assert result['valid'] and result['rows'] == 2 and result['warnings']
