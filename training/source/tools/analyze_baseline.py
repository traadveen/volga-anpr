"""Count observed legacy stages without estimating unannotated detection recall.

Usage: python -B tools/analyze_baseline.py
Consumes existing inference artifacts only; no model import or inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(trace_path, manifest_path, predictions_path):
    images = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    boxes = [box for image in images for box in image.get('detector_boxes_before_geometry_filter', [])]
    observations = [item for image in images for item in image.get('observations', [])]
    emissions = [item for image in images for item in image.get('emissions', [])]
    reasons = Counter()
    for box in boxes:
        width, height = max(1., box['bbox'][2]), max(1., box['bbox'][3])
        for name, condition in [('aspect_below_1.6', width / height < 1.6),
                                ('aspect_above_6.5', width / height > 6.5),
                                ('width_below_50', width < 50), ('height_below_14', height < 14)]:
            if condition:
                reasons[name] += 1

    matched = []
    unmatched = []
    no_emission = []
    for image in images:
        available = list(image.get('observations', []))
        for emission in image.get('emissions', []):
            bbox = emission['bbox']
            if isinstance(bbox, str):
                bbox = json.loads(bbox)
            # Match exported provenance, never assume rows retain detection order.
            observation = next((obs for obs in available if obs['bbox'] == bbox
                                and obs['det_conf'] == emission['det_conf']
                                and obs['raw_ocr'] == emission['raw_ocr']), None)
            if observation is None:
                unmatched.append({'image': image['image'], 'emission': emission})
                continue
            available.remove(observation)
            matched.append({'image': image['image'], 'bbox': bbox,
                            'raw_ocr': observation['raw_ocr'],
                            'ocr_stage_text': observation['postprocessed_ocr'],
                            'final_text': emission['plate_num']})
        no_emission.extend({'image': image['image'], 'bbox': obs['bbox'],
                            'raw_ocr': obs['raw_ocr'], 'ocr_stage_text': obs['postprocessed_ocr']}
                           for obs in available)

    source_hashes = {}
    for relative, expected in manifest.get('source_hashes', {}).items():
        path = ROOT / Path(relative.replace('\\', '/'))
        actual = sha256(path) if path.is_file() else None
        source_hashes[relative] = {'expected': expected, 'actual': actual, 'matches': actual == expected}
    with predictions_path.open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream, delimiter=';'))
    # DictReader contains strings while trace contains numbers; compare serialized
    # scalar fields using their CSV representation.
    trace_as_csv = [{key: str(value) for key, value in row.items()} for row in emissions]
    return {
        'experiment': manifest.get('experiment'),
        'scope': 'Counts on this trace only; no GT bbox available, so detector recall/precision/FP are not estimated. Detector boxes are after YOLO confidence/NMS and before legacy geometry filtering.',
        'artifacts': {'trace': str(trace_path), 'manifest': str(manifest_path), 'predictions': str(predictions_path)},
        'integrity': {
            'trace_sha256_matches_manifest': sha256(trace_path) == manifest['trace_sha256'],
            'prediction_sha256_matches_manifest': sha256(predictions_path) == manifest['prediction_sha256'],
            'csv_matches_trace_emissions': csv_rows == trace_as_csv,
            'manifest_image_count_matches_trace': manifest['images'] == len(images),
            'manifest_prediction_count_matches_trace': manifest['predictions'] == len(emissions),
            'source_hashes': source_hashes,
            'unmatched_emission_count': len(unmatched),
        },
        'images': {
            'total': len(images), 'status_counts': dict(Counter(image['status'] for image in images)),
            'with_detector_boxes': sum(bool(image.get('detector_boxes_before_geometry_filter')) for image in images),
            'with_geometry_accepted_boxes': sum(any(box['legacy_geometry_pass'] for box in image.get('detector_boxes_before_geometry_filter', [])) for image in images),
            'with_any_geometry_rejected_box': sum(any(not box['legacy_geometry_pass'] for box in image.get('detector_boxes_before_geometry_filter', [])) for image in images),
            'all_detector_boxes_geometry_rejected': sum(bool(image.get('detector_boxes_before_geometry_filter')) and not any(box['legacy_geometry_pass'] for box in image['detector_boxes_before_geometry_filter']) for image in images),
            'with_emissions': sum(bool(image.get('emissions')) for image in images),
            'ocr_observations_per_image': dict(sorted(Counter(len(image.get('observations', [])) for image in images).items())),
            'emissions_per_image': dict(sorted(Counter(len(image.get('emissions', [])) for image in images).items())),
        },
        'stages': {
            'detector_boxes_before_geometry_filter': len(boxes),
            'geometry_accepted_boxes': sum(box['legacy_geometry_pass'] for box in boxes),
            'geometry_rejected_boxes': sum(not box['legacy_geometry_pass'] for box in boxes),
            'geometry_rejection_reasons_nonexclusive': dict(reasons),
            'ocr_observations': len(observations), 'empty_raw_ocr': sum(not obs['raw_ocr'] for obs in observations),
            'emissions': len(emissions), 'observations_without_emission': len(no_emission),
            'emissions_containing_hash': sum('#' in item['plate_num'] for item in emissions),
        },
        'text_changes': {
            'note': 'Different text does not imply improved or worsened accuracy. OCR-stage text precedes tracker normalization; final text is the exported plate_num.',
            'raw_to_ocr_stage_changed_all_observations': sum(obs['raw_ocr'] != obs['postprocessed_ocr'] for obs in observations),
            'raw_to_final_changed_emissions': sum(item['raw_ocr'] != item['final_text'] for item in matched),
            'ocr_stage_to_final_changed_emissions': sum(item['ocr_stage_text'] != item['final_text'] for item in matched),
            'matched_emission_count': len(matched),
            'changed_emission_examples': [item for item in matched if item['raw_ocr'] != item['final_text']],
            'observations_without_emission': no_emission,
        },
        'unmatched_emissions': unmatched,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT / 'reports/baseline_predictions.trace.jsonl')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'reports/baseline_predictions.manifest.json')
    parser.add_argument('--predictions', type=Path, default=ROOT / 'reports/baseline_predictions.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/baseline_stage_counts.json')
    args = parser.parse_args(argv)
    result = analyze(args.trace.resolve(), args.manifest.resolve(), args.predictions.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['images', 'stages']}, ensure_ascii=False, indent=2))
    print(f'Saved: {args.output}')
    checks = result['integrity']
    failed = any(checks[key] is not True for key in ['trace_sha256_matches_manifest', 'prediction_sha256_matches_manifest',
                 'csv_matches_trace_emissions', 'manifest_image_count_matches_trace', 'manifest_prediction_count_matches_trace'])
    failed = failed or any(not value['matches'] for value in checks['source_hashes'].values()) or bool(checks['unmatched_emission_count'])
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
