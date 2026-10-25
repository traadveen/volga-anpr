"""Run an explicitly supplied, reviewed organizer scorer without modifying it.

Only the four organizer output fields are projected from diagnostic predictions.
No normalization, deduplication, thresholding or label/type correction is applied.
The scorer remains external to the submission dataset and is hashed in provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

COLUMNS = ('image', 'plate_num', 'plate_type', 'confidence')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_predictions(source, destination):
    with Path(source).open(encoding='utf-8-sig', newline='') as src:
        reader = csv.DictReader(src, delimiter=';')
        if not set(COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError('Diagnostic predictions must contain ' + ';'.join(COLUMNS))
        rows = list(reader)
        if any(None in row or any(row.get(c) is None for c in COLUMNS) for row in rows):
            raise ValueError('Malformed prediction row')
    with Path(destination).open('w', encoding='utf-8', newline='') as dst:
        writer = csv.DictWriter(dst, fieldnames=COLUMNS, delimiter=';', lineterminator='\n')
        writer.writeheader()
        writer.writerows({c: row[c] for c in COLUMNS} for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorer', required=True, type=Path, help='Path to the unmodified, reviewed organizer evaluate.py')
    parser.add_argument('--ground-truth', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--output', default=Path('reports/baseline_official'), type=Path)
    args = parser.parse_args(argv)
    if not args.scorer.is_file() or not args.ground_truth.is_file() or not args.images.is_dir():
        parser.error('Scorer, ground-truth file and image directory must exist locally')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    projected = args.output.with_suffix('.input.csv')
    result = args.output.with_suffix('.json')
    details = args.output.with_suffix('.details.csv')
    try:
        project_predictions(args.predictions, projected)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    command = [sys.executable, str(args.scorer.resolve()), '--gt', str(args.ground_truth.resolve()),
               '--pred', str(projected.resolve()), '--images', str(args.images.resolve()),
               '--json', str(result.resolve()), '--details', str(details.resolve()), '--max-list', '10']
    proc = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    args.output.with_suffix('.log.txt').write_text(proc.stdout + proc.stderr, encoding='utf-8')
    provenance = {
        'command': command, 'exit_code': proc.returncode,
        'scorer_sha256': file_hash(args.scorer), 'ground_truth_sha256': file_hash(args.ground_truth),
        'diagnostic_predictions_sha256': file_hash(args.predictions),
        'projected_predictions_sha256': file_hash(projected),
        'projection': 'four original columns, rows and values unchanged',
        'meaning': 'Score of the downloaded public evaluator v1.0; text-similarity recall is not spatial detector recall',
    }
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print(proc.stdout, end='')
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end='')
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
