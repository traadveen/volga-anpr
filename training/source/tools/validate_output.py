"""Validate submission CSV bytes and schema without changing predictions."""
from pathlib import Path, PureWindowsPath
import argparse
import csv
import io
import json
import math
import re

COLUMNS = ('image', 'plate_num', 'plate_type', 'confidence')
LETTERS = 'ABEKMHOPCTYX#'
PATTERNS = {
    'type1': re.compile(rf'[{LETTERS}][0-9#]{{3}}[{LETTERS}]{{2}}[0-9#]{{2,3}}\Z'),
    'type1a': re.compile(rf'[{LETTERS}][0-9#]{{3}}[{LETTERS}]{{2}}[0-9#]{{2,3}}\Z'),
    'type1b': re.compile(rf'[{LETTERS}]{{2}}[0-9#]{{5,6}}\Z'),
    'other': re.compile(r'[A-Z0-9#]{0,12}\Z'),
}


def validate_rows(rows, image_names=None):
    errors = []
    warnings = []
    seen = set()
    for line, row in enumerate(rows, 2):
        if tuple(row) != COLUMNS:
            errors.append(f'Line {line}: expected exactly four columns')
            continue
        name = row['image']
        if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or '\\' in name or PureWindowsPath(name).drive:
            errors.append(f'Line {line}: image must be a filename without a path')
        if image_names is not None and name not in image_names:
            errors.append(f'Line {line}: image is absent from the input directory')
        kind, text = row['plate_type'], row['plate_num']
        if kind not in PATTERNS:
            errors.append(f'Line {line}: unsupported plate_type')
        elif not isinstance(text, str) or PATTERNS[kind].fullmatch(text) is None:
            errors.append(f'Line {line}: text does not match the declared Russian format')
        try:
            if isinstance(row['confidence'], bool):
                raise ValueError()
            confidence = float(row['confidence'])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError()
        except (ValueError, TypeError):
            errors.append(f'Line {line}: confidence must be finite and within 0..1')
        identity = tuple(str(row[k]) for k in COLUMNS)
        if identity in seen:
            warnings.append(f'Line {line}: identical prediction row repeated; retained unchanged')
        seen.add(identity)
    return dict(valid=not errors, rows=len(rows), errors=errors, warnings=warnings,
                format_profile='russian_v1', predictions_modified=False)


def validate_file(path, image_names=None):
    try:
        content = Path(path).read_bytes().decode('utf-8')
    except (OSError, UnicodeError) as error:
        return dict(valid=False, rows=0, errors=[str(error)], warnings=[], predictions_modified=False)
    reader = csv.DictReader(io.StringIO(content, newline=''), delimiter=';')
    if tuple(reader.fieldnames or ()) != COLUMNS:
        return dict(valid=False, rows=0, errors=['Header must be image;plate_num;plate_type;confidence'],
                    warnings=[], predictions_modified=False)
    try:
        return validate_rows(list(reader), image_names)
    except csv.Error as error:
        return dict(valid=False, rows=0, errors=[str(error)], warnings=[], predictions_modified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='CSV file')
    parser.add_argument('--images', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    names = {p.name for p in args.images.iterdir() if p.is_file()} if args.images else None
    result = validate_file(args.input, names)
    if args.report:
        if args.report.resolve() == args.input.resolve():
            parser.error('Report must not overwrite predictions')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
