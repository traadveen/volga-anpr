"""Canonical, lossless dataset metadata and strict label/geometry parsing."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

VERSION = '2.0.0'
FIELDS = ('image', 'plate_num', 'plate_type', 'bbox', 'quad', 'is_vehicle',
          'is_synthetic', 'source', 'license', 'conditions')
EXTRA_FIELDS = ('annotation_kind', 'source_split', 'group_id', 'license_review')
TYPES = ('type1', 'type1a', 'type1b', 'other')
TARGETS = TYPES[:3]
KINDS = ('full', 'ocr_crop', 'detection_only', 'negative')
SPLITS = ('train', 'val', 'test')
ALPHABET = 'ABEKMHOPCTYX'
CYR_MAP = str.maketrans('АВЕКМНОРСТУХавекмнорстух', ALPHABET + ALPHABET)
DATA_LICENSES = ('CC-BY-4.0', 'CC0-1.0')
FORMAT_PROFILES = ('legacy_v2', 'russian_v1')


def normalize_plate(text: str) -> str:
    """Only transliterate case/lookalikes. Never remove characters or repair digits."""
    return text.translate(CYR_MAP).upper()


def plate_error(text: str) -> str | None:
    if len(text) not in (8, 9):
        return 'Expected LDDDLLRR or LDDDLLRRR (8 or 9 characters)'
    for char, kind in zip(text, 'LDDDLL' + 'D' * (len(text) - 6)):
        if char != '#' and char not in (ALPHABET if kind == 'L' else '0123456789'):
            return 'Invalid ASCII alphabet or positional mask'
    if len(text) == 9 and text[6] not in '127#':
        return 'First digit of a three-digit region must be 1, 2, 7 or #'
    return None


def current_plate_error(text: str, plate_type: str, *, partial=True) -> str | None:
    """Dataset identity policy; inference retains its independent mask contract."""
    from src.plate_formats import valid_plate
    if not valid_plate(text,plate_type,partial=partial):
        return 'Invalid ASCII alphabet or type-specific positional mask'
    if text[5 if plate_type=='type1b' else 6:] in ('00','000'):
        return 'Region 00/000 is invalid'
    if set(text)=={'#'}:
        return 'All plate positions are unknown'
    return None


def relative_path(value: str) -> str:
    if '\\' in value or ':' in value or '\x00' in value:
        raise ValueError('Use a relative POSIX path without drive or NUL')
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or str(path) != value or path.name in ('', '.'):
        raise ValueError('Invalid relative image path')
    return value


def resolve_image(root: Path, image: str) -> Path:
    path = (root.resolve() / relative_path(image)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Image resolves outside dataset root')
    return path


def number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Geometry requires finite JSON numbers')
    return float(value)


def box(value: str) -> tuple[float, ...] | None:
    if not value:
        return None
    raw = json.loads(value)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError('bbox must be JSON [x,y,width,height]')
    result = tuple(number(x) for x in raw)
    x, y, w, h = result
    if x < 0 or y < 0 or w <= 0 or h <= 0 or not math.isfinite(w * h) or w * h == 0:
        raise ValueError('bbox must have nonnegative origin and positive finite area')
    if not math.isfinite(x + w) or not math.isfinite(y + h):
        raise ValueError('bbox endpoints overflow')
    return result


def quad(value: str) -> tuple[tuple[float, float], ...] | None:
    if not value:
        return None
    raw = json.loads(value)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError('quad must contain four [x,y] points')
    points = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('quad must contain four [x,y] points')
        points.append(tuple(number(x) for x in point))
    # Positive cross-products are clockwise in image coordinates (y increases down).
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        cross = (b[0]-a[0]) * (c[1]-b[1]) - (b[1]-a[1]) * (c[0]-b[0])
        if not math.isfinite(cross) or cross <= 0:
            raise ValueError('quad must be strictly convex, clockwise and nondegenerate')
    # Image-coordinate convention: TL is the smallest x+y, then y/x as ties.
    first = min(range(4), key=lambda i: (sum(points[i]), points[i][1], points[i][0]))
    if first != 0:
        raise ValueError('quad must start at top-left (minimum x+y)')
    return tuple(points)


def flag(value: str, allow_unknown=False) -> bool | None:
    if value == '1':
        return True
    if value == '0':
        return False
    if allow_unknown and value == '':
        return None
    raise ValueError('Boolean must be 0 or 1' + ('; empty is explicitly unknown' if allow_unknown else ''))


@dataclass(frozen=True)
class Annotation:
    image: str
    plate_num: str
    plate_type: str
    bbox: tuple[float, ...] | None
    quad: tuple[tuple[float, float], ...] | None
    is_vehicle: bool | None
    is_synthetic: bool
    source: str
    license: str
    conditions: tuple[str, ...] = ()
    annotation_kind: str = 'full'
    source_split: str = ''
    group_id: str = ''
    license_review: str = ''

    def row(self) -> dict[str, str]:
        result = {name: getattr(self, name) for name in (*FIELDS, *EXTRA_FIELDS)}
        result['bbox'] = json.dumps(self.bbox, separators=(',', ':')) if self.bbox else ''
        result['quad'] = json.dumps(self.quad, separators=(',', ':')) if self.quad else ''
        result['is_vehicle'] = '' if self.is_vehicle is None else str(int(self.is_vehicle))
        result['is_synthetic'] = str(int(self.is_synthetic))
        result['conditions'] = '|'.join(self.conditions)
        return result


def parse_row(row: dict[str, str], *, format_profile='legacy_v2', allow_legacy_zero_region=False) -> Annotation:
    if format_profile not in FORMAT_PROFILES:
        raise ValueError('Unknown format_profile')
    kind = row.get('annotation_kind', '') or 'full'
    if kind not in KINDS:
        raise ValueError('Unknown annotation_kind')
    ptype, text = row['plate_type'], row['plate_num']
    if ptype not in TYPES and not (kind in ('detection_only', 'negative') and ptype == ''):
        raise ValueError('Unknown plate_type')
    bounds, corners = box(row['bbox']), quad(row['quad'])
    vehicle = flag(row['is_vehicle'], allow_unknown=kind in ('ocr_crop', 'detection_only'))
    synthetic = flag(row['is_synthetic'])
    if not row['source'].strip() or row['source'] != row['source'].strip():
        raise ValueError('source must be nonempty without surrounding whitespace')
    if row['license'] not in DATA_LICENSES:
        raise ValueError('Unknown/unapproved data license; requires explicit review before import')
    if kind in ('full', 'ocr_crop') and ptype in TARGETS:
        if format_profile == 'russian_v1':
            error = current_plate_error(text, ptype)
            # Raw archives may retain these rows solely for whole-frame quarantine.
            # Normal validation and all new generated datasets keep the strict default.
            if allow_legacy_zero_region is True and error=='Region 00/000 is invalid':
                error=None
        else:
            error = plate_error(text)
        if error:
            raise ValueError(error)
    if kind != 'negative' and bounds is None:
        raise ValueError('Object annotations require bbox (quad alone does not replace bbox)')
    if kind == 'negative' and (text or ptype or bounds or corners or vehicle is not False):
        raise ValueError('Negative image requires empty object fields and is_vehicle=0')
    if kind == 'detection_only' and text:
        raise ValueError('detection_only cannot claim OCR ground truth')
    if kind == 'ocr_crop' and ptype not in TARGETS:
        raise ValueError('ocr_crop requires an explicitly reviewed target type')
    split = row.get('source_split', '')
    if split and split not in SPLITS:
        raise ValueError('source_split must be train/val/test or empty')
    tags = row['conditions'].split('|') if row['conditions'] else []
    if any(not tag or tag != tag.strip() for tag in tags) or len(tags) != len(set(tags)):
        raise ValueError('conditions must contain distinct nonempty | separated tags')
    return Annotation(relative_path(row['image']), text, ptype, bounds, corners, vehicle,
                      synthetic, row['source'], row['license'], tuple(sorted(tags)), kind,
                      split, row.get('group_id', ''), row.get('license_review', ''))


def read_metadata(path: Path, *, format_profile='legacy_v2', allow_legacy_zero_region=False) -> tuple[list[Annotation], list[dict]]:
    if format_profile not in FORMAT_PROFILES:
        raise ValueError('Unknown format_profile')
    annotations, issues = [], []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';')
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(FIELDS) - set(fields) or set(fields) - set((*FIELDS, *EXTRA_FIELDS)):
            raise ValueError('Invalid meta.csv header; required canonical columns, optional documented extensions only')
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('Malformed CSV row')
                annotations.append(parse_row(row, format_profile=format_profile,
                                             allow_legacy_zero_region=allow_legacy_zero_region))
            except (ValueError, TypeError) as error:
                issues.append({'code': 'invalid_metadata', 'line': reader.line_num,
                               'image': row.get('image', ''), 'message': str(error)})
    return annotations, issues


def write_metadata(path: Path, annotations) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(*FIELDS, *EXTRA_FIELDS), delimiter=';', lineterminator='\n')
        writer.writeheader()
        writer.writerows(annotation.row() for annotation in annotations)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
