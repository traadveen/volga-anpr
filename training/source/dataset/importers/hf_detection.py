"""YOLO normalized boxes -> pixel boxes, with missing text/context preserved."""
import math
from pathlib import Path

from ..schema import Annotation, TYPES
from ..validation import fingerprint
from .hf_ocr import EXTENSIONS

SOURCE_ID = 'AY000554/Car_plate_detecting_dataset'


def prepare(source_root: Path, splits, review: dict, class_map: dict):
    if not class_map or any(not key.isdecimal() or value not in (*TYPES, '') for key, value in class_map.items()):
        raise ValueError('Explicit class map {"0":""} (unknown type) or target/other type is required')
    pairs = []
    for split in splits:
        images, labels = source_root / split / 'images', source_root / split / 'labels'
        if not images.is_dir() or not labels.is_dir():
            raise ValueError(f'Expected extracted {split}/images and {split}/labels')
        used_labels = set()
        for image in sorted(p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS):
            relative = image.relative_to(images)
            label = (labels / relative).with_suffix('.txt')
            if not image.resolve().is_relative_to(source_root.resolve()) or not label.resolve().is_relative_to(source_root.resolve()):
                raise ValueError('Source path resolves outside source-root')
            if not label.is_file():
                raise ValueError(f'Missing YOLO label: {label}; absent file is not a negative example')
            used_labels.add(label)
            shape = fingerprint(image)
            name = f'images/real/hf_detection/{split}/' + relative.as_posix()
            lines = [line for line in label.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            if not lines:
                pairs.append((Annotation(name, '', '', None, None, False, False, review['source'], review['license'], (), 'negative', split), image))
            for line in lines:
                pieces = line.split()
                if len(pieces) != 5 or pieces[0] not in class_map:
                    raise ValueError(f'{label}: unknown class or invalid five-column YOLO box')
                cx, cy, w, h = map(float, pieces[1:])
                if any(not math.isfinite(v) for v in (cx, cy, w, h)) or w <= 0 or h <= 0 or cx-w/2 < 0 or cy-h/2 < 0 or cx+w/2 > 1 or cy+h/2 > 1:
                    raise ValueError(f'{label}: YOLO box outside normalized image bounds')
                bbox = ((cx-w/2)*shape['width'], (cy-h/2)*shape['height'], w*shape['width'], h*shape['height'])
                record = Annotation(name, '', class_map[pieces[0]], bbox, None, None, False,
                                    review['source'], review['license'], (), 'detection_only', split)
                pairs.append((record, image))
        orphaned = set(labels.rglob('*.txt')) - used_labels
        if orphaned:
            raise ValueError(f'{split}: orphaned YOLO labels: {len(orphaned)}')
    return pairs
