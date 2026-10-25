"""Local integrity checks, content fingerprints and task coverage. No model imports."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image

from .schema import SPLITS, TARGETS, VERSION, read_metadata, resolve_image, sha256


def fingerprint(path: Path) -> dict:
    with Image.open(path) as opened:
        opened.load()
        rgb = opened.convert('RGB')
        pixels = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode() + rgb.tobytes()).hexdigest()
        return {'sha256': sha256(path), 'pixel_sha256': pixels, 'width': rgb.width,
                'height': rgb.height, 'bytes': path.stat().st_size}


def excluded_hashes(manifests) -> set[str]:
    hashes = set()
    for path in manifests:
        manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        entries = manifest.get('files', manifest.get('images', []))
        if not isinstance(entries, list):
            raise ValueError(f'Invalid exclusion manifest: {path}')
        hashes.update(item['sha256'] for item in entries if 'sha256' in item)
    return hashes


def validate_dataset(root: Path, meta: Path | None = None, check_splits=True, excluded=(), frozen_images=None, *, format_profile='legacy_v2') -> dict:
    root = root.resolve()
    meta = meta or root / 'meta.csv'
    records, errors = read_metadata(meta, format_profile=format_profile)
    warnings = []
    images = {}
    seen, by_image = set(), defaultdict(list)
    blocked = excluded_hashes(excluded)

    def issue(destination, code, image='', message='', **extra):
        destination.append(dict(code=code, image=image, message=message, **extra))

    for record in records:
        key = tuple(record.row().items())
        if key in seen:
            issue(errors, 'duplicate_metadata', record.image, 'Identical metadata row')
        seen.add(key)
        by_image[record.image].append(record)
        prefix = 'images/synthetic/' if record.is_synthetic else 'images/real/'
        if not record.image.startswith(prefix):
            issue(errors, 'wrong_image_partition', record.image, f'Expected {prefix}')
        if record.annotation_kind in ('ocr_crop', 'detection_only'):
            issue(warnings, 'incomplete_annotation', record.image,
                  f'{record.annotation_kind} is not full-frame end-to-end ground truth')
        if '#' in record.plate_num and not record.group_id:
            issue(warnings, 'partial_identity', record.image,
                  'Unknown symbols are not an identity; add group_id for related frames')
        if record.license_review:
            try:
                review_path = resolve_image(root, record.license_review)
                review = json.loads(review_path.read_text(encoding='utf-8'))
                if review.get('license') != record.license or review.get('source') != record.source or review.get('approved') is not True:
                    raise ValueError('License review does not match annotation')
                evidence = resolve_image(root, review['evidence_file'])
                if sha256(evidence) != review['evidence_sha256']:
                    raise ValueError('License evidence hash mismatch')
            except (OSError, ValueError, KeyError, TypeError) as error:
                issue(errors, 'invalid_license_review', record.image, str(error))
        elif not record.is_synthetic:
            issue(warnings, 'missing_license_review', record.image,
                  'Source/license declared, but no linked evidence; import tooling requires review')
        if record.image not in images:
            try:
                image = resolve_image(root, record.image)
                if frozen_images is None:
                    images[record.image] = fingerprint(image)
                else:
                    # Reuse decoded pixel identity only when current file bytes
                    # exactly match a previously fully validated manifest.
                    previous = frozen_images[record.image]
                    if sha256(image) != previous['sha256'] or image.stat().st_size != previous['bytes']:
                        raise ValueError('Frozen image bytes changed')
                    images[record.image] = previous.copy()
                if images[record.image]['sha256'] in blocked:
                    issue(errors, 'excluded_evaluation_image', record.image, 'Image belongs to a protected evaluation package')
            except (OSError, ValueError, KeyError, Image.DecompressionBombError) as error:
                issue(errors, 'unreadable_image', record.image, str(error))
                images[record.image] = None
        shape = images[record.image]
        if shape:
            width, height = shape['width'], shape['height']
            if record.bbox:
                x, y, w, h = record.bbox
                if x+w > width + 1e-6 or y+h > height + 1e-6:
                    issue(errors, 'bbox_outside_image', record.image, 'bbox exceeds stored image dimensions')
            if record.quad:
                if any(x < 0 or y < 0 or x > width or y > height for x, y in record.quad):
                    issue(errors, 'quad_outside_image', record.image, 'quad exceeds stored image dimensions')
                if record.bbox:
                    x, y, w, h = record.bbox
                    if any(px < x-1e-6 or py < y-1e-6 or px > x+w+1e-6 or py > y+h+1e-6 for px, py in record.quad):
                        issue(errors, 'quad_outside_bbox', record.image, 'bbox does not enclose quad')
    for image, rows in by_image.items():
        if len(rows) > 1 and any(row.annotation_kind == 'negative' for row in rows):
            issue(errors, 'contradictory_negative', image, 'Negative image also has objects')
        if len({(row.source, row.license, row.is_synthetic, row.source_split) for row in rows}) > 1:
            issue(errors, 'inconsistent_image_metadata', image, 'Source/license/partition/split disagree within image')
        objects = Counter((r.bbox, r.quad) for r in rows if r.annotation_kind != 'negative')
        if any(n > 1 for n in objects.values()):
            issue(errors, 'duplicate_object_geometry', image, 'Same geometry annotated more than once')
    digest_groups = defaultdict(list)
    for image, info in images.items():
        if info:
            digest_groups[info['pixel_sha256']].append(image)
    duplicates = [sorted(names) for names in digest_groups.values() if len(names) > 1]
    for names in duplicates:
        issue(warnings, 'duplicate_image_content', message='Identical decoded RGB pixels', images=names)

    split_info = {}
    split_rows = []
    if check_splits:
        membership = defaultdict(set)
        for split in SPLITS:
            path = root / 'splits' / f'{split}.csv'
            if not path.exists():
                issue(errors, 'missing_split', message=str(path))
                continue
            subset, subset_errors = read_metadata(path, format_profile=format_profile)
            errors.extend({**error, 'split': split} for error in subset_errors)
            split_info[split] = {'rows': len(subset), 'images': len({r.image for r in subset}), 'sha256': sha256(path)}
            for record in subset:
                if tuple(record.row().items()) not in seen:
                    issue(errors, 'split_row_not_in_meta', record.image, split)
                if record.source_split and record.source_split != split:
                    issue(errors, 'source_split_changed', record.image, f'Protected {record.source_split} moved into {split}')
                membership[record.image].add(split)
                split_rows.append(record)
        if Counter(tuple(r.row().items()) for r in split_rows) != Counter(tuple(r.row().items()) for r in records):
            issue(errors, 'split_coverage', message='Split row multiset differs from meta.csv')
        for image, names in membership.items():
            if len(names) > 1:
                issue(errors, 'image_leakage', image, ','.join(sorted(names)))
        identity_splits = defaultdict(set)
        for record in records:
            identities = []
            if record.plate_num and '#' not in record.plate_num:
                identities.append('plate:' + record.plate_num)
            if record.group_id:
                identities.append('group:' + record.group_id)
            info = images.get(record.image)
            if info:
                identities.append('pixels:' + info['pixel_sha256'])
            for key in identities:
                identity_splits[key].update(membership[record.image])
        for key, names in identity_splits.items():
            if len(names) > 1:
                issue(errors, 'identity_leakage', message=f'{key} crosses {sorted(names)}')
        manifest_path = root / 'splits' / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('format_profile', 'legacy_v2') != format_profile:
                issue(errors, 'split_format_profile_mismatch', message='Split format profile differs from requested validation profile')
            if manifest.get('meta_sha256') != sha256(meta):
                issue(errors, 'stale_split_manifest', message='meta.csv hash changed')
            for split, info in split_info.items():
                if manifest.get('splits', {}).get(split, {}).get('sha256') != info['sha256']:
                    issue(errors, 'stale_split_manifest', message=f'{split}.csv hash changed')
            if manifest.get('image_manifest') != {name: info for name, info in sorted(images.items()) if info}:
                issue(errors, 'stale_image_manifest', message='Image content/shape/bytes changed after split')
        else:
            issue(warnings, 'missing_split_manifest', message='Run split_dataset to create provenance')
    type_counts = Counter(r.plate_type for r in records if r.plate_type)
    unique = {t: len({r.plate_num for r in records if r.plate_type == t and r.plate_num and '#' not in r.plate_num}) for t in TARGETS}
    full = sum(r.annotation_kind == 'full' for r in records)
    if not records:
        issue(warnings, 'empty_dataset', message='Infrastructure only; no training or accuracy coverage')
    return {'schema_version': VERSION, 'format_profile': format_profile, 'valid': not errors, 'ready_for_e2e_evaluation': not errors and full > 0,
            'dataset_root': str(root), 'meta_sha256': sha256(meta), 'counts': {'rows': len(records),
            'images': len(images), 'by_type': dict(type_counts), 'unique_complete_plates': unique,
            'annotation_kinds': dict(Counter(r.annotation_kind for r in records)),
            'synthetic_images': len({r.image for r in records if r.is_synthetic}), 'full_rows': full},
            'coverage': {'missing_target_types': [t for t in TARGETS if not type_counts[t]],
                         'condition_counts': dict(Counter(tag for r in records for tag in r.conditions))},
            'images': {name: info for name, info in sorted(images.items()) if info},
            'duplicate_content_groups': duplicates, 'splits': split_info, 'errors': errors, 'warnings': warnings}
