"""Reviewed source licenses and transactional publication of import collections."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile

from ..schema import DATA_LICENSES, parse_row, resolve_image, sha256, write_metadata
from ..splitting import make_split


def license_review(path: Path, source_id: str) -> tuple[dict, Path]:
    review = json.loads(path.read_text(encoding='utf-8'))
    if review.get('source_id') != source_id or review.get('approved') is not True:
        raise ValueError('Source license review is absent, unapproved or belongs to another dataset')
    if review.get('license') not in DATA_LICENSES or review.get('redistribution_allowed') is not True:
        raise ValueError('License is unknown or incompatible with the data distribution policy')
    for field in ['source', 'revision', 'attribution', 'review_date']:
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f'License review requires {field}')
    evidence = (path.parent / review['evidence_file']).resolve()
    if not evidence.is_file() or sha256(evidence) != review.get('evidence_sha256'):
        raise ValueError('Local license evidence missing or changed')
    if review.get('conflict_detected') and not review.get('conflict_resolution', '').strip():
        raise ValueError('Conflicting license declarations require a recorded resolution')
    return review, evidence


def publish(destination: Path, pairs, review: dict, evidence: Path, excluded=(), seed=42):
    """Publish a new collection only after complete validation; never overwrite datasets."""
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Import destination already exists; select a new version, never overwrite')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.import-', dir=destination.parent) as temporary:
        stage = Path(temporary) / 'collection'
        stage.mkdir()
        (stage / 'labels').mkdir()
        (stage / 'images/synthetic').mkdir(parents=True)
        (stage / 'imports').mkdir()
        shutil.copyfile(evidence, stage / 'imports/license_evidence.txt')
        saved_review = {**review, 'evidence_file': 'imports/license_evidence.txt'}
        (stage / 'imports/license_review.json').write_text(json.dumps(saved_review, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        records, copied = [], {}
        for record, source in pairs:
            # Validate untrusted metadata before any copy/path creation.
            record = parse_row(record.row())
            if record.source != review['source'] or record.license != review['license']:
                raise ValueError('Annotation source/license differs from reviewed source')
            source = source.resolve()
            target = resolve_image(stage, record.image)
            if not record.image.startswith('images/real/') or record.is_synthetic:
                raise ValueError('External import images must use images/real/')
            if record.image in copied and copied[record.image] != source:
                raise ValueError('Colliding destination names with different source paths')
            if record.image not in copied:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                copied[record.image] = source
            records.append(replace(record, license_review='imports/license_review.json'))
        if not records:
            raise ValueError('No images/annotations found; refusing empty external import')
        write_metadata(stage / 'meta.csv', records)
        (stage / 'LICENSE').write_text('Per-record license: '+review['license']+'\nAttribution: '+review['attribution']+
                                     '\nSource: '+review['source']+'\nRetain original attribution/evidence in imports/.\n', encoding='utf-8')
        (stage / 'README.md').write_text('# Imported collection\n\nSource splits are protected. Missing labels/context remain explicitly unknown.\n'
                                       'See imports/license_review.json and splits/manifest.json. This collection is not automatically merged into the competition dataset.\n', encoding='utf-8')
        manifest = make_split(stage, seed=seed, excluded=excluded)
        # No absolute staging path enters the split manifest; original paths remain in provenance.
        provenance = {'source_id': review['source_id'], 'source': review['source'], 'revision': review['revision'],
                      'license': review['license'], 'records': len(records),
                      'images': [{'image': name, 'original_path': str(source), 'sha256': sha256(source)}
                                 for name, source in sorted(copied.items())],
                      'split_sha256': sha256(stage / 'splits/manifest.json'), 'seed': seed}
        (stage / 'imports/provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        os.replace(stage, destination)
    return manifest
