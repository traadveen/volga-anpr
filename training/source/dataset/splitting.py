"""Image/plate/content-connected splits, preserving external source partitions."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .schema import SPLITS, VERSION, read_metadata, sha256, write_metadata
from .validation import validate_dataset


def connected_groups(records, image_manifest):
    images = sorted({record.image for record in records})
    parent = {image: image for image in images}

    def find(image):
        while parent[image] != image:
            parent[image] = parent[parent[image]]
            image = parent[image]
        return image

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    identities = {}
    for record in records:
        keys = ['pixels:' + image_manifest[record.image]['pixel_sha256']]
        if record.plate_num and '#' not in record.plate_num:
            keys.append('plate:' + record.plate_num)
        if record.group_id:
            keys.append('group:' + record.group_id)
        for key in keys:
            if key in identities:
                union(record.image, identities[key])
            else:
                identities[key] = record.image
    groups = defaultdict(list)
    for record in records:
        groups[find(record.image)].append(record)
    return [sorted(group, key=lambda r: json.dumps(r.row(), sort_keys=True)) for _, group in sorted(groups.items())]


def make_split(root: Path, seed=42, ratios=(.7, .15, .15), excluded=(), *, format_profile='legacy_v2') -> dict:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if len(ratios) != 3 or any(not isinstance(v, (float, int)) or isinstance(v, bool) or not 0 < v < 1 for v in ratios) or abs(sum(ratios)-1) > 1e-9:
        raise ValueError('Three finite positive ratios summing to one are required')
    root = root.resolve()
    validation = validate_dataset(root, check_splits=False, excluded=excluded, format_profile=format_profile)
    if not validation['valid']:
        raise ValueError('Dataset invalid: ' + '; '.join(e['code'] for e in validation['errors'][:8]))
    records, _ = read_metadata(root / 'meta.csv', format_profile=format_profile)
    groups = connected_groups(records, validation['images'])

    def features(group):
        counts = Counter({'images': len({r.image for r in group})})
        for record in group:
            counts['kind:' + record.annotation_kind] += 1
            if record.plate_type:
                counts['type:' + record.plate_type] += 1
            for tag in record.conditions:
                counts['condition:' + tag] += 1
        return counts

    totals = sum((features(group) for group in groups), Counter())
    assigned, counts = {name: [] for name in SPLITS}, {name: Counter() for name in SPLITS}
    free = []
    assignments = []
    for group in groups:
        protected = {r.source_split for r in group if r.source_split}
        if len(protected) > 1:
            raise ValueError('Source-split leakage: a connected plate/image/content group spans ' + ','.join(sorted(protected)))
        if protected:
            split = next(iter(protected))
            assigned[split].extend(group)
            counts[split].update(features(group))
            assignments.append({'images': sorted({r.image for r in group}), 'split': split, 'protected': True})
        else:
            free.append(group)
    # The seed determines only stable tie/order keys, independent of CSV input order.
    def tie_key(group):
        names = '|'.join(sorted({r.image for r in group}))
        return hashlib.sha256(f'{seed}:{names}'.encode()).hexdigest()

    free.sort(key=lambda group: (-len({r.image for r in group}), -len(group), tie_key(group)))
    for group in free:
        delta = features(group)

        def cost(index):
            name = SPLITS[index]
            result = 0.
            for feature, amount in delta.items():
                target = totals[feature] * ratios[index]
                before = counts[name][feature] - target
                result += ((before+amount)**2 - before**2) / max(1., target)
            return result

        chosen = min(range(3), key=lambda index: (cost(index), index))
        split = SPLITS[chosen]
        assigned[split].extend(group)
        counts[split].update(delta)
        assignments.append({'images': sorted({r.image for r in group}), 'split': split, 'protected': False})
    manifest = {'schema_version': VERSION, 'format_profile': format_profile, 'seed': seed, 'requested_ratios': dict(zip(SPLITS, ratios)),
                'algorithm': 'connected components (image/complete plate/group_id/RGB pixel hash), protected source split, deterministic greedy feature balance v1',
                'meta_sha256': sha256(root / 'meta.csv'), 'image_manifest': validation['images'],
                'groups': sorted(assignments, key=lambda item: item['images']), 'splits': {},
                'warnings': validation['warnings'], 'training_data_present': bool(records),
                'partial_plate_policy': '# labels are not identities; related partial labels require group_id',
                'protected_source_split_policy': 'Never automatically mix or reassign external train/val/test'}
    if len(groups) < 3 and records:
        manifest['warnings'].append({'code': 'too_few_groups', 'message': 'Fewer than three connected groups; all three nonempty splits cannot be guaranteed'})
    manifest['excluded_manifests'] = [{'sha256': sha256(Path(path)), 'path': str(Path(path).resolve())} for path in excluded]
    for split in SPLITS:
        assigned[split].sort(key=lambda r: json.dumps(r.row(), sort_keys=True))
        manifest['splits'][split] = {'rows': len(assigned[split]), 'images': len({r.image for r in assigned[split]}),
                                    'features': dict(sorted(counts[split].items()))}
    destination = root / 'splits'
    destination.mkdir(exist_ok=True)
    # Manifest is committed last; interruption is detectable by hash validation.
    with tempfile.TemporaryDirectory(prefix='.split-', dir=root) as temporary:
        stage = Path(temporary)
        for split in SPLITS:
            write_metadata(stage / f'{split}.csv', assigned[split])
            manifest['splits'][split]['sha256'] = sha256(stage / f'{split}.csv')
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        for name in [*(f'{split}.csv' for split in SPLITS), 'manifest.json']:
            os.replace(stage / name, destination / name)
    result = validate_dataset(root, excluded=excluded, format_profile=format_profile)
    if not result['valid']:
        raise RuntimeError('Split failed its own integrity check: ' + str(result['errors']))
    return manifest
