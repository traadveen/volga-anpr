"""Reuse verified train/val boxes without inventing verified plate orientation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
from collections import Counter
from tools.phase_i_review import Review, DEFAULT, atomic_json, geometry
from src.inference_io import file_sha256
from tools.phase_h_common import now


def build(review, statement):
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError('An explicit human bbox confirmation is required')
    review.guard.verify()
    state = review.state()
    rows = []
    for row in review.rows:
        if row.get('bbox_verified') not in (True, 'True'):
            raise ValueError('Unverified source bbox: ' + row['review_id'])
        if file_sha256(Path(row['image'])) != row['source_sha256']:
            raise ValueError('Source bytes changed: ' + row['review_id'])
        x1, y1, x2, y2 = row['bbox']
        corners, _, _ = geometry([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], row['image_size'])
        previous = state['decisions'].get(row['review_id'], {})
        verified = previous.get('quad_verified') is True
        rows.append({
            **{k:copy.deepcopy(row[k]) for k in ('review_id','split','source_sha256','bbox','image_size')},
            'bbox_verified':True,
            'quad':copy.deepcopy(previous['quad']) if verified else corners,
            'quad_verified':verified,
            'quad_source':previous['quad_source'] if verified else 'verified_bbox_corner_draft',
            'obb_verified':bool(previous.get('obb_verified')) if verified else False,
            'training_region_complete':bool(previous.get('training_region_complete')) if verified else False,
            'previous_review_status':previous.get('status'),
            'previous_operation_id':previous.get('operation_id'),
        })
    review.guard.verify()
    if review.state()['revision'] != state['revision']:
        raise ValueError('Human review advanced during export; retry without overwriting it')
    return {
        'created_at':now(), 'scope':'frozen train/val only',
        'human_bbox_statement':statement.strip(),
        'confirmation_scope':'bbox only; no confirmation of plate orientation or full-scene annotation completeness',
        'index_sha256':review.index_sha, 'geometry_state_revision':state['revision'],
        'counts':{'bbox_verified':len(rows),
                  'preserved_human_quad':sum(r['quad_verified'] for r in rows),
                  'bbox_corner_drafts':sum(not r['quad_verified'] for r in rows),
                  'per_split':dict(Counter(r['split'] for r in rows))},
        'training_started':False, 'test_images_read':False, 'records':rows,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statement', required=True)
    args = parser.parse_args()
    review = Review()
    with review.transaction():
        result = build(review, args.statement)
        target = DEFAULT / 'bbox_geometry_drafts.json'
        if target.exists():
            raise ValueError('Export already exists; preserve it before generating a new snapshot')
        atomic_json(target, result)
    print(result['counts'])
