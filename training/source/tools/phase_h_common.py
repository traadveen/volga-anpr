"""Phase H split guards and shared BEST-G representation. No debug-data imports."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
import cv2
import numpy as np
from src.inference_io import file_sha256
from src.multiline_g import ConfigG, candidates, prepare_g, candidate_score
from src.image_anpr_g import gray_for_ocr
from src.image_anpr import current_postprocess
from src.geometry_f import canonical_rectify, validate_quad
from src.real_data_g import valid_complete_plate
from src.plate_formats import valid_plate, yellow_postprocess

REPORT = ROOT/'reports/phase_h'
CONFIG = ROOT/'configs/phase_h'
MODELS = ROOT/'models/phase_h'
SPLIT = ROOT.parent/'datasets/reports/data3_2026_10_18/frozen_v1/manifest.json'
D_WEIGHTS = ROOT/'models/phase_d/exp_d_01_finetune/best.pt'
SEED = 20261018


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def yes(value):
    return value is True or str(value).lower() in ('true','1')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def guard_check():
    guard = read(REPORT/'split_lock.json')
    for path,sha in guard['hashes'].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError('Frozen input changed: '+path)
    return guard


def load_split(name):
    if name not in ('train','val','untouched_test'):
        raise ValueError('Only named frozen memberships are allowed')
    guard_check()
    if name == 'untouched_test':
        final = read(CONFIG/'final_h_freeze.json')
        if final['status'] != 'FINAL-H-FROZEN' or final['split_manifest_sha256'] != file_sha256(SPLIT):
            raise ValueError('Independent test requires a valid FINAL-H freeze')
        for path,sha in final['hashes'].items():
            if file_sha256(Path(path)) != sha:
                raise ValueError('FINAL-H changed before test')
    entry = read(SPLIT)['splits'][name]
    if file_sha256(Path(entry['path'])) != entry['sha256']:
        raise ValueError('Membership checksum mismatch')
    rows = read_rows(entry['path'])
    if len(rows) != entry['records']:
        raise ValueError('Membership count mismatch')
    return rows


def image(path, expected_sha256=None):
    encoded = np.fromfile(path,dtype=np.uint8)
    if expected_sha256 and hashlib.sha256(encoded.tobytes()).hexdigest() != expected_sha256:
        raise ValueError('Image changed after split freeze: '+str(path))
    data = cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError('Unreadable image: '+str(path))
    return data


def bbox(row):
    value = row.get('bbox_xyxy') or row['bbox']
    return json.loads(value) if isinstance(value,str) else value


def target_choices(frame, row, config, observed_quad=None):
    """Known GT type for OCR-only, shared label-free split/stitch and normalization.

    Observed quad is never called GT. Without a reliable quad, use the verified
    bbox crop; no orientation is manufactured from that axis-aligned annotation.
    """
    kind = row['plate_type']
    if observed_quad is not None:
        crop,_ = canonical_rectify(frame,observed_quad,kind,'observed_frozen_detector_not_GT')
    else:
        a,b,c,d = bbox(row)
        h,w = frame.shape[:2]
        crop = frame[max(0,int(b)):min(h,int(np.ceil(d))),max(0,int(a)):min(w,int(np.ceil(c)))]
        if not crop.size:
            raise ValueError('Empty verified bbox crop')
    gray = gray_for_ocr(crop)
    cfg = ConfigG(**config['split'])
    return candidates(gray,kind,cfg) if config['multi'] else [prepare_g(gray,kind,cfg)]


def decode_choices(ocr, choices):
    decoded = ocr.recognize_candidates([c[0] for c in choices])
    scores = [candidate_score(v[0],v[3],c[2]['structural_confidence'],valid_complete_plate(v[0]))
              for c,v in zip(choices,decoded)]
    index = max(range(len(scores)),key=lambda i:scores[i])
    return decoded[index],choices[index][2],index


def postprocess(raw, probs, kind, profile):
    if profile == 'russian_v1' and kind == 'type1b':
        return yellow_postprocess(raw,probs)[0]
    return current_postprocess(raw,probs)


def audit():
    REPORT.mkdir(parents=True,exist_ok=True)
    CONFIG.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)
    if (REPORT/'split_lock.json').exists():
        guard_check()
        return read(REPORT/'data_usable_summary.json')
    manifest = read(SPLIT)
    hashes = {str(SPLIT):file_sha256(SPLIT),**manifest['frozen_asset_hashes']}
    # Test membership is hashed as opaque bytes only; no test rows/images read.
    for entry in manifest['splits'].values():
        if file_sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Frozen split changed')
        hashes[entry['path']] = entry['sha256']
    for path in [ROOT/'configs/phase_g/best_g.json',D_WEIGHTS,
                 ROOT.parent/'datasets/reports/frozen_best_g_lock.json']:
        if path.exists():
            hashes[str(path)] = file_sha256(path)
    result = {}
    issues = []
    for name in ('train','val'):
        rows = read_rows(manifest['splits'][name]['path'])
        counts = Counter()
        files = {}
        for r in rows:
            counts[r['plate_type']] += 1
            for flag in ('ocr_usable','detector_usable','bbox_verified'):
                counts[flag] += yes(r.get(flag))
            quad = r.get('quad')
            reliable_quad = False
            if quad and quad != 'null':
                counts['quad_or_polygon_available'] += 1
                try:
                    validate_quad(json.loads(quad) if isinstance(quad,str) else quad)
                    reliable_quad = yes(r.get('quad_GT_verified'))
                except (ValueError,TypeError):
                    pass
            counts['verified_quad_obb_usable'] += reliable_quad and yes(r.get('detector_usable'))
            if not yes(r.get('verified')) or not valid_plate(r['plate_num'],r['plate_type']):
                issues.append({'review_id':r['review_id'],'code':'invalid_human_full_label'})
            for key,sha_key in [('image','source_sha256'),('crop','crop_sha256')]:
                files[r[key]] = r[sha_key]
        for path,sha in files.items():
            if file_sha256(Path(path)) != sha:
                issues.append({'path':path,'code':'file_hash_mismatch'})
        result[name] = {'objects':len(rows),**{k:int(counts[k]) for k in
            ('ocr_usable','detector_usable','bbox_verified','quad_or_polygon_available',
             'verified_quad_obb_usable','type1','type1a','type1b')},'files_rehashed':len(files)}
    if issues:
        write(REPORT/'data_usable_issues.json',issues)
        raise ValueError('Training-label audit failed')
    write(REPORT/'split_lock.json',{'created_at':now(),'hashes':hashes,
          'test_access':'Only opaque membership checksum; no test samples read',
          'membership_or_GT_changed':False})
    result.update(created_at=now(),issues=issues,detector_finetuning_allowed=False,
                  detector_decision='No verified quad/polygon/OBB. Bbox annotations alone do not authorize OBB training.',
                  authorized_by='Explicit user PHASE H request; only frozen train, selection only real val',
                  split_manifest_sha256=file_sha256(SPLIT),test_read=False)
    write(REPORT/'data_usable_summary.json',result)
    return result
