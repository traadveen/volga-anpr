"""Resumable scale-up of the reviewed Phase B renderer and readability gates."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image

from dataset.schema import Annotation, parse_row, read_metadata, sha256, write_metadata
from dataset.splitting import make_split
from dataset.generator.config import load_config
from dataset.generator.generate import plate_number
from dataset.generator.render import ASSETS, render_plate
from dataset.generator.augment import geometry, degrade
from dataset.generator.verify import bank, check_sample
from src.multiline import MultilineConfig
from src.ocr_preprocess import prepare_ocr_input
from ocr.crnn import DEFAULT_ALPHABET, CTCLabelConverter
from training.crnn_v2 import validate_label

SEEDS = {'train': 20261002, 'val': 27183, 'test': 31415}
TYPES = {'type1': 20000, 'type1a': 30000, 'type1b': 20000}
TIERS = {'clean': .2, 'medium': .5, 'hard': .3}


def tier_config(base, tier):
    config = deepcopy(base)
    if tier == 'hard':
        config['geometry'].update(scale=[.60, .78], rotation_deg=[-9, 9], perspective_fraction=[.025, .08])
        config['augmentation'].update(probability=1., brightness=[.70, 1.15],
            contrast=[.80, 1.12], motion_blur_px=[2, 5], defocus_blur_px=[.4, 1.],
            gaussian_blur_px=[.4, 1.], low_resolution_scale=[.35, .55],
            jpeg_quality=[60, 80], exposure=[.70, 1.20], noise_sigma=[1., 4.],
            glare_strength=[.1, .2], shadow_strength=[.1, .25],
            dirt_density=[.00015, .0005], occlusion_fraction=[.015, .04])
    elif tier == 'medium':
        config['augmentation']['probability'] = 1.
    elif tier != 'clean':
        raise ValueError(tier)
    return config


def plan(counts=TYPES, excluded=()):
    """One unique identity per sample, separate held-out RNGs, exact type/tier totals."""
    seen = set(excluded)
    jobs = []
    for split, ratio in [('val', .1), ('test', .1), ('train', .8)]:
        rng = np.random.default_rng(SEEDS[split])
        for kind, total in counts.items():
            amount = round(total * ratio)
            levels = ['clean'] * round(amount*.2) + ['medium'] * round(amount*.5)
            levels += ['hard'] * (amount-len(levels))
            rng.shuffle(levels)
            for tier in levels:
                text = plate_number(rng, int(rng.integers(2, 4)))
                while text in seen:
                    text = plate_number(rng, int(rng.integers(2, 4)))
                seen.add(text)
                jobs.append({'index': len(jobs), 'plate_num': text, 'plate_type': kind,
                             'tier': tier, 'split': split, 'seed': SEEDS[split]})
    return jobs


def initialize_worker(root, base_config, multiline):
    global WORK_ROOT, CONFIGS, TEMPLATES, MULTILINE
    cv2.setNumThreads(1)
    WORK_ROOT = Path(root)
    base = load_config(base_config)
    CONFIGS = {tier: tier_config(base, tier) for tier in TIERS}
    TEMPLATES = bank(WORK_ROOT)
    MULTILINE = MultilineConfig.load(multiline)


def generate_one(job):
    root = WORK_ROOT
    index = job['index']; kind = job['plate_type']; text = job['plate_num']
    _, minimum = validate_label(text, CTCLabelConverter(DEFAULT_ALPHABET))
    if minimum > 39:
        raise ValueError('CTC target exceeds frozen CRNN output length')
    done = root / 'generator/completed' / f'{index:06}.json'
    if done.exists():
        record = json.loads(done.read_text(encoding='utf-8'))
        if record['job'] != job:
            raise ValueError('Resume generation plan changed')
        for field in ('image', 'canonical', 'footprint'):
            if sha256(root/record['sample'][field]) != record['sample'][field+'_sha256']:
                raise ValueError('Published intermediate pixels changed')
        return record
    config = CONFIGS[job['tier']]
    plate, ink, characters = render_plate(text, kind, config['supersample'])
    name = f'images/synthetic/{index:06}_{kind}.png'
    canonical = f'generator/reference/{index:06}.png'
    footprint = f'generator/footprints/{index:06}.png'
    plate.save(root/canonical)
    rejected = []
    for attempt in range(30):
        rng = np.random.default_rng(np.random.SeedSequence([job['seed'], index, attempt]))
        clean = job['tier'] == 'clean'
        image, protected, corners, matrix, geo, mask = geometry(plate, ink, kind, config, rng, clean)
        image, effects = degrade(image, protected, config, rng, clean, mask)
        image.save(root/name); mask.save(root/footprint)
        bbox = tuple(np.r_[corners.min(axis=0), corners.max(axis=0)-corners.min(axis=0)].tolist())
        tags = tuple(sorted({'synthetic', job['tier'], 'plate_crop'} | {e['effect'] for e in effects}))
        row = Annotation(name, text, kind, bbox, tuple(tuple(float(v) for v in p) for p in corners),
                         None, True, 'volga-phase-d-large-v1', 'CC-BY-4.0', tags, 'ocr_crop',
                         job['split'], f'plate:{text}')
        sample = {'image': name, 'plate_type': kind, 'canonical': canonical, 'footprint': footprint,
                  'canonical_size': list(plate.size), 'characters': characters,
                  'homography': matrix.tolist(), 'geometry': geo, 'augmentation': effects,
                  'clean': clean, 'difficulty': job['tier'], 'attempt': attempt,
                  **{f+'_sha256': sha256(root/p) for f,p in
                     [('image',name),('canonical',canonical),('footprint',footprint)]}}
        problems = check_sample(root, row, sample, TEMPLATES)
        prepared = prepare_ocr_input(np.array(image)[:,:,::-1].copy(), kind,
                    homography=matrix, canonical_size=plate.size, multiline_config=MULTILINE)
        if kind == 'type1a' and prepared.diagnostics['effective_mode'] != 'split_and_stitch':
            problems.append('Type1a cannot split/stitch')
        if prepared.normalized.shape != (32,160) or not np.isfinite(prepared.normalized).all():
            problems.append('Invalid normalized OCR input')
        if not problems:
            break
        rejected.append(problems)
    else:
        raise ValueError(f'No valid sample after 30 attempts: {job}: {problems}')
    record = {'job':job, 'sample':sample, 'rejected':rejected,
              'preprocessing_passed':True, 'row':row.row()}
    temporary = done.with_suffix('.tmp')
    temporary.write_text(json.dumps(record), encoding='utf-8'); os.replace(temporary,done)
    return record


def generate_large(destination, workers=3, counts=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Immutable dataset already exists')
    root = destination.with_name(destination.name+'.building')
    counts = counts or TYPES
    base = Path('configs/synthetic_v2.json').resolve()
    multiline = Path('configs/multiline_v2.json').resolve()
    excluded_rows, _ = read_metadata(Path('D:/volga/generated_data/synthetic_b_v3/meta.csv'))
    jobs = plan(counts, (r.plate_num for r in excluded_rows))
    header = {'version':'phase-d-large-v1', 'counts':counts, 'seeds':SEEDS,
              'difficulty_ratios':TIERS, 'jobs':jobs, 'base_config_sha256':sha256(base),
              'multiline_config_sha256':sha256(multiline), 'script_sha256':sha256(Path(__file__))}
    plan_path = root/'generator/plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != header:
            raise ValueError('Generation plan/source changed; use a new dataset version')
    else:
        for folder in ('images/real','images/synthetic','labels','generator/reference',
                       'generator/footprints','generator/completed'):
            (root/folder).mkdir(parents=True,exist_ok=True)
        shutil.copytree(ASSETS,root/'generator/assets')
        shutil.copyfile(base,root/'generator/config.json')
        plan_path.write_text(json.dumps(header,indent=2),encoding='utf-8')
    start=time.perf_counter(); records=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,
            initargs=(str(root),str(base),str(multiline))) as executor:
        for record in executor.map(generate_one,jobs,chunksize=8):
            records.append(record)
            if len(records)%100==0:
                print(json.dumps({'generated':len(records),'total':len(jobs),
                                  'elapsed_seconds':round(time.perf_counter()-start,1)}),flush=True)
    # Recreate typed rows with the schema parser rather than maintaining a second CSV schema.
    write_metadata(root/'meta.csv',(parse_row(r['row']) for r in records))
    rows, issues=read_metadata(root/'meta.csv')
    if issues:raise ValueError(issues[:3])
    manifest={'generator_version':'phase-d-large-v1','count':len(rows),'seeds':SEEDS,
              'config_sha256':sha256(root/'generator/config.json'),
              'glyph_manifest_sha256':sha256(root/'generator/assets/manifest.json'),
              'source_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
              'plan_sha256':sha256(plan_path),'samples':[r['sample'] for r in records],
              'readability_policy':'Original Phase B raster-to-GT gate on ALL final images; reject unrecognizable hard examples',
              'oracle_geometry':True,'difficulty_config':{t:tier_config(load_config(base),t) for t in TIERS}}
    (root/'generator/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    split=make_split(root,seed=20261002,ratios=(.8,.1,.1))
    validation={'valid':True,'images_checked':len(rows),'raster_ground_truth_all':True,
                'preprocessing_all':True,'type1a_stitch_all':True,'rejected_attempts':sum(len(r['rejected']) for r in records),
                'by_type':dict(Counter(r.plate_type for r in rows)),
                'by_difficulty':dict(Counter(r['job']['tier'] for r in records)),
                'seconds':time.perf_counter()-start,'visual_review':'PENDING; do not train yet'}
    (root/'generator/verification.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    shutil.copyfile(Path('D:/volga/generated_data/synthetic_b_v3/LICENSE'),root/'LICENSE')
    os.replace(root,destination)
    return validation
