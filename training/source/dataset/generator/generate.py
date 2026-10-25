"""Versioned, offline synthetic crop collections with pixel-to-GT publication gates."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

from ..schema import ALPHABET, Annotation, TARGETS, sha256, write_metadata
from ..splitting import make_split
from .augment import degrade, geometry
from .config import DEFAULT, load_config
from .render import ASSETS, render_plate
from .verify import bank, check_sample, verify_collection


def plate_number(rng, region_length):
    letters=lambda n: ''.join(rng.choice(list(ALPHABET),n))
    digits=lambda n: ''.join(str(v) for v in rng.integers(0,10,n))
    region=digits(2) if region_length==2 else str(rng.choice([1,2,7]))+digits(2)
    return letters(1)+digits(3)+letters(2)+region


def generate(destination, count, seed=42, plate_type='all', config_path=DEFAULT, variants=2, profile='mixed', excluded=()):
    if isinstance(count,bool) or not isinstance(count,int) or count<=0:raise ValueError('count must be a positive integer')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('seed must be integer 0..2^32-1')
    if isinstance(variants,bool) or not isinstance(variants,int) or not 1<=variants<=16:raise ValueError('variants must be integer 1..16')
    if plate_type not in (*TARGETS,'all') or profile not in ('mixed','clean'):raise ValueError('Unknown plate type/profile')
    config=load_config(config_path);destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Output exists; use a new dataset version, never overwrite')
    destination.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();timings=[];rejections=[]
    with tempfile.TemporaryDirectory(prefix='.synthetic-',dir=destination.parent) as temporary:
        root=Path(temporary)/'collection';root.mkdir()
        for folder in ['images/real','images/synthetic','labels','generator/reference','generator/footprints']:(root/folder).mkdir(parents=True)
        shutil.copytree(ASSETS,root/'generator/assets')
        templates=bank(root)
        (root/'generator/config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
        rows=[];samples=[];seen=set();number_rng=np.random.default_rng(seed)
        types=TARGETS if plate_type=='all' else (plate_type,)
        for index in range(count):
            sample_start=time.perf_counter();group=index//variants
            if index%variants==0:
                kind=types[group%len(types)];region_length=2+group%2
                text=plate_number(number_rng,region_length)
                while text in seen:text=plate_number(number_rng,region_length)
                seen.add(text)
            name=f'images/synthetic/{index:06}_{kind}.png'
            canonical=f'generator/reference/{index:06}.png';footprint=f'generator/footprints/{index:06}.png'
            plate,ink,characters=render_plate(text,kind,config['supersample'])
            plate.save(root/canonical)
            clean=profile=='clean' or np.random.default_rng(np.random.SeedSequence([seed,index,999])).random()<config['clean_fraction']
            for attempt in range(12):
                rng=np.random.default_rng(np.random.SeedSequence([seed,index,attempt]))
                image,protected,corners,matrix,geo,mask=geometry(plate,ink,kind,config,rng,clean)
                image,effects=degrade(image,protected,config,rng,clean,mask)
                image.save(root/name);mask.save(root/footprint)
                bbox=tuple(np.r_[corners.min(axis=0),corners.max(axis=0)-corners.min(axis=0)].tolist())
                tags=tuple(sorted({'synthetic','clean' if clean else 'mixed','plate_crop'}|{e['effect'] for e in effects}))
                row=Annotation(name,text,kind,bbox,tuple(tuple(float(v) for v in p) for p in corners),None,True,
                               f'volga-synthetic-v2:seed{seed}','CC-BY-4.0',tags,'ocr_crop','',f'plate:{text}')
                sample={'image':name,'plate_type':kind,'canonical':canonical,'footprint':footprint,
                        'characters':characters,'homography':matrix.tolist(),'geometry':geo,
                        'augmentation':effects,'clean':bool(clean),'attempt':attempt,
                        **{field+'_sha256':sha256(root/path) for field,path in [('image',name),('canonical',canonical),('footprint',footprint)]}}
                problems=check_sample(root,row,sample,templates)
                if not problems:break
                rejections.append({'index':index,'attempt':attempt,'errors':problems})
            else:raise ValueError(f'No readable label-faithful sample after 12 attempts: {index}: {problems}')
            rows.append(row);samples.append(sample);timings.append((time.perf_counter()-sample_start)*1000)
        write_metadata(root/'meta.csv',rows)
        source_dir=Path(__file__).parent
        manifest={'generator_version':'2.0.0','seed':seed,'count':count,'requested_type':plate_type,'variants':variants,'profile':profile,
                  'config_sha256':sha256(root/'generator/config.json'),
                  'source_sha256':{p.name:sha256(p) for p in sorted(source_dir.glob('*.py'))},
                  'glyph_manifest_sha256':sha256(root/'generator/assets/manifest.json'),
                  'versions':{'numpy':np.__version__,'opencv':cv2.__version__,'pillow':pillow_version},
                  'plate_mask':'LDDDLLRR/RRR for all target types; three-digit region starts 1/2/7',
                  'annotation_policy':'Synthetic plate crops, unknown vehicle context; not full-scene detector/context data',
                  'visual_review':'required before downstream use; recorded separately in PHASE B report',
                  'rejected_attempts':rejections,'samples':samples}
        (root/'generator/manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        verification=verify_collection(root)
        if not verification['valid']:raise ValueError('Ground-truth verification failed before publication: '+str(verification['errors'][:3]))
        (root/'generator/verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
        split=make_split(root,seed=seed,excluded=excluded)
        (root/'LICENSE').write_text('Authored procedural synthetic crops: CC BY 4.0.\nAttribution: Volga ANPR V2 procedural generator.\nhttps://creativecommons.org/licenses/by/4.0/\nNo external images/system fonts are included.\n',encoding='utf-8')
        (root/'README.md').write_text('# Synthetic PHASE B collection\n\nPlate crops only. No vehicle context ground truth or real-world model accuracy is implied.\nmeta.csv is canonical. See generator/manifest.json, verification.json and splits/manifest.json.\nVisual review must be completed before training. No stitching/training/inference changes in PHASE B.\n',encoding='utf-8')
        os.replace(root,destination)
    return {'output':str(destination),'images':count,'by_type':dict(Counter(r.plate_type for r in rows)),
            'unique_plates':len(seen),'rejected_attempts':len(rejections),'verification':verification,
            'split_counts':{k:{field:v[field] for field in ['images','rows']} for k,v in split['splits'].items()},
            'generation_ms':{'mean':float(np.mean(timings)),'p95':float(np.percentile(timings,95)),'max':float(max(timings))},
            'total_seconds':time.perf_counter()-started,'timing_meaning':'CPU dataset generation+sample checks, not ANPR inference latency'}
