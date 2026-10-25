"""Validate each frozen sample while building common normalized OCR inputs.

The generator fully checked decoded pixels/geometry/metadata and splits twice.
Recheck unchanged metadata/split hashes and every image byte hash, then decode
and preprocess each image once. Parallel workers do not change sample order.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from dataset.schema import read_metadata,sha256
from ocr.crnn import CTCLabelConverter
from src.multiline import MultilineConfig
from src.ocr_preprocess import prepare_ocr_input
from training.crnn_v2 import TrainConfig,validate_label


def initialize():
    import cv2
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(1)


def validate_input(job):
    index,path,kind,sample,previous,multiline=job
    content=Path(path).read_bytes();digest=hashlib.sha256(content).hexdigest()
    if digest!=sample['image_sha256'] or digest!=previous['sha256'] or len(content)!=previous['bytes']:
        raise ValueError(f'Frozen image bytes changed: {path}')
    with Image.open(io.BytesIO(content)) as image:
        if image.size!=(previous['width'],previous['height']):raise ValueError('Image dimensions changed')
        pixels=np.asarray(image.convert('RGB'))[:,:,::-1].copy()
    if sample['plate_type']!=kind:raise ValueError('Layout provenance mismatch')
    prepared=prepare_ocr_input(pixels,kind,homography=sample['homography'],
        canonical_size=sample['canonical_size'],multiline_config=MultilineConfig(**multiline))
    if prepared.crop.size==0 or prepared.normalized.shape!=(32,160) or not np.isfinite(prepared.normalized).all():
        raise ValueError('Empty/invalid/nonfinite OCR input')
    stitched=kind=='type1a'
    if stitched and prepared.diagnostics['effective_mode']!='split_and_stitch':raise ValueError('type1a stitch failed')
    return index,prepared.normalized,stitched


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();root=Path(args.root);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();config=TrainConfig(dataset_root=str(root),cache_images=False)
    rows,issues=read_metadata(root/'meta.csv')
    if issues:raise ValueError(issues)
    split_manifest=json.loads((root/'splits/manifest.json').read_text())
    if split_manifest['meta_sha256']!=sha256(root/'meta.csv'):raise ValueError('Frozen metadata changed')
    for split in ('train','val','test'):
        if split_manifest['splits'][split]['sha256']!=sha256(root/'splits'/f'{split}.csv'):
            raise ValueError('Frozen split changed')
    verification=json.loads((root/'generator/verification.json').read_text())
    if not verification['valid'] or verification['images_checked']!=len(rows) or not verification['raster_ground_truth_all']:
        raise ValueError('Prior full raster/integrity validation missing')
    samples={r['image']:r for r in json.loads((root/'generator/manifest.json').read_text())['samples']}
    converter=CTCLabelConverter(config.alphabet);minimums=[]
    for row in rows:
        _,minimum=validate_label(row.plate_num,converter)
        if row.plate_type not in ('type1','type1a','type1b') or minimum>39:raise ValueError('Invalid type/CTC target')
        minimums.append(minimum)
    path=root/'generator/ocr_inputs.npy';temporary=path.with_suffix('.tmp')
    if path.exists() or temporary.exists():raise ValueError('Frozen OCR cache already exists; never overwrite')
    cache=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(rows),1,32,160))
    multiline=json.loads(Path(config.multiline_config).read_text())
    jobs=((i,str(root/r.image),r.plate_type,samples[r.image],split_manifest['image_manifest'][r.image],multiline) for i,r in enumerate(rows))
    split_count=0
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as executor:
        for count,(index,normalized,stitched) in enumerate(executor.map(validate_input,jobs,chunksize=32),1):
            cache[index,0]=normalized;split_count+=stitched
            if count%1000==0:print(json.dumps({'validated':count,'total':len(rows),'seconds':round(time.perf_counter()-started,1)}),flush=True)
    cache.flush();del cache;os.replace(temporary,path)
    sources=('src/ocr_preprocess.py','src/multiline.py','ocr/crnn.py',config.multiline_config)
    contract={'cache_sha256':sha256(path),'meta_sha256':sha256(root/'meta.csv'),
        'image_order':[r.image for r in rows],'preprocessing_hashes':{str(Path(p).resolve()):sha256(Path(p)) for p in sources},
        'shape':[len(rows),1,32,160],'dtype':'float32','geometry':'known generator homography then Phase C stitch','additional_augmentation':False}
    path.with_suffix('.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    report={'status':'PASS','images_checked':len(rows),'integrity_valid':True,'raster_ground_truth':True,
        'integrity_protocol':'Generation fully checked decoded pixels, geometry, metadata and split; hashes bind unchanged metadata/splits, all current image bytes rechecked while decoding/preprocessing once.',
        'labels_alphabet_type_ctc_all':True,'preprocessing_all':True,'type1a_stitch_count':split_count,
        'ctc_timesteps':39,'minimum_ctc_spare_timesteps':39-max(minimums),'normalized_cache':str(path.resolve()),
        'meta_sha256':contract['meta_sha256'],'cache_sha256':contract['cache_sha256'],
        'split_manifest_sha256':sha256(root/'splits/manifest.json'),
        'seconds':time.perf_counter()-started,'counts':{'rows':len(rows),'by_type':dict(Counter(r.plate_type for r in rows))},'visual_gate':'PENDING'}
    (output/'dataset_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
