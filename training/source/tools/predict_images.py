"""Directory/single-image ANPR entry point, independent of the legacy video path."""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from dataset.schema import sha256
from src.image_anpr import (ImageConfig,ImageANPRPipeline,FrozenCRNNOCR,YOLOImageDetector,write_predictions)
from tools.baseline_predict import deny_network,summary

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/image_anpr_e.json')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--repeat',type=int,default=1)
    args=parser.parse_args()
    if args.repeat<1 or args.warmup<0:parser.error('Invalid repeat/warmup')
    config_data=json.loads(args.config.read_text(encoding='utf-8'))
    config=replace(ImageConfig(**config_data['config']),debug=args.debug)
    detector_path=ROOT/config_data['detector'];ocr_path=ROOT/config_data['ocr']
    if not detector_path.is_file() or not ocr_path.is_file():parser.error('Local weights required')
    files=sorted(p for p in args.input.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}) if args.input.is_dir() else [args.input]
    if not files or any(not p.is_file() for p in files):parser.error('No input images')
    if len({p.name for p in files})!=len(files):parser.error('Duplicate image basenames')
    os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.baseline_yolo')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    with deny_network():
        start=time.perf_counter()
        reader=FrozenCRNNOCR(ocr_path,args.device)
        detector=YOLOImageDetector(detector_path,args.device)
        pipeline=ImageANPRPipeline(detector,reader,config)
        load_ms=(time.perf_counter()-start)*1000
        def load(path):
            image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError(f'Image decode failed: {path}')
            return image
        for i in range(args.warmup):pipeline.process(load(files[i%len(files)]),image_name=files[i%len(files)].name)
        outputs=[];timings=[];read_timings=[];rows=[];reference=None
        stage_values={name:[] for name in ('detector','crop','routing','rectification','deskew','split_stitch','OCR','postprocess')}
        for repeat in range(args.repeat):
            current=[]
            for index,path in enumerate(files):
                t0=time.perf_counter();image=load(path);read_ms=(time.perf_counter()-t0)*1000
                out=pipeline.process(image,image_name=path.name,
                    debug_directory=args.output.with_suffix('')/'debug'/path.stem if args.debug else None)
                current.extend(out['predictions']);timings.append(out['total_ms']);read_timings.append(read_ms+out['total_ms'])
                stage_values['detector'].append(out['detector_ms'])
                for name in stage_values:
                    if name!='detector':stage_values[name].append(sum(r['stage_ms'].get(name,0) for r in out['detections']))
                if repeat==0:outputs.append(out)
                if (index+1)%25==0:print(f'{args.device} repeat{repeat+1}: {index+1}/{len(files)}',flush=True)
            if reference is not None and current!=reference:raise RuntimeError('Repeated images changed predictions')
            reference=current
            if repeat==0:rows=current
        write_predictions(args.output,rows)
        args.output.with_suffix('.trace.json').write_text(json.dumps(outputs,indent=2),encoding='utf-8')
        manifest={'config':asdict(config),'device':args.device,'images':len(files),'predictions':len(rows),
            'warmup_images':args.warmup,'repeats':args.repeat,'model_load_ms':load_ms,
            'pipeline_ms':summary(timings),'read_plus_pipeline_ms':summary(read_timings),
            'stage_ms_per_image':{k:summary(v) for k,v in stage_values.items()},
            'scope':'Per-image stage wall times with CUDA synchronization, warmup excluded, no debug artifacts unless explicitly requested; includes live YOLO, excludes model load and CSV IO',
            'weights':{str(p):sha256(p) for p in (detector_path,ocr_path)},
            'sources':{str(p):sha256(p) for p in (ROOT/'src/image_anpr.py',ROOT/'src/ocr_preprocess.py',ROOT/'src/multiline.py',Path(__file__))},
            'prediction_sha256':sha256(args.output),'images_sha256':{p.name:sha256(p) for p in files}}
        args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'device':args.device,'predictions':len(rows),'pipeline_ms':manifest['pipeline_ms']}))


if __name__=='__main__':main()
