"""PHASE C conditional crop ablations, without changing legacy inference.

Preparation freezes the crop pool before OCR. Selection uses synthetic val
only; test and public diagnostic samples never select a configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset.schema import read_metadata, sha256
from src.multiline import METHODS, MODES, MultilineConfig, prepare_multiline, recognize_prepared


def save_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def read_bgr(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))[:,:,::-1].copy()


def write_bgr(path,image):
    Image.fromarray(image[:,:,::-1]).save(path)


def distance(reference,prediction):
    previous=list(range(len(prediction)+1))
    for i,a in enumerate(reference,1):
        current=[i]
        for j,b in enumerate(prediction,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def timing(values):
    return {key:float(value) for key,value in zip(('mean','median','p90','p95','max'),
            (np.mean(values),np.median(values),*np.percentile(values,[90,95]),max(values)))}


def summarize(rows):
    characters=sum(len(row['label']) for row in rows)
    errors=sum(distance(row['label'],row['prediction']) for row in rows)
    return {'count':len(rows),'correct':sum(row['label']==row['prediction'] for row in rows),
            'full_plate_accuracy':sum(row['label']==row['prediction'] for row in rows)/len(rows),
            'characters':characters,'edit_distance':errors,'cer':errors/characters,
            'character_accuracy_1_minus_cer':1-errors/characters,
            'total_ms':timing([v for row in rows for v in row.get('timing_repeats',{}).get('total_ms',[row['total_ms']])]),
            'preprocess_ms':timing([v for row in rows for v in row.get('timing_repeats',{}).get('preprocess_ms',[row['preprocess_ms']])]),
            'ocr_ms':timing([v for row in rows for v in row.get('timing_repeats',{}).get('ocr_ms',[row['ocr_ms']])]),
            'effective_modes':dict(Counter(row['diagnostics']['effective_mode'] for row in rows)),
            'used_methods':dict(Counter((row['diagnostics'].get('split') or {}).get('used_method','none') for row in rows))}


def gallery(records, output, prefix, config, mode='split_and_stitch', method='horizontal_projection'):
    """Review input, top/bottom reading order, and transformed OCR input."""
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for page,start in enumerate(range(0,len(records),12),1):
        selected=records[start:start+12]
        sheet=Image.new('RGB',(1050,170*len(selected)),(224,228,232))
        draw=ImageDraw.Draw(sheet)
        for i,record in enumerate(selected):
            image=read_bgr(output/record['crop'])
            p=prepare_multiline(image,mode=mode,method=method,config=config)
            draw.text((8,i*170+3),f"{record['id']} | {record['label']} | {p.effective_mode} | "
                      f"{p.split.used_method if p.split else '-'}",font=font,fill='black')
            previews=[image]
            if p.row_boxes:
                previews += [image[y0:y1,x0:x1] for x0,y0,x1,y1 in p.row_boxes]
            else:
                previews += [image,image]
            previews += [p.inputs[0]]
            for j,array in enumerate(previews):
                panel=Image.fromarray(array[:,:,::-1]);panel.thumbnail((250,135))
                sheet.paste(panel,(8+j*260,i*170+27))
        sheet.save(output/f'{prefix}_{page:02d}.png')


def prepare(args):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    (output/'inputs').mkdir(exist_ok=True)
    registry=json.loads((ROOT/'reports/phase_b/accepted_dataset.json').read_text(encoding='utf-8'))
    root=Path(registry['accepted_dataset'])
    if not all(registry['gates'].values()):raise ValueError('PHASE B gates not passed')
    for relative,key in [('meta.csv','meta_sha256'),('generator/manifest.json','generator_manifest_sha256'),
                         ('splits/manifest.json','split_manifest_sha256')]:
        if sha256(root/relative)!=registry[key]:raise ValueError(f'Accepted data changed: {relative}')
    rows,issues=read_metadata(root/'meta.csv')
    if issues:raise ValueError(issues)
    membership={}
    for split in ('train','val','test'):
        subset,issues=read_metadata(root/f'splits/{split}.csv')
        if issues:raise ValueError(issues)
        membership.update({row.image:split for row in subset})
    samples={s['image']:s for s in json.loads((root/'generator/manifest.json').read_text(encoding='utf-8'))['samples']}
    records=[];integrity=[]
    config=MultilineConfig.load(args.config)
    for row in rows:
        if row.plate_type!='type1a':continue
        sample=samples[row.image]
        if sha256(root/row.image)!=sample['image_sha256']:raise ValueError('Synthetic image changed')
        raw=read_bgr(root/row.image)
        # Common oracle rectification in this controlled experiment, NOT a new
        # predicted rectifier. Labels/characters never enter prepare_multiline.
        with Image.open(root/sample['canonical']) as canonical:w,h=canonical.size
        restored=cv2.warpPerspective(raw,np.linalg.inv(np.asarray(sample['homography'])),(w,h))
        crop=cv2.resize(restored,(w//2,h//2),interpolation=cv2.INTER_AREA)
        name=f"synthetic_{Path(row.image).stem}"
        path=output/'inputs'/f'{name}.png';write_bgr(path,crop)
        record={'id':name,'label':row.plate_num,'pool':f"synthetic_{membership[row.image]}",
                'source_image':str(root/row.image),'source_sha256':sha256(root/row.image),
                'crop':path.relative_to(output).as_posix(),'crop_sha256':sha256(path),
                'geometry':'oracle generator homography -> 290x170; same crop for all modes',
                'conditions':row.conditions}
        records.append(record)
        for method in METHODS:
            p=prepare_multiline(crop,mode='split_and_stitch',method=method,config=config)
            failures=[]
            if p.effective_mode=='whole_plate':
                failures.append('unexpected synthetic fallback')
            else:
                for c in sample['characters']:
                    x0,y0,x1,y1=[int(v//2) for v in c['box']]
                    cell=cv2.cvtColor(crop[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
                    _,ink=cv2.threshold(cell,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
                    ys,xs=np.where(ink>0)
                    bx0,by0,bx1,by1=p.row_boxes[c['row']]
                    retained=np.mean((xs+x0>=bx0)&(xs+x0<bx1)&(ys+y0>=by0)&(ys+y0<by1))
                    if retained<.995:failures.append({'character_index':c['index'],'ink_retained':float(retained)})
                # Stitched row transforms must be invertible, keep top before
                # bottom, and map every retained ink point inside its own tile.
                if p.stitch_transforms[0][0,2]>=p.stitch_transforms[1][0,2]:failures.append('reading order')
                if any(np.linalg.det(m)<=0 for m in p.stitch_transforms):failures.append('invalid stitch transform')
            integrity.append({'id':name,'method':method,'failures':failures,'diagnostics':p.diagnostics()})
    public=Path(args.public)
    labels=defaultdict(list)
    with (public/'debug_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f,delimiter=';'):labels[row['image']].append(row)
    trace_path=ROOT/'reports/baseline_predictions.trace.jsonl'
    for line in trace_path.read_text(encoding='utf-8').splitlines():
        trace=json.loads(line);ground=labels[trace['image']]
        boxes=trace.get('detector_boxes_before_geometry_filter',[])
        if len(ground)!=1 or ground[0]['plate_type']!='type1a' or '#' in ground[0]['plate_num'] or len(boxes)!=1:continue
        box=boxes[0];x,y,w,h=box['bbox']
        if not .9<=w/h<=2.4:continue
        original=public/'debug set'/trace['image']
        if sha256(original)!=trace['sha256']:raise ValueError('Public image changed')
        raw=read_bgr(original)
        x0,y0=max(0,int(np.floor(x))),max(0,int(np.floor(y)))
        x1,y1=min(raw.shape[1],int(np.ceil(x+w))),min(raw.shape[0],int(np.ceil(y+h)))
        rawcrop=raw[y0:y1,x0:x1]
        crop=cv2.resize(rawcrop,(round(rawcrop.shape[1]*170/rawcrop.shape[0]),170),interpolation=cv2.INTER_LINEAR)
        name=f"public_{Path(trace['image']).stem}"
        path=output/'inputs'/f'{name}.png';write_bgr(path,crop)
        records.append({'id':name,'label':ground[0]['plate_num'],'pool':'public_diagnostic',
                        'source_image':str(original),'source_sha256':trace['sha256'],
                        'crop':path.relative_to(output).as_posix(),'crop_sha256':sha256(path),
                        'bbox_xyxy':[x0,y0,x1,y1],'legacy_geometry_pass':box['legacy_geometry_pass'],
                        'geometry':'saved detector bbox; no quad GT, no deskew; resize preserving aspect to h170',
                        'review_required':True})
    manifest={'scope':'conditional type1a OCR ablation; public evaluation only; not spatial or end-to-end metrics',
              'selection':'synthetic_val only; public/test not used for tuning',
              'phase_b_registry_sha256':sha256(ROOT/'reports/phase_b/accepted_dataset.json'),
              'baseline_trace_sha256':sha256(trace_path),'public_labels_sha256':sha256(public/'debug_labels.csv'),
              'config_sha256':sha256(args.config),'config':config.__dict__,'records':records}
    save_json(output/'inputs_manifest.json',manifest)
    save_json(output/'ink_integrity.json',{'checks':len(integrity),'images':sum(r['pool'].startswith('synthetic') for r in records),
              'valid':all(not r['failures'] for r in integrity),'failed_checks':sum(bool(r['failures']) for r in integrity),
              'meaning':'source character ink retained and top/bottom geometric order; not OCR model accuracy',
              'results':integrity})
    synthetic=[r for r in records if r['pool'].startswith('synthetic')]
    # All validation samples, plus degradation extremes from train/test.
    selection=[r for r in synthetic if r['pool']=='synthetic_val']
    gallery(selection,output,'synthetic_review',config)
    gallery([r for r in records if r['pool']=='public_diagnostic'],output,'public_review',config)
    print(json.dumps({'counts':dict(Counter(r['pool'] for r in records)),
                      'ink_failed_checks':sum(bool(r['failures']) for r in integrity)},indent=2))


def run(args):
    import torch
    from ocr.crnn import CRNNRecognizer
    output=args.output
    manifest=json.loads((output/'inputs_manifest.json').read_text(encoding='utf-8'))
    integrity=json.loads((output/'ink_integrity.json').read_text(encoding='utf-8'))
    review=json.loads((output/'visual_review.json').read_text(encoding='utf-8'))
    if not integrity['valid'] or not review['accepted']:raise ValueError('Input integrity/visual review failed')
    if review['inputs_manifest_sha256']!=sha256(output/'inputs_manifest.json'):raise ValueError('Reviewed inputs changed')
    if manifest['config_sha256']!=sha256(args.config):raise ValueError('Frozen config changed')
    config=MultilineConfig.load(args.config)
    torch.set_num_threads(4);torch.manual_seed(42);cv2.setNumThreads(1)
    weight=ROOT/'models/ocr_crnn_nomeroff_clean.pt'
    if sha256(weight)!='273c3b990e1be9e16b48b628c61d88c0e5b351b973705e6a5e88ac5a5f5dc2a5':raise ValueError('Baseline weight changed')
    started=time.perf_counter();reader=CRNNRecognizer(str(weight),device='cpu');load_ms=(time.perf_counter()-started)*1000
    def recognize(image):
        item=reader.readtext(image,detail=1)[0]
        return item[1],item[2],item[3]['char_probs']
    records=[r for r in manifest['records'] if r['pool']!='synthetic_train' and r['id'] not in review.get('excluded_ids',[])]
    for _ in range(5):recognize(read_bgr(output/records[0]['crop']))
    configurations=[('whole_plate','none')]+[(mode,method) for mode in MODES[1:] for method in METHODS]
    predictions=[];random=np.random.default_rng(42)
    # Random order per image reduces systematic cold/order bias. Inputs decoded
    # before timing; all modes share that exact array and frozen OCR model.
    for index,record in enumerate(records):
        path=output/record['crop']
        if sha256(path)!=record['crop_sha256']:raise ValueError('Frozen crop changed')
        image=read_bgr(path)
        for j in random.permutation(len(configurations)):
            mode,method=configurations[j]
            repeats={'total_ms':[],'preprocess_ms':[],'ocr_ms':[]}
            texts=[]
            for _ in range(3):
                start=time.perf_counter()
                p=prepare_multiline(image,mode=mode,method='fixed_ratio' if method=='none' else method,config=config)
                split_end=time.perf_counter()
                prediction,confidence,probs=recognize_prepared(p,recognize)
                end=time.perf_counter()
                texts.append(prediction)
                repeats['preprocess_ms'].append((split_end-start)*1000)
                repeats['ocr_ms'].append((end-split_end)*1000)
                repeats['total_ms'].append((end-start)*1000)
            if len(set(texts))!=1:raise ValueError('Non-deterministic OCR prediction')
            predictions.append({**record,'mode':mode,'method':method,'prediction':prediction,
                                'confidence_uncalibrated':confidence,'char_probs':probs,
                                **{key:float(np.mean(value)) for key,value in repeats.items()},
                                'timing_repeats':repeats,'diagnostics':p.diagnostics()})
        if (index+1)%10==0:print(f"processed {index+1}/{len(records)}",flush=True)
    grouped=defaultdict(list)
    for row in predictions:grouped[(row['pool'],row['mode'],row['method'])].append(row)
    metrics={'|'.join(key):summarize(rows) for key,rows in sorted(grouped.items())}
    def select(mode):
        # Accuracy, CER, then preprocessing latency: identical OCR inputs need
        # not choose split algorithms by unrelated CPU scheduling noise in OCR.
        keys=[key for key in grouped if key[0]=='synthetic_val' and key[1]==mode]
        return min(keys,key=lambda key:(-metrics['|'.join(key)]['full_plate_accuracy'],
                                        metrics['|'.join(key)]['cer'],metrics['|'.join(key)]['preprocess_ms']['mean'],
                                        METHODS.index(key[2]) if key[2] in METHODS else -1))
    selection={mode:{'mode':mode,'method':select(mode)[2],'selected_on':'synthetic_val only'} for mode in MODES}
    save_json(output/'selection.json',selection)
    # Legacy-pass subset remains a conditional crop experiment, not emissions.
    for mode,method in configurations:
        subset=[r for r in predictions if r['pool']=='public_diagnostic' and r['mode']==mode
                and r['method']==method and r['legacy_geometry_pass']]
        if subset:metrics[f'public_legacy_geometry_subset|{mode}|{method}']=summarize(subset)
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    fields=('pool','mode','method','count','correct','full_plate_accuracy','cer','character_accuracy_1_minus_cer',
            'mean_ms','median_ms','p90_ms','p95_ms','max_ms','preprocess_mean_ms','ocr_mean_ms')
    with (output/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for key,value in metrics.items():
            pool,mode,method=key.split('|')
            writer.writerow({'pool':pool,'mode':mode,'method':method,**{k:value[k] for k in fields[3:8]},
                             **{f'{k}_ms':v for k,v in value['total_ms'].items()},
                             'preprocess_mean_ms':value['preprocess_ms']['mean'],'ocr_mean_ms':value['ocr_ms']['mean']})
    save_json(output/'metrics.json',{'metrics':metrics,'selection':selection,
        'runtime':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
                   'opencv':cv2.__version__,'numpy':np.__version__,'threads':4,'device':'cpu','warmup':5,
                   'load_ms':load_ms,'timing_repeats_per_image_configuration':3,'vram':'not applicable: CPU execution'},
        'timing_scope':'warm crop preprocessing + frozen CRNN greedy + raw concatenation; excludes file IO, detector, rectification, model load',
        'scope':manifest['scope'],'model_sha256':sha256(weight),'manifest_sha256':sha256(output/'inputs_manifest.json'),
        'config_sha256':sha256(args.config),'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in
                       (ROOT/'src/multiline.py',ROOT/'tools/benchmark_multiline.py',ROOT/'ocr/crnn.py')}})
    for mode,value in selection.items():
        key=f"synthetic_test|{mode}|{value['method']}"
        print(key,json.dumps(metrics[key],ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,default=ROOT/'reports/phase_c')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/multiline_v2.json')
    parser.add_argument('--public',type=Path,default=Path('D:/volga/external_data/organizer_debug/Полуфинал - public'))
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else run)(args)


if __name__=='__main__':main()
