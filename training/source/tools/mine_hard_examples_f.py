"""Targeted next iteration from TRAIN only; validation/test labels never mined."""
from collections import Counter
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from src.geometry_f import bbox_iou
from src.image_anpr import FrozenCRNNOCR
from src.image_anpr_f import DetectorF,ImagePipelineF
from tools.phase_e import ROOT,BEST,read,write,load_image
from tools.phase_f_setup import DATA,REPORT
from dataset.schema import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--teacher',default='F02_balanced');parser.add_argument('--name',default='F02_hard_v2');args=parser.parse_args()
    config=read(ROOT/'configs/phase_f'/f'{args.teacher}.json')
    dataset_root=Path(config.get('dataset_root',Path(config['data']).parent))
    training=read(REPORT/'training'/f'{args.teacher}.json');torch.set_num_threads(4);cv2.setNumThreads(1)
    model=DetectorF(training['best_checkpoint'],confidence=.05,imgsz=640)
    pipe=ImagePipelineF(model,FrozenCRNNOCR(BEST,'cuda'))
    dataset=read(dataset_root/'manifest.json');cases=[];seen=set()
    for r in dataset['records']:
        if r['split']!='train' or r['sha256'] in seen:continue
        seen.add(r['sha256']);frame=load_image(dataset_root/'detect/images/train'/r['image']);ds=model.detect(frame)
        reasons=set();targets=[]
        for a in r['annotations']:
            scores=sorted([(bbox_iou(d.bbox,a['bbox']),d.confidence) for d in ds],reverse=True)
            overlap,confidence=scores[0] if scores else (0,0)
            failure='FALSE_NEGATIVE' if overlap<.1 else 'BAD_BBOX' if overlap<.5 else 'LOW_CONFIDENCE' if confidence<.5 else None
            if failure:reasons.add(failure)
            best_box=max(ds,key=lambda d:bbox_iou(d.bbox,a['bbox']),default=None)
            if best_box is not None and overlap>=.1:
                x=a['bbox'];y=best_box.bbox
                inter=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
                coverage=inter/max(1,(x[2]-x[0])*(x[3]-x[1]))
                if coverage<.94:reasons.add('CLIPPED_PLATE')
            typed=max(((bbox_iou(d.bbox,a['bbox']),d.class_name) for d in ds),default=(0,None))
            if typed[0]>=.5 and typed[1]!=a['plate_type']:reasons.add('CLASS_CONFUSION')
            if a['bbox'][2]-a['bbox'][0]<80:reasons.add('SMALL_PLATE')
            if abs(a.get('angle',0))>15:reasons.add('ANGLED_PLATE')
            if a.get('occlusion'):reasons.add('PARTIAL_OCCLUSION')
            targets.append({'plate_type':a['plate_type'],'best_IoU':overlap,'confidence':confidence,'failure':failure})
        for d in ds:
            if d.confidence>=.5 and max((bbox_iou(d.bbox,a['bbox']) for a in r['annotations']),default=0)<.5:
                reasons.add('FALSE_POSITIVE')
        if np.median(frame)<60:reasons.add('LOW_LIGHT')
        # Real train scenes are few, so inspect split behavior only on these here.
        if not r['is_synthetic']:
            out=pipe.process(frame,detections=[d for d in ds if d.confidence>=.5],image_name=r['image'])
            for d in out['detections']:
                if d.get('plate_type')=='type1a' and d.get('preprocessing',{}).get('effective_mode')!='split_and_stitch':
                    reasons.add('TYPE1A_SPLIT_FAILURE')
        if reasons:
            priority=sum({'FALSE_NEGATIVE':5,'BAD_BBOX':4,'FALSE_POSITIVE':4,'LOW_CONFIDENCE':3,
                'TYPE1A_SPLIT_FAILURE':4,'CLASS_CONFUSION':4,'CLIPPED_PLATE':4}.get(reason,1) for reason in reasons)
            cases.append({'image':r['image'],'is_synthetic':r['is_synthetic'],'reasons':sorted(reasons),
                'priority':priority,'targets':targets,'negative_kind':r.get('negative_kind')})
    cases.sort(key=lambda r:(r['is_synthetic'],-r['priority'],r['image']))
    selected=cases[:240]
    folder=dataset_root.parent/('detector_'+args.name.lower());folder.mkdir(exist_ok=False)
    originals=[str(dataset_root/'detect/images/train'/r['image']) for r in dataset['records'] if r['split']=='train']
    extras=[str(dataset_root/'detect/images/train'/r['image']) for r in selected for _ in range(3)]
    (folder/'train.txt').write_text('\n'.join(originals+extras)+'\n',encoding='utf-8')
    (folder/'detect.yaml').write_text('path: '+str(folder).replace('\\','/')+'\ntrain: train.txt\nval: '+
        str(dataset_root/'detect/images/val').replace('\\','/')+'\nnames:\n  0: type1\n  1: type1a\n  2: type1b\n',encoding='utf-8')
    result={'scope':'TRAIN ONLY — unique scenes from frozen teacher dataset; existing validation/test untouched.',
        'model_sha256':training['checkpoint_sha256'],'dataset_sha256':sha256(dataset_root/'manifest.json'),
        'cases':cases,'selected':selected,'selected_count':len(selected),'extra_repeat_factor':3,
        'failure_counts':dict(Counter(reason for r in cases for reason in r['reasons'])),
        'hard_dataset':str(folder),'training_index_sha256':sha256(folder/'train.txt'),
        'no_random_dataset_growth':True,'no_test_mining':True}
    report=REPORT/f'hard_mining_{args.name}.json';write(report,result)
    config.update(name=args.name,initial_weights=training['best_checkpoint'],data=str(folder/'detect.yaml'))
    config['train'].update(name=args.name,epochs=8,patience=3,lr0=.0003,warmup_epochs=1.,close_mosaic=2)
    config['budget']='Eight-epoch targeted TRAIN-only continuation from verified teacher; no random dataset growth or validation/test mining'
    config['hard_mining_manifest_sha256']=sha256(report)
    write(ROOT/'configs/phase_f'/f'{args.name}.json',config)
    print(result['failure_counts'],flush=True)


if __name__=='__main__':main()
