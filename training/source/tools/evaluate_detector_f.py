"""Detector/AP and fixed-OCR development tournament with explicit subset scope."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from dataset.schema import sha256
from src.geometry_f import bbox_iou,quad_errors,validate_quad,quad_iou
from src.image_anpr import Detection,FrozenCRNNOCR,ImageConfig,write_predictions
from src.image_anpr_f import DetectorF,HybridDetectorF,ImagePipelineF,RouterF
from tools.phase_e import ROOT,IMAGES,BEST,read,write,load_image,official,summary
from tools.phase_f_setup import DATA,REPORT

TYPES=('type1','type1a','type1b')


def average_precision(tp,fp,gt_count):
    if not gt_count:return None
    tp=np.cumsum(tp);fp=np.cumsum(fp)
    recall=tp/gt_count;precision=tp/np.maximum(1,tp+fp)
    # 101 recall samples, monotonic precision envelope; not framework fitness.
    return float(np.mean([max(precision[recall>=r],default=0) for r in np.linspace(0,1,101)]))


def detection_metrics(records,predictions, typed=False, threshold=.5, polygon=False):
    targets={r['image']:r['annotations'] for r in records}
    flat=sorted([(name,p) for name,ps in predictions.items() for p in ps],key=lambda pair:-pair[1]['confidence'])
    total=sum(len(v) for v in targets.values())
    def overlap(p,t):
        if polygon:
            if p.get('quad') is None:return 0.
            try:return quad_iou(p['quad'],t['quad'])
            except ValueError:return 0.
        return bbox_iou(p['bbox'],t['bbox'])
    def match(iou_threshold,kind=None,operating=False):
        used=set();tp=[];fp=[];pairs=[]
        for name,p in flat:
            if operating and p['confidence']<threshold:continue
            if kind is not None and p['routed_type']!=kind:continue
            candidates=[]
            for i,t in enumerate(targets.get(name,[])):
                if (name,i) in used:continue
                if kind is not None and t['plate_type']!=kind:continue
                if typed and p['routed_type']!=t['plate_type']:continue
                candidates.append((overlap(p,t),i,t))
            best=max(candidates,key=lambda a:a[0],default=(0,None,None))
            correct=best[1] is not None and best[0]>=iou_threshold
            tp.append(int(correct));fp.append(int(not correct))
            if correct:
                used.add((name,best[1]));pairs.append((name,p,best[2],best[0]))
        return tp,fp,pairs
    tp,fp,pairs=match(.5,operating=True);ntp=sum(tp);nfp=sum(fp)
    result={'GT_count':total,'prediction_count':len(tp),'TP':ntp,'FP':nfp,'FN':total-ntp,
        'Precision':ntp/max(1,ntp+nfp),'Recall':ntp/max(1,total),
        'F1':2*ntp/max(1,2*ntp+nfp+total-ntp),
        'FP_per_image':nfp/max(1,len(records)),'confidence_threshold':threshold,
        'matching':'One-to-one confidence-sorted at IoU>=0.5; '+('typed' if typed else 'class agnostic'),
        'AP_protocol':'101-point interpolated AP, greedy per-image matching at IoU0.50:0.05:0.95, predictions conf>=0.001; class-agnostic AP comparable to old single-class detector',
        'per_type':{},'IoU_matched_distribution':summary([v[3] for v in pairs]) if pairs else None}
    aps=[]
    for t in np.arange(.5,.951,.05):
        tp_all,fp_all,_=match(float(t));aps.append(average_precision(tp_all,fp_all,total))
    result['mAP50']=aps[0];result['mAP50_95']=float(np.mean(aps)) if total else None
    for kind in TYPES:
        n=sum(t['plate_type']==kind for ts in targets.values() for t in ts)
        recovered=sum(t['plate_type']==kind for _,_,t,_ in pairs)
        ktp,kfp,_=match(.5,kind=kind,operating=True);a=sum(ktp);b=sum(kfp)
        kap=[]
        for t in np.arange(.5,.951,.05):
            atp,afp,_=match(float(t),kind=kind)
            kap.append(average_precision(atp,afp,n))
        result['per_type'][kind]={'GT_count':n,'Recall':recovered/n if n else None,'miss_count':n-recovered,
            'classification_aware':{'Precision':a/max(1,a+b) if n else None,'Recall':a/n if n else None,
                'F1':2*a/max(1,2*a+b+n-a) if n else None,
                'AP50':kap[0],'AP50_95':float(np.mean(kap)) if n else None}}
    if typed:
        present=[v['classification_aware'] for v in result['per_type'].values() if v['GT_count']]
        result['mAP50']=float(np.mean([v['AP50'] for v in present])) if present else None
        result['mAP50_95']=float(np.mean([v['AP50_95'] for v in present])) if present else None
    sizes={k:[] for k in ('tiny','small','medium','large')}
    for r in records:
        for t in r['annotations']:
            width=t['bbox'][2]-t['bbox'][0]
            k='tiny' if width<40 else 'small' if width<80 else 'medium' if width<160 else 'large'
            best=max((overlap(p,t) for p in predictions.get(r['image'],[]) if p['confidence']>=threshold),default=0)
            sizes[k].append(best)
    result['size_analysis']={k:{'count':len(v),'Recall':sum(x>=.5 for x in v)/len(v) if v else None,
        'bad_localization_count':sum(.1<=x<.5 for x in v),'no_overlap_or_miss':sum(x<.1 for x in v)} for k,v in sizes.items()}
    result['bad_localization_count']=sum(v['bad_localization_count'] for v in result['size_analysis'].values())
    geometry=[];invalid=0;missing=0
    for name,p,t,overlap_value in pairs:
        if p.get('quad') is not None:
            try:geometry.append(quad_errors(p['quad'],t['quad']))
            except ValueError:invalid+=1
        else:missing+=1
    result['quad_quality']={'matched_valid_quads':len(geometry),'matched_invalid_quads':invalid,
        'matched_missing_quads':missing,
        'corner_pixel_error':float(np.mean([r['corner_pixel_error'] for r in geometry])) if geometry else None,
        'normalized_corner_error':float(np.mean([r['normalized_corner_error'] for r in geometry])) if geometry else None,
        'quad_IoU':float(np.mean([r['quad_iou'] for r in geometry])) if geometry else None,
        'scope':'Only matched valid quads; report missing/invalid quads separately, not perfect errors for absent predictions'}
    negative_images=[r['image'] for r in records if not r['annotations']]
    result['negatives']={'images':len(negative_images),
        'images_with_FP':sum(any(p['confidence']>=threshold for p in predictions.get(n,[])) for n in negative_images),
        'negative_image_FPR':sum(any(p['confidence']>=threshold for p in predictions.get(n,[])) for n in negative_images)/len(negative_images) if negative_images else None}
    return result


def serialize_detections(frame,detections):
    router=RouterF('geometry');h,w=frame.shape[:2];rows=[]
    for d in detections:
        x1,y1,x2,y2=map(int,d.bbox);crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
        kind=router.classify(crop,d).plate_type if crop.size else 'other'
        row=asdict(d);row['routed_type']=kind;rows.append(row)
    return rows


def evaluate(name,weights,imgsz=640,geometry='bbox',routing='geometry',benchmark=True,stitch=True,filter_override=None,old_type1_weights=None,old_type1_imgsz=1024):
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    folder=REPORT/'experiments'/name;folder.mkdir(parents=True,exist_ok=True)
    config={'name':name,'weights':str(Path(weights).resolve()),'weights_sha256':sha256(Path(weights)),
        'imgsz':imgsz,'geometry':geometry,'routing':routing,'fixed_OCR_sha256':sha256(BEST),
        'dataset_sha256':sha256(DATA/'manifest.json'),'confidence':.5,'iou':.5}
    if not stitch:config['stitch']=False
    if filter_override is not None:config['filter_override']=filter_override
    if old_type1_weights:
        config['hybrid_secondary']={'weights':str(Path(old_type1_weights).resolve()),'weights_sha256':sha256(Path(old_type1_weights)),'imgsz':old_type1_imgsz,'role':'old_type1_only'}
    if (folder/'result.json').exists():
        old=read(folder/'result.json');assert old['config']==config;return old
    detector=DetectorF(weights,imgsz=imgsz,confidence=.001)
    if old_type1_weights:
        detector=HybridDetectorF(detector,DetectorF(old_type1_weights,imgsz=old_type1_imgsz,confidence=.001))
    dataset=read(DATA/'manifest.json');records=[r for r in dataset['records'] if r['split']=='val']
    detections={}
    for r in records:
        frame=load_image(DATA/'detect/images/val'/r['image'])
        detections[r['image']]=serialize_detections(frame,detector.detect(frame))
    # Geometric typed output for old detector; predicted class for multi-class detector.
    typed_predictions={k:[dict(p,routed_type=p['class_name'] if p['class_name'] in TYPES else p['routed_type']) for p in ps]
                       for k,ps in detections.items()}
    synth=[r for r in records if r['is_synthetic']];real=[r for r in records if not r['is_synthetic']]
    metrics={}
    for label,group in [('synthetic_val',synth),('real_development_val',real),('mixed_val',records)]:
        ps={r['image']:detections[r['image']] for r in group}
        typed={r['image']:typed_predictions[r['image']] for r in group}
        metrics[label]={'class_agnostic':detection_metrics(group,ps),'typed':detection_metrics(group,typed,typed=True),
            'polygon':detection_metrics(group,ps,polygon=True) if geometry!='bbox' else None,
            'scope':'Real validation consists of previously exposed public scenes' if label=='real_development_val' else 'Synthetic/mixed validation, not real generalization'}
    write(folder/'validation_predictions.json',detections)
    write(folder/'detector_metrics.json',metrics)
    # End-to-end independent images, exact frozen E preprocessing and CRNN.
    detector.confidence=.5;ocr=FrozenCRNNOCR(BEST,'cuda')
    e=read(ROOT/'configs/image_anpr_e.json')['config'];e['debug']=False
    e['stitch']=stitch
    if filter_override is not None:e['geometry_filter']=filter_override
    pipe=ImagePipelineF(detector,ocr,geometry=geometry,routing=routing,image_config=ImageConfig(**e))
    frames=[(p.name,load_image(p)) for p in sorted(IMAGES.glob('*.jpg'))]
    for n,frame in frames[:5]:pipe.process(frame,image_name=n)
    traces=[];predictions=[];total=[]
    repeats=3 if benchmark else 1
    for repeat in range(repeats):
        current=[]
        for n,frame in frames:
            ocr.synchronize();t=time.perf_counter();out=pipe.process(frame,image_name=n);ocr.synchronize()
            elapsed=(time.perf_counter()-t)*1000;total.append(elapsed)
            current.extend(out['predictions'])
            if repeat==0:traces.append(out);predictions.extend(out['predictions'])
        if repeat:
            assert [(r['image'],r['plate_num'],r['plate_type']) for r in current]==[(r['image'],r['plate_num'],r['plate_type']) for r in predictions]
    write_predictions(folder/'predictions.csv',predictions);write(folder/'trace.json',{'records':traces})
    score,_=official(folder/'predictions.csv',folder/'score')
    from tools.score_development_slices_f import score as score_slices
    slices=score_slices(folder/'predictions.csv')
    # Validation end-to-end truth is exact on synthetic, scene-disjoint but exposed on real.
    val_fpa={};val_cases=[]
    for label,group in [('synthetic_val',synth),('real_development_val',real)]:
        correct=0;target_count=0;per_type=Counter();per_type_gt=Counter()
        for r in group:
            frame=load_image(DATA/'detect/images/val'/r['image'])
            ds=[Detection(tuple(p['bbox']),p['confidence'],p['class_id'],p['class_name'],p.get('quad'),p.get('quad_source'))
                for p in detections[r['image']] if p['confidence']>=.5]
            out=pipe.process(frame,image_name=r['image'],detections=ds);used=set()
            for target in r['annotations']:
                target_count+=1;per_type_gt[target['plate_type']]+=1
                matches=[(bbox_iou(ds[p['detection_index']].bbox,target['bbox']),i,p) for i,p in enumerate(out['predictions']) if i not in used]
                overlap,i,p=max(matches,key=lambda v:v[0],default=(0,None,None))
                full=overlap>=.5 and p['plate_num']==target['plate_num'] and p['plate_type']==target['plate_type']
                if overlap>=.5:used.add(i)
                correct+=int(full);per_type[target['plate_type']]+=int(full)
                val_cases.append({'image':r['image'],'source':label,'plate_type':target['plate_type'],
                    'GT_width':target['bbox'][2]-target['bbox'][0],'correct':bool(full),'IoU':overlap,
                    'prediction':p['plate_num'] if p else None,'GT':target['plate_num']})
        val_fpa[label]={'correct':correct,'total':target_count,'FPA':correct/target_count if target_count else None,
            'per_type':{k:{'correct':per_type[k],'total':per_type_gt[k]} for k in TYPES}}
    result={'config':config,'detector_validation':metrics,'public_development':score,'development_slices':slices,'validation_e2e':val_fpa,
        'GPU_latency_ms':summary(total),'GPU_peak_allocated_MB':torch.cuda.max_memory_allocated()/2**20,
        'GPU_peak_reserved_MB':torch.cuda.max_memory_reserved()/2**20,
        'latency_scope':'Warmup5 then 3 passes of100 preloaded public images, synchronized live detector+OCR compute, no file reading/debug/model load',
        'quad_rectification_attempts':sum(len(r['detections']) for r in traces),
        'quad_rectification_success':sum(d.get('perspective_rectification',False) for r in traces for d in r['detections']),
        'type1a_inputs':sum(d.get('plate_type')=='type1a' and d.get('status')=='ocr' for r in traces for d in r['detections']),
        'type1a_split_success':sum(d.get('plate_type')=='type1a' and d.get('preprocessing',{}).get('effective_mode')=='split_and_stitch' for r in traces for d in r['detections']),
        'public_contains_training_images':dataset['real_import']['original_train_images'],
        'generalization':'GENERALIZATION NOT YET VERIFIED','OCR_trained':False}
    write(folder/'validation_e2e_cases.json',val_cases);write(folder/'result.json',result)
    write(ROOT/'configs/phase_f'/f'{name}_inference.json',config)
    print(json.dumps({'name':name,'FPA':score['full_match_total'],'GPU_ms':result['GPU_latency_ms'],
                      'real_recall':metrics['real_development_val']['class_agnostic']['Recall']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--imgsz',type=int,default=640);p.add_argument('--geometry',default='bbox');p.add_argument('--routing',default='geometry')
    p.add_argument('--no-stitch',action='store_true');p.add_argument('--filter',default=None)
    p.add_argument('--old-type1-weights',default=None);p.add_argument('--old-type1-imgsz',type=int,default=1024)
    args=p.parse_args();evaluate(args.name,args.weights,args.imgsz,args.geometry,args.routing,stitch=not args.no_stitch,filter_override=args.filter,old_type1_weights=args.old_type1_weights,old_type1_imgsz=args.old_type1_imgsz)
