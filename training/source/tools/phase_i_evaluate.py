"""Full frozen REAL VAL evaluation for detector-only pilots; never access test."""
from pathlib import Path
import os,sys
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import time
import cv2
import numpy as np
import torch
from tools.phase_i_dataset import REPORT,DATA,prepare,verify_selected
from tools.phase_i_review import atomic_json
from tools.phase_i_prepare import size_bin
from tools.phase_h_common import ROOT,read,load_split,image,now,bbox
from tools.phase_h_evaluate import match_objects
from src.image_anpr_f import DetectorF
from src.image_anpr_g import OCRG,ImagePipelineG
from src.multiline_g import ConfigG
from src.plate_formats import apply_russian_profile
from src.inference_io import file_sha256,offline_network


def oriented_iou(a,b):
    a=np.array(a,np.float32);b=np.array(b,np.float32)
    inter,_=cv2.intersectConvexConvex(a,b)
    union=abs(cv2.contourArea(a))+abs(cv2.contourArea(b))-inter
    return float(inter/union) if union>0 else 0.


def angle_error(a,b):
    def angle(q):
        q=np.array(q,float);return np.degrees(np.arctan2(*(q[1]-q[0])[::-1]))
    return float(abs((angle(a)-angle(b)+90)%180-90))


def summarize(records):
    n=len(records);matched=[r for r in records if r['detector_matched']]
    correct=sum(r['e2e_correct'] for r in records)
    geometry=[r for r in records if r['geometry_GT_available']]
    localized=[r for r in geometry if r['OBB_IoU'] is not None]
    return {'objects':n,'detected':len(matched),'recall':len(matched)/n if n else None,
        'correct':correct,'FPA':correct/n if n else None,
        'type_correct_given_detection':sum(r['type_correct'] for r in matched)/len(matched) if matched else None,
        'OCR_text_given_detection':sum(r['text_correct'] for r in matched)/len(matched) if matched else None,
        'geometry_GT_objects':len(geometry),'geometry_matched_objects':len(localized),
        'OBB_IoU_mean_given_match':float(np.mean([r['OBB_IoU'] for r in localized])) if localized else None,
        'angle_error_mean_deg_given_match':float(np.mean([r['angle_error_degrees'] for r in localized])) if localized else None,
        'oriented_recall_at_0_5':sum((r['OBB_IoU'] or 0)>=.5 for r in geometry)/len(geometry) if geometry else None}


def evaluate(name,checkpoint,report=REPORT,data=DATA):
    report=Path(report)
    manifest=prepare(report,data);verify_selected(manifest)
    output=report/'evaluations'/name/'val_metrics.json'
    if output.exists():
        old=read(output)
        if old['detector_sha256']!=file_sha256(checkpoint):raise ValueError('Evaluation checkpoint changed')
        return old
    cfg=read(ROOT/'configs/phase_h/final_h.json')
    if file_sha256(Path(cfg['ocr']))!=cfg['ocr_sha256']:raise ValueError('FINAL-H OCR changed')
    rows=load_split('val');grouped=defaultdict(list)
    for r in rows:grouped[r['source_sha256']].append(r)
    snapshot=read(report/'geometry_snapshot.json')
    val_geometry={r['review_id']:snapshot['decisions'][r['review_id']]
        for r in rows if r['review_id'] in manifest['selected_decision_sha256']}
    detector=DetectorF(str(checkpoint),'cuda',cfg['imgsz'],cfg['confidence'],cfg['iou'])
    ocr=OCRG(cfg['ocr'],'cuda')
    pipe=ImagePipelineG(detector,ocr,ConfigG(**cfg['split']),multi=cfg['multi'],
                        postprocess=cfg['postprocess'],expansion=cfg['expansion'])
    first=image(next(iter(grouped.values()))[0]['image'])
    def process(frame,sha,detections=None):
        result=pipe.process(frame,image_name=sha,detections=detections)
        return apply_russian_profile(result) if cfg['recognition_profile']=='russian_v1' else result
    for _ in range(5):process(first,'warmup')
    ocr.synchronize();torch.cuda.reset_peak_memory_stats()
    records=[];traces=[];timings=[];detections={}
    for index,(sha,gt) in enumerate(grouped.items()):
        frame=image(gt[0]['image'],sha)
        # The model receives pixels only; GT enters matching after prediction.
        ocr.synchronize();start=time.perf_counter()
        ds=detector.detect(frame)
        result=process(frame,sha,ds)
        ocr.synchronize();timings.append((time.perf_counter()-start)*1000)
        matched=match_objects(gt,ds)
        by_index={d['detection_index']:d for d in result['detections']}
        detections[sha]=[asdict(d) for d in ds];traces.append(result)
        for j,row in enumerate(gt):
            match=matched.get(j);obs=by_index.get(match,{})
            detected=match is not None
            type_ok=detected and obs.get('plate_type')==row['plate_type']
            text_ok=detected and obs.get('final_ocr')==row['plate_num']
            good=type_ok and text_ok
            geom=val_geometry.get(row['review_id']);iou=angle=None
            if geom and detected and ds[match].quad is not None:
                iou=oriented_iou(geom['obb'],ds[match].quad)
                angle=angle_error(geom['obb'],ds[match].quad)
            failure='CORRECT' if good else 'MISS' if not detected else 'WRONG_CLASS' if not type_ok else \
                'BAD_OBB' if geom and iou is not None and iou<.5 else \
                'BAD_ANGLE' if angle is not None and angle>15 else 'OCR_OR_RECTIFICATION'
            records.append({'review_id':row['review_id'],'gt_type':row['plate_type'],'gt_text':row['plate_num'],
                'detector_matched':detected,'type_correct':bool(type_ok),'text_correct':bool(text_ok),
                'e2e_correct':bool(good),'size_bin_640':size_bin(bbox(row),list(frame.shape[1::-1]))[0],
                'geometry_GT_available':geom is not None,'OBB_IoU':iou,'angle_error_degrees':angle,
                'failure':failure,'observed':obs})
        if index%25==0:print(name,index+1,'/',len(grouped),flush=True)
    overall=summarize(records)
    if name=='I_DET_0_FINAL_H_reference' and (overall['detected']!=38 or overall['correct']!=28):
        raise ValueError('Evaluator did not reproduce the frozen FINAL-H 38/93 recall and 28/93 FPA baseline')
    result={'created_at':now(),'name':name,'detector':str(checkpoint),'detector_sha256':file_sha256(checkpoint),
        'OCR':cfg['ocr'],'OCR_sha256':cfg['ocr_sha256'],'recognition_profile':cfg['recognition_profile'],
        'resolution':cfg['imgsz'],'confidence':cfg['confidence'],'nms_iou':cfg['iou'],
        'overall':overall,'per_type':{k:summarize([r for r in records if r['gt_type']==k]) for k in ('type1','type1a','type1b')},
        'by_size':{k:summarize([r for r in records if r['size_bin_640']==k]) for k in ('tiny','small','medium','large')},
        'failures':dict(Counter(r['failure'] for r in records)),
        'performance':{'images':len(timings),'warmup':5,'GPU_mean_ms':float(np.mean(timings)),
            'GPU_p95_ms':float(np.percentile(timings,95)),'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,'GPU':torch.cuda.get_device_name(0),
            'scope':'Detector + rectification + frozen FINAL-H OCR + postprocessing; excludes image decode/disk; one measured full-val pass.',
            'jury_1050Ti_measured':False},
        'geometry_scope':{'objects':len(val_geometry),'counts':manifest['real_counts']['val'],
            'limitation':'Priority human review subset; not representative full-val geometry.'},
        'precision':None,'FPR':None,'scene_GT':'PARTIAL selected objects; additional detected plates are not labelled false positives',
        'test_read':False,'public_debug_used':False,'selected_geometry_state_revision':manifest['state_revision']}
    verify_selected(manifest)
    atomic_json(output,result);atomic_json(output.with_name('val_objects.json'),records)
    atomic_json(output.with_name('val_traces.json'),traces);atomic_json(output.with_name('val_detections.json'),detections)
    print(name,'recall/FPA',overall['recall'],overall['FPA'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    with offline_network():evaluate(args.name,args.checkpoint)
