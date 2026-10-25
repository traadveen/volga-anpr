"""Selected-object metrics and failure attribution, without complete-scene claims."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
from collections import Counter,defaultdict
from dataclasses import asdict
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tools.phase_h_common import (ROOT,REPORT,CONFIG,SPLIT,read,write,load_split,image,bbox,
    target_choices,decode_choices,postprocess,guard_check,now)
from src.image_anpr import Detection
from src.image_anpr_f import DetectorF
from src.image_anpr_g import OCRG,ImagePipelineG
from src.multiline_g import ConfigG
from src.geometry_f import bbox_iou,validate_quad
from src.plate_formats import apply_russian_profile


def match_objects(rows,detections,threshold=.5):
    """Maximize valid match cardinality before IoU, independent of text/type."""
    if not rows or not detections:
        return {}
    overlaps = np.array([[bbox_iou(bbox(r),d.bbox) for d in detections] for r in rows])
    cost = np.where(overlaps >= threshold,min(overlaps.shape)+1+overlaps,0)
    a,b = linear_sum_assignment(cost,maximize=True)
    return {int(i):int(j) for i,j in zip(a,b) if overlaps[i,j]>=threshold}


def metric(rows):
    def summarize(rs):
        count = len(rs)
        ratio = lambda k: sum(bool(r[k]) for r in rs)/count if count else None
        matched = [r for r in rs if r['detector_matched']]
        square = [r for r in matched if r['gt_type']=='type1a']
        return {'objects':count,'correct':sum(r['e2e_correct'] for r in rs),
                'FPA':ratio('e2e_correct'),'OCR_good_bbox_raw_FPA':ratio('good_raw_correct'),
                'OCR_good_bbox_post_FPA':ratio('good_post_correct'),
                'selected_object_detector_recall':ratio('detector_matched'),
                'type_accuracy_all_targets':ratio('type_correct'),
                'type_accuracy_given_detection':sum(r['type_correct'] for r in matched)/len(matched) if matched else None,
                'type1a_structural_split_success':sum(r['split_success'] for r in square)/len(square) if square else None,
                'split_success_denominator':len(square),'split_metric_is_proxy_without_row_GT':True}
    return {'overall':summarize(rows),
            'per_type':{k:summarize([r for r in rows if r['gt_type']==k]) for k in ('type1','type1a','type1b')},
            'waterfall':dict(Counter(r['failure'] for r in rows))}


def evaluate(name,checkpoint,profile='frozen_g',split='val',device='cuda',cache=True,save=True):
    rows = load_split(split)
    cfg = read(ROOT/'configs/phase_g/best_g.json')
    detector_cache_path = REPORT/('detections_'+split+'.json')
    if cache and detector_cache_path.exists():
        frozen_cache = read(detector_cache_path)
        if frozen_cache['split_manifest_sha256'] != guard_check()['hashes'][str(SPLIT)]:
            raise ValueError('Detection cache from another split')
        saved_detections = frozen_cache['images']
        detector = None
    else:
        saved_detections = {}
        detector = DetectorF(cfg['detector'],device,cfg['imgsz'],cfg['confidence'],cfg['iou'])
    ocr = OCRG(str(checkpoint),device)
    pipeline = ImagePipelineG(detector,ocr,ConfigG(**cfg['split']),multi=cfg['multi'],
                              postprocess=cfg['postprocess'],expansion=cfg['expansion'])
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['source_sha256']].append(r)
    records = []
    traces = []
    timings = []
    for sha,gt in grouped.items():
        frame = image(gt[0]['image'],sha)
        if sha in saved_detections:
            ds = [Detection(**d) for d in saved_detections[sha]]
        else:
            ds = detector.detect(frame)
            saved_detections[sha] = [asdict(d) for d in ds]
        output = pipeline.process(frame,image_name=sha,detections=ds)
        if profile == 'russian_v1':
            output = apply_russian_profile(output)
        traces.append(output)
        timings.append(output['total_ms'])
        matched = match_objects(gt,ds)
        by_index = {d['detection_index']:d for d in output['detections']}
        for i,row in enumerate(gt):
            choices = target_choices(frame,row,cfg)
            (raw,conf,probs,logp),diag,_ = decode_choices(ocr,choices)
            good_final = postprocess(raw,probs,row['plate_type'],profile)
            j = matched.get(i)
            observed = by_index.get(j,{})
            detected = j is not None
            correct_type = detected and observed.get('plate_type') == row['plate_type']
            correct = correct_type and observed.get('final_ocr') == row['plate_num']
            split_ok = observed.get('preprocessing',{}).get('effective_mode') == 'split_and_stitch'
            if correct:
                failure = 'CORRECT'
            elif not detected:
                failure = 'DETECTOR_MISS'
            elif not correct_type:
                failure = 'WRONG_TYPE'
            elif ds[j].quad is None:
                failure = 'BAD_OBB'
            elif observed.get('G_fallback'):
                failure = 'BAD_RECTIFICATION'
            elif observed.get('raw_ocr') == row['plate_num']:
                failure = 'POSTPROCESS'
            elif row['plate_type']=='type1a' and not split_ok:
                failure = 'TYPE1A_SPLIT'
            elif raw != row['plate_num']:
                failure = 'OCR_GOOD_INPUT'
            else:
                failure = 'UNKNOWN'
            records.append({'review_id':row['review_id'],'gt_type':row['plate_type'],
                'gt_text':row['plate_num'],'detector_matched':detected,'type_correct':bool(correct_type),
                'e2e_correct':bool(correct),'good_raw_correct':raw==row['plate_num'],
                'good_post_correct':good_final==row['plate_num'],'split_success':split_ok,
                'good_crop_raw':raw,'good_crop_final':good_final,'prediction':observed,
                'failure':failure,'good_crop_preprocessing':diag,
                'good_crop_input_scope':'Verified bbox crop; no GT quad available'})
    summary = metric(records)
    summary.update(model=name,profile=profile,split=split,created_at=now(),
        source_checkpoint=str(checkpoint),image_count=len(grouped),
        cached_pipeline_timing_ms={'mean':float(np.mean(timings)),'p95':float(np.percentile(timings,95)),
                                  'scope':'Diagnostic; excludes detector and good-crop OCR'},
        scope='PARTIAL-GT selected objects; precision/FP per image/complete-scene recall not computed')
    if save:
        write(REPORT/'evaluations'/name/(split+'_metrics.json'),summary)
        write(REPORT/'evaluations'/name/(split+'_objects.json'),records)
        write(REPORT/'evaluations'/name/(split+'_traces.json'),traces)
    if detector is not None:
        write(detector_cache_path,{'split_manifest_sha256':guard_check()['hashes'][str(SPLIT)],'images':saved_detections})
    guard_check()
    print(name,split,'FPA',summary['overall']['FPA'],'good bbox OCR',summary['overall']['OCR_good_bbox_raw_FPA'],flush=True)
    return summary,records


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',default='H0_BEST_G')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--profile',choices=['frozen_g','russian_v1'],default='frozen_g')
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    from src.inference_io import offline_network
    with offline_network():
        evaluate(args.name,args.checkpoint or read(ROOT/'configs/phase_g/best_g.json')['ocr'],args.profile)
