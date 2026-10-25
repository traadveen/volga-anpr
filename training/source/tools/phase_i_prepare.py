"""Train/val-only geometry audit, annotation proposals and bbox baseline."""
from pathlib import Path
import sys,os
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import json
import math
import cv2
import torch
from PIL import Image
from tools.phase_h_common import ROOT,SPLIT,read,write,load_split,guard_check,image,bbox,now
from tools.phase_h_evaluate import match_objects
from src.image_anpr import Detection
from src.image_anpr_f import DetectorF
from src.geometry_f import bbox_iou
from src.inference_io import file_sha256,offline_network

REPORT=ROOT/'reports/phase_i'
CONFIG=ROOT/'configs/phase_i'


def size_bin(box,shape,imgsz=640):
    width,height=shape
    short=min(box[2]-box[0],box[3]-box[1])*imgsz/max(width,height)
    return ('tiny' if short<8 else 'small' if short<16 else 'medium' if short<32 else 'large'),short


def freeze_inputs():
    guard_check()
    REPORT.mkdir(parents=True,exist_ok=True)
    CONFIG.mkdir(parents=True,exist_ok=True)
    (ROOT/'models/phase_i').mkdir(parents=True,exist_ok=True)
    target=REPORT/'input_guard.json'
    if target.exists():
        value=read(target)
        for path,sha in value['hashes'].items():
            if file_sha256(Path(path))!=sha:
                raise ValueError('Frozen input changed: '+path)
        return value
    hashes={**read(ROOT/'reports/phase_h/split_lock.json')['hashes'],
            **read(ROOT/'configs/phase_h/final_h_freeze.json')['hashes']}
    for p in [ROOT/'configs/phase_h/final_h_freeze.json',ROOT/'reports/phase_h/final_test.json',
              ROOT/'reports/phase_h/final_test_run.json',ROOT/'reports/phase_h/split_lock.json',
              ROOT.parent/'datasets/reports/data2_review_state.json']:
        hashes[str(p)]=file_sha256(p)
    value={'created_at':now(),'hashes':hashes,'test_samples_read':False,
           'test_policy':'No test images, GT rows or new inference until FINAL-I; old H test already measured, so never call a future reuse a fresh untouched test.',
           'annotations':'New geometry sidecar only; frozen labels, membership, source/duplicate groups unchanged'}
    write(target,value)
    return value


def prepare():
    freeze_inputs()
    if (REPORT/'annotation_index.json').exists():
        return read(REPORT/'geometry_summary.json')
    policy={'imgsz':640,'threshold':.5,'IoU_match':.5,'annotation_proposal_confidence':.05,
        'annotation_proposal_min_bbox_IoU':.1,'proposal_is_GT':False,
        'size_bins':'Shorter GT bbox edge after resize longest frame edge to640: tiny<8, small[8,16), medium[16,32), large>=32 pixels. Diagnostic bins, not accuracy targets.',
        'resolution_followup':'640 vs960 only if at least half of val misses fall into tiny/small bins; no test inspection.',
        'training_gate':'Human-verified geometry required; never train from an empty sidecar or auto proposals. Preserve val membership and FINAL-H OCR.',
        'full_scene_GT':'Unknown for train/val; isolated training regions require separate human completeness confirmation before ordinary YOLO export.'}
    write(CONFIG/'annotation_policy.json',policy)
    rows=[]
    summary={}
    shapes={}
    for split in ('train','val'):
        inputs=load_split(split)
        counts=Counter()
        for row in inputs:
            sha=row['source_sha256']
            if sha not in shapes:
                if file_sha256(Path(row['image']))!=sha:
                    raise ValueError('Image hash mismatch')
                with Image.open(row['image']) as im:
                    shapes[sha]=list(im.size)
            box=bbox(row)
            bucket,edge=size_bin(box,shapes[sha])
            quad=json.loads(row['quad']) if row.get('quad') else None
            counts[row['plate_type']]+=1
            counts['quad_available']+=bool(quad)
            counts['verified_quad']+=bool(quad) and str(row.get('quad_GT_verified')).lower()=='true'
            counts['verified_OBB']+=0
            counts[bucket]+=1
            rows.append({k:row.get(k,'') for k in ('review_id','sample_id','source_image_id','image',
                'plate_num','plate_type','source_sha256','source','source_family','original_image_group_id',
                'duplicate_group_id','detector_usable','ocr_usable','bbox_verified')} |
                {'split':split,'bbox':box,'image_size':shapes[sha],'size_bin_640':bucket,
                 'short_edge_pixels_640':edge,'quad':quad,'quad_verified':False,
                 'proposal_quad':None,'proposal_source':'none','proposal_verified':False})
        summary[split]={'objects':len(inputs),**{k:counts[k] for k in
            ('type1','type1a','type1b','quad_available','verified_quad','verified_OBB','tiny','small','medium','large')}}
    cfg=read(ROOT/'configs/phase_g/best_g.json')
    detector=DetectorF(cfg['detector'],'cuda',640,.05,cfg['iou'])
    # Existing H val detections are a byte-bound .5 baseline, not new test access.
    cached=read(ROOT/'reports/phase_h/detections_val.json')
    if cached['split_manifest_sha256']!=file_sha256(SPLIT):
        raise ValueError('Baseline cache from another split')
    frames=defaultdict(list)
    for row in rows:
        frames[row['source_sha256']].append(row)
    for index,(sha,gt) in enumerate(frames.items()):
        frame=image(gt[0]['image'],sha)
        low=detector.detect(frame)
        high=[Detection(**d) for d in cached['images'][sha]] if gt[0]['split']=='val' else [d for d in low if d.confidence>=.5]
        matched=match_objects(gt,high)
        for i,row in enumerate(gt):
            row['baseline_detected']=i in matched
            row['baseline_detection']=asdict(high[matched[i]]) if i in matched else None
            eligible=[d for d in low if d.quad is not None and bbox_iou(row['bbox'],d.bbox)>=.1]
            if eligible:
                best=max(eligible,key=lambda d:(bbox_iou(row['bbox'],d.bbox),d.confidence))
                row.update(proposal_quad=[list(p) for p in best.quad],
                    proposal_source='BEST_G_frozen_OBB_annotation_conf_0.05_NOT_GT',
                    proposal_confidence=best.confidence,proposal_bbox_IoU=bbox_iou(row['bbox'],best.bbox),
                    proposal_class=best.class_name)
            row['priority_reason']='DETECTOR_MISS' if not row['baseline_detected'] else 'LOCALIZATION_OR_HARD_CASE' if row.get('proposal_bbox_IoU',0)<.75 else 'NORMAL'
        if index%60==0:
            print('Geometry proposals',index+1,'/',len(frames),flush=True)
    rows.sort(key=lambda r:({'type1b':0,'type1a':1,'type1':2}[r['plate_type']],
        r['baseline_detected'],r.get('proposal_bbox_IoU',0)>=.75,r['short_edge_pixels_640'],r['split']!='val',r['review_id']))
    write(REPORT/'annotation_index.json',{'created_at':now(),'records':rows,
        'policy':policy,'split_checksum':file_sha256(SPLIT),'proposal_weights_sha256':cfg['detector_sha256'],
        'training_started':False,'human_verified_geometry_created':0,'test_read':False})
    val=[r for r in rows if r['split']=='val']
    def recall(rs):
        return {'objects':len(rs),'matched':sum(r['baseline_detected'] for r in rs),
                'recall':sum(r['baseline_detected'] for r in rs)/len(rs) if rs else None}
    baseline={'created_at':now(),'model':'BEST-G F03 frozen','resolution':640,'confidence':.5,
        'matching':'One-to-one bbox IoU>=.5, geometry-only matching; no type/text matching',
        'overall':recall(val),'per_type':{k:recall([r for r in val if r['plate_type']==k]) for k in ('type1','type1a','type1b')},
        'by_size':{k:recall([r for r in val if r['size_bin_640']==k]) for k in ('tiny','small','medium','large')},
        'OBB_IoU':None,'angle_error':None,'geometry_metric_status':'Awaiting human quad GT',
        'precision':None,'precision_status':'Full scene GT unavailable; extra plates not false positives by default',
        'end_to_end_FINAL_H_FPA':read(ROOT/'reports/phase_h/evaluations/H4_H1_H_OCR_1_real_heavy/val_metrics.json')['overall']['FPA'],
        'source':'Frozen H .5 detector cache verified against split checksum; not low-confidence annotation proposals',
        'test_used':False}
    misses=[r for r in val if not r['baseline_detected']]
    tiny=sum(r['size_bin_640'] in ('tiny','small') for r in misses)
    baseline['resolution_960_justified']=bool(misses and tiny/len(misses)>=.5)
    baseline['tiny_small_miss_count']=tiny
    baseline['total_miss_count']=len(misses)
    write(REPORT/'detector_baseline.json',baseline)
    summary.update(created_at=now(),proposal_count=sum(r['proposal_quad'] is not None for r in rows),
        no_proposal_count=sum(r['proposal_quad'] is None for r in rows),
        human_verified_geometry=0,training_allowed=False,
        training_blocker='Human geometry review and training-region completeness not yet confirmed',
        test_read=False,split_changed=False)
    write(REPORT/'geometry_summary.json',summary)
    freeze_inputs()
    print(json.dumps(summary,indent=2),flush=True)
    print('BASELINE',json.dumps(baseline,ensure_ascii=True),flush=True)
    return summary


if __name__=='__main__':
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    with offline_network():
        prepare()
