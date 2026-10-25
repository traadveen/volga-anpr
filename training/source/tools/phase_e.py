"""Reproducible Phase E development ablations with frozen weights/detections.

Public GT is confined to this evaluation tool and diagnostic oracle adapters.
It never enters the production image pipeline or checkpoint selection.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from dataset.schema import sha256
from src.image_anpr import (Detection, ImageConfig, ImageANPRPipeline, FrozenCRNNOCR,
                           GeometryPlateRouter, Route, YOLOImageDetector, write_predictions)
from tools.baseline_predict import deny_network, summary

PUBLIC = ROOT.parent/'external_data/organizer_debug/Полуфинал - public'
IMAGES = PUBLIC/'debug set'
REPORT = ROOT/'reports/phase_e'
BEST = ROOT/'models/phase_d/exp_d_01_finetune/best.pt'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def load_image(path):
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unreadable image {path}')
    return image


def scorer():
    path=PUBLIC/'evaluate.py'
    spec=importlib.util.spec_from_file_location('phase_e_frozen_scorer',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def official(csv_path, output):
    module=scorer();problems=[]
    labels, images=module.load_gt(str(PUBLIC/'debug_labels.csv'),problems)
    images |= {p.name for p in IMAGES.iterdir() if p.suffix.lower() in module.IMAGE_EXTS}
    preds=module.load_pred(str(csv_path),images,problems)
    result=module.evaluate(labels,images,preds,{'type1':.15,'type1a':.35,'type1b':.35,'precision':.15})
    module.write_details(str(Path(output).with_suffix('.details.csv')),result)
    serial={k:v for k,v in result.items() if not k.startswith('_')}
    serial['format_problems']=problems
    write(Path(output).with_suffix('.official.json'),serial)
    return serial, result['_matches']


def freeze():
    REPORT.mkdir(parents=True,exist_ok=True)
    path=REPORT/'baseline_freeze.json'
    if path.exists():
        frozen=read(path)
        for name,digest in frozen['hashes'].items():
            assert sha256(Path(name))==digest, f'Frozen file changed: {name}'
        return frozen
    baseline=read(ROOT/'reports/baseline_predictions.manifest.json')
    for name,digest in baseline['source_hashes'].items():
        assert sha256(ROOT/name)==digest
    for name,info in baseline['models'].items():
        assert sha256(ROOT/name)==info['sha256']
    assert sha256(BEST)=='876d8510d507d983d7d911b4e5b135113e98145fe7d4184d74ae9a09262191a9'
    paths=[ROOT/name for name in baseline['source_hashes']]
    paths += [ROOT/name for name in baseline['models']]
    paths += [BEST,ROOT/'reports/baseline_predictions.csv',ROOT/'reports/phase_d/real_best_capture.csv',
              ROOT/'src/multiline.py',ROOT/'src/ocr_preprocess.py',ROOT/'configs/multiline_v2.json',
              PUBLIC/'debug_labels.csv',PUBLIC/'evaluate.py']
    paths += sorted(IMAGES.glob('*.jpg'))
    frozen={'created_utc':datetime.now(timezone.utc).isoformat(),
        'hashes':{str(p):sha256(p) for p in paths},
        'baselines':{'A':'old YOLO/CRNN + old geometry/deskew/postprocess',
                     'B':'old YOLO + Phase D BEST CRNN + old geometry/deskew/postprocess'},
        'scope':'Phase E uses public as a development set for classical inference parameters; no claim of independent test generalization',
        'selection_policy':'overall official FPA, then no type1 regression below8/15, then official score; exact ties prefer fewer components/less padding. Oracle configurations never eligible.',
        'predeclared_candidates':{'filter':['old','none','relaxed','type_aware'],
            'deskew':['old','none','corrected-sign-only'], 'padding':[0,.03,.05,.06,.08,.12],
            'router':'bbox ratio<=3 type1a else type1, explicit classes override; no classifier training',
            'rectification':'observed quad only; none provided by current YOLO'},
        'no_new_training':True}
    write(path,frozen)
    traces=[json.loads(s) for s in (ROOT/'reports/baseline_predictions.trace.jsonl').read_text().splitlines()]
    accepted=read(ROOT/'reports/phase_d/real_input_review/crop_review.json')['records']
    rejected=read(ROOT/'reports/phase_d/rejected_input_review/manifest.json')['records']
    detections=[]
    for trace in traces:
        for index,box in enumerate(trace['detector_boxes_before_geometry_filter']):
            x,y,w,h=box['bbox'];coords=[x,y,x+w,y+h]
            match=None
            for r in accepted:
                if r['image']==trace['image'] and np.allclose(r['bbox_xyxy'],list(map(int,coords)),atol=1.):
                    match={'ground_truth':r['matched_ground_truth'],'plate_type':r['matched_type'],
                           'crop_quality':r['crop_quality'],'review_source':'Phase D complete/cropped actual input review'}
                    break
            if match is None:
                for r in rejected:
                    if r['image']==trace['image'] and np.allclose(r['bbox'],box['bbox'],atol=.001):
                        match={'ground_truth':r['matched_ground_truth'],
                               'plate_type':'type1a' if r['matched_ground_truth'] else None,
                               'crop_quality':'REJECTED_BEFORE_OCR_REQUIRES_NEW_CROP_REVIEW',
                               'review_source':'Phase D all rejected proposals review'}
                        break
            assert match is not None,(trace['image'],coords)
            detections.append({'image':trace['image'],'index':index,'bbox':coords,
                'confidence':box['confidence'],'class_id':box['class_id'],**match})
    assert len(detections)==83
    write(REPORT/'detections.json',{'source_sha256':sha256(ROOT/'reports/baseline_predictions.trace.jsonl'),
                                   'records':detections,'manual_pairing_not_official_spatial_GT':True})
    attribution()
    print('E0-E1 frozen: 100 images,83 detections,115 scored targets',flush=True)
    return frozen


def attribution():
    module=scorer();labels,_=module.load_gt(str(PUBLIC/'debug_labels.csv'),[])
    cases=read(ROOT/'reports/phase_d/real_best_analysis/errors.json')
    refs=read(REPORT/'detections.json')['records']
    _,matches=official(ROOT/'reports/phase_d/real_best_capture.csv',REPORT/'baseline_B')
    matching={m.gt.line:m for m in matches}
    rows=[]
    for gt in labels:
        if not gt.is_target:
            continue
        records=[r for r in refs if r['image']==gt.image and r['ground_truth']==gt.plate]
        case=next((r for r in cases if r.get('gt') and r['gt']['line']==gt.line),None)
        m=matching[gt.line]
        category='CORRECT' if m.full_ok else 'UNKNOWN'
        obs=[]
        if case:
            obs=case['observations']
            primary=case['stage_category']
            if primary=='DETECTION_FAILURE': category='DETECTOR_MISS'
            elif case['rejected_matched_target_proposals']: category='GEOMETRY_FILTER_REJECTION'
            elif primary=='BAD_GEOMETRY': category='DETECTOR_BAD_BOX'
            elif primary=='TYPE1A_SPLIT_FAILURE': category='TYPE_NOT_ROUTED'
            elif primary=='OCR_FAILURE': category='OCR_FAILURE'
            elif primary=='POSTPROCESS_FAILURE': category='POSTPROCESS_FAILURE'
        pairedobs=[o for o in obs if o.get('manual_crop_review',{}).get('matched_ground_truth')==gt.plate]
        rows.append({'image':gt.image,'ground_truth':gt.plate,'plate_type':gt.ptype,
            'detected':bool(records),'bbox':json.dumps([r['bbox'] for r in records]),
            'geometry_filter_pass':any(r['crop_quality']!='REJECTED_BEFORE_OCR_REQUIRES_NEW_CROP_REVIEW' for r in records),
            'deskew_applied':'unknown_per_target' if pairedobs else 'not_reached',
            'route':'type1' if pairedobs else 'not_reached', 'ocr_input_exists':bool(pairedobs),
            'raw_ocr':' | '.join(o.get('raw_prediction','') for o in pairedobs),
            'final_ocr':m.pred.plate if m.pred else '', 'failure_category':category,
            'notes':case['attribution'] if case else 'Official wildcard/type-aware baseline match; manual spatial mapping is scoped to reviewed proposals'})
    columns=list(rows[0])
    with (REPORT/'error_attribution.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,delimiter=';');writer.writeheader();writer.writerows(rows)
    assert len(rows)==115
    write(REPORT/'error_attribution_summary.json',{'total_scored_targets':115,
        'categories':dict(Counter(r['failure_category'] for r in rows)),
        'input_error_records':139,'literal_GT_count':118,'ignored_GT_count':3,
        'notes':'One row per scored target, using official wildcard/ignore semantics. Spatial matching is manual and incomplete; UNKNOWN never charged to OCR.'})


class OracleRouter:
    """Evaluation-only adapter. Not exported by src/image_anpr.py."""
    def __init__(self, mapping): self.mapping=mapping
    def classify(self,crop,detection):
        kind=self.mapping.get(tuple(detection.bbox))
        if kind is None:
            # Do not use absence of a manual match as a GT-based detection filter.
            # Unpaired proposals retain the ordinary automatic routing.
            automatic=GeometryPlateRouter().classify(crop,detection)
            return Route(automatic.plate_type,'unpaired_automatic_fallback_no_GT_type',automatic.aspect_ratio)
        return Route(kind,'ORACLE_TYPE_ROUTING_DIAGNOSTIC_ONLY',crop.shape[1]/crop.shape[0])


def evaluate_config(name, config, ocr, *, oracle=False, debug=False):
    path=REPORT/'ablations'/name
    path.mkdir(parents=True,exist_ok=True)
    if (path/'result.json').exists():
        result=read(path/'result.json')
        assert result['config']==asdict(config) and result['checkpoint_sha256']==sha256(BEST)
        return result
    refs=read(REPORT/'detections.json')['records'];group=defaultdict(list)
    for r in refs:group[r['image']].append(r)
    allrows=[];outputs=[];elapsed=[]
    for i,file in enumerate(sorted(IMAGES.glob('*.jpg'))):
        reference=group[file.name]
        detections=[Detection(tuple(r['bbox']),r['confidence'],r['class_id']) for r in reference]
        router=OracleRouter({tuple(r['bbox']):r['plate_type'] for r in reference}) if oracle else None
        pipeline=ImageANPRPipeline(None,ocr,replace(config,debug=debug),router)
        output=pipeline.process(load_image(file),image_name=file.name,detections=detections,
            debug_directory=path/'debug'/file.stem if debug else None)
        outputs.append(output);allrows.extend(output['predictions']);elapsed.append(output['total_ms'])
    predictions=path/'predictions.csv';write_predictions(predictions,allrows)
    metrics,_=official(predictions,path/'score')
    write(path/'trace.json',outputs)
    observations=[r for out in outputs for r in out['detections']]
    retained=[r for r in observations if r.get('geometry_filter_pass')]
    matched_retained=0;unpaired_retained=0
    for r in retained:
        ref=next(x for x in refs if x['image']==r['image'] and x['index']==r['detection_index'])
        matched_retained+=bool(ref['ground_truth']);unpaired_retained+=not bool(ref['ground_truth'])
    result={'name':name,'config':asdict(config),'checkpoint_sha256':sha256(BEST),'official':metrics,
        'detections':len(observations),'retained':len(retained),
        'manually_paired_target_proposals_retained':matched_retained,
        'unpaired_proposals_retained':unpaired_retained,
        'false_detections_retained':None,
        'retention_note':'Unpaired proposals are partial/unknown and cannot all be called false positives. No global IoU recall without spatial GT.',
        'ocr_calls':sum(r['status']=='ocr' for r in observations),
        'type1a_routed':sum(r.get('plate_type')=='type1a' and r['status']=='ocr' for r in observations),
        'type1a_stitched':sum(r.get('preprocessing',{}).get('effective_mode')=='split_and_stitch' for r in observations),
        'cached_pipeline_ms':summary(elapsed),'debug_artifacts_written':debug,
        'latency_scope':'Cached detector outputs; excludes detector and source-image IO; debug artifact writes included if enabled; exploratory, not final benchmark',
        'oracle':oracle,'predictions':str(predictions),'trace':str(path/'trace.json')}
    write(path/'result.json',result)
    print(f"{name}: {metrics['full_match_total']}/115; type1={metrics['per_type']['type1']['full_match']}/15; type1a={metrics['per_type']['type1a']['full_match']}/100; retained={len(retained)},stitch={result['type1a_stitched']}",flush=True)
    return result


def rank(result):
    m=result['official']
    return (m['per_type']['type1']['full_match']>=8,m['full_match_total'],m['score'],
            result['config']['deskew']=='none',-result['config']['padding'],
            result['config']['geometry_filter']=='none')


def experiments(device):
    freeze();torch.set_num_threads(4);cv2.setNumThreads(1)
    ocr=FrozenCRNNOCR(BEST,device)
    original=ImageConfig(geometry_filter='old',deskew='old',routing='constant',stitch=False)
    results={}
    results['image_independent_old']=evaluate_config('image_independent_old',original,ocr)
    for mode in ('old','none','relaxed','type_aware'):
        name='filter_'+mode
        results[name]=evaluate_config(name,replace(original,geometry_filter=mode),ocr)
    stage=max((results['filter_'+m] for m in ('old','none','relaxed','type_aware')),key=rank)
    config=ImageConfig(**stage['config'])
    # Compare no deskew first; corrected candidate only changes the legacy sign.
    for mode in ('none','old','corrected'):
        name='deskew_'+mode
        results[name]=evaluate_config(name,replace(config,deskew=mode),ocr)
    stage=max((results['deskew_'+m] for m in ('none','old','corrected')),key=rank)
    config=ImageConfig(**stage['config'])
    for padding in (0,.03,.05,.06,.08,.12):
        name=f'padding_{round(padding*100):02}'
        results[name]=evaluate_config(name,replace(config,padding=padding),ocr)
    stage=max((results[f'padding_{round(p*100):02}'] for p in (0,.03,.05,.06,.08,.12)),key=rank)
    config=ImageConfig(**stage['config'])
    results['type_routing_only']=evaluate_config('type_routing_only',replace(config,routing='geometry',stitch=False),ocr)
    results['type1a_stitch']=evaluate_config('type1a_stitch',replace(config,routing='geometry',stitch=True),ocr,debug=True)
    # Joint grid avoids attributing padding/deskew improvements to the router.
    for padding in (0,.03,.05,.06,.08,.12):
        for deskew in ('none','old','corrected'):
            name=f'joint_stitch_p{round(padding*100):02}_{deskew}'
            results[name]=evaluate_config(name,replace(config,routing='geometry',stitch=True,padding=padding,deskew=deskew),ocr)
    for mode in ('old','none','relaxed','type_aware'):
        name='routed_filter_'+mode
        routed=max((r for r in results.values() if r['config']['routing']=='geometry' and r['config']['stitch']),key=rank)
        results[name]=evaluate_config(name,replace(ImageConfig(**routed['config']),geometry_filter=mode),ocr)
    routed=max((r for r in results.values() if r['config']['routing']=='geometry' and r['config']['stitch']),key=rank)
    for padding in (0,.03,.05,.06,.08,.12):
        for deskew in ('none','old','corrected'):
            name=f'joint_full_p{round(padding*100):02}_{deskew}'
            results[name]=evaluate_config(name,replace(ImageConfig(**routed['config']),padding=padding,deskew=deskew),ocr)
    best=max(results.values(),key=rank)
    config=ImageConfig(**best['config'])
    results['raw_ocr']=evaluate_config('raw_ocr_final',replace(config,postprocess='raw'),ocr)
    results['postprocess']=evaluate_config('postprocess_final',replace(config,postprocess='current'),ocr)
    chosen=max((best,results['raw_ocr'],results['postprocess']),key=rank)
    # Overall full accuracy decides postprocessing, never count changes as gains.
    chosen_config=ImageConfig(**chosen['config'])
    results['BEST_CLASSICAL_E']=evaluate_config('BEST_CLASSICAL_E_final',chosen_config,ocr,debug=True)
    results['oracle_type_routing']=evaluate_config('oracle_type_routing_paired',chosen_config,ocr,oracle=True,debug=True)
    write(REPORT/'experiments.json',results)
    write(ROOT/'configs/image_anpr_e.json',{'name':'BEST-CLASSICAL-E','config':asdict(chosen_config),
        'detector':'models/yolo11_plate.pt','ocr':'models/phase_d/exp_d_01_finetune/best.pt',
        'development_set':'public debug; independent generalization not measured',
        'no_GT_routing':True,'no_temporal_logic':True})
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','experiments'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    with deny_network():
        if args.stage=='freeze':freeze()
        else:experiments(args.device)


if __name__=='__main__':main()
