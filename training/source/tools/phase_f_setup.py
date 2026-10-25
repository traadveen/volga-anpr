"""Freeze E, publish bounded scene experiments, and run the pre-training quad gate."""
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from dataset.schema import sha256
from dataset.generator.scenes_f import generate
from src.geometry_f import canonical_rectify, bbox_from_quad, order_obb
from src.image_anpr import FrozenCRNNOCR, current_postprocess
from src.ocr_preprocess import prepare_ocr_input
from tools.phase_e import ROOT, IMAGES, BEST, read, write, load_image
from tools.benchmark_multiline import distance

REPORT=ROOT/'reports/phase_f'
DATA=Path(os.environ.get('PHASE_F_DATA',str(ROOT.parent/'generated_data/detector_f_v2')))


def freeze():
    file=REPORT/'freeze.json'
    if file.exists():
        frozen=read(file)
        for path,digest in frozen['hashes'].items():
            assert sha256(Path(path))==digest, f'Frozen E changed: {path}'
        return frozen
    paths=list((ROOT/'reports/phase_e').rglob('*'))
    paths=[p for p in paths if p.is_file()]
    paths += [ROOT/'configs/image_anpr_e.json',BEST,ROOT/'models/yolo11_plate.pt',
              ROOT/'src/image_anpr.py', ROOT/'src/ocr_preprocess.py',ROOT/'src/multiline.py',
              ROOT/'tools/predict_images.py',ROOT/'docs/phase_e_report.md',ROOT/'reports/baseline_predictions.csv']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
        'hashes':{str(p):sha256(p) for p in paths},'fixed_OCR':str(BEST),
        'policy':'No E report/config/source/weight changes; public is development only; no OCR training',
        'selection_policy':'Real development end-to-end FPA, no type1 regression below8/15; prefer <=60ms GPU compute, reject >=100ms; ties simpler/faster. Untouched real test never used for selection.'}
    write(file,result);print(f'Frozen {len(paths)} E artifacts',flush=True)
    return result


def oracle():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=read(ROOT/'reports/phase_e/oracle_expansion/manifest.json')
    assert manifest['review_status']=='PASS'
    ocr=FrozenCRNNOCR(BEST,'cuda');rows=[]
    for r in manifest['records']:
        frame=load_image(IMAGES/r['image']);assert sha256(IMAGES/r['image'])==r['source_sha256']
        x1,y1,x2,y2=map(int,r['detector_bbox'])
        quad,_=canonical_rectify(frame,r['manual_quad'],r['plate_type'],'manual_GT_evaluation_only')
        for mode,crop in [('bbox',frame[y1:y2,x1:x2]),('oracle_quad',quad)]:
            h,w=crop.shape[:2];scale=max(480/w,160/h)
            if scale>1:crop=cv2.resize(crop,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_CUBIC)
            p=prepare_ocr_input(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),r['plate_type'])
            raw,conf,probs=ocr.recognize(p.normalized)
            rows.append({'id':r['id'],'image':r['image'],'plate_type':r['plate_type'],'ground_truth':r['ground_truth'],
                'mode':mode,'raw':raw,'postprocessed':current_postprocess(raw,probs),
                'valid_split':p.diagnostics.get('effective_mode')=='split_and_stitch','preprocessing':p.diagnostics})
    metrics={}
    for mode in ('bbox','oracle_quad'):
        metrics[mode]={}
        for kind in ('all','type1','type1a'):
            group=[r for r in rows if r['mode']==mode and (kind=='all' or r['plate_type']==kind)]
            metrics[mode][kind]={'count':len(group),'raw_correct':sum(r['raw']==r['ground_truth'] for r in group),
                'post_correct':sum(r['postprocessed']==r['ground_truth'] for r in group),
                'raw_FPA':sum(r['raw']==r['ground_truth'] for r in group)/len(group),
                'raw_CER':sum(distance(r['raw'],r['ground_truth']) for r in group)/sum(len(r['ground_truth']) for r in group),
                'successful_splits':sum(r['valid_split'] for r in group) if kind=='type1a' else None}
    gain=metrics['oracle_quad']['all']['raw_FPA']-metrics['bbox']['all']['raw_FPA']
    result={'annotation_sha256':sha256(ROOT/'reports/phase_e/oracle_expansion/manifest.json'),
        'OCR_sha256':sha256(BEST),'scope':'31 selected visible development plates; supplied GT type to both; no detection score or generalization claim',
        'metrics':metrics,'rows':rows,'absolute_raw_FPA_gain':gain,
        'gate':{'keypoint_experiment_justified':gain>=.10,'threshold_predeclared':.10,
            'decision':'Bounded four-corner challenger justified by >=10 percentage-point conditional OCR gain' if gain>=.10 else 'Do not train keypoints'},
        'completed_before_detector_training':True}
    write(REPORT/'oracle_quad.json',result);print(json.dumps(metrics),flush=True)


def import_real():
    manifest=read(DATA/'manifest.json')
    if 'real_import' in manifest:return
    quads=read(ROOT/'reports/phase_e/oracle_expansion/manifest.json')['records']
    with (IMAGES.parent/'debug_labels.csv').open(encoding='utf-8-sig') as f:
        labels=list(csv.DictReader(f,delimiter=';'))
    counts=Counter(r['image'] for r in labels);by_image={}
    for r in quads:by_image.setdefault(r['image'],[]).append(r)
    complete=sorted(name for name,rs in by_image.items() if len(rs)==counts[name])
    # Deterministic scene-disjoint split. The only fully annotated type1 scene is validation.
    val={'img_010.jpg'}|set([n for n in complete if n!='img_010.jpg'][::3])
    added=[]
    for name in complete:
        split='val' if name in val else 'train';frame=load_image(IMAGES/name);h,w=frame.shape[:2]
        annotations=[];lines={t:[] for t in ('detect','obb','pose')}
        for r in by_image[name]:
            q=np.float32(r['manual_quad']);box=np.array(bbox_from_quad(q));obb=order_obb(cv2.boxPoints(cv2.minAreaRect(q)))
            a={'plate_type':r['plate_type'],'plate_num':r['ground_truth'],'bbox':box.tolist(),'quad':q.tolist(),
               'obb':obb.tolist(),'quad_source':'human_plate_border_phase_E_reused_under_phase_F_training_scope'}
            annotations.append(a);c=('type1','type1a','type1b').index(r['plate_type'])
            xywh=np.r_[(box[:2]+box[2:])/2,box[2:]-box[:2]]/[w,h,w,h]
            lines['detect'].append(' '.join(map(str,[c,*xywh])))
            lines['obb'].append(' '.join(map(str,[c,*(obb/[w,h]).ravel()])))
            lines['pose'].append(' '.join(map(str,[c,*xywh,*[v for p in q/[w,h] for v in (*p,2)]])))
        copies=6 if split=='train' else 1
        for copy in range(copies):
            output=f'real_{Path(name).stem}_{copy}.jpg'
            for task in lines:
                os.link(IMAGES/name,DATA/task/'images'/split/output)
                (DATA/task/'labels'/split/Path(output).with_suffix('.txt')).write_text('\n'.join(lines[task])+'\n',encoding='utf-8')
            added.append({'image':output,'original_image':name,'split':split,'width':w,'height':h,
                'is_synthetic':False,'domain':'organizer_public_DEVELOPMENT','annotations':annotations,
                'sha256':sha256(IMAGES/name),'replicate':copy})
    manifest['records']+=added
    manifest['real_import']={'unique_frames':len(complete),'train_frames':len(complete)-len(val),'val_frames':len(val),
        'train_repeat_factor':6,'real_test':0,'exhaustiveness':'All literal organizer GT records in each chosen scene have reviewed manual quad; partial annotation frames excluded',
        'original_train_images':[n for n in complete if n not in val],'original_val_images':sorted(val),
        'scope':'Public DEVELOPMENT source, already exposed in E. Full 115-plate public evaluation includes some training images and cannot establish generalization.',
        'new_untouched_real_test':'UNAVAILABLE'}
    manifest['composition']['real_unique_scenes']=len(complete)
    manifest['composition']['real_scenes']=len(complete)
    manifest['composition']['real_training_replicates']=sum(r['split']=='train' for r in added)
    manifest['composition']['real_validation_scenes']=len(val)
    manifest['limitations'][1]='Real validation exists only from already exposed public development; untouched real holdout unavailable'
    write(DATA/'manifest.json',manifest);write(REPORT/'dataset_composition.json',{
        'dataset':str(DATA),'manifest_sha256':sha256(DATA/'manifest.json'),
        'composition':manifest['composition'],'real_import':manifest['real_import'],'limitations':manifest['limitations']})
    print(json.dumps(manifest['real_import']),flush=True)


def configs():
    folder=ROOT/'configs/phase_f';folder.mkdir(parents=True,exist_ok=True)
    common={'epochs':24,'imgsz':640,'batch':8,'workers':0,'device':0,'optimizer':'AdamW',
        'lr0':.001,'weight_decay':.0005,'patience':5,'seed':20261003,'deterministic':True,
        'amp':False,'mosaic':.3,'close_mosaic':4,'fliplr':0.,'flipud':0.,'degrees':0.,
        'plots':False,'cache':False,'save':True,'pretrained':True,'project':str(ROOT/'runs/phase_f')}
    candidates=[('F02_finetune','detect','models/yolo11_plate.pt'),
                ('F02_clean','detect','yolo11n.pt'),('F03_obb','obb','models/yolo11_plate.pt'),
                ('F04_pose','pose','models/yolo11_plate.pt')]
    for name,task,initial in candidates:
        if (folder/f'{name}.json').exists():continue
        config={'name':name,'task':task,'architecture':f'yolo11n{ "-"+task if task!="detect" else ""}.yaml',
            'initial_weights':str(ROOT/initial),'data':str(DATA/f'{task}.yaml'),'train':dict(common,name=name),
            'fixed_OCR':str(BEST),'real_test_access':'none','initialization':'Transfer matching layers; task/class heads reinitialized as needed',
            'selection':'Validation only; real public is development, not untouched test'}
        write(folder/f'{name}.json',config)
    if not (folder/'F01_old.json').exists():
        write(folder/'F01_old.json',{'weights':str(ROOT/'models/yolo11_plate.pt'),'imgsz':1024,
            'confidence':.5,'iou':.5,'fixed_OCR':str(BEST),'image_config':str(ROOT/'configs/image_anpr_e.json')})


if __name__=='__main__':
    action=sys.argv[1]
    if action=='freeze':freeze()
    elif action=='oracle':oracle()
    elif action=='dataset':
        if not DATA.exists():generate(DATA)
        import_real();configs()
    elif action=='verify-freeze':print(len(freeze()['hashes']))
    else:raise ValueError(action)
