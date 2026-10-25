"""One post-freeze FINAL-I pass over the 40 frozen selected test objects; no tuning."""
from pathlib import Path
import argparse,json,os,sys,time
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter,defaultdict
from dataclasses import asdict
import cv2,numpy as np,torch
from tools.phase_h_common import ROOT,SPLIT,read,load_split,image,bbox,now
from tools.phase_h_evaluate import match_objects
from tools.phase_i_evaluate import summarize
from tools.phase_i_prepare import size_bin
from tools.phase_i_review import atomic_json
from src.inference_io import file_sha256,load_runtime_config,offline_network
from src.image_anpr_f import DetectorF
from src.image_anpr_g import OCRG,ImagePipelineG
from src.multiline_g import ConfigG
from src.plate_formats import apply_russian_profile

REPORT=ROOT/'reports/phase_i'
RUN=REPORT/'test_20261019'
RECEIPT=REPORT/'final_test_run.json'
FREEZE=ROOT/'configs/phase_i/final_i_freeze.json'


def claim_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create blocks simultaneous launches and automatic retries after a failure.
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.flush();os.fsync(stream.fileno())


def verify_freeze():
    frozen=read(FREEZE)
    if frozen['status']!='FINAL-I-FROZEN':raise ValueError('Freeze FINAL-I before independent evaluation')
    if frozen['selection_uses_test'] or frozen['test_prediction_runs_before_freeze']!=0:
        raise ValueError('Invalid post-freeze test provenance')
    for path,sha in frozen['hashes'].items():
        if file_sha256(Path(path))!=sha:raise ValueError('FINAL-I frozen input changed: '+path)
    if file_sha256(SPLIT)!=frozen['split_manifest_sha256']:raise ValueError('Frozen split changed')
    return frozen


def score_scene(rows,detections,output,shape):
    matched=match_objects(rows,detections)
    observed={r['detection_index']:r for r in output['detections']}
    records=[]
    for index,row in enumerate(rows):
        match=matched.get(index);prediction=observed.get(match,{})
        detected=match is not None
        type_ok=detected and prediction.get('plate_type')==row['plate_type']
        text_ok=detected and prediction.get('final_ocr')==row['plate_num']
        correct=type_ok and text_ok
        failure='CORRECT' if correct else 'DETECTOR_MISS' if not detected else \
            'WRONG_TYPE' if not type_ok else 'OCR_OR_RECTIFICATION'
        records.append({'review_id':row['review_id'],'gt_type':row['plate_type'],'gt_text':row['plate_num'],
            'detector_matched':detected,'type_correct':bool(type_ok),'text_correct':bool(text_ok),
            'e2e_correct':bool(correct),'size_bin_640':size_bin(bbox(row),shape)[0],
            'geometry_GT_available':False,'OBB_IoU':None,'angle_error_degrees':None,
            'failure':failure,'observed':prediction})
    return records


def main(preflight_only=False):
    if RECEIPT.exists():raise ValueError('FINAL-I test already claimed; use saved results, never rerun automatically')
    verify_freeze()
    cfg=load_runtime_config(ROOT/'configs/phase_i/final_i.json',ROOT)
    if read(REPORT/'final_test.json')['status']!='NOT_RUN':raise ValueError('Unexpected existing test result')
    rows=load_split('untouched_test')
    if len(rows)!=40 or Counter(r['plate_type'] for r in rows)!={'type1a':20,'type1b':20}:
        raise ValueError('Expected exactly the frozen 20 type1a +20 type1b objects')
    grouped=defaultdict(list)
    for row in rows:grouped[row['source_sha256']].append(row)
    if len(grouped)!=40:raise ValueError('Expected 40 frozen images')
    for sha,group in grouped.items():
        if file_sha256(Path(group[0]['image']))!=sha:raise ValueError('Test image bytes changed')
    history=read(ROOT/'reports/phase_h/final_test.json')
    if history['status']!='COMPLETE' or history['freeze_sha256']!=file_sha256(ROOT/'configs/phase_h/final_h_freeze.json'):
        raise ValueError('Historical H baseline is not bound to its freeze')
    for metric in history['results'].values():
        if metric['overall']['objects']!=40:raise ValueError('Historical baseline membership mismatch')
    sources=[Path(__file__),ROOT/'tools/phase_i_evaluate.py',ROOT/'tools/phase_h_evaluate.py']
    bindings={str(p):file_sha256(p) for p in sources}
    plan={'status':'PREFLIGHT_PASS','created_at':now(),'freeze_sha256':file_sha256(FREEZE),
        'objects':40,'images':40,'input_hashes':list(grouped),'source_hashes':bindings,
        'model':'FINAL-I','scope':'PARTIAL-GT selected objects; previously used in H; not a fresh untouched test',
        'old_model_inference':False,'historical_comparison':'Reuse frozen aggregate BEST_G_TEST / FINAL_H_TEST results',
        'new_model_passes':1,'warmup_test_passes':0,'threshold_tuning':False,'oracle_crop_OCR':False,
        'full_scene_precision_FPR':False,'human_GT_modified':False}
    atomic_json(RUN/'test_plan.json',plan)
    if preflight_only:
        print('PREFLIGHT PASS: 40 frozen objects/images; inference not started',flush=True);return
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no fallback rerun was authorized')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    # Model construction does not run inference; catch load problems before claiming the only run.
    detector=DetectorF(cfg['detector'],'cuda',cfg['imgsz'],cfg['confidence'],cfg['iou'])
    ocr=OCRG(cfg['ocr'],'cuda')
    pipe=ImagePipelineG(detector,ocr,ConfigG(**cfg['split']),multi=cfg['multi'],
        postprocess=cfg['postprocess'],expansion=cfg['expansion'])
    receipt={**plan,'status':'STARTED','started_at':now(),'completed_images':0}
    claim_once(RECEIPT,receipt)
    records=[];timings=[]
    try:
        torch.cuda.reset_peak_memory_stats()
        for n,(sha,gt) in enumerate(grouped.items()):
            frame=image(gt[0]['image'],sha)
            # The prediction path receives pixels and an opaque image hash only, never GT fields.
            ocr.synchronize();start=time.perf_counter()
            ds=detector.detect(frame)
            output=pipe.process(frame,image_name=sha,detections=ds)
            if cfg['recognition_profile']=='russian_v1':output=apply_russian_profile(output)
            ocr.synchronize();elapsed=(time.perf_counter()-start)*1000
            atomic_json(RUN/'images'/(sha+'.json'),{'source_sha256':sha,'detections':[asdict(d) for d in ds],
                'trace':output,'GPU_ms':elapsed,'freeze_sha256':receipt['freeze_sha256']})
            timings.append(elapsed)
            records.extend(score_scene(gt,ds,output,list(frame.shape[1::-1])))
            receipt.update(completed_images=n+1);atomic_json(RECEIPT,receipt)
            if n%10==0:print('FINAL-I TEST',n+1,'/',len(grouped),flush=True)
        verify_freeze()
        if file_sha256(FREEZE)!=receipt['freeze_sha256']:raise ValueError('Freeze changed during inference')
        for path,sha in bindings.items():
            if file_sha256(Path(path))!=sha:raise ValueError('Evaluator changed during inference')
        metrics={'created_at':now(),'model':'FINAL-I','overall':summarize(records),
            'per_type':{k:summarize([r for r in records if r['gt_type']==k]) for k in ('type1','type1a','type1b')},
            'by_size':{k:summarize([r for r in records if r['size_bin_640']==k]) for k in ('tiny','small','medium','large')},
            'failures':dict(Counter(r['failure'] for r in records)),
            'performance_diagnostic':{'GPU_mean_ms':float(np.mean(timings)),'GPU_p95_ms':float(np.percentile(timings,95)),
                'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,'GPU':torch.cuda.get_device_name(0),
                'scope':'One pass, no test-image warmup; includes first-image costs; not the VAL benchmark'},
            'scope':plan['scope'],'precision':None,'FPR':None,'complete_scene_recall':None}
        atomic_json(RUN/'test_objects.json',records);atomic_json(RUN/'test_metrics.json',metrics)
        result={'status':'COMPLETE','created_at':now(),'scope':plan['scope'],
            'freeze_sha256':receipt['freeze_sha256'],'FINAL_I_TEST':metrics,
            'historical_baselines':history['results'],'historical_baseline_sha256':file_sha256(ROOT/'reports/phase_h/final_test.json'),
            'previous_H_evaluation_exists':True,'fresh_untouched_independent_claim_allowed':False,
            'test_prediction_runs_in_phase_i':1,'new_model_images_inferred':40,'baseline_models_rerun':False,
            'changes_after_test':False,'image_level_precision_computed':False,'false_positives_per_image_computed':False}
        atomic_json(REPORT/'final_test.json',result)
        receipt.update(status='COMPLETE',completed_at=now(),final_test_sha256=file_sha256(REPORT/'final_test.json'),
            test_objects_sha256=file_sha256(RUN/'test_objects.json'),new_model_passes=1)
        atomic_json(RECEIPT,receipt)
        print('FINAL-I TEST COMPLETE',metrics['overall'],flush=True)
    except Exception as error:
        receipt.update(status='FAILED_NO_AUTOMATIC_RETRY',error=str(error),failed_at=now())
        atomic_json(RECEIPT,receipt)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    with offline_network():main(args.preflight_only)
