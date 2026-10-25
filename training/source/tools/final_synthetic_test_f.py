"""One post-lock held-out synthetic evaluation, never used for model selection."""
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import cv2
from src.image_anpr import FrozenCRNNOCR,ImageConfig
from src.image_anpr_f import DetectorF,ImagePipelineF
from src.geometry_f import bbox_iou
from tools.phase_e import ROOT,read,write,load_image
from tools.phase_f_setup import REPORT,DATA
from tools.evaluate_detector_f import detection_metrics,serialize_detections
from dataset.schema import sha256

def main():
    output=REPORT/'final_synthetic_test.json';assert not output.exists(),'Do not silently evaluate test again'
    config=read(ROOT/'configs/phase_f/best_f.json');lock=read(REPORT/'selection_lock.json')
    assert sha256(ROOT/'configs/phase_f/best_f.json')==lock['config_sha256']
    torch.set_num_threads(4);cv2.setNumThreads(1)
    ds=DetectorF(config['detector'],device='cuda',imgsz=config['imgsz'],confidence=.001)
    pipe=ImagePipelineF(ds,FrozenCRNNOCR(config['ocr'],'cuda'),geometry=config['geometry'],routing=config['routing'],image_config=ImageConfig(**config['image_config']))
    records=[r for r in read(DATA/'manifest.json')['records'] if r['split']=='test'];preds={};cases=[]
    for r in records:
        frame=load_image(DATA/'detect/images/test'/r['image']);observed=ds.detect(frame);preds[r['image']]=serialize_detections(frame,observed)
        output_rows=pipe.process(frame,image_name=r['image'],detections=[d for d in observed if d.confidence>=.5]);used=set()
        det=[d for d in observed if d.confidence>=.5]
        for target in r['annotations']:
            overlap,index,pred=max(((bbox_iou(det[p['detection_index']].bbox,target['bbox']),i,p) for i,p in enumerate(output_rows['predictions']) if i not in used),default=(0,None,None))
            if overlap>=.5:used.add(index)
            correct=overlap>=.5 and pred['plate_type']==target['plate_type'] and pred['plate_num']==target['plate_num']
            cases.append({'image':r['image'],'plate_type':target['plate_type'],'correct':bool(correct),'IoU':overlap,
                'ground_truth':target['plate_num'],'prediction':pred['plate_num'] if pred else None})
    correct=sum(r['correct'] for r in cases)
    write(output,{'selected_experiment':config['selected_experiment'],'selection_lock_sha256':sha256(REPORT/'selection_lock.json'),
        'images':len(records),'GT_targets':len(cases),'full_match':correct,'paired_FPA':correct/len(cases),
        'per_type':{k:{'correct':sum(r['correct'] for r in cases if r['plate_type']==k),'total':sum(r['plate_type']==k for r in cases)} for k in ('type1','type1a','type1b')},
        'runtime_safety_amendments':lock.get('amendments',[]),
        'evaluation_history':'One completed evaluation; an earlier attempt stopped on a runtime exception with no published aggregate metrics' if lock.get('amendments') else 'One completed evaluation',
        'detector':detection_metrics(records,preds),'cases':cases,
        'scope':'Untouched synthetic split used once after development selection lock; authored procedural domain, cannot establish real generalization',
        'independent_real_test':'UNAVAILABLE','generalization':'GENERALIZATION NOT YET VERIFIED'})
    print('Synthetic post-lock test',correct,'/',len(cases),flush=True)

if __name__=='__main__':main()
