"""Frozen synthetic validation and once-only post-lock OCR crop test."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import cv2,numpy as np,torch
from tools.phase_g_common import ROOT,REPORT,read,write
from tools.phase_e import load_image
from tools.phase_g_metrics import ocr_metrics
from src.image_anpr import Detection,FrozenCRNNOCR,current_postprocess
from src.image_anpr_g import OCRG,ImagePipelineG,gray_for_ocr
from src.multiline_g import ConfigG,candidates,candidate_score
from src.real_data_g import valid_complete_plate
from src.ocr_preprocess import prepare_ocr_input
from src.geometry_f import bbox_iou
from dataset.schema import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['val','test']);args=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    provisional=read(REPORT/'provisional_best_g.json');spec=provisional['result']['config'];ocr=OCRG(provisional['checkpoint'],'cuda');cfg=ConfigG(**spec['split'])
    if args.mode=='val':
        data=ROOT.parent/'generated_data/detector_f_v2';manifest=read(data/'manifest.json')
        records=[r for r in manifest['records'] if r['split']=='val' and r['is_synthetic']]
        cache=read(ROOT/'reports/phase_f/experiments/F03_obb_transfer_640_detector/validation_predictions.json')
        pipeline=ImagePipelineG(None,ocr,cfg,multi=spec['multi'],postprocess=spec['postprocess'],expansion=spec['expansion']);cases=[]
        for r in records:
            ds=[Detection(tuple(d['bbox']),d['confidence'],d['class_id'],d['class_name'],d.get('quad'),d.get('quad_source')) for d in cache[r['image']] if d['confidence']>=spec['threshold']]
            out=pipeline.process(load_image(data/'detect/images/val'/r['image']),image_name=r['image'],detections=ds);used=set()
            for target in r['annotations']:
                overlap,index,pred=max(((bbox_iou(ds[v['detection_index']].bbox,target['bbox']),i,v) for i,v in enumerate(out['predictions']) if i not in used),default=(0,None,None))
                if overlap>=.5:used.add(index)
                correct=overlap>=.5 and pred['plate_type']==target['plate_type'] and pred['plate_num']==target['plate_num']
                cases.append({'image':r['image'],'plate_type':target['plate_type'],'ground_truth':target['plate_num'],'correct':bool(correct),'prediction':pred['plate_num'] if pred else '', 'IoU':overlap})
        f=read(ROOT/'reports/phase_f/experiments/F03_obb_transfer_640_detector/result.json')['validation_e2e']['synthetic_val']
        result={'scope':'Paired E2E synthetic validation, exposed authored F val scenes, NOT held-out F test or real generalization. Reuses exact frozen F observed detector proposals; GT supplied only to evaluator.', 'detector_cached':True,'baseline_F':f,'BEST_G':{'correct':sum(r['correct'] for r in cases),'total':len(cases),'per_type':{k:{'correct':sum(r['correct'] for r in cases if r['plate_type']==k),'total':sum(r['plate_type']==k for r in cases)} for k in ('type1','type1a','type1b')}},'cases':cases}
        write(REPORT/'synthetic_validation_regression.json',result);print('Syntheticval',result['BEST_G'],flush=True)
    else:
        dest=REPORT/'terminal_synthetic_ocr_test.json';assert not dest.exists(),'Test aggregate already evaluated; never silently rerun'
        lock=read(REPORT/'selection_lock.json');assert sha256(ROOT/'configs/phase_g/best_g.json')==lock['config_sha256']
        for path,expected in lock['hashes'].items():assert sha256(Path(path))==expected,path
        data=ROOT.parent/'generated_data/targeted_g_v1';freeze=read(data/'test/freeze.json')
        for path,expected in freeze['files'].items():assert sha256(Path(path))==expected,path
        baseline=FrozenCRNNOCR(read(ROOT/'configs/phase_f/best_f.json')['ocr'],'cuda');rows=[]
        for r in read(data/'manifest.json')['records']:
            if r['split']!='test':continue
            gray=gray_for_ocr(load_image(Path(r['image'])));prepared=prepare_ocr_input(gray,r['plate_type'])
            raw,conf,probs=baseline.recognize(prepared.normalized);post=current_postprocess(raw,probs)
            cs=candidates(gray,r['plate_type'],cfg);decoded=ocr.recognize_candidates([c[0] for c in cs])
            idx=max(range(len(cs)),key=lambda i:candidate_score(decoded[i][0],decoded[i][3],cs[i][2]['structural_confidence'],valid_complete_plate(decoded[i][0])))
            gr,gc,gp,_=decoded[idx]
            rows.append({'image':r['image'],'ground_truth':r['plate_num'],'plate_type':r['plate_type'],'F_raw':raw,'F_post':post,'G_raw':gr,'G_post':current_postprocess(gr,gp)})
        result={'scope':'Once-only new90 identity-disjoint targeted synthetic OCR crops after BEST-G lock. Known supplied layout, no detector or scene E2E; authored yellow type1b is not real type1b. Comparing complete F and G OCR preprocessing+weights.', 'selection_lock_sha256':sha256(REPORT/'selection_lock.json'),'evaluation_count':1,'metrics':{k:{field:ocr_metrics([r for r in rows if k=='ALL' or r['plate_type']==k],field) for field in ('F_raw','F_post','G_raw','G_post')} for k in ('ALL','type1','type1a','type1b')},'rows':rows,'test_used_to_select':False}
        write(dest,result);write(data/'test/evaluation_receipt.json',{'result':str(dest),'result_sha256':sha256(dest),'selection_lock_sha256':result['selection_lock_sha256'],'evaluation_count':1})
        print('TerminalOCR',result['metrics']['ALL'],flush=True)

if __name__=='__main__':main()
