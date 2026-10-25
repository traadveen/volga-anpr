"""Bounded checkpoint/threshold/compute tradeoffs on exposed val only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import cv2,torch
from tools.phase_g_common import ROOT,REPORT,CONFIG,IMAGES,read,write
from tools.phase_e import load_image
from tools.phase_g_experiments import run,cached_detections
from src.image_anpr_g import OCRG
from src.multiline_g import ConfigG
from src.geometry_f import bbox_iou,bbox_from_quad

def main():
    print('Loading G checkpoint and threshold diagnostics',flush=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    training=read(REPORT/'training/tournament.json');control=read(REPORT/'training/inference_control.json')['config']
    frames={p.name:load_image(p) for p in sorted(IMAGES.glob('*.jpg'))};cache=cached_detections();results=[]
    # Exact metric ties use balanced replay before the threshold/latency sweep.
    winner=max([r for r in training if r['best_checkpoint']],key=lambda r:(tuple(r['best_key']),r['name']=='G06_balanced'))
    write(REPORT/'training/selected_checkpoint.json',{'candidate':winner,'tie_break':'At equal real_val/full FPA, prefer balanced replay over real-heavy to retain synthetic diversity; stated after exposed training diagnostics, no independent holdout claim.'})
    ocr=OCRG(winner['best_checkpoint'],'cuda');ocr.checkpoint_path=winner['best_checkpoint'];cfg=ConfigG(**control['split'])
    results.append(run('G07_trained_control',cfg,ocr,frames,cache,multi=True,postprocess=control['postprocess']))
    for limit in (3,5):results.append(run(f'G07_multi{limit}',replace(cfg,max_candidates=limit),ocr,frames,cache,multi=True,postprocess=control['postprocess']))
    for method in ('B_projection','C_adaptive'):results.append(run('G07_single_'+method,replace(cfg,method=method),ocr,frames,cache,multi=False,postprocess=control['postprocess']))
    results.append(run('G07_trained_mask_preserve',cfg,ocr,frames,cache,multi=True,postprocess='mask_preserve'))
    write(REPORT/'trained_inference_tournament.json',results)
    eligible=[r for r in results if r['full']['per_type']['type1']['full_match']>=11 and r['full']['per_type']['type1a']['full_match']>=40]
    best=max(eligible,key=lambda r:(r['real_val_exposed']['full_match_total'],r['full']['full_match_total'],-r['config']['split']['max_candidates'] if r['config']['multi'] else 0))
    chosen=best['config'];cfg=ConfigG(**chosen['split']);low=cached_detections(True);curves=[]
    dense=set(read(ROOT/'reports/phase_f/dataset_composition.json')['real_import']['original_val_images'])
    from tools.phase_g_common import spatial_references
    refs=[r for r in spatial_references() if r['image'] in dense]
    for threshold in (.2,.35,.5,.65,.8):
        result=run(f'G08_conf_{round(threshold*100)}',cfg,ocr,frames,low,multi=chosen['multi'],postprocess=chosen['postprocess'],threshold=threshold)
        tp=fp=0
        for name in dense:
            gt=[r for r in refs if r['image']==name];ds=[d for d in low[name] if d.confidence>=threshold];used=set()
            for d in sorted(ds,key=lambda d:-d.confidence):
                matches=[(bbox_iou(d.bbox,bbox_from_quad(r['quad'])),i) for i,r in enumerate(gt) if i not in used]
                value,i=max(matches,default=(0,None))
                if value>=.5:tp+=1;used.add(i)
                else:fp+=1
        spatial={'GT':len(refs),'frames':len(dense),'TP':tp,'FP':fp,'Recall':tp/len(refs),'Precision':tp/(tp+fp) if tp+fp else None,'FP_per_image':fp/len(dense),
            'scope':'8 previously exposed densely annotated F val frames /9 target GT only; class-agnostic IoU>=.5. Not global80-frame recall or FP/image.'}
        result['scoped_detector_real_val']=spatial;write(REPORT/'experiments'/result['config']['name']/'result.json',result);curves.append(result)
    write(REPORT/'threshold_curves.json',curves)
    # Include accuracy/FP tradeoff: must not increase scoped spatial FP/image
    # or organizer val false positives relative to the selected 0.5 control.
    control_curve=next(r for r in curves if r['config']['threshold']==.5)
    eligible=[r for r in curves if r['full']['per_type']['type1']['full_match']>=11 and r['full']['per_type']['type1a']['full_match']>=40
        and r['scoped_detector_real_val']['FP_per_image']<=control_curve['scoped_detector_real_val']['FP_per_image']
        and r['real_val_exposed']['false_positives']<=control_curve['real_val_exposed']['false_positives']]
    best=max(eligible,key=lambda r:(r['real_val_exposed']['full_match_total'],r['full']['full_match_total'],-abs(r['config']['threshold']-.5)))
    write(REPORT/'provisional_best_g.json',{'checkpoint':winner['best_checkpoint'],'result':best,'runtime_gates':'PENDING warm full-pipeline timing, VRAM, CPU/GPU parity, regression and frozen baseline verification',
        'independent_real_status':'GENERALIZATION NOT YET VERIFIED'})
    print('Provisional G',best['config']['name'],best['full']['full_match_total'],best['real_val_exposed']['full_match_total'],flush=True)

if __name__=='__main__':main()
