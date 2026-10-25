"""Target-level cascade accounting, paired router/postprocess and geometry checks."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from tools.phase_e import ROOT,REPORT,IMAGES,read,write,official,scorer,load_image
from src.image_anpr import padded_crop,deskew_estimate,rotate
from tools.benchmark_multiline import distance


def main():
    results=read(REPORT/'experiments.json');best=results['BEST_CLASSICAL_E']
    refs=read(REPORT/'detections.json')['records']
    traces=read(best['trace']);observations=[r for t in traces for r in t['detections']]
    module=scorer();gt,_=module.load_gt(str(ROOT.parent/'external_data/organizer_debug/Полуфинал - public/debug_labels.csv'),[])
    metrics,matches=official(Path(best['predictions']),REPORT/'BEST_CLASSICAL_E')
    matching={m.gt.line:m for m in matches};rows=[]
    prediction_order=[p for t in traces for p in t['predictions']]
    confusion=Counter();paired_post=[];global_post=Counter()
    for obs in observations:
        ref=next(r for r in refs if r['image']==obs['image'] and r['index']==obs['detection_index'])
        label=ref['ground_truth'];kind=ref['plate_type']
        if label and obs.get('plate_type'):
            confusion[(kind,obs['plate_type'],ref['crop_quality'])]+=1
        if obs['status']!='ocr':continue
        raw,post=obs['raw_ocr'],obs['postprocessed_ocr']
        same_gt=[g for g in gt if g.image==obs['image'] and g.is_target]
        before=any(module.compare_plates(g.plate,raw,g.ptype)[1] and g.ptype==obs['plate_type'] for g in same_gt)
        after=any(module.compare_plates(g.plate,post,g.ptype)[1] and g.ptype==obs['plate_type'] for g in same_gt)
        effect='fixed' if after and not before else 'broken' if before and not after else 'unchanged' if raw==post else 'changed_without_full_gain'
        global_post[effect]+=1
        if label:
            paired_post.append({'image':obs['image'],'detection_index':obs['detection_index'],
                'ground_truth':label,'GT_plate_type':kind,'route':obs['plate_type'],
                'RAW_OCR':raw,'POSTPROCESSED_OCR':post,'effect':effect,
                'crop_quality':ref['crop_quality'],'split':obs['preprocessing']['effective_mode']})
    for g in gt:
        if not g.is_target:continue
        match=matching[g.line]
        paired=[r for r in refs if r['image']==g.image and r['ground_truth']==g.plate]
        obs=[r for r in observations if any(r['image']==p['image'] and r['detection_index']==p['index'] for p in paired)]
        okobs=[r for r in obs if r.get('geometry_filter_pass') and r['status']=='ocr']
        category='UNKNOWN';note='Partial/masked GT without verified spatial correspondence; no OCR attribution'
        if match.full_ok:
            category='CORRECT';note='Official readable positions and type both correct'
        elif not paired and ('#' not in g.plate or not any(r['image']==g.image for r in observations)):
            category='DETECTOR_MISS';note='All83 proposals manually reviewed; no corresponding complete target found'
        elif paired and not okobs:
            category='GEOMETRY_FILTER_REJECTION';note='Corresponding target proposal fails selected relaxed dimensions/aspect filter'
        elif paired and all(r['crop_quality']=='CLIPPED_TARGET' for r in paired):
            category='DETECTOR_BAD_BOX';note='Visible target characters outside detector crop, not a CRNN error'
        elif okobs and all(r['plate_type']!=g.ptype for r in okobs):
            category='TYPE_NOT_ROUTED';note='Geometric router produces wrong target type; no reliable full-input OCR attribution'
        elif okobs and g.ptype=='type1a' and all(r['preprocessing']['effective_mode']!='split_and_stitch' for r in okobs):
            category='TYPE1A_BAD_SPLIT';note='Correct type requested, but Phase C structural row evidence fails; whole crop fallback is not successful conversion'
        elif okobs:
            raw_good=any(module.compare_plates(g.plate,r['raw_ocr'],g.ptype)[1] and r['plate_type']==g.ptype for r in okobs)
            post_good=any(module.compare_plates(g.plate,r['postprocessed_ocr'],g.ptype)[1] and r['plate_type']==g.ptype for r in okobs)
            if raw_good and not post_good:
                category='POSTPROCESS_FAILURE';note='Correct raw readable positions damaged by postprocessing on paired input'
            else:
                category='OCR_FAILURE';note='Manual full crop/input review and successful same-format stitch (if type1a); raw OCR misses readable glyphs'
        chosen=okobs[0] if okobs else obs[0] if obs else None
        if chosen is None and match.full_ok:
            emitted=prediction_order[match.pred.line-2]
            chosen=next(r for r in observations if r['image']==emitted['image'] and r['detection_index']==emitted['detection_index'])
        detected=bool(paired)
        # A fully correct official wildcard match is evidence of a processed
        # detection even if a manual text correspondence could not be fixed.
        if match.full_ok:detected=True
        rows.append({'image':g.image,'ground_truth':g.plate,'plate_type':g.ptype,'detected':detected,
            'bbox':json.dumps([r['bbox'] for r in paired] or ([chosen['bbox']] if chosen else [])),'geometry_filter_pass':bool(okobs) or match.full_ok,
            'deskew_applied':chosen.get('deskew_applied') if chosen else 'unknown' if match.full_ok else 'not_reached',
            'route':chosen.get('plate_type') if chosen else match.pred.ptype if match.full_ok else 'not_reached',
            'ocr_input_exists':bool(okobs) or match.full_ok,'raw_ocr':' | '.join(r.get('raw_ocr','') for r in okobs) or chosen.get('raw_ocr','') if chosen else '',
            'final_ocr':match.pred.plate if match.pred else '', 'failure_category':category,'notes':note})
    baseline=REPORT/'error_attribution_baseline.csv'
    if not baseline.exists():shutil.copyfile(REPORT/'error_attribution.csv',baseline)
    baseline_summary=REPORT/'error_attribution_baseline_summary.json'
    if not baseline_summary.exists():shutil.copyfile(REPORT/'error_attribution_summary.json',baseline_summary)
    with (REPORT/'error_attribution.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter=';');writer.writeheader();writer.writerows(rows)
    counts=Counter(r['failure_category'] for r in rows);assert sum(counts.values())==115
    summary={'count':115,'correct':metrics['full_match_total'],'categories':dict(counts),
        'scope':'One primary observable cause per officially scored target; 3 ignored literal GT excluded. Earliest confirmed gate wins, wrong crop never charged to OCR.',
        'manual_mapping_limitations':'Public has no official spatial GT; manual proposal pairing, readable-position scorer and review support scoped attribution. UNKNOWN remains explicit.',
        'proposal_coverage':{'all_proposals':83,'manually_paired_proposals':sum(bool(r['ground_truth']) for r in refs),
            'unique_manually_paired_targets':len({(r['image'],r['ground_truth']) for r in refs if r['ground_truth']}),
            'unpaired_partial_proposals':sum(not bool(r['ground_truth']) for r in refs)},
        'false_rejection_rate_paired_proposals':sum(r['failure_category']=='GEOMETRY_FILTER_REJECTION' for r in rows)/sum(bool(r['ground_truth']) for r in refs)}
    write(REPORT/'error_attribution_summary.json',summary)
    write(REPORT/'router_analysis.json',{'confusion_rows':[{'GT_type':a,'predicted_type':b,'crop_quality':c,'count':n} for (a,b,c),n in confusion.items()],
        'auto_FPA':best['official']['full_match_total']/115,
        'oracle_type_FPA':results['oracle_type_routing']['official']['full_match_total']/115,
        'scope':'Manually paired detections, separated by complete vs clipped quality. A clipped square-plate row can look monoline; do not confuse box failure with classifier capacity.',
        'error_examples':[r for r in rows if r['failure_category']=='TYPE_NOT_ROUTED']})
    complete=[]
    for r in paired_post:
        if r['crop_quality']=='CLIPPED_TARGET':continue
        if r['GT_plate_type']=='type1a' and r['split']!='split_and_stitch':continue
        if r['route']!=r['GT_plate_type']:continue
        complete.append(r)
    post={'counts':dict(global_post),'paired_rows':paired_post,'complete_correct_preprocessing_rows':complete,
        'raw_FPA':results['raw_ocr']['official']['full_match_total']/115,
        'post_FPA':results['postprocess']['official']['full_match_total']/115,
        'scope':'Effects on same detection, readable GT positions and type; paired CER limited to manually localized full inputs with successful type1a conversion.',
        'paired_good_input_metrics':{}}
    for kind in ('all','type1','type1a'):
        group=[r for r in complete if kind=='all' or r['GT_plate_type']==kind]
        post['paired_good_input_metrics'][kind]={mode:{'count':len(group),'correct':sum(r[mode]==r['ground_truth'] for r in group),
            'cer':sum(distance(r['ground_truth'],r[mode]) for r in group)/sum(len(r['ground_truth']) for r in group)}
            for mode in ('RAW_OCR','POSTPROCESSED_OCR')}
    write(REPORT/'postprocess_analysis.json',post)
    with (REPORT/'postprocess_cases.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired_post[0]),delimiter=';');writer.writeheader();writer.writerows(paired_post)
    # The sign test is fully label-free and independent of public OCR outcomes.
    image=np.full((180,400),255,np.uint8)
    for y in (45,75,105,135):cv2.line(image,(40,y),(360,y),0,2)
    deskew=[]
    def measure_unrestricted_control(gray):
        lines=cv2.HoughLines(cv2.Canny(gray,50,120),1,np.pi/180,60)
        values=[] if lines is None else [float(theta*180/np.pi-90) for _,theta in lines[:,0]]
        values=[v for v in values if -45<=v<=45]
        return float(np.median(values)) if values else None
    for angle in (-15,-10,-5,5,10,15):
        before=rotate(image,angle);estimated=deskew_estimate(before)
        deskew.append({'input_rotation':angle,'estimated':estimated,
            'after_old_estimated_angle':deskew_estimate(rotate(before,-estimated)),
            'after_corrected_estimated_angle':deskew_estimate(rotate(before,estimated)),
            'after_old_control_angle_wider_window':measure_unrestricted_control(rotate(before,-estimated)),
            'measurement_note':'The legacy +/-20 degree estimation window reports0 if the actual residual moves outside that window; control measurement uses+/-45, not a production algorithm.'})
    write(REPORT/'deskew_quality.json',{'synthetic_line_controls':deskew,
        'real_same_routing_padding_filter':{mode:results[f'joint_full_p00_{mode}']['official'] for mode in ('none','old','corrected')},
        'decision':'NO_DESKEW: old loses3 full plates; corrected fixes sign controls but ties no-deskew FPA and adds complexity. No perspective correction inferred from bbox.'})
    for k in ('legacy_A_cpu','legacy_B_cpu'):
        if (REPORT/f'{k}.csv').exists():official(REPORT/f'{k}.csv',REPORT/k)
    print(json.dumps({'cascade':dict(counts),'postprocess':dict(global_post),'good_input_count':len(complete)},indent=2))


if __name__=='__main__':main()
