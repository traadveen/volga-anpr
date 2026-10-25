"""Save all literal errors and separate observed pipeline gates from OCR causes."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.evaluation import load_ground_truth,load_predictions,evaluate


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gt',required=True)
    parser.add_argument('--predictions',required=True);parser.add_argument('--images',required=True)
    parser.add_argument('--captures',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--crop-review',default='reports/phase_d/real_input_review/crop_review.json')
    parser.add_argument('--rejected-review',default='reports/phase_d/rejected_input_review/manifest.json')
    args=parser.parse_args();pred_path=Path(args.predictions);output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    gt=load_ground_truth(args.gt);pred=load_predictions(pred_path)
    diagnostic=evaluate(gt,pred)
    (output/'literal_evaluation.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    traces={r['image']:r for r in (json.loads(line) for line in pred_path.with_suffix('.trace.jsonl').read_text().splitlines())}
    captures=json.loads(Path(args.captures).read_text())['observations']
    reviewed=json.loads(Path(args.crop_review).read_text())['records']
    rejected=json.loads(Path(args.rejected_review).read_text())['records']
    for r in rejected:
        if traces[r['image']]['sha256']!=r['source_sha256']:raise ValueError('Rejected crop review source changed')
        if not any(b['bbox']==r['bbox'] and not b['legacy_geometry_pass'] for b in traces[r['image']]['detector_boxes_before_geometry_filter']):
            raise ValueError('Reviewed rejected detector proposal changed')
    review_by_input={(r['image'],tuple(r['bbox_xyxy'])):r for r in reviewed}
    by_image=defaultdict(list)
    paired=[]
    for record in captures:
        review=review_by_input.get((record['image'],tuple(record['bbox_xyxy'])))
        digest=hashlib.sha256(Path(record['OCR_input']).read_bytes()).hexdigest()
        if not review or review['ocr_input_sha256']!=digest:
            raise ValueError('Crop review is not applicable to these actual input pixels')
        record['manual_crop_review']=review
        alphabet='0123456789ABCEHKMOPTXY';previous=None;decoded=[]
        for token in record['raw_ctc_tokens']:
            if token!=previous and token<len(alphabet):decoded.append(alphabet[token])
            previous=token
        record['raw_prediction']=''.join(decoded)
        by_image[record['image']].append(record)
        if review['crop_quality']=='COMPLETE_CORRECT_CROP':
            trace_observation=next(o for o in traces[record['image']]['observations']
                if o['bbox'][:2]==record['bbox_xyxy'][:2] and o['bbox'][2:]==[record['bbox_xyxy'][2]-record['bbox_xyxy'][0],record['bbox_xyxy'][3]-record['bbox_xyxy'][1]])
            paired.append({'image':record['image'],'ground_truth':review['matched_ground_truth'],
                'plate_type':review['matched_type'],'raw':record['raw_prediction'],
                'postprocessed':trace_observation['postprocessed_ocr'],'confidence':record['confidence']})
    all_gt=defaultdict(list)
    for r in gt:
        if r.target:all_gt[r.image].append(r.plate_num)
    post_counts=Counter();post_rows=[]
    for trace in traces.values():
        for emission in trace.get('observations',[]):
            raw=emission['raw_ocr'];final=emission['postprocessed_ocr'];labels=all_gt[trace['image']]
            before=raw in labels;after=final in labels
            category='fixed' if after and not before else 'damaged' if before and not after else 'unchanged' if raw==final else 'changed_without_exact_gain'
            post_counts[category]+=1
            post_rows.append({'image':trace['image'],'bbox':emission['bbox'],'raw':raw,'final':final,
                              'raw_text_exact':before,'final_text_exact':after,'effect':category})
    cases=[]
    for index,error in enumerate(diagnostic['errors']):
        name=error['image'];trace=traces[name];actual=error['gt']
        folder=output/'errors'/f'case_{index:04}';folder.mkdir(parents=True,exist_ok=True)
        source=Path(args.images)/name;shutil.copyfile(source,folder/('original'+source.suffix))
        observations=[]
        for number,record in enumerate(by_image[name]):
            item=record.copy()
            for field in ('detected_crop','rectified_crop','OCR_input'):
                if item.get(field):
                    target=folder/f'observation_{number}_{field}.png'
                    shutil.copyfile(item[field],target);item[field]=str(target)
            observations.append(item)
        boxes=trace.get('detector_boxes_before_geometry_filter',[])
        matched=[r for r in by_image[name] if actual and r['manual_crop_review']['matched_ground_truth']==actual['plate_num']]
        blocked=[r for r in rejected if r['image']==name and actual and r['matched_ground_truth']==actual['plate_num']]
        for n,r in enumerate(blocked):
            shutil.copyfile(r['crop'],folder/f'rejected_target_{n}_crop.png')
        if actual is None and error['prediction'] and error['prediction']['bbox']:
            x,y,w,h=error['prediction']['bbox']
            matched=[r for r in by_image[name] if r['bbox_xyxy']==[x,y,x+w,y+h]]
        paired_gt=actual['plate_num'] if actual else matched[0]['manual_crop_review']['matched_ground_truth'] if matched else None
        paired_type=actual['plate_type'] if actual else matched[0]['manual_crop_review']['matched_type'] if matched else None
        primary=None;certainty='unresolved: GT has no spatial boxes; do not invent OCR attribution'
        if actual and not boxes:
            primary='DETECTION_FAILURE';certainty='confirmed observed gate: no YOLO proposal; GT localization unannotated'
        elif blocked:
            primary='BAD_GEOMETRY';certainty='manually matched target proposal rejected by frozen geometry filter before OCR'
        elif matched and all(r['manual_crop_review']['crop_quality']!='COMPLETE_CORRECT_CROP' for r in matched):
            primary='BAD_GEOMETRY';certainty='manual spatial/text pairing: actual crop is missing target characters before OCR'
        elif paired_gt and any(r['raw_ocr']==paired_gt and r['postprocessed_ocr']!=paired_gt for r in trace.get('observations',[])):
            primary='POSTPROCESS_FAILURE';certainty='exact same-image GT text destroyed in an emitted row'
        elif matched and paired_type=='type1a':
            primary='TYPE1A_SPLIT_FAILURE';certainty='type1a conversion/type output absent in frozen legacy pipeline; no claim that a split was attempted'
        elif matched and paired_type=='type1' and any(r['raw_prediction']!=paired_gt for r in matched):
            primary='OCR_FAILURE';certainty='manually verified complete correct crop, wrong raw CTC output'
        elif actual and '#' not in actual['plate_num'] and not matched:
            primary='DETECTION_FAILURE';certainty='all83 detector proposals manually reviewed; no proposal matched this complete target. Manual correspondence, not official bbox recall.'
        # OCR_FAILURE requires a manually verified complete, correct input.
        # Unknown or clipped crops are retained without assigning this cause.
        case={'id':index,**error,'stage_category':primary,'attribution':certainty,
              'detected_bboxes':boxes,'observations':observations,
              'original_image':str(folder/('original'+source.suffix)),
              'ground_truth':paired_gt,
              'raw_predictions':[o['raw_ocr'] for o in trace.get('observations',[])],
              'final_predictions':[e['plate_num'] for e in trace.get('emissions',[])],
              'plate_type':paired_type,'rejected_matched_target_proposals':blocked,
              'conversion_status':'NOT_ATTEMPTED: legacy constant type1, no Phase C stitch',
              'GT_spatial_pairing_available':bool(matched),
              'confidence':[o['confidence'] for o in observations]}
        (folder/'case.json').write_text(json.dumps(case,indent=2),encoding='utf-8');cases.append(case)
    (output/'errors.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    paired_metrics={}
    paired_distance_effects=Counter()
    for row in paired:
        before=distance(row['ground_truth'],row['raw']);after=distance(row['ground_truth'],row['postprocessed'])
        row.update({'raw_edit_distance':before,'postprocessed_edit_distance':after})
        paired_distance_effects['improved' if after<before else 'worsened' if after>before else 'same_distance']+=1
    for kind in ('all','type1','type1a'):
        group=[r for r in paired if kind=='all' or r['plate_type']==kind]
        if group:paired_metrics[kind]={mode:{'count':len(group),'correct':sum(r[mode]==r['ground_truth'] for r in group),
            'cer':sum(distance(r['ground_truth'],r[mode]) for r in group)/sum(len(r['ground_truth']) for r in group)}
            for mode in ('raw','postprocessed')}
    post={'scope':'All fixed OCR calls, exact text membership in same-image GT. Additional paired text/CER metrics only for 24 manually reviewed complete crops; type1a has no stitch in legacy. Type output considered separately in E2E.',
          'rows':post_rows,'counts':{k:post_counts[k] for k in ('fixed','damaged','unchanged','changed_without_exact_gain')},'raw_exact_count':sum(r['raw_text_exact'] for r in post_rows),
          'final_exact_count':sum(r['final_text_exact'] for r in post_rows),'paired_complete_crops':paired,
          'paired_metrics':paired_metrics,'paired_distance_effects':dict(paired_distance_effects)}
    (output/'postprocess_ablation.json').write_text(json.dumps(post,indent=2),encoding='utf-8')
    summary={'literal_metrics':diagnostic['metrics'],'errors_saved':len(cases),
        'stage_categories':dict(Counter(r['stage_category'] or 'UNRESOLVED' for r in cases)),
        'missed_GT_categories':dict(Counter(r['stage_category'] or 'UNRESOLVED_PARTIAL_GT' for r in cases if r['gt'] is not None)),
        'category_count_scope':'Literal FN and unmatched output cases are separate error records; primary cause counts must not be interpreted as unique target plates.',
        'OCR_FAILURE_not_assigned_to_unverified_crops':True,'postprocess':post['counts'],
        'limitations':'Stage failures are observable; complete causal decomposition/detector recall requires GT boxes or further manual annotation. Internal literal GT count includes organizer-ignored rows.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
