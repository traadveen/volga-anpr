"""Apply the predeclared val-only pilot gates; preserve FINAL-H and pending FINAL-I."""
from pathlib import Path
import sys,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_i_dataset import REPORT,read,verify_selected
from tools.phase_i_review import atomic_json
from tools.phase_h_common import ROOT,now
from src.inference_io import file_sha256


def csv_write(path,records):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def main(report=REPORT):
    if (ROOT/'configs/phase_i/final_i_freeze.json').exists():
        raise ValueError('FINAL-I already frozen; do not rerun model selection')
    report=Path(report)
    manifest=read(report/'dataset_manifest.json');verify_selected(manifest)
    policy=read(report/'selection_policy.json');names=policy['candidates']
    metrics=[read(report/'evaluations'/n/'val_metrics.json') for n in names];base=metrics[0]
    table=[];decisions=[];errors=[];paired=[]
    for m in metrics:
        reasons=[];overall=m['overall'];perf=m['performance']
        if m['name']!=names[0]:
            for key in ('detected','correct'):
                if overall[key]<base['overall'][key]:reasons.append('overall_'+key+'_regression')
            if not any(overall[k]>base['overall'][k] for k in ('detected','correct')):reasons.append('no_strict_overall_gain')
            for kind in ('type1','type1a','type1b'):
                for key in ('detected','correct'):
                    if m['per_type'][kind][key]<base['per_type'][kind][key]:reasons.append(kind+'_'+key+'_regression')
            if perf['GPU_mean_ms']>=100:reasons.append('mean_GPU_gate_failed')
            if perf['peak_reserved_MiB']>=4096:reasons.append('4GB_memory_gate_failed')
        status='KEEP_FROZEN_REFERENCE' if m['name']==names[0] else 'REJECT_FOR_PROMOTION' if reasons else 'ELIGIBLE_PILOT_ONLY'
        decisions.append({'model':m['name'],'status':status,'reasons':reasons})
        table.append({'MODEL':m['name'],'VAL_DETECTED':overall['detected'],'VAL_OBJECTS':overall['objects'],
            'VAL_DETECTOR_RECALL':overall['recall'],'TYPE1A_RECALL':m['per_type']['type1a']['recall'],
            'TYPE1B_RECALL':m['per_type']['type1b']['recall'],'VAL_CORRECT':overall['correct'],
            'VAL_END_TO_END_FPA':overall['FPA'],'TYPE1A_CORRECT':m['per_type']['type1a']['correct'],
            'TYPE1B_CORRECT':m['per_type']['type1b']['correct'],'GPU_MEAN':perf['GPU_mean_ms'],
            'GPU_P95':perf['GPU_p95_ms'],'VRAM_ALLOCATED_MiB':perf['peak_allocated_MiB'],
            'VRAM_RESERVED_MiB':perf['peak_reserved_MiB'],'KEEP_REJECT':status,
            'TIMING_SOURCE':'New full REAL VAL pass on GTX1060; same frozen FINAL-H OCR, no disk decode'})
        records=read(report/'evaluations'/m['name']/'val_objects.json')
        baseline={r['review_id']:r for r in read(report/'evaluations'/names[0]/'val_objects.json')}
        for r in records:
            if r['failure']!='CORRECT':errors.append({'model':m['name'],'review_id':r['review_id'],
                'type':r['gt_type'],'size':r['size_bin_640'],'failure':r['failure'],
                'OBB_IoU_on_human_subset':r['OBB_IoU'],'angle_error_deg_on_human_subset':r['angle_error_degrees']})
            if m['name']!=names[0] and r['gt_type']=='type1a':
                old=baseline[r['review_id']]
                if old['e2e_correct'] and not r['e2e_correct']:
                    paired.append({'model':m['name'],'review_id':r['review_id'],'text':r['gt_text'],
                        'baseline_observed':old['observed'],'candidate_observed':r['observed'],
                        'candidate_detected':r['detector_matched'],'candidate_type_correct':r['type_correct'],
                        'note':'Val diagnostic only; never add this image/GT to train or tune per-image routing.'})
    csv_write(report/'detector_comparison.csv',table);csv_write(ROOT/'reports/phase_i/detector_comparison.csv',table)
    csv_write(report/'error_attribution.csv',errors)
    atomic_json(report/'type1a_regressions.json',paired)
    has_type1a=manifest['real_counts']['train'].get('type1a',0)>0
    summary={'created_at':now(),'scope':'PHASE I bounded geometry pilot, not completed full detector adaptation',
        'decisions':decisions,'retained_pipeline':'FINAL-H with frozen BEST-G/F03 detector',
        'FINAL_I_selected':False,'test_run':False,'OCR_modified':False,'split_modified':False,
        'real_counts':manifest['real_counts'],'real_unique_plates':manifest['real_unique_plates'],
        'next_requirement':'Assess full VAL per-type gates and failures before any further experiment or FINAL-I freeze.' if has_type1a else
            'Explicit human type1a quad and region-completeness confirmation. Prepared batch:40 train +10 val, a review scope, not an official minimum.',
        'new_ground_truth_created_automatically':0,'selection_policy_sha256':file_sha256(report/'selection_policy.json'),
        'metrics':metrics}
    atomic_json(report/'summary.json',summary)
    final_path=ROOT/'reports/phase_i/final_val.json'
    old=read(final_path)
    old.update(status='BALANCED_PILOT_COMPLETE_FINAL_I_PENDING_DECISION' if has_type1a else 'PILOT_COMPLETE_FINAL_I_PENDING_TYPE1A_GEOMETRY',
        final_I_selected=False,pilot_summary=str(report/'summary.json'),retained_pipeline=summary['retained_pipeline'])
    atomic_json(final_path,old)
    print(decisions)


if __name__=='__main__':main()
