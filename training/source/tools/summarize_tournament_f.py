"""Comparable development ranking; real public scorer is kept untouched."""
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,read,write
from tools.phase_f_setup import REPORT
from tools.score_development_slices_f import score

def rows():
    baseline=read(REPORT/'baseline_development_slices.json')['untrained_exposed_development']
    entries=[]
    for path in sorted((REPORT/'experiments').glob('*/result.json')):
        r=read(path);s=score(path.parent/'predictions.csv');r['development_slices']=s;write(path,r)
        c=r['config'];d=r['detector_validation']['real_development_val']['class_agnostic'];pub=r['public_development'];u=s['untrained_exposed_development']
        latency=r['GPU_latency_ms']['mean'];t1=pub['per_type']['type1']['full_match'];u1=u['per_type']['type1']['full_match']
        reasons=[]
        if c['name'].startswith(('F02_finetune','F02_balanced','F02_clean_v2')):reasons.append('INVALID intended initialization: pretrained=False reset loaded weights')
        if t1<8:reasons.append('type1 regression on full exposed development')
        if u1<baseline['per_type']['type1']['full_match']:reasons.append('type1 regression on common untrained-exposed slice')
        if u['full_match_total']<baseline['full_match_total']:reasons.append('FPA regression on common untrained-exposed slice')
        if latency>=100:reasons.append('GPU mean exceeds local 100ms budget')
        row={'experiment':c['name'],'geometry':c['geometry'],'routing':c['routing'],'imgsz':c['imgsz'],
            'precision_real9':d['Precision'],'recall_real9':d['Recall'],'F1_real9':d['F1'],
            'mAP50_real9':d['mAP50'],'mAP50_95_real9':d['mAP50_95'],
            **{k+'_recall_real9':d['per_type'][k]['Recall'] for k in ('type1','type1a','type1b')},
            'mean_IoU_real9':d['IoU_matched_distribution']['mean'] if d['IoU_matched_distribution'] else None,
            'corner_pixel_error_real9':d['quad_quality']['corner_pixel_error'],
            'corner_normalized_error_real9':d['quad_quality']['normalized_corner_error'],
            'quad_IoU_real9':d['quad_quality']['quad_IoU'],
            'public_full_match':pub['full_match_total'],'public_GT':pub['gt_target_plates'],'type1_full_match':t1,
            'type1a_full_match':pub['per_type']['type1a']['full_match'],'untrained_full_match':u['full_match_total'],
            'untrained_GT':u['gt_target_plates'],'untrained_type1_full_match':u1,
            'structural_split_success':r['type1a_split_success'],'type1a_inputs':r['type1a_inputs'],
            'rectification_success':r['quad_rectification_success'],'rectification_attempts':r['quad_rectification_attempts'],
            'GPU_mean_ms':latency,'GPU_p95_ms':r['GPU_latency_ms']['p95'],
            'GPU_peak_allocated_MB':r['GPU_peak_allocated_MB'],
            'synthetic_FP_per_image':r['detector_validation']['synthetic_val']['class_agnostic']['FP_per_image'],
            'synthetic_negative_FPR':r['detector_validation']['synthetic_val']['class_agnostic']['negatives']['negative_image_FPR'],
            'synthetic_type1b_full_match':r['validation_e2e']['synthetic_val']['per_type']['type1b']['correct'],
            'eligible':not reasons,'rejection_reasons':' | '.join(reasons),'generalization':'NOT VERIFIED'}
        entries.append(row)
    return entries

def main():
    entries=rows();path=REPORT/'model_tournament.csv'
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    eligible=[r for r in entries if r['eligible']]
    # One target in this small exposed slice is a near tie, not evidence for
    # a slower/more complex geometry. Preserve synthetic rare-type support.
    maximum=max((r['untrained_full_match'] for r in eligible),default=0)
    near=[r for r in eligible if r['untrained_full_match']>=maximum-1]
    near.sort(key=lambda r:(r['GPU_mean_ms']>60,r['synthetic_type1b_full_match']==0,
        r['geometry']!='bbox',r['routing']=='combined',r['GPU_mean_ms'],-r['untrained_full_match']))
    remainder=[r for r in eligible if r not in near]
    remainder.sort(key=lambda r:(-r['untrained_full_match'],r['GPU_mean_ms']))
    eligible=near+remainder
    write(REPORT/'provisional_ranking.json',{'rows':entries,'eligible_ranked':eligible,
        'ranking_scope':'Common 87 untrained-exposed targets; previously exposed public, no independent real holdout. Full115 includes28 training targets.',
        'spatial_scope':'Only9 manually annotated real validation targets: one type1/eight type1a; no real type1b. Not global detector recall.',
        'split_scope':'Structural split acceptance on predicted inputs, not manually verified correct glyph-row split rate',
        'latency_scope':'Final candidates must be benchmarked serially without training before lock'})
    print([(r['experiment'],r['untrained_full_match'],r['public_full_match'],r['type1_full_match'],round(r['GPU_mean_ms'],1)) for r in eligible],flush=True)

if __name__=='__main__':main()
