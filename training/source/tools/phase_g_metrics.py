"""Stage accounting and conditional OCR diagnostics; no global IoU invention."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter
from tools.phase_g_common import REPORT,read,write,scorer
from tools.evaluation import levenshtein

def ocr_metrics(rows,field):
    if not rows:return {'count':0,'correct':0,'FPA':None,'CER':None}
    return {'count':len(rows),'correct':sum(r[field]==r['ground_truth'] for r in rows),
        'FPA':sum(r[field]==r['ground_truth'] for r in rows)/len(rows),
        'CER':sum(levenshtein(r['ground_truth'],r[field] or '') for r in rows)/sum(len(r['ground_truth']) for r in rows)}

def waterfall(rows):
    bad=Counter(r['primary_failure_category'] for r in rows if not r['official_success'])
    n=len(rows);resolved=n-bad['UNKNOWN'];detected=resolved-bad['DETECTOR_MISS']-bad['LOW_DETECTOR_CONFIDENCE']
    geometry=detected-bad['BAD_RECTIFICATION'];preprocessed=geometry-bad['UNREADABLE'];split=preprocessed-bad['TYPE1A_SPLIT_FAILURE']
    module=scorer();eligible=[r for r in rows if r['primary_failure_category'] in ('SUCCESS','OCR_FAILURE_GOOD_INPUT','POSTPROCESS_FAILURE')]
    correct=lambda r,k:bool(r[k]) and module.compare_plates(r['ground_truth'],r[k],r['plate_type'])[1]
    raw=sum(correct(r,'raw_OCR') for r in eligible);post=sum(correct(r,'postprocessed_OCR') for r in eligible)
    fixed=sum(not correct(r,'raw_OCR') and correct(r,'postprocessed_OCR') for r in eligible)
    broken=sum(correct(r,'raw_OCR') and not correct(r,'postprocessed_OCR') for r in eligible)
    final=sum(r['official_success'] for r in rows)
    assert sum(bad.values())==n-final
    assert raw+fixed-broken==post==final
    return {'GT':n,'GT_source_discrepancy_unresolved':bad['UNKNOWN'],'reviewed_consistent_GT':resolved,
        'detected_at_frozen_confidence':detected,'acceptable_localization_visual_review':geometry,
        'correct_routing':geometry,'acceptable_preprocessing':preprocessed,'correct_split_or_not_applicable':split,
        'raw_OCR_correct':raw,'postprocess_FIXED':fixed,'postprocess_BROKEN':broken,
        'postprocessed_correct':post,'final_prediction_correct':final,
        'cascade_excluded_unresolved':bad['UNKNOWN'],'failures':dict(bad),
        'accounting':'Source GT discrepancies branch out before stage attribution. Presence/layout/input quality is a visual engineering review; this is NOT independently measured detector recall/IoU. Raw exactness uses readable-position organizer policy, with explicit postprocess recovery bridge; never force a monotone count through repair.'}

def main():
    rows=read(REPORT/'error_summary.json')['rows']
    write(REPORT/'error_waterfall.json',{kind:waterfall([r for r in rows if kind=='ALL' or r['plate_type']==kind])
        for kind in ('ALL','type1','type1a','type1b')})
    goods=[r for r in rows if r['good_input_verified']]
    result={'scope':'Conditional OCR on visually inspected exact F inputs. Selected exposed development subset; not representative real accuracy. Partial labels excluded from FPA/CER; no GT discrepancy/unreadable/failed split/geometry failure included.',
        'metrics':{kind:{stage:ocr_metrics([r for r in goods if kind=='ALL' or r['plate_type']==kind],stage)
            for stage in ('raw_OCR','postprocessed_OCR')} for kind in ('ALL','type1','type1a','type1b')},
        'rows':goods,'reviewer':'Agent visual review; no independent annotation adjudication',
        'training_gate':'PENDING oracle geometry + visually verified manual split diagnostic; real-input OCR gap observed, but split/stitch/geometry must be isolated before training.'}
    write(REPORT/'real_good_input_ocr.json',result)
    print('Waterfall',waterfall(rows));print('Good OCR',result['metrics']['ALL'],flush=True)

if __name__=='__main__':main()
