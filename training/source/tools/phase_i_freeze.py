"""Select the eligible balanced-round detector on VAL and freeze FINAL-I, without test inference."""
from pathlib import Path
import shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_h_common import ROOT,SPLIT,read,now
from tools.phase_i_review import Review,atomic_json
from tools.phase_i_dataset import verify_selected
from src.inference_io import file_sha256

REPORT=ROOT/'reports/phase_i/balanced_20261019'
CONFIG=ROOT/'configs/phase_i'


def eligible(metric,baseline):
    keys=('detected','correct')
    return (all(metric['overall'][k]>=baseline['overall'][k] for k in keys)
        and any(metric['overall'][k]>baseline['overall'][k] for k in keys)
        and all(metric['per_type'][t][k]>=baseline['per_type'][t][k]
                for t in ('type1','type1a','type1b') for k in keys)
        and metric['performance']['GPU_mean_ms']<100
        and metric['performance']['peak_reserved_MiB']<4096)


def freeze():
    target=CONFIG/'final_i_freeze.json'
    if target.exists():
        existing=read(target)
        for path,sha in existing['hashes'].items():
            if file_sha256(Path(path))!=sha:raise ValueError('Existing FINAL-I changed: '+path)
        return existing
    review=Review();manifest=read(REPORT/'dataset_manifest.json');verify_selected(manifest,review)
    policy=read(REPORT/'selection_policy.json');summary=read(REPORT/'summary.json')
    if summary['selection_policy_sha256']!=file_sha256(REPORT/'selection_policy.json'):
        raise ValueError('Selection policy changed after comparison')
    if policy['dataset_manifest_sha256']!=file_sha256(REPORT/'dataset_manifest.json'):
        raise ValueError('Selection dataset changed')
    metrics=[read(REPORT/'evaluations'/name/'val_metrics.json') for name in policy['candidates']]
    baseline=metrics[0]
    if (baseline['overall']['objects'],baseline['overall']['detected'],baseline['overall']['correct'])!=(93,38,28):
        raise ValueError('Frozen REAL VAL baseline not reproduced')
    if any(m['overall']['objects']!=93 for m in metrics):raise ValueError('Candidates evaluated on different memberships')
    candidates=[m for m in metrics[1:] if eligible(m,baseline)]
    if not candidates:raise ValueError('No candidate passes the predeclared VAL gates')
    selected=max(candidates,key=lambda m:(m['overall']['FPA'],m['overall']['recall'],-m['performance']['GPU_mean_ms']))
    test=read(ROOT/'reports/phase_i/final_test.json')
    if test['status']!='NOT_RUN':raise ValueError('Test state must be inspected before freezing')
    source=Path(selected['detector'])
    if file_sha256(source)!=selected['detector_sha256']:raise ValueError('Selected detector changed')
    training=read(REPORT/'training'/(selected['name']+'.json'))
    if not (training['training_complete'] and training['epochs_completed']==4
            and training['dataset_manifest_sha256']==file_sha256(REPORT/'dataset_manifest.json')
            and training['checkpoint_sha256']==selected['detector_sha256']
            and training['initialization'][0]['all_bit_exact_before_first_batch']):
        raise ValueError('Selected checkpoint is not the declared completed four-epoch pilot')
    original=read(ROOT/'configs/phase_h/final_h.json')
    for m in metrics:
        if (m['OCR_sha256'],m['recognition_profile'],m['resolution'],m['confidence'],m['nms_iou'])!=(
                original['ocr_sha256'],original['recognition_profile'],original['imgsz'],original['confidence'],original['iou']):
            raise ValueError('Comparison changed frozen OCR/profile/threshold/resolution')
    weights=ROOT/'models/phase_i/FINAL_I.pt'
    cfg_path=CONFIG/'final_i.json'
    if weights.exists() or cfg_path.exists():raise ValueError('Incomplete FINAL-I export exists; never overwrite it')
    weights.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,weights)
    if file_sha256(weights)!=selected['detector_sha256']:raise ValueError('FINAL-I copy differs')
    cfg={**original,'name':'FINAL-I','detector':str(weights),'detector_sha256':file_sha256(weights),
        'selection_basis':'Frozen DATA-3 REAL VAL only; per-type protection; two fixed balanced-round variants',
        'promotion_scope':'Phase I VAL-selected detector; frozen FINAL-H OCR; BEST-G and FINAL-H preserved',
        'generalization':'NOT EVALUATED ON TEST IN PHASE I',
        'runtime_gates':{'measured_hardware':'GTX1060 6GB; not jury GTX1050Ti',
            'GPU_mean_ms':selected['performance']['GPU_mean_ms'],'GPU_p95_ms':selected['performance']['GPU_p95_ms']}}
    atomic_json(cfg_path,cfg)
    selection={'created_at':now(),'selected':selected['name'],'detector_sha256':selected['detector_sha256'],
        'source_checkpoint':str(source),'policy_sha256':file_sha256(REPORT/'selection_policy.json'),
        'dataset_manifest_sha256':file_sha256(REPORT/'dataset_manifest.json'),
        'selection_uses_test':False,'candidates':[{ 'name':m['name'],'eligible':eligible(m,baseline)} for m in metrics[1:]],
        'selected_val_metrics':selected,'previous_pipeline_preserved':'FINAL-H / BEST-G F03'}
    atomic_json(ROOT/'reports/phase_i/selection.json',selection)
    final_val={'status':'FINAL-I-FROZEN-VAL-ONLY','selected':selected['name'],'final_I_selected':True,
        'metrics':selected,'baseline':baseline,'selection':str(ROOT/'reports/phase_i/selection.json'),
        'test_used':False,'phase_i_complete':False}
    atomic_json(ROOT/'reports/phase_i/final_val.json',final_val)
    test.update(reason='FINAL-I frozen from VAL; test explicitly deferred in this run; no inference launched',
        test_prediction_runs_in_phase_i=0,test_images_read_in_phase_i=0)
    atomic_json(ROOT/'reports/phase_i/final_test.json',test)
    summary.update(FINAL_I_selected=True,selected=selected['name'],retained_pipeline='FINAL-I detector with frozen FINAL-H OCR',
        previous_pipeline_preserved='FINAL-H with BEST-G/F03',
        next_requirement='Separate post-freeze independent test decision; no test run in this continuation')
    atomic_json(REPORT/'summary.json',summary)
    files=[cfg_path,weights,source,ROOT/'reports/phase_i/selection.json',ROOT/'reports/phase_i/final_val.json',
        REPORT/'selection_policy.json',REPORT/'dataset_manifest.json',REPORT/'summary.json',
        ROOT/'tools/phase_i_freeze.py',ROOT/'tools/predict_images_i.py',ROOT/'tools/predict_images_g.py']
    files += [REPORT/'evaluations'/name/'val_metrics.json' for name in policy['candidates']]
    hashes={**review.guard.hashes,**{str(p):file_sha256(p) for p in files}}
    value={'status':'FINAL-I-FROZEN','created_at':now(),'selected':selected['name'],
        'hashes':hashes,'split_manifest_sha256':file_sha256(SPLIT),'selection_uses_test':False,
        'test_prediction_runs_before_freeze':0,'changes_after_test_forbidden':True,
        'scope':'VAL-selected detector; PARTIAL-GT independent test not run',
        'performance_gate':{'1060_mean_lt_100ms':True,'1050Ti_gate':'UNVERIFIED: hardware unavailable'},
        'training_source':'90 explicitly human-verified train geometries; controlled authored synthetic replay',
        'OCR':'FINAL-H frozen; no OCR adaptation'}
    atomic_json(target,value);print('FROZEN',selected['name'],selected['overall'],flush=True)
    return value


if __name__=='__main__':freeze()
