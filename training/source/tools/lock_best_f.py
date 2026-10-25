"""Freeze the choice before one final synthetic test or any future real holdout."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,BEST,read,write
from tools.phase_f_setup import REPORT
from dataset.schema import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args()
    output=ROOT/'configs/phase_f/best_f.json'
    assert not output.exists(),'Selection must not be silently changed after lock/test'
    r=read(REPORT/'experiments'/a.experiment/'result.json');c=r['config']
    ranking=read(REPORT/'provisional_ranking.json');row=next(v for v in ranking['rows'] if v['experiment']==a.experiment)
    assert row['eligible'],'Does not satisfy no-regression/speed rule'
    image_config=read(ROOT/'configs/image_anpr_e.json')['config'];image_config['debug']=False
    image_config['stitch']=c.get('stitch',True)
    if c.get('filter_override'):image_config['geometry_filter']=c['filter_override']
    export=read(REPORT/'export/comparison.json');assert export['selected_experiment']==a.experiment
    config={'name':'BEST-F-DEVELOPMENT','selected_experiment':a.experiment,'detector':c['weights'],
        'detector_sha256':sha256(Path(c['weights'])),'ocr':str(BEST),'ocr_sha256':sha256(BEST),
        'imgsz':c['imgsz'],'geometry':c['geometry'],'routing':c['routing'],'confidence':c['confidence'],'iou':c['iou'],
        'image_config':image_config,'default_backend':'onnx' if export['decision']=='KEEP CUDA export' else 'pytorch','onnx_detector':export['onnx_path'],
        'onnx_sha256':export['onnx_sha256'],'onnx_accepted':export['decision']=='KEEP CUDA export',
        'generalization':'GENERALIZATION NOT YET VERIFIED','selection_basis':'Previously exposed common untrained-development slice; fixed CRNN; no final synthetic test or untouched real test used'}
    config['inference_source_sha256']={str(p):sha256(p) for p in [ROOT/'src/image_anpr_f.py',ROOT/'src/geometry_f.py',ROOT/'src/image_anpr.py',ROOT/'src/ocr_preprocess.py',ROOT/'src/multiline.py']}
    write(output,config)
    from datetime import datetime,timezone
    write(REPORT/'selection_lock.json',{'utc':datetime.now(timezone.utc).isoformat(),'experiment':a.experiment,
        'config_sha256':sha256(output),'development_selection_row':row,'independent_real_test':'UNAVAILABLE',
        'synthetic_test_access_before_lock':'NONE','policy':read(REPORT/'selection_policy.json')})
    print('Locked',a.experiment,flush=True)

if __name__=='__main__':main()
