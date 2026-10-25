"""Finite diagnostic tournament on declared exposed validation only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time
from dataclasses import asdict
import cv2,numpy as np,torch
from tools.phase_g_common import ROOT,REPORT,CONFIG,IMAGES,read,write,freeze,scorer
from tools.phase_e import load_image,official,summary
from src.image_anpr import Detection,write_predictions
from src.image_anpr_g import OCRG,ImagePipelineG
from src.multiline_g import ConfigG,METHODS

def cached_detections(low=False):
    records=read(REPORT/'audit/lowconf_proposals.json')['records'] if low else read(REPORT/'audit/artifact_index.json')['records']
    if low:return {r['image']:[Detection(**d) for d in r['detections']] for r in records}
    return {r['image']:[Detection(tuple(d['bbox']),d['detector_confidence'],class_name=d['detector_class'],quad=d['observed_quad'],quad_source=d['quad_source']) for d in r['detections']] for r in records}

def evaluate_rows(predictions,folder):
    folder.mkdir(parents=True,exist_ok=True);write_predictions(folder/'predictions.csv',predictions)
    full,matches=official(folder/'predictions.csv',folder/'official')
    train=set(read(ROOT/'reports/phase_f/dataset_composition.json')['real_import']['original_train_images'])
    val=set(p.name for p in IMAGES.glob('*.jpg'))-train
    module=scorer();issues=[];labels,_=module.load_gt(str(IMAGES.parent/'debug_labels.csv'),issues)
    write_predictions(folder/'val_predictions.csv',[p for p in predictions if p['image'] in val])
    preds=module.load_pred(str(folder/'val_predictions.csv'),val,issues)
    result=module.evaluate([g for g in labels if g.image in val],val,preds,{'type1':.15,'type1a':.35,'type1b':.35,'precision':.15})
    serial={k:v for k,v in result.items() if not k.startswith('_')};write(folder/'val.official.json',serial)
    return {'full':full,'real_val_exposed':serial,'format_issues':issues}

def run(name,config,ocr,frames,cache,*,multi=False,postprocess='current',expansion=None,threshold=.5):
    folder=REPORT/'experiments'/name;spec={'name':name,'split':asdict(config),'multi':multi,'postprocess':postprocess,'expansion':expansion or {},'threshold':threshold,
        'data_scope':'real_val_exposed; full public/train scores diagnostic only','ocr_checkpoint':str(ocr.checkpoint_path) if hasattr(ocr,'checkpoint_path') else 'Frozen D BEST','no_GT_in_inference':True}
    write(CONFIG/(name+'.json'),spec)
    pipeline=ImagePipelineG(None,ocr,config,multi=multi,postprocess=postprocess,expansion=expansion);rows=[];preds=[]
    for n,f in frames.items():
        r=pipeline.process(f,image_name=n,detections=[d for d in cache[n] if d.confidence>=threshold]);rows.append(r);preds.extend(r['predictions'])
    result=evaluate_rows(preds,folder);result.update(config=spec,cached_prepared_OCR_latency=summary([r['total_ms'] for r in rows]))
    write(folder/'trace.json',{'records':rows});write(folder/'result.json',result)
    print(name,'full',result['full']['full_match_total'],'val',result['real_val_exposed']['full_match_total'],'cached ms',round(result['cached_prepared_OCR_latency']['mean'],2),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['split','followup'],default='split');args=p.parse_args()
    print('Loading bounded G tournament runtime',flush=True);c=freeze()['baseline'];torch.set_num_threads(4);cv2.setNumThreads(1)
    ocr=OCRG(c['ocr'],'cuda');ocr.checkpoint_path=c['ocr'];frames={p.name:load_image(p) for p in sorted(IMAGES.glob('*.jpg'))};cache=cached_detections();results=[]
    if args.stage=='split':
        for method in METHODS:results.append(run('G01_'+method,ConfigG(method=method),ocr,frames,cache))
        results.append(run('G02_multi4',ConfigG(),ocr,frames,cache,multi=True))
        write(REPORT/'split_tournament.json',results)
        baseline=read(ROOT/'reports/phase_f/experiments/F03_obb_transfer_640_detector/trace.json')['records']
        control=read(REPORT/'experiments/G01_A_current_F/trace.json')['records'];diff=[]
        for a,b in zip(baseline,control):
            for x,y in zip(a['detections'],b['detections']):
                if (x['normalized_sha256'],x['raw_ocr'],x['final_ocr'])!=(y['normalized_sha256'],y['raw_ocr'],y['final_ocr']):diff.append((a['image'],x['detection_index']))
        write(REPORT/'F_adapter_parity.json',{'status':'PASS' if not diff else 'FAIL','differences':diff,'same_count':all(len(a['detections'])==len(b['detections']) for a,b in zip(baseline,control))})
        assert not diff,'G baseline adapter must reproduce frozen F'
    else:
        split=read(REPORT/'split_tournament.json')
        eligible=[r for r in split if r['full']['per_type']['type1']['full_match']>=11 and r['full']['per_type']['type1a']['full_match']>=40]
        best=max(eligible,key=lambda r:(r['real_val_exposed']['full_match_total'],-r['cached_prepared_OCR_latency']['mean']))
        cfg=ConfigG(**best['config']['split']);multi=best['config']['multi'];base=best['config']['name']
        write(REPORT/'provisional_split_choice.json',{'experiment':base,'basis':'Exposed real_val, protected type1/type1a floor; not final promotion'})
        for gap in (0,4,16):results.append(run(f'G03_gap_{gap}',ConfigG(**dict(asdict(cfg),stitch_gap=gap)),ocr,frames,cache,multi=multi))
        for height in (48,80):results.append(run(f'G03_height_{height}',ConfigG(**dict(asdict(cfg),stitch_height=height)),ocr,frames,cache,multi=multi))
        for padding in (0,8):results.append(run(f'G03_padding_{padding}',ConfigG(**dict(asdict(cfg),stitch_padding=padding)),ocr,frames,cache,multi=multi))
        results.append(run('G04_mask_preserve',cfg,ocr,frames,cache,multi=multi,postprocess='mask_preserve'))
        for expand in (.02,.04,.06):results.append(run(f'G05_expand_{round(expand*100)}',cfg,ocr,frames,cache,multi=multi,expansion={'type1a':expand}))
        write(REPORT/'followup_tournament.json',results)

if __name__=='__main__':main()
