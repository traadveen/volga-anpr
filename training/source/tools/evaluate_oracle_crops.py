"""Small fixed public subset: automatic bbox vs manually rectified same plates."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from ocr.crnn import CRNNRecognizer
from src.ocr_preprocess import prepare_ocr_input
from tools.benchmark_multiline import distance
from dataset.schema import sha256


def summarize(cases):
    cer=sum(distance(c['ground_truth'],c['prediction']) for c in cases)/sum(len(c['ground_truth']) for c in cases)
    return {'count':len(cases),'correct':sum(c['ground_truth']==c['prediction'] for c in cases),
            'full_plate_accuracy':sum(c['ground_truth']==c['prediction'] for c in cases)/len(cases),
            'cer':cer,'character_accuracy':1-cer}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',nargs='+',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(4)
    manifest_path=Path('reports/phase_d/oracle_crops/manifest.json')
    oracle=json.loads(manifest_path.read_text())
    assert oracle['visual_review']['status']=='PASS'
    auto={r['id']:r for r in json.loads(Path('reports/phase_c/inputs_manifest.json').read_text())['records']}
    result={'selection_allowed':False,'scope':'Eight hand-reviewed type1a plates, same known type/C stitch for both inputs; no whole-image claim',
            'manifest_sha256':sha256(manifest_path),'device':args.device,'results':{}}
    for path in args.weights:
        reader=CRNNRecognizer(path,device=args.device);cases={'auto_crop':[],'oracle_crop':[]}
        for record in oracle['records']:
            assert record['source_sha256']==sha256(Path(record['source_image']))
            for mode in cases:
                crop=Path('reports/phase_c')/auto[record['id']]['crop'] if mode=='auto_crop' else Path(record['rectified_crop'])
                image=np.asarray(Image.open(crop).convert('RGB'))[:,:,::-1].copy()
                prepared=prepare_ocr_input(image,'type1a')
                raw=reader.readtext(prepared.crop,detail=1)[0]
                cases[mode].append({'id':record['id'],'image':record['image'],'ground_truth':record['ground_truth'],
                    'prediction':raw[1],'confidence':raw[2],'diagnostics':prepared.diagnostics,
                    'crop':str(crop),'crop_sha256':sha256(crop)})
        result['results'][path]={'checkpoint_sha256':sha256(Path(path)),
            **{mode:{'metrics':summarize(rows),'cases':rows} for mode,rows in cases.items()}}
        del reader
        if args.device=='cuda':torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({p:{m:v[m]['metrics'] for m in ('auto_crop','oracle_crop')} for p,v in result['results'].items()},indent=2))


if __name__=='__main__':main()
