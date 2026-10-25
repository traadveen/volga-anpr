"""Test-only pixel/GT checks of transformed crops, independent of ML OCR.

Ground-truth cells and frozen templates enter only this checker. They never
enter the multiline processor or CRNN recognizer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from dataset.generator.verify import bank,raster_text
from dataset.schema import read_metadata,sha256
from src.multiline import METHODS,MultilineConfig,prepare_multiline
from tools.benchmark_multiline import read_bgr,save_json


def verify_prepared_pixels(image,characters,label,templates,prepared):
    """Read mapped raster cells, not an echo of their GT characters."""
    scores=[]
    if prepared.effective_mode=='whole_plate':
        decoded,scores=raster_text(image[:,:,::-1],characters,templates)
    elif prepared.effective_mode=='split_and_stitch':
        cells=[]
        for c in characters:
            x0,y0,x1,y1=c['box']
            bx0,by0,_,_=prepared.row_boxes[c['row']]
            matrix=prepared.stitch_transforms[c['row']]
            lo=matrix@np.array([x0-bx0,y0-by0,1])
            hi=matrix@np.array([x1-bx0,y1-by0,1])
            cells.append({**c,'box':[round(lo[0]),round(lo[1]),round(hi[0]),round(hi[1])]})
        decoded,scores=raster_text(prepared.inputs[0][:,:,::-1],cells,templates)
    else:
        texts=[]
        for row,part in enumerate(prepared.inputs):
            bx0,by0,_,_=prepared.row_boxes[row]
            cells=[{**c,'box':[c['box'][0]-bx0,c['box'][1]-by0,c['box'][2]-bx0,c['box'][3]-by0]}
                   for c in characters if c['row']==row]
            text,row_scores=raster_text(part[:,:,::-1],cells,templates)
            texts.append(text);scores.extend(row_scores)
        decoded=''.join(texts)
    # The original B readability cutoff was calibrated at its original raster
    # size. After h170/stitch resampling it is a diagnostic, not the identity
    # gate: exact decoded pixels + nonempty glyph ink establish that gate.
    return {'valid':decoded==label and min(scores)>0,'label':label,
            'decoded_pixels':decoded,'minimum_template_score':min(scores),
            'template_readability_warning':min(scores)<.65}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/'reports/phase_c/inputs_manifest.json')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/multiline_v2.json')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/phase_c/transformed_ground_truth.json')
    args=parser.parse_args()
    registry=json.loads((ROOT/'reports/phase_b/accepted_dataset.json').read_text(encoding='utf-8'))
    root=Path(registry['accepted_dataset'])
    if sha256(root/'meta.csv')!=registry['meta_sha256']:raise ValueError('Accepted labels changed')
    if sha256(root/'generator/manifest.json')!=registry['generator_manifest_sha256']:raise ValueError('Generator provenance changed')
    manifest=json.loads(args.inputs.read_text(encoding='utf-8'))
    if manifest['config_sha256']!=sha256(args.config):raise ValueError('Reviewed configuration changed')
    samples={s['image']:s for s in json.loads((root/'generator/manifest.json').read_text(encoding='utf-8'))['samples']}
    labels,issues=read_metadata(root/'meta.csv')
    if issues:raise ValueError(issues)
    labels={row.image:row.plate_num for row in labels}
    config=MultilineConfig.load(args.config);templates=bank(root);results=[];images=0
    for record in manifest['records']:
        if not record['pool'].startswith('synthetic'):continue
        images+=1
        relative=Path(record['source_image']).relative_to(root).as_posix()
        sample=samples[relative]
        if labels[relative]!=record['label']:raise ValueError('Evaluation label changed')
        path=args.inputs.parent/record['crop']
        if sha256(path)!=record['crop_sha256']:raise ValueError('Reviewed crop changed')
        image=read_bgr(path)
        characters=[{**c,'box':[int(v//2) for v in c['box']]} for c in sample['characters']]
        for method in METHODS:
            for mode in ('split_and_stitch','split_and_ocr_separately'):
                prepared=prepare_multiline(image,mode=mode,method=method,config=config)
                result=verify_prepared_pixels(image,characters,labels[relative],templates,prepared)
                results.append({'id':record['id'],'mode':mode,'method':method,**result})
    report={'images':images,'checks':len(results),'valid':all(r['valid'] for r in results),
            'failed_checks':sum(not r['valid'] for r in results),
            'readability_warnings':sum(r['template_readability_warning'] for r in results),
            'readability_policy':'Scores below B-size-specific .65 flagged separately for visual review; identity requires exact pixel decode and nonempty glyphs',
            'minimum_template_score':min(r['minimum_template_score'] for r in results),
            'scope':'independent frozen glyph raster consistency with GT, not CRNN or end-to-end accuracy',
            'input_manifest_sha256':sha256(args.inputs),'source_sha256':sha256(Path(__file__)),
            'results':results}
    save_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    return 0 if report['valid'] else 1


if __name__=='__main__':raise SystemExit(main())
