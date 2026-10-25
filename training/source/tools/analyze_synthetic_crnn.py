"""Save every frozen-test OCR error and its measured edit operations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image


def alignment(a,b):
    matrix=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):matrix[i][0]=i
    for j in range(len(b)+1):matrix[0][j]=j
    for i,x in enumerate(a,1):
        for j,y in enumerate(b,1):matrix[i][j]=min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+(x!=y))
    i,j=len(a),len(b);ops=[]
    while i or j:
        if i and j and matrix[i][j]==matrix[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]!=b[j-1]:ops.append({'kind':'substitution','position':i-1,'expected':a[i-1],'actual':b[j-1]})
            i-=1;j-=1
        elif i and matrix[i][j]==matrix[i-1][j]+1:
            ops.append({'kind':'deletion','position':i-1,'expected':a[i-1],'actual':''});i-=1
        else:
            ops.append({'kind':'insertion','position':i,'expected':'','actual':b[j-1]});j-=1
    return list(reversed(ops))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--predictions',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    samples={r['image']:r for r in json.loads((root/'generator/manifest.json').read_text())['samples']}
    contract=json.loads((root/'generator/ocr_inputs.json').read_text())
    positions={image:i for i,image in enumerate(contract['image_order'])}
    cache=np.load(root/'generator/ocr_inputs.npy',mmap_mode='r',allow_pickle=False)
    predictions=json.loads(Path(args.predictions).read_text());errors=[];confusions=Counter();operations=Counter()
    for record in predictions:
        if record['prediction']==record['ground_truth']:continue
        folder=output/f'case_{len(errors):04}';folder.mkdir(exist_ok=True)
        sample=samples[record['image']];shutil.copyfile(root/record['image'],folder/'original.png')
        normalized=cache[positions[record['image']],0]
        Image.fromarray(np.clip((normalized+1)*127.5,0,255).astype(np.uint8)).save(folder/'ocr_input.png')
        edits=alignment(record['ground_truth'],record['prediction'])
        for edit in edits:
            operations[edit['kind']]+=1
            if edit['kind']=='substitution':confusions[edit['expected']+'->'+edit['actual']]+=1
        case={**record,'difficulty':sample['difficulty'],'augmentation':sample['augmentation'],
            'edits':edits,'region_error':any(e['position']>=6 for e in edits),
            'original_image':str((folder/'original.png').resolve()),'OCR_input':str((folder/'ocr_input.png').resolve()),
            'scope':'Known synthetic oracle geometry/stitch; augmentation association is not proof of error cause'}
        (folder/'case.json').write_text(json.dumps(case,indent=2),encoding='utf-8');errors.append(case)
    summary={'count':len(predictions),'errors':len(errors),'by_type':dict(Counter(r['plate_type'] for r in errors)),
        'by_difficulty':dict(Counter(r['difficulty'] for r in errors)),'region_error_count':sum(r['region_error'] for r in errors),
        'edit_operations':dict(operations),'substitution_confusions':dict(confusions.most_common()),'selection_allowed':False}
    (output/'errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))


if __name__=='__main__':main()
