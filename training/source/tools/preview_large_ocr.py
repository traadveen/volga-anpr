"""Random, labeled original/stitch contact sheets for reviewed synthetic data."""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from training.crnn_v2 import TrainConfig,OCRDatasetV2
from dataset.schema import read_metadata,write_metadata,sha256


def preview(root,output,seed=71895,count=100):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    rows,issues=read_metadata(root/'meta.csv')
    if issues:raise ValueError(issues)
    selections={kind:rng.sample([i for i,r in enumerate(rows) if r.plate_type==kind],
        min(count,sum(r.plate_type==kind for r in rows))) for kind in ('type1','type1a','type1b')}
    indices=sorted(i for selected in selections.values() for i in selected)
    selected_path=(output/'selected.csv').resolve()
    write_metadata(selected_path,(rows[i] for i in indices))
    config=TrainConfig(dataset_root=str(root),cache_images=False)
    data=OCRDatasetV2(config,str(selected_path));positions={index:local for local,index in enumerate(indices)}
    manifest={'seed':seed,'root':str(root.resolve()),'meta_sha256':sha256(root/'meta.csv'),
              'selected_fingerprint':data.fingerprint,'sheets':[]}
    for kind in ('type1','type1a','type1b'):
        selected=selections[kind]
        for mode in ('original','stitched') if kind=='type1a' else ('original',):
            for page,start in enumerate(range(0,len(selected),25),1):
                subset=selected[start:start+25];width,height=300,200
                sheet=Image.new('RGB',(5*width,5*height),'#d9dde0');draw=ImageDraw.Draw(sheet)
                records=[]
                for cell,index in enumerate(subset):
                    row=data.rows[positions[index]];sample=data.samples[row.image];x=cell%5*width;y=cell//5*height
                    if mode=='stitched':
                        crop=data.input_for(positions[index]).crop
                        image=Image.fromarray(crop[:,:,::-1] if crop.ndim==3 else crop)
                    else:
                        image=Image.open(root/row.image).convert('RGB')
                    image.thumbnail((width-8,height-42))
                    sheet.paste(image,(x+(width-image.width)//2,y+34+(height-38-image.height)//2))
                    caption=f"{index:06} {row.plate_num} | {sample['difficulty']}"
                    draw.text((x+5,y+5),caption,font=font,fill='black')
                    records.append({'index':index,'image':row.image,'ground_truth':row.plate_num,
                                    'difficulty':sample['difficulty']})
                name=f'{kind}_{mode}_{page:02}.png';sheet.save(output/name)
                manifest['sheets'].append({'file':name,'type':kind,'mode':mode,'samples':records})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'sheets':len(manifest['sheets']),'count_per_type':count,'seed':seed}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();preview(args.root,args.output)
