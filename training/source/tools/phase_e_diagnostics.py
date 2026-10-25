"""Visual and scoped causal diagnostics; no production inputs from GT."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.phase_e import ROOT,REPORT,IMAGES,read,write,load_image
from src.image_anpr import padded_crop,save_image

FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
SMALL=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)


def place(canvas,path,box):
    if not path:return
    im=Image.open(path).convert('RGB');im.thumbnail((box[2]-box[0],box[3]-box[1]))
    canvas.paste(im,(box[0],box[1]))


def previews():
    results=read(REPORT/'experiments.json');best=results['BEST_CLASSICAL_E']
    traces=read(best['trace']);refs=read(REPORT/'detections.json')['records']
    records=[]
    for trace in traces:
        for r in trace['detections']:
            ref=next(x for x in refs if x['image']==r['image'] and x['index']==r['detection_index'])
            records.append({**r,'manual_reference':ref})
    folder=REPORT/'input_review';folder.mkdir(parents=True,exist_ok=True)
    for page in range((len(records)+5)//6):
        canvas=Image.new('RGB',(1240,990),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(records[page*6:(page+1)*6]):
            x=(i%2)*620;y=(i//2)*330;ref=r['manual_reference'];art=r.get('artifacts',{})
            draw.text((x+8,y+5),f"{page*6+i:02}: {r['image']} det {r['detection_index']} -> {r.get('plate_type')}",fill='black',font=FONT)
            draw.text((x+8,y+29),f"GT {ref['ground_truth']}  quality {ref['crop_quality']}",fill='black',font=SMALL)
            place(canvas,art.get('03_padded_crop'),(x+8,y+55,x+605,y+220))
            place(canvas,art.get('09_ocr_input'),(x+8,y+225,x+605,y+275))
            draw.text((x+8,y+279),f"raw: {r.get('raw_ocr')} final: {r.get('final_ocr')}",fill='black',font=SMALL)
            split=r.get('preprocessing',{}).get('effective_mode','not reached')
            draw.text((x+8,y+301),f"{split}; ratio={r.get('aspect_ratio',0):.2f}",fill='black',font=SMALL)
        canvas.save(folder/f'review_{page+1:02}.png')
    write(folder/'manifest.json',{'records':records,'count':len(records),'best_configuration':best['name'],
        'review_status':'PENDING','scope':'All proposals, padded crop and actual normalized OCR input; GT captions only for human diagnostics'})
    split_folder=REPORT/'type1a_split';split_folder.mkdir(parents=True,exist_ok=True)
    split=[]
    for r in records:
        if r.get('plate_type')!='type1a' or r['status']!='ocr':continue
        d=r['preprocessing'];rows=d.get('row_boxes',[])
        split.append({'image':r['image'],'index':r['detection_index'],'ground_truth':r['manual_reference']['ground_truth'],
            'split':d['split'],'row_boxes':rows,'effective_mode':d['effective_mode'],
            'both_rows_exist':len(rows)==2,'nonempty_rows':len(rows)==2 and all(b[3]>b[1] and b[2]>b[0] for b in rows),
            'valid_heights':len(rows)==2 and all(b[3]-b[1]>=8 for b in rows),
            'characters_not_clipped':None,'flag_RUS_not_destructive':None,
            'limitation':'No character spatial GT on real data: automatic row evidence is structural, not proof all true glyphs survived. Manual review follows.',
            'artifacts':r.get('artifacts',{})})
    write(split_folder/'diagnostics.json',{'records':split,'attempted':len(split),
        'successful':sum(s['both_rows_exist'] for s in split),'review_status':'PENDING'})
    # Expansion chosen by visibility/coverage, never by new-model prediction.
    old=read(ROOT/'reports/phase_d/oracle_crops/manifest.json')['records']
    old_images={r['image'] for r in old}
    complete=[r for r in refs if r['crop_quality']=='COMPLETE_CORRECT_CROP' and r['ground_truth']
              and r['image'] not in old_images]
    extra_a=[r for r in refs if r['crop_quality']=='REJECTED_BEFORE_OCR_REQUIRES_NEW_CROP_REVIEW'
             and r['ground_truth'] and r['image'] not in old_images][:8]
    # All existing complete monoline candidates plus additional two-row examples.
    candidates=[r for r in complete if r['plate_type']=='type1'][:9]
    candidates += [r for r in complete if r['plate_type']=='type1a'][:6]
    candidates += extra_a
    annfolder=REPORT/'oracle_expansion';annfolder.mkdir(parents=True,exist_ok=True)
    annotations=[]
    for r in candidates:
        source=load_image(IMAGES/r['image']);x1,y1,x2,y2=r['bbox'];w=x2-x1;h=y2-y1
        a=max(0,int(x1-.45*w));b=max(0,int(y1-.45*h));c=min(source.shape[1],int(x2+.45*w));d=min(source.shape[0],int(y2+.45*h))
        region=source[b:d,a:c];scale=min(4.,650/max(region.shape[:2]))
        im=Image.fromarray(cv2.cvtColor(region,cv2.COLOR_BGR2RGB));im=im.resize((round(im.width*scale),round(im.height*scale)))
        can=Image.new('RGB',(im.width,max(100,im.height)+60),'white');can.paste(im,(0,60));draw=ImageDraw.Draw(can)
        draw.text((5,5),f"{r['image']} {r['ground_truth']} {r['plate_type']}",font=FONT,fill='black')
        draw.text((5,28),f"source offset ({a},{b}), scale {scale:.4f}; bbox {list(map(int,r['bbox']))}",font=SMALL,fill='black')
        for gx in range(a,c,10):
            xx=(gx-a)*scale
            draw.line((xx,60,xx,can.height),fill=(170,210,210),width=1)
            draw.text((xx+1,62),str(gx),font=SMALL,fill='blue')
        for gy in range(b,d,10):
            yy=60+(gy-b)*scale
            draw.line((0,yy,can.width,yy),fill=(170,210,210),width=1)
            draw.text((1,yy+1),str(gy),font=SMALL,fill='blue')
        filename=f"{r['image'].split('.')[0]}_{r['index']}.png";can.save(annfolder/filename)
        annotations.append({**r,'preview':str(annfolder/filename),'offset':[a,b],'scale':scale,'preview_top':60})
    write(annfolder/'candidates.json',{'selected_before_oracle_weight_evaluation':True,
        'selection_basis':'All complete type1 and visible complete/rejected type1a; not based on OCR success',
        'records':annotations,'original_eight':old})
    print(json.dumps({'review_records':len(records),'sheets':(len(records)+5)//6,'type1a_split_attempts':len(split),'oracle_expansion_candidates':len(annotations)}))


if __name__=='__main__':previews()
