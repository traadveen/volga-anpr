"""Rebuild exact frozen inputs and visual evidence without changing BEST-F."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.phase_g_common import ROOT,REPORT,IMAGES,freeze,read,write
from tools.phase_e import load_image
from src.geometry_f import canonical_rectify
from src.image_anpr import save_image,padded_crop
from src.ocr_preprocess import prepare_ocr_input

FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)

def gray_input(crop):
    h,w=crop.shape[:2];scale=max(480/w,160/h)
    if scale>1:crop=cv2.resize(crop,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)

def artifacts(frame,d,folder):
    folder.mkdir(parents=True,exist_ok=True)
    tight,_=padded_crop(frame,d['bbox'],0)
    save_image(folder/'crop.png',tight)
    if d.get('perspective_rectification'):
        crop,_=canonical_rectify(frame,d['observed_quad'],d['plate_type'],d['quad_source'])
    else:crop=tight
    save_image(folder/'rectified.png',crop)
    gray=gray_input(crop);prepared=prepare_ocr_input(gray,d['plate_type'])
    actual=hashlib.sha256(prepared.normalized.tobytes()).hexdigest()
    assert actual==d['normalized_sha256'],(folder,actual,d['normalized_sha256'])
    for name,box in zip(('top','bottom'),prepared.diagnostics['row_boxes']):
        a,b,c,e=box;save_image(folder/f'{name}.png',gray[b:e,a:c])
    save_image(folder/'stitched.png',prepared.crop)
    save_image(folder/'normalized.png',np.clip((prepared.normalized+1)*127.5,0,255).astype(np.uint8))
    np.save(folder/'normalized.npy',prepared.normalized)
    return {name:str(folder/f'{name}.png') if (folder/f'{name}.png').exists() else None
            for name in ('crop','rectified','top','bottom','stitched','normalized')}

def paste(canvas,path,box):
    im=Image.open(path).convert('RGB');im.thumbnail((box[2],box[3]))
    canvas.paste(im,(box[0],box[1]))

def main():
    freeze()
    refs=read(ROOT/'reports/phase_f/attribution/remaining_failures.json')['rows']
    traces=read(ROOT/'reports/phase_f/experiments/F03_obb_transfer_640_detector/trace.json')['records']
    output=[];pages=[]
    for trace in traces:
        name=trace['image'];frame=load_image(IMAGES/name)
        gt=[r for r in refs if r['image']==name]
        failed=[r for r in gt if not r['official_success']]
        ds=[]
        for d in trace['detections']:
            a=artifacts(frame,d,REPORT/'audit/inputs'/Path(name).stem/str(d['detection_index'])) if d['status']=='ocr' else {}
            ds.append(dict(d,artifacts=a))
        output.append({'image':name,'GT':gt,'detections':ds})
        if not failed:continue
        marked=frame.copy()
        for d in ds:
            x1,y1,x2,y2=map(int,d['bbox'])
            cv2.rectangle(marked,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.putText(marked,str(d['detection_index']),(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,255),2)
        overview=REPORT/'audit'/f'{Path(name).stem}_overview.jpg';save_image(overview,marked)
        h=max(800,130+len(ds)*260);canvas=Image.new('RGB',(1600,h),'#eeeeee');draw=ImageDraw.Draw(canvas)
        draw.text((10,10),name+' FAILED: '+', '.join(r['ground_truth'] for r in failed),font=FONT,fill='black')
        draw.text((10,38),'Green numbered boxes = actual F detections. GT location is NOT inferred from text matching.',font=FONT,fill='black')
        paste(canvas,overview,(10,75,740,h-85))
        for i,d in enumerate(ds):
            y=80+i*260
            draw.text((770,y),f"D{d['detection_index']} {d.get('detector_class')} conf={d['detector_confidence']:.3f} raw={d.get('raw_ocr')} final={d.get('final_ocr')}",font=FONT,fill='black')
            draw.text((770,y+24),str(d.get('preprocessing',{}).get('effective_mode')),font=FONT,fill='black')
            if d.get('artifacts'):
                paste(canvas,d['artifacts']['rectified'],(770,y+50,390,150))
                paste(canvas,d['artifacts']['normalized'],(1180,y+50,400,100))
                draw.text((1180,y+156),'Actual 32x160 OCR input (enlarged)',font=FONT,fill='black')
        path=REPORT/'audit'/f'{Path(name).stem}.jpg';canvas.save(path,quality=95);pages.append(str(path))
    write(REPORT/'audit/artifact_index.json',{'records':output,'contact_sheets':pages,'exact_normalized_hash_verification':'PASS'})
    html='<html><meta charset="utf-8"><title>G failure review</title><h1>Frozen F failure review</h1><p>Exposed development; green boxes are model proposals, not GT.</p>'
    for p in pages:html+=f'<h2>{Path(p).stem}</h2><img style="max-width:100%" src="{Path(p).name}">'
    (REPORT/'audit/index.html').write_text(html,encoding='utf-8')
    print('Exact input hashes verified:',sum(len(r['detections']) for r in output),'detections;',len(pages),'failure sheets',flush=True)

if __name__=='__main__':main()
