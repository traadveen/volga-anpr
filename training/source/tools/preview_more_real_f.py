"""Human annotation aids for completely labeling scenes containing real type1."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.phase_e import ROOT,IMAGES,read,write,load_image
from tools.phase_f_setup import REPORT

NAMES=['img_018.jpg','img_026.jpg','img_032.jpg','img_033.jpg','img_087.jpg','img_092.jpg','img_097.jpg']


def main():
    folder=REPORT/'real_annotation_v2';folder.mkdir(exist_ok=True)
    refs=read(ROOT/'reports/phase_e/detections.json')['records']
    new=read(REPORT/'experiments/F02_finetune_1024/trace.json')['records']
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    entries=[]
    for name in NAMES:
        frame=load_image(IMAGES/name);h,w=frame.shape[:2]
        canvas=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(canvas)
        for x in range(0,w,100):draw.line((x,0,x,h),fill=(120,180,200),width=1);draw.text((x+2,8),str(x),font=font,fill='red')
        for y in range(0,h,100):draw.line((0,y,w,y),fill=(120,180,200),width=1);draw.text((5,y+2),str(y),font=font,fill='red')
        canvas.thumbnail((1400,1200));canvas.save(folder/f'{Path(name).stem}_full.jpg')
        # Candidate locations are aids only; actual human border points define GT.
        boxes=[r['bbox'] for r in refs if r['image']==name and r['plate_type']=='type1a']
        boxes += [r['bbox'] for t in new if t['image']==name for r in t['detections'] if r.get('plate_type')=='type1a']
        seen=[]
        for box in boxes:
            if any(np.allclose(box,b,atol=20) for b in seen):continue
            seen.append(box);x1,y1,x2,y2=map(int,box)
            x1=max(0,x1-40);y1=max(0,y1-40);x2=min(w,x2+40);y2=min(h,y2+40)
            crop=Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)).resize(((x2-x1)*3,(y2-y1)*3))
            d=ImageDraw.Draw(crop)
            for x in range((x1//10+1)*10,x2,10):
                d.line(((x-x1)*3,0,(x-x1)*3,crop.height),fill=(80,180,220),width=1)
                d.text(((x-x1)*3+1,0),str(x),font=font,fill='red')
            for y in range((y1//10+1)*10,y2,10):
                d.line((0,(y-y1)*3,crop.width,(y-y1)*3),fill=(80,180,220),width=1)
                d.text((0,(y-y1)*3+1),str(y),font=font,fill='red')
            file=folder/f'{Path(name).stem}_proposal_{len(seen)-1}.jpg';crop.save(file)
            entries.append({'image':name,'candidate':str(file),'source_bounds':[x1,y1,x2,y2]})
    write(folder/'preview_manifest.json',entries)
    print(entries)

if __name__=='__main__':main()
