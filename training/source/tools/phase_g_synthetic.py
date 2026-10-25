"""Small targeted OCR crop set with independently decoded raster identities."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.phase_g_common import ROOT,REPORT,CONFIG,IMAGES,read,write,SEED
from dataset.generator.render import render_plate,ASSETS
from dataset.generator.verify import raster_text
from dataset.schema import ALPHABET,sha256

OUT=ROOT.parent/'generated_data/targeted_g_v1'

def templates(style,thickness):
    result={}
    for ch in ALPHABET+'0123456789':
        if style=='authored':mask=np.array(Image.open(ASSETS/f'{ch}.png').convert('L'))
        else:
            image=Image.new('L',(240,260));draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',190)
            draw.text((10,0),ch,font=font,fill=255);bounds=image.getbbox();glyph=image.crop(bounds).resize((50,84))
            image=Image.new('L',(64,96));image.paste(glyph,(7,6));mask=np.array(image)
        if thickness:mask=(cv2.dilate if thickness>0 else cv2.erode)(mask,np.ones((3,3),np.uint8),iterations=1)
        result[ch]=mask
    return result

def main():
    recipe=read(CONFIG/'targeted_synthetic.json');rng=np.random.default_rng(SEED);forbidden=set()
    for file in (ROOT.parent/'generated_data/synthetic_d_large_v1/meta.csv',IMAGES.parent/'debug_labels.csv'):
        forbidden|={r['plate_num'] for r in csv.DictReader(file.open(encoding='utf-8-sig'),delimiter=';')}
    for r in read(ROOT.parent/'generated_data/detector_f_v2/manifest.json')['records']:
        forbidden|={a['plate_num'] for a in r['annotations']}
    records=[];banks={(style,thick):templates(style,thick) for style in ('authored','arial') for thick in (-1,0,1)}
    for i in range(360):
        kind=('type1','type1a','type1b')[i%3];split='train' if i<180 else 'val' if i<270 else 'test'
        while True:
            text=str(rng.choice(list(ALPHABET)))+f'{int(rng.integers(0,1000)):03}'+''.join(rng.choice(list(ALPHABET),2))+str(rng.choice([14,38,55,72,98,125,134,196,797]))
            if text not in forbidden:break
        forbidden.add(text);style=('authored','arial')[i%2];thick=(-1,0,1)[(i//2)%3];bank=banks[(style,thick)]
        plate,_,chars=render_plate(text,kind,2);array=np.array(plate);color=(246,194,41) if kind=='type1b' else (241,240,231)
        original_boxes=[c['box'][:] for c in chars]
        for c in chars:
            a,b,z,t=c['box'];array[b:t,a:z]=color
        for c,old in zip(chars,original_boxes):
            a,b,z,t=old;shift=int(rng.choice([-2,0,2]))*2 if c['index']<6 else 0
            a+=shift;z+=shift
            if c['index']>=6:
                center=(a+z)/2;half=(z-a)/2*float(rng.choice([.96,1.,1.04]));a,z=round(center-half),round(center+half)
            mask=cv2.resize(bank[text[c['index']]],(z-a,t-b),interpolation=cv2.INTER_LANCZOS4)
            alpha=mask.astype(float)/255.;array[b:t,a:z]=np.rint(np.array(color)[None,None,:]*(1-alpha[:,:,None])+np.array([19,21,20])*alpha[:,:,None]).astype(np.uint8)
            c['box']=[a,b,z,t]
        canonical=Image.fromarray(array);binary_bank={ch:mask>127 for ch,mask in bank.items()}
        decoded,scores=raster_text(canonical,chars,binary_bank)
        assert decoded==text and min(scores)>.72,(i,text,decoded,scores)
        width,height=canonical.size
        assert all(0<c['box'][0]<c['box'][2]<width and 0<c['box'][1]<c['box'][3]<height for c in chars)
        groups=[chars[:4],chars[4:]] if kind=='type1a' else [chars]
        assert all(a['box'][2]<b['box'][0] for group in groups for a,b in zip(group,group[1:]))
        bounds=recipe['source_resolution_width_by_type'].get(kind) or [100,260];small=int(rng.integers(bounds[0],bounds[1]+1));sh=max(16,round(small*height/width))
        degraded=cv2.resize(cv2.cvtColor(array,cv2.COLOR_RGB2BGR),(small,sh),interpolation=cv2.INTER_AREA)
        sigma=float(rng.choice([0,.35,.65]))
        if sigma:degraded=cv2.GaussianBlur(degraded,(3,3),sigma)
        brightness=float(rng.choice([.65,.85,1.]));degraded=np.clip(degraded.astype(float)*brightness,0,255).astype(np.uint8)
        quality=int(rng.choice([60,80,95]));_,encoded=cv2.imencode('.jpg',degraded,[cv2.IMWRITE_JPEG_QUALITY,quality])
        path=OUT/split/f'{i:03}_{kind}.jpg';path.parent.mkdir(parents=True,exist_ok=True);encoded.tofile(path)
        source=OUT/'canonical'/f'{i:03}.png';source.parent.mkdir(parents=True,exist_ok=True);canonical.save(source)
        records.append({'id':i,'image':str(path),'canonical':str(source),'split':split,'plate_num':text,'plate_type':kind,
            'font':style,'thickness':thick,'characters':chars,'raster_decoded':decoded,'raster_scores':scores,
            'width':small,'height':sh,'brightness':brightness,'sigma':sigma,'JPEG_quality':quality,
            'image_sha256':sha256(path),'canonical_sha256':sha256(source),'GT_source':'Actual rendered raster decoded without echoing label; font-specific templates constrained by positional mask',
            'is_vehicle':None,'scope':'OCR-only authored crop; no full scene detection or real type1b claim'})
    manifest={'records':records,'seed':SEED,'count':360,'raster_GT_checks':'360/360 PASS','identity_disjoint_from_D_F_public':True,'visual_review':'PENDING',
        'type1b':'Authored yellow control; real type1b unavailable','test_policy':'New90 test crops frozen before BEST-G lock, once terminal OCR-only comparison after selection'}
    write(OUT/'manifest.json',manifest);write(REPORT/'targeted_synthetic_manifest.json',manifest)
    for page in range(3):
        canvas=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(canvas)
        selected=[r for r in records if r['split']=='train'][page*12:(page+1)*12]
        for j,r in enumerate(selected):
            x=j%3*400;y=j//3*225;draw.text((x,y),r['plate_num']+' '+r['plate_type']+' '+r['font'],fill='black')
            image=Image.open(r['image']);scale=min(370/image.width,180/image.height,3);image=image.resize((round(image.width*scale),round(image.height*scale)));canvas.paste(image,(x,y+25))
        canvas.save(REPORT/f'targeted_synthetic_review_{page+1}.jpg',quality=95)
    write(OUT/'test/freeze.json',{'files':{r['image']:r['image_sha256'] for r in records if r['split']=='test'},'evaluation':'NOT_RUN'})
    print('Generated360 targeted OCR crops; independent raster GT checks PASS; review train samples before acceptance',flush=True)

if __name__=='__main__':main()
