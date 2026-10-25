"""Authored full scenes, not cropped plates; explicit procedural-domain limits."""
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from dataset.generator.render import render_plate, ASSETS
from dataset.generator.generate import plate_number
from dataset.generator.verify import raster_text
from dataset.schema import sha256
from src.geometry_f import validate_quad, bbox_from_quad, order_obb

TYPES = ('type1', 'type1a', 'type1b')
NEGATIVES = ('vehicle_without_plate', 'billboard', 'road_sign', 'shop_sign',
             'screen', 'printed_plate', 'text', 'background')


def background(rng, width=960, height=640):
    style = int(rng.integers(0, 4))
    colors = [(112, 143, 163), (100, 107, 98), (46, 55, 67), (158, 161, 158)]
    im = Image.new('RGB', (width, height), colors[style]); d = ImageDraw.Draw(im)
    horizon = int(rng.integers(180, 330))
    for _ in range(20):
        x = int(rng.integers(-30, width)); y = int(rng.integers(0, horizon))
        color = tuple(int(v) for v in rng.integers(40, 160, 3))
        d.rectangle((x, y, x+int(rng.integers(20, 180)), horizon+40), fill=color)
    d.rectangle((0, horizon, width, height), fill=(60, 63, 66))
    for _ in range(8):
        x = int(rng.integers(0, width)); y = int(rng.integers(horizon, height))
        d.line((x, y, x+20, y+60), fill=(143, 143, 136), width=3)
    return im


def vehicle(draw, box, rng):
    x1, y1, x2, y2 = box; w, h = x2-x1, y2-y1
    body = tuple(int(v) for v in rng.integers(30, 220, 3))
    draw.rounded_rectangle(box, radius=max(3, int(w*.09)), fill=body, outline=(25,25,25), width=3)
    draw.polygon([(x1+w*.2,y1),(x2-w*.2,y1),(x2-w*.1,y1+h*.35),(x1+w*.1,y1+h*.35)], fill=(38,49,61))
    draw.rectangle((x1+w*.15,y1+h*.55,x2-w*.15,y1+h*.66), fill=(24,26,28))
    for x in (x1+w*.06, x2-w*.24):
        draw.rectangle((x,y1+h*.44,x+w*.18,y1+h*.56), fill=(217,214,181))
    draw.rectangle((x1+w*.03,y2-h*.06,x2-w*.03,y2), fill=(26,29,32))
    for x in (x1+w*.04,x2-w*.16):
        draw.rectangle((x,y2-h*.02,x+w*.12,y2+h*.08), fill=(16,17,19))


def compose(rng, kinds, numbers, negative_kind=None):
    im = background(rng); draw = ImageDraw.Draw(im); width,height = im.size
    records = []
    if negative_kind:
        if negative_kind == 'vehicle_without_plate':
            vehicle(draw, (200,220,720,520), rng)
        elif negative_kind == 'road_sign':
            draw.rectangle((455,260,465,560),fill=(160,160,158))
            draw.ellipse((370,110,550,290),fill=(192,35,30))
            draw.ellipse((390,130,530,270),fill=(236,234,227))
            draw.text((422,178),'60',fill=(20,20,20),font_size=58)
            draw.rectangle((410,310,530,390),fill=(34,83,137))
            draw.text((424,320),'P',fill='white',font_size=58)
        elif negative_kind == 'shop_sign':
            draw.rectangle((110,140,850,550),fill=(168,150,132),outline=(40,40,40),width=5)
            draw.rectangle((130,165,830,260),fill=(214,217,204))
            draw.text((220,182),'AUTO SHOP 24',fill=(25,45,36),font_size=53)
            for x in (150,600):draw.rectangle((x,300,x+170,490),fill=(61,85,106),outline=(200,200,200),width=8)
            draw.rectangle((405,295,545,540),fill=(51,72,86),outline=(205,205,200),width=8)
        elif negative_kind == 'billboard':
            draw.rectangle((250,360,267,565),fill=(146,144,140));draw.rectangle((682,360,699,565),fill=(146,144,140))
            draw.rectangle((150,130,790,360),fill=(223,191,95),outline=(48,43,32),width=12)
            draw.text((195,170),'RENT  8 800 555 01 23',fill=(40,28,20),font_size=45)
            draw.text((275,250),'SALE 50%',fill=(160,35,30),font_size=55)
        elif negative_kind == 'text':
            draw.rectangle((80,180,880,430),fill=(226,223,211))
            for j,text in enumerate(('BUILDING 123   ENTRANCE 4','OPEN 9:00 - 21:00','PARKING 75 RUB / HOUR')):
                draw.text((130,205+65*j),text,fill=(25,25,25),font_size=37)
        elif negative_kind != 'background':
            draw.rectangle((180,180,740,400), fill=(190,188,176), outline='black', width=8)
            if negative_kind == 'printed_plate':
                plate, _, _ = render_plate('A123BC777', 'type1', 1)
                im.paste(plate.resize((390,84)), (260,250))
            else:
                draw.text((230,210), 'NEWS  21:45', fill=(20,25,30), font_size=45)
                if negative_kind == 'screen':
                    draw.rectangle((190,190,730,390), outline=(20,20,20), width=20)
                    draw.rectangle((405,400,515,420),fill=(20,20,20))
                    draw.rectangle((350,420,570,435),fill=(20,20,20))
        return np.asarray(im).copy(), records
    # Separate lanes prevent plate-to-plate occlusion; each plate has vehicle context.
    for index, (kind, text) in enumerate(zip(kinds, numbers)):
        plate, ink, chars = render_plate(text, kind, 1)
        cw, ch = plate.size
        plate_w = float(rng.choice([32,48,64,96,140,200,260]))
        if len(kinds) > 1: plate_w = min(plate_w, 140)
        plate_h = plate_w*ch/cw
        slot_width = width/len(kinds)
        cx = slot_width*(index+.5)+float(rng.uniform(-slot_width*.13,slot_width*.13))
        cy = float(rng.uniform(300,540))
        car_w = max(70, plate_w*2.8); car_h = max(90, plate_h*3.7)
        x1=max(8,cx-car_w*.5); x2=min(width-8,cx+car_w*.5)
        y1=max(30,cy-car_h*.72); y2=min(height-15,cy+car_h*.25)
        vehicle(draw, (x1,y1,x2,y2), rng)
        # Rotated perspective surface corners from an actual homography.
        angle = float(rng.uniform(-28,28)); a=np.deg2rad(angle)
        local=np.float32([[-plate_w/2,-plate_h/2],[plate_w/2,-plate_h/2],
                          [plate_w/2,plate_h/2],[-plate_w/2,plate_h/2]])
        rotation=np.float32([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        quad=local@rotation.T+np.float32([cx,cy])
        quad += rng.uniform(-.06,.06,(4,2))*np.float32([plate_w,plate_h])
        # Keep the complete physical plate surface inside the image. Adjust the
        # destination quad before building H, never clip an already rendered quad.
        low=quad.min(axis=0);high=quad.max(axis=0)
        offset=np.maximum(np.array([8.,8.])-low,0)-np.maximum(high-np.array([width-9.,height-9.]),0)
        quad+=offset
        validate_quad(quad, (height,width))
        src=np.float32([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]])
        matrix=cv2.getPerspectiveTransform(src,quad.astype(np.float32))
        scene=np.asarray(im).copy()
        warped=cv2.warpPerspective(np.array(plate),matrix,(width,height))
        mask=cv2.warpPerspective(np.full((ch,cw),255,np.uint8),matrix,(width,height))
        alpha=mask[...,None]/255.;scene=np.rint(scene*(1-alpha)+warped*alpha).astype(np.uint8)
        im=Image.fromarray(scene); draw=ImageDraw.Draw(im)
        occ = bool(rng.random()<.12)
        if occ:
            px,py=quad[0];draw.rectangle((px,py,px+plate_w*.06,py+plate_h*.5),fill=(63,64,62))
        obb=order_obb(cv2.boxPoints(cv2.minAreaRect(quad.astype(np.float32))))
        records.append({'plate_type':kind,'plate_num':text,'bbox':bbox_from_quad(quad),
            'quad':quad.tolist(),'obb':obb.tolist(),'homography':matrix.tolist(),
            'canonical_size':[cw,ch],'characters':chars,'occlusion':occ,'angle':angle,
            'quad_source':'exact_generator_homography','plate_width':plate_w})
    arr=np.asarray(im).copy(); brightness=float(rng.uniform(.45,1.15))
    arr=np.clip(arr.astype(float)*brightness+rng.normal(0,rng.uniform(0,3),arr.shape),0,255).astype(np.uint8)
    if rng.random()<.25: arr=cv2.GaussianBlur(arr,(3,3),float(rng.uniform(.3,1.1)))
    return arr,records


def generate(destination, counts=(1600,240,240), seed=20261003):
    destination=Path(destination)
    if destination.exists(): raise ValueError('Never overwrite a published collection')
    destination.mkdir(parents=True)
    templates={c:np.array(Image.open(ASSETS/f'{c}.png').convert('L').resize((64,96)))>127
               for c in '0123456789ABCEHKMOPTXY'}
    seen=set(); rows=[]; canonical_checks=0
    for split, count in zip(('train','val','test'),counts):
        for task in ('detect','obb','pose'):
            (destination/task/'images'/split).mkdir(parents=True)
            (destination/task/'labels'/split).mkdir(parents=True)
        rng=np.random.default_rng(np.random.SeedSequence([seed, {'train':0,'val':1,'test':2}[split]]))
        for i in range(count):
            negative=NEGATIVES[(i//5)%len(NEGATIVES)] if i%5==0 else None
            kinds=[TYPES[(i+j)%3] for j in range(2 if i%13==0 else 1)] if not negative else []
            numbers=[]
            for kind in kinds:
                text=plate_number(rng,2+i%2)
                while text in seen:text=plate_number(rng,2+i%2)
                seen.add(text);numbers.append(text)
                plate,_,chars=render_plate(text,kind,1)
                decoded,scores=raster_text(plate,chars,templates)
                if decoded!=text or min(scores)<.76: raise ValueError('Canonical raster/GT mismatch')
                canonical_checks+=1
            rgb,annotations=compose(rng,kinds,numbers,negative)
            name=f'{split}_{i:05}.jpg'; h,w=rgb.shape[:2]
            path=destination/'detect/images'/split/name
            cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])[1].tofile(path)
            for task in ('obb','pose'):
                import os
                os.link(path,destination/task/'images'/split/name)
            labels={t:[] for t in ('detect','obb','pose')}
            for a in annotations:
                c=TYPES.index(a['plate_type']);box=np.array(a['bbox']);q=np.array(a['quad']); obb=np.array(a['obb'])
                xywh=np.r_[(box[:2]+box[2:])/2,box[2:]-box[:2]]/np.array([w,h,w,h])
                labels['detect'].append(' '.join([str(c)]+[f'{v:.8f}' for v in xywh]))
                labels['obb'].append(' '.join([str(c)]+[f'{v:.8f}' for v in (obb/[w,h]).ravel()]))
                labels['pose'].append(' '.join([str(c)]+[f'{v:.8f}' for v in xywh]+
                    [f'{v:.8f}' for point in q/[w,h] for v in (*point,2)]))
                a['bbox_normalized']=xywh.tolist()
                src=np.float32([[0,0],[a['canonical_size'][0]-1,0],
                    [a['canonical_size'][0]-1,a['canonical_size'][1]-1],[0,a['canonical_size'][1]-1]])
                check=cv2.perspectiveTransform(src[None],np.array(a['homography']))[0]
                assert np.allclose(check,q,atol=1e-3)
            for task in labels:
                (destination/task/'labels'/split/Path(name).with_suffix('.txt')).write_text('\n'.join(labels[task])+'\n',encoding='utf-8')
            rows.append({'image':name,'split':split,'width':w,'height':h,'is_synthetic':True,
                'domain':'authored_procedural_vehicle_scene','negative_kind':negative,'annotations':annotations,
                'sha256':sha256(path)})
        print(f'{split}: {count} scenes',flush=True)
    for task in ('detect','obb','pose'):
        text=f'path: {str((destination/task).resolve()).replace(chr(92),"/")}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: type1\n  1: type1a\n  2: type1b\n'
        if task=='pose':text+='kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\n'
        (destination/f'{task}.yaml').write_text(text,encoding='utf-8')
    manifest={'version':'scene-f-v1','seed':seed,'records':rows,'counts':dict(zip(('train','val','test'),counts)),
        'composition':{'synthetic_scenes':len(rows),'real_scenes':0,'negative_scenes':sum(bool(r['negative_kind']) for r in rows),
            'plates_by_type':dict(Counter(a['plate_type'] for r in rows for a in r['annotations']))},
        'canonical_raster_checks':canonical_checks,'split_policy':'Distinct plate identities, seeds and procedural scene instances; source domain shared',
        'quad_policy':'Actual homography, TL TR BR BL; OBB is explicit min-area rectangle approximation',
        'license':'CC-BY-4.0 authored procedural assets','source_hashes':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('render.py')]},
        'limitations':['Procedural vehicles are stylized, not real photographs','No real validation or untouched real holdout is available',
            'Tiny/occluded/blurred plates have exact authored identity but may be unreadable','Synthetic metrics do not establish real generalization']}
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest
