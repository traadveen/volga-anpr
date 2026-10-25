"""Current-format procedural scenes, pixel gates, and unassigned identities."""
from collections import Counter
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image,ImageDraw

from dataset.schema import ALPHABET,Annotation,sha256,write_metadata
from dataset.validation import validate_dataset
from dataset.generator.render import ASSETS,render_plate
from dataset.generator.scenes_f import background,vehicle,compose as legacy_negative,NEGATIVES
from dataset.generator.type1b_v1 import templates,read_yellow_raster
from dataset.generator.verify import raster_text
from src.geometry_f import validate_quad,bbox_from_quad
from src.plate_formats import valid_plate

TYPES=('type1','type1a','type1b')
SCHEDULE=('type1b','type1a','type1','type1a','type1b')


def plate_number(rng,kind,region_length):
    prefix=''.join(rng.choice(list(ALPHABET),2 if kind=='type1b' else 1))
    number=''.join(rng.choice(list('0123456789'),3))
    suffix='' if kind=='type1b' else ''.join(rng.choice(list(ALPHABET),2))
    region='0'*region_length
    while set(region)=={'0'}:region=''.join(rng.choice(list('0123456789'),region_length))
    return prefix+number+suffix+region


def read_plate(image,kind,characters,bank):
    return (read_yellow_raster(image,characters,bank) if kind=='type1b'
            else raster_text(image,characters,bank))


def compose(rng,objects,bank,*,profile='clean'):
    if profile not in ('clean','mixed'):raise ValueError('Unknown scene profile')
    if not 1<=len(objects)<=2:raise ValueError('One or two vehicles required')
    scene=background(rng);width,height=scene.size;draw=ImageDraw.Draw(scene)
    proofs=[];canvases=[]
    for index,(kind,text) in enumerate(objects):
        plate,ink,characters=render_plate(text,kind,2,format_profile='russian_v1')
        decoded,scores=read_plate(plate,kind,characters,bank)
        if decoded!=text or min(scores)<.76:raise ValueError('Canonical raster differs from label')
        cw,ch=plate.size;slot=width/len(objects)
        plate_w=float(rng.uniform(165,205) if len(objects)==1 else rng.uniform(135,155))
        plate_h=plate_w*ch/cw;cx=slot*(index+.5);cy=float(rng.uniform(420,500))
        car_w=min(slot-18,plate_w*2.7);car_h=max(170,plate_h*3.3)
        body=(cx-car_w/2,max(30,cy-car_h*.76),cx+car_w/2,min(height-15,cy+car_h*.24))
        vehicle(draw,body,rng)
        angle=float(rng.uniform(-8,8) if profile=='clean' else rng.uniform(-16,16))
        a=np.deg2rad(angle);rotation=np.float32([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        local=np.float32([[-plate_w/2,-plate_h/2],[plate_w/2,-plate_h/2],[plate_w/2,plate_h/2],[-plate_w/2,plate_h/2]])
        quad=local@rotation.T+np.float32([cx,cy])
        if profile=='mixed':quad+=rng.uniform(-.025,.025,(4,2))*[plate_w,plate_h]
        validate_quad(quad,(height,width))
        if any(not (body[0]<=x<=body[2] and body[1]<=y<=body[3]) for x,y in quad):
            raise ValueError('Plate surface left its procedural vehicle')
        corners=np.float32([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]])
        matrix=cv2.getPerspectiveTransform(corners,quad.astype(np.float32))
        surface=cv2.warpPerspective(np.asarray(plate),matrix,(width,height))
        alpha=cv2.warpPerspective(np.full((ch,cw),255,np.uint8),matrix,(width,height))[...,None]/255
        pixels=np.asarray(scene);pixels=np.rint(pixels*(1-alpha)+surface*alpha).astype(np.uint8)
        scene=Image.fromarray(pixels);draw=ImageDraw.Draw(scene)
        ink_points=[]
        for entry in characters:
            x0,y0,x1,y1=entry['box'];points=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
            ink_points.append(cv2.perspectiveTransform(points[None],matrix)[0].tolist())
        proofs.append({'plate_num':text,'plate_type':kind,'quad':quad.tolist(),'bbox_xyxy':list(map(float,bbox_from_quad(quad))),
            'homography':matrix.tolist(),'canonical_size':[cw,ch],'characters':characters,'character_quads':ink_points,
            'canonical_read':decoded,'canonical_scores':scores,'vehicle_bbox_xyxy':list(body),'angle':angle,
            'quad_source':'exact_generator_homography'})
        canvases.append(plate)
    pixels=np.asarray(scene).copy();brightness=1.;warm_cast=False
    if profile=='mixed':
        brightness=float(rng.uniform(.75,1.08));warm_cast=bool(rng.random()<.35)
        scale=np.array([1.,.96,.84] if warm_cast else [1.,1.,1.])*brightness
        pixels=np.clip(pixels*scale,0,255).astype(np.uint8)
        if rng.random()<.25:pixels=cv2.GaussianBlur(pixels,(3,3),.35)
    for proof in proofs:
        cw,ch=proof['canonical_size']
        rectified=cv2.warpPerspective(pixels,np.linalg.inv(np.array(proof['homography'])),(cw,ch))
        text,scores=read_plate(Image.fromarray(rectified),proof['plate_type'],proof['characters'],bank)
        if text!=proof['plate_num'] or min(scores)<.56:raise ValueError('Projected scene characters do not match label')
        proof.update(scene_read=text,scene_scores=scores)
    return pixels,proofs,canvases,{'brightness':brightness,'warm_cast':warm_cast,'profile':profile}


def verify_record(root,record,bank):
    root=Path(root);path=root/record['image']
    if sha256(path)!=record['sha256']:raise ValueError('Scene bytes changed')
    with Image.open(path) as image:pixels=np.asarray(image.convert('RGB'))
    height,width=pixels.shape[:2]
    labels=(root/'labels'/Path(record['image']).with_suffix('.txt').name).read_text('utf-8').splitlines()
    if len(labels)!=len(record['annotations']):raise ValueError('Detector labels and object count disagree')
    for proof,label in zip(record['annotations'],labels):
        text=proof['plate_num'];kind=proof['plate_type']
        if not valid_plate(text,kind):raise ValueError('Wrong type-specific format')
        if [c['index'] for c in proof['characters']]!=list(range(len(text))):raise ValueError('Wrong character order')
        quad=validate_quad(np.array(proof['quad']),(height,width));matrix=np.array(proof['homography']);cw,ch=proof['canonical_size']
        corners=np.float32([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]])
        if not np.allclose(cv2.perspectiveTransform(corners[None],matrix)[0],quad,atol=1e-3):raise ValueError('Homography and quad disagree')
        if not np.allclose(bbox_from_quad(quad),proof['bbox_xyxy'],atol=1e-3):raise ValueError('Box and quad disagree')
        x0,y0,x1,y1=proof['bbox_xyxy'];expected=[(x0+x1)/2/width,(y0+y1)/2/height,(x1-x0)/width,(y1-y0)/height]
        values=label.split()
        if len(values)!=5 or values[0]!=str(TYPES.index(kind)) or not np.allclose(list(map(float,values[1:])),expected,atol=1e-8):
            raise ValueError('Normalized detector label disagrees with scene')
        a,b,c,d=proof['vehicle_bbox_xyxy']
        if any(not (a<=x<=c and b<=y<=d) for x,y in quad):raise ValueError('Quad lies outside procedural vehicle')
        canonical=root/proof['canonical'];
        if sha256(canonical)!=proof['canonical_sha256']:raise ValueError('Canonical bytes changed')
        with Image.open(canonical) as image:decoded,scores=read_plate(image.convert('RGB'),kind,proof['characters'],bank)
        if decoded!=text or min(scores)<.76:raise ValueError('Canonical pixels disagree with label')
        rectified=cv2.warpPerspective(pixels,np.linalg.inv(matrix),(cw,ch))
        decoded,scores=read_plate(Image.fromarray(rectified),kind,proof['characters'],bank)
        if decoded!=text or min(scores)<.56:raise ValueError('Scene pixels disagree with label')
    return True


def package_generator(stage):
    """Include executable source and exact glyphs, not references to this machine."""
    project=Path(__file__).resolve().parents[2];sources=[]
    for directory in ('dataset','dataset/generator'):
        sources.extend(p.relative_to(project) for p in (project/directory).glob('*.py'))
    sources.extend(map(Path,('src/__init__.py','src/geometry_f.py','src/plate_formats.py','tools/generate_current_scenes.py')))
    sources.extend(p.relative_to(project) for p in ASSETS.iterdir() if p.is_file())
    hashes={}
    for relative in sorted(sources):
        source=project/relative;target=stage/'generator/source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        hashes[relative.as_posix()]=sha256(source);shutil.copy2(source,target)
        if sha256(target)!=hashes[relative.as_posix()]:raise ValueError('Generator source changed during packaging')
    (stage/'generator/requirements.txt').write_text('numpy==2.2.3\nopencv-python==5.0.0.93\nPillow==12.0.0\n',encoding='utf-8')
    return hashes


def generate(output,count=32,seed=20261015,profile='mixed',negative_every=6):
    if type(count) is not int or count<1 or count>20000:raise ValueError('count must be integer 1..20000')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('seed must be integer 0..2^32-1')
    if profile not in ('clean','mixed'):raise ValueError('Unknown scene profile')
    if type(negative_every) is not int or not 0<=negative_every<=1000:raise ValueError('negative_every must be integer 0..1000')
    output=Path(output).resolve()
    if output.exists():raise ValueError('Output exists; published collections are preserved')
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='current-scenes-',dir=output.parent) as temporary:
        temporary=Path(temporary).resolve()
        if not temporary.is_relative_to(output.parent):raise ValueError('Staging directory escaped parent')
        stage=temporary/'scenes';(stage/'images/synthetic').mkdir(parents=True);(stage/'images/real').mkdir()
        (stage/'labels').mkdir();(stage/'generator/reference').mkdir(parents=True)
        shutil.copytree(ASSETS,stage/'generator/assets');bank=templates();rows=[];records=[];seen=set();rejections=Counter()
        positive_index=0
        for index in range(count):
            rng=np.random.default_rng(np.random.SeedSequence([seed,index]));negative=NEGATIVES[(index//negative_every)%len(NEGATIVES)] if negative_every and index%negative_every==0 else None
            objects=[]
            if not negative:
                for offset in range(2 if index%7==0 else 1):
                    kind=SCHEDULE[(positive_index+offset)%len(SCHEDULE)]
                    text=plate_number(rng,kind,2+(index+offset)%2)
                    while text in seen:text=plate_number(rng,kind,2+(index+offset)%2)
                    seen.add(text);objects.append((kind,text))
                positive_index+=1
            if negative:
                pixels,_=legacy_negative(rng,[],[],negative);proofs=[];canvases=[];effects={'profile':profile}
            else:
                for attempt in range(24):
                    try:pixels,proofs,canvases,effects=compose(rng,objects,bank,profile=profile);break
                    except ValueError as error:rejections[str(error)]+=1
                else:raise ValueError(f'No readable scene for image {index}')
            name=f'{index:05d}.png';relative='images/synthetic/'+name;path=stage/relative
            Image.fromarray(pixels).save(path,compress_level=1)
            conditions=('procedural_scene',profile)+(('warm_cast',) if effects.get('warm_cast') else ())
            if not proofs:rows.append(Annotation(relative,'','',None,None,False,True,'own_generator:scene_current_v1','CC-BY-4.0',conditions,annotation_kind='negative'))
            labels=[];height,width=pixels.shape[:2]
            for offset,(proof,canonical) in enumerate(zip(proofs,canvases)):
                reference=f'generator/reference/{index:05d}_{offset}.png';canonical.save(stage/reference)
                proof.update(canonical=reference,canonical_sha256=sha256(stage/reference))
                x0,y0,x1,y1=proof['bbox_xyxy']
                xywh=[(x0+x1)/2/width,(y0+y1)/2/height,(x1-x0)/width,(y1-y0)/height]
                labels.append(str(TYPES.index(proof['plate_type']))+' '+' '.join(f'{v:.9f}' for v in xywh))
                rows.append(Annotation(relative,proof['plate_num'],proof['plate_type'],(x0,y0,x1-x0,y1-y0),
                    tuple(map(tuple,proof['quad'])),True,True,'own_generator:scene_current_v1','CC-BY-4.0',conditions,
                    group_id=f'scene-{seed}-{index:05d}'))
            (stage/'labels'/Path(name).with_suffix('.txt')).write_text('\n'.join(labels)+('\n' if labels else ''),encoding='utf-8')
            record={'image':relative,'sha256':sha256(path),'negative_kind':negative,'annotations':proofs,'effects':effects,'split':'unassigned'}
            verify_record(stage,record,bank);records.append(record)
            if count>=100 and (index+1)%100==0:print(f'Generated and pixel-checked {index+1}/{count} scenes',flush=True)
        write_metadata(stage/'meta.csv',rows)
        validation=validate_dataset(stage,check_splits=False,format_profile='russian_v1')
        if not validation['valid']:raise ValueError('Generated metadata validation failed: '+str(validation['errors'][:3]))
        manifest={'version':'scene-current-v1','seed':seed,'count':count,'profile':profile,'negative_every':negative_every,'format_profile':'russian_v1',
            'records':records,'rejected_attempts':dict(rejections),'plates_by_type':dict(Counter(p['plate_type'] for r in records for p in r['annotations'])),
            'negative_scenes':sum(bool(r['negative_kind']) for r in records),'split':'unassigned',
            'is_synthetic':True,'real_images':0,'training_started':False,'final_split_created':False,
            'source_sha256':package_generator(stage),'metadata_sha256':sha256(stage/'meta.csv'),
            'limitations':['Stylized procedural vehicles; no proof of real-world generalization','Template reading is a raster consistency gate, not an independent OCR benchmark','Clean/mixed data are development controls, not untouched real test']}
        (stage/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        (stage/'README.md').write_text('Current-format authored procedural scenes. CC BY 4.0.\n'
            'Attribution: Volga ANPR procedural generator and glyph assets. Glyphs approximate a plate font.\n'
            'All vehicle context is synthetic. No human photographs or external backgrounds.\n'
            'No train/val/test assignment; no training. Not an independent accuracy benchmark.\n'
            'meta.csv is authoritative. labels/*.txt: normalized YOLO detection boxes, class0 type1,1 type1a,2 type1b; negative files empty.\n'
            'Install generator/requirements.txt with Python3.12. Reproduce in a new directory:\n'
            f'python generator/source/tools/generate_current_scenes.py --count {count} --seed {seed} --profile {profile} --negative-every {negative_every} --output NEW_DIRECTORY\n'
            'Exact image hashes require the recorded library versions. Organizer validator has not been run.\n',encoding='utf-8')
        (stage/'LICENSE').write_text('Authored procedural images and glyphs: Volga ANPR procedural generator, CC BY 4.0.\n'
            'https://creativecommons.org/licenses/by/4.0/\nNo third-party photographs included.\n',encoding='utf-8')
        stage.rename(output)
    return manifest
