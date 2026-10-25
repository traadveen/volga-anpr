"""Authored yellow LLDDDRR/RRR crops, separate from frozen synthetic v2."""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from src.plate_formats import ALPHABET,valid_plate,position_mask
from dataset.generator.render import ASSETS
from dataset.schema import Annotation,sha256,write_metadata


def render_type1b(text,supersample=2):
    if not valid_plate(text,'type1b'):raise ValueError('A yellow plate needs AA00077 / AA000777')
    if type(supersample) is not int or not 1<=supersample<=4:raise ValueError('supersample must be integer 1..4')
    s=supersample;plate=Image.new('RGB',(520*s,112*s),(246,194,41));ink=Image.new('L',plate.size);draw=ImageDraw.Draw(plate)
    draw.rounded_rectangle((s,s,518*s,110*s),radius=6*s,outline=(24,25,23),width=2*s)
    characters=[]
    def put(index,x,y,w,h):
        box=[round(v*s) for v in (x,y,x+w,y+h)]
        with Image.open(ASSETS/(text[index]+'.png')) as glyph:
            mask=glyph.convert('L').resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
        plate.paste((19,21,20),tuple(box),mask);ink.paste(mask,tuple(box))
        characters.append({'index':index,'row':0,'box':box})
    for index,x in enumerate((25,88)):put(index,x,22,48,70)
    for index,x in enumerate((181,244,307),2):put(index,x,14,50,82)
    draw.line(((383*s,5*s),(383*s,106*s)),fill=(30,31,28),width=2*s)
    region=text[5:];rw=29 if len(region)==3 else 43;gap=8
    start=390+(124-(rw*len(region)+gap*(len(region)-1)))/2
    for i in range(len(region)):put(5+i,start+i*(rw+gap),15,rw,59)
    draw.text((410*s,82*s),'RUS',font=ImageFont.load_default(size=13*s),fill=(28,29,28))
    for x,y in ((12,55),(371,55)):
        draw.ellipse(((x-2)*s,(y-2)*s,(x+2)*s,(y+2)*s),fill=(96,96,91))
    return plate,ink,characters


def read_yellow_raster(image,characters,templates):
    """Read pixels using type/length positions, with no proposed text as input."""
    gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY);mask=position_mask('type1b',len(characters));text=[];scores=[]
    if [c['index'] for c in characters]!=list(range(len(mask))):raise ValueError('Incorrect character order')
    for entry,slot in zip(characters,mask):
        x0,y0,x1,y1=entry['box'];cell=gray[y0:y1,x0:x1]
        if not cell.size:raise ValueError('Empty character cell')
        _,binary=cv2.threshold(cell,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        normalized=cv2.resize(binary,(64,96),interpolation=cv2.INTER_AREA)>127
        def distance(ch):
            expected=templates[ch]
            return float(np.count_nonzero(normalized^expected)/max(1,np.count_nonzero(normalized|expected)))
        best=min(ALPHABET if slot=='L' else '0123456789',key=lambda ch:(distance(ch),ch))
        text.append(best);scores.append(1-distance(best))
    return ''.join(text),scores


def templates():
    values={}
    for char in ALPHABET+'0123456789':
        with Image.open(ASSETS/(char+'.png')) as glyph:values[char]=np.array(glyph.convert('L').resize((64,96)))>127
    return values


def generate(output,count=24,seed=20261015):
    if type(count) is not int or not 1<=count<=10000:raise ValueError('count must be integer 1..10000')
    if type(seed) is not int or seed<0:raise ValueError('seed must be a nonnegative integer')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be new or empty; original sets are preserved')
    rng=np.random.default_rng(seed);reference=templates();records=[]
    images=output/'images/synthetic';images.mkdir(parents=True,exist_ok=True)
    for index in range(count):
        # Deliberately test both supported region lengths, without claiming issuance.
        length=2+index%2;text=''.join(rng.choice(list(ALPHABET),2))+''.join(rng.choice(list('0123456789'),3+length))
        image,ink,chars=render_type1b(text)
        decoded,scores=read_yellow_raster(image,chars,reference)
        if decoded!=text or min(scores)<.76:raise ValueError('Rendered pixels do not match their authored ground truth')
        path=images/(f'{index:05d}.png');image.save(path)
        records.append({'image':path.relative_to(output).as_posix(),'plate_num':text,'plate_type':'type1b',
            'is_synthetic':True,'is_vehicle':None,'annotation_kind':'ocr_crop','source':'own_generator:type1b_v1',
            'split':'unassigned','image_sha256':sha256(path),'characters':chars,'raster_decoded':decoded,
            'raster_scores':scores,'bbox':[0,0,image.width,image.height],
            'quad':[[0,0],[image.width,0],[image.width,image.height],[0,image.height]]})
    manifest={'schema_version':'type1b-prototype-1','seed':seed,'count':count,'format':'LLDDDRR/RRR',
        'records':records,'glyph_sha256':{c:sha256(ASSETS/(c+'.png')) for c in ALPHABET+'0123456789'},
        'training_started':False,'final_split_created':False,'real_images':0,'supersample':2,
        'scope':'Authored pixel/format control crops, not real vehicle scenes or a final submission dataset'}
    annotations=[Annotation(r['image'],r['plate_num'],r['plate_type'],tuple(r['bbox']),
                 tuple(map(tuple,r['quad'])),None,True,r['source'],'CC-BY-4.0',('canonical_crop',),
                 annotation_kind='ocr_crop') for r in records]
    write_metadata(output/'meta.csv',annotations)
    manifest.update(format_profile='russian_v1',metadata_file='meta.csv',license='CC-BY-4.0')
    (output/'ATTRIBUTION.md').write_text('Authored control images: volga-anpr type1b_v1 generator. CC BY 4.0.\n'
        'Glyph shapes: original programmatic approximations stored in dataset/generator/assets; glyph byte hashes in manifest.json.\n'
        'No third-party vehicle photographs. No claim that generated plate numbers have been issued.\n',encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest
