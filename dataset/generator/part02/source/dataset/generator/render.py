"""Plate layouts in canonical pixels; every target glyph has an auditable ink crop."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..schema import TARGETS, plate_error

ASSETS=Path(__file__).parent/'assets'
CANONICAL={'type1':(520,112),'type1a':(290,170),'type1b':(520,112)}


def render_plate(text, plate_type, supersample=2, *, format_profile='legacy_v2'):
    if format_profile not in ('legacy_v2','russian_v1'):
        raise ValueError('Unknown format_profile')
    if format_profile=='russian_v1':
        from src.plate_formats import valid_plate
        error=not valid_plate(text,plate_type)
    else:error=plate_error(text)
    if plate_type not in TARGETS or error or '#' in text:
        raise ValueError('Generator requires a fully known valid target plate')
    if isinstance(supersample,bool) or not isinstance(supersample,int) or not 1<=supersample<=4:
        raise ValueError('supersample must be integer 1..4')
    if format_profile=='russian_v1' and plate_type=='type1b':
        from .type1b_v1 import render_type1b
        return render_type1b(text,supersample)
    width,height=CANONICAL[plate_type]; s=supersample
    color=(246,194,41) if plate_type=='type1b' else (241,240,231)
    plate=Image.new('RGB',(width*s,height*s),color); ink=Image.new('L',plate.size)
    draw=ImageDraw.Draw(plate)
    draw.rounded_rectangle((s,s,(width-2)*s,(height-2)*s),radius=6*s,outline=(24,25,23),width=2*s)
    chars=[]
    def put(index,x,y,w,h,row):
        box=[int(x*s),int(y*s),int((x+w)*s),int((y+h)*s)]
        mask=Image.open(ASSETS/f'{text[index]}.png').convert('L').resize((int(w*s),int(h*s)),Image.Resampling.LANCZOS)
        plate.paste((19,21,20),tuple(box),mask); ink.paste(mask,tuple(box))
        chars.append({'index':index,'row':row,'box':box})
    region=text[6:]
    if plate_type=='type1a':
        put(0,24,28,42,48,0)
        for i,x in enumerate([82,144,206],1): put(i,x,12,43,66,0)
        put(4,20,99,40,55,1);put(5,78,99,40,55,1)
        draw.line([(134*s,91*s),(134*s,165*s)],fill=(30,31,28),width=2*s)
        rw=31 if len(region)==3 else 40; gap=9
        start=141+(140-(rw*len(region)+gap*(len(region)-1)))/2
        for i in range(len(region)): put(6+i,start+i*(rw+gap),100,rw,42,1)
        rus=(155,149);flag=(217,151)
        bolts=[(12,84),(278,84)]
    else:
        put(0,20,36,42,55,0)
        for i,x in enumerate([78,134,190],1): put(i,x,16,44,76,0)
        put(4,248,36,42,55,0);put(5,304,36,42,55,0)
        draw.line([(365*s,5*s),(365*s,106*s)],fill=(30,31,28),width=2*s)
        rw=32 if len(region)==3 else 43;gap=10
        start=371+(143-(rw*len(region)+gap*(len(region)-1)))/2
        for i in range(len(region)): put(6+i,start+i*(rw+gap),15,rw,59,0)
        rus=(390,82);flag=(443,83);bolts=[(12,55),(354,55)]
    # Auxiliary small print is outside target character cells.
    draw.text((rus[0]*s,rus[1]*s),'RUS',font=ImageFont.load_default(size=13*s),fill=(28,29,28))
    if plate_type!='type1b':
        fx,fy=flag
        for band,col in enumerate([(248,248,247),(33,69,145),(180,40,39)]):
            draw.rectangle((fx*s,(fy+band*3)*s,(fx+25)*s,(fy+band*3+3)*s),fill=col)
    for x,y in bolts:
        draw.ellipse(((x-2)*s,(y-2)*s,(x+2)*s,(y+2)*s),fill=(96,96,91))
        draw.line(((x-1)*s,y*s,(x+1)*s,y*s),fill=(33,34,32),width=s)
    return plate, ink, sorted(chars,key=lambda c:c['index'])


def project(points, matrix):
    p=np.c_[np.asarray(points,dtype=np.float64),np.ones(len(points))]@matrix.T
    return (p[:,:2]/p[:,2:]).tolist()
