"""Read raster symbols against a frozen, visually reviewed alphabet; never echo labels."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ..schema import ALPHABET, read_metadata, resolve_image, sha256


def bank(root):
    manifest=json.loads((root/'generator/assets/manifest.json').read_text(encoding='utf-8'))
    result={}
    if set(manifest['glyphs'])!=set(ALPHABET+'0123456789'):
        raise ValueError('Glyph bank incomplete')
    for char,expected in manifest['glyphs'].items():
        path=root/'generator/assets'/f'{char}.png'
        if sha256(path)!=expected:raise ValueError(f'Glyph asset changed: {char}')
        with Image.open(path) as im:result[char]=np.array(im.convert('L').resize((64,96)))>127
    return result


def raster_text(image, characters, templates):
    """Nearest binary glyph; candidates constrained only by fixed mask positions."""
    gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    chars=[];scores=[]
    for entry in characters:
        x0,y0,x1,y1=entry['box']
        crop=gray[y0:y1,x0:x1]
        if crop.size==0:raise ValueError('Empty character cell')
        _,ink=cv2.threshold(crop,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        normalized=cv2.resize(ink,(64,96),interpolation=cv2.INTER_AREA)>127
        candidates=ALPHABET if entry['index'] in (0,4,5) else '0123456789'
        def distance(char):
            expected=templates[char]
            return float(np.count_nonzero(normalized^expected)/max(1,np.count_nonzero(normalized|expected)))
        best=min(candidates,key=lambda char:(distance(char),char))
        chars.append(best);scores.append(1-distance(best))
    return ''.join(chars),scores


def check_sample(root, row, sample, templates, verify_hashes=True):
    errors=[]
    def fail(message):errors.append(message)
    if sample['image']!=row.image or sample['plate_type']!=row.plate_type:fail('image/type differs from raster provenance')
    if not row.is_synthetic or row.is_vehicle is not None or row.annotation_kind!='ocr_crop':fail('Synthetic crop must not invent vehicle context')
    characters=sample['characters'];n=len(row.plate_num)
    if [c['index'] for c in characters]!=list(range(n)):return ['Target character count/order differs from ground truth']
    with Image.open(resolve_image(root,sample['canonical'])) as im:canonical=im.convert('RGB')
    with Image.open(resolve_image(root,row.image)) as im:final=im.convert('RGB')
    color=np.median(np.asarray(canonical).reshape(-1,3),axis=0)
    if bool(color[0]-color[2]>100)!=(row.plate_type=='type1b'):fail('Canonical plate color differs from type')
    expected_ratio=290/170 if row.plate_type=='type1a' else 520/112
    if abs(canonical.width/canonical.height-expected_ratio)>1e-6:fail('Canonical layout dimensions differ from type')
    if verify_hashes:
        for field in ('image','canonical','footprint'):
            if sha256(resolve_image(root,sample[field]))!=sample[field+'_sha256']:fail(f'{field} pixels changed')
    if row.plate_type=='type1a':
        if [c['row'] for c in characters]!=[0]*4+[1]*(n-4):fail('type1a requires A123 / BC+region rows')
        if max(c['box'][3] for c in characters[:4])>=min(c['box'][1] for c in characters[4:]):fail('Rows overlap or type1a is already stitched')
    elif any(c['row']!=0 for c in characters):fail('Long plate must have a single target row')
    for group in ([characters[:4],characters[4:]] if row.plate_type=='type1a' else [characters]):
        if any(a['box'][2]>=b['box'][0] for a,b in zip(group,group[1:])):fail('Character cells overlap or reading order is wrong')
    w,h=canonical.size
    for c in characters:
        x0,y0,x1,y1=c['box']
        if not 0<=x0<x1<=w or not 0<=y0<y1<=h:fail('Character cell leaves plate')
    decoded,scores=raster_text(canonical,characters,templates)
    if decoded!=row.plate_num:fail(f'Canonical raster reads {decoded}, metadata says {row.plate_num}')
    if min(scores)<.76:fail('Canonical glyph is missing or malformed')
    matrix=np.array(sample['homography'],dtype=np.float64)
    src=np.array([[0,0],[w,0],[w,h],[0,h]],dtype=np.float64)
    homogeneous=np.c_[src,np.ones(4)]@matrix.T
    projected=homogeneous[:,:2]/homogeneous[:,2:]
    if not np.allclose(projected,row.quad,atol=1e-4):fail('Quad differs from raster transform')
    expected_bbox=np.r_[projected.min(axis=0),projected.max(axis=0)-projected.min(axis=0)]
    if not np.allclose(expected_bbox,row.bbox,atol=1e-4):fail('bbox is not quad envelope')
    restored=cv2.warpPerspective(np.array(final),np.linalg.inv(matrix),(w,h),flags=cv2.INTER_LINEAR)
    decoded_final,final_scores=raster_text(restored,characters,templates)
    if decoded_final!=row.plate_num:fail(f'Final raster reads {decoded_final}, metadata says {row.plate_num}')
    if min(final_scores)<.65:fail('Final glyph readability below conservative synthetic gate')
    with Image.open(resolve_image(root,sample['footprint'])) as im:mask=np.array(im)>32
    yy,xx=np.where(mask)
    if not len(xx):fail('Projected plate footprint is empty')
    else:
        actual=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
        expected=np.r_[projected.min(axis=0),projected.max(axis=0)]
        if np.max(np.abs(actual-expected))>3:fail('Projected plate footprint differs from quad envelope')
    return errors


def verify_collection(root):
    root=Path(root).resolve();rows,issues=read_metadata(root/'meta.csv')
    manifest=json.loads((root/'generator/manifest.json').read_text(encoding='utf-8'))
    templates=bank(root);samples={s['image']:s for s in manifest['samples']}
    errors=[{'image':e.get('image',''),'errors':[e['message']]} for e in issues]
    if manifest['count']!=len(rows) or len(samples)!=len(manifest['samples']) or set(samples)!={r.image for r in rows}:
        errors.append({'image':'','errors':['Raster provenance population differs from metadata']})
    for path,key in [('generator/config.json','config_sha256'),('generator/assets/manifest.json','glyph_manifest_sha256')]:
        if sha256(root/path)!=manifest[key]:errors.append({'image':'','errors':[f'{path} changed']})
    for row in rows:
        try:problems=check_sample(root,row,samples[row.image],templates)
        except (OSError,ValueError,KeyError,TypeError,cv2.error) as error:problems=[str(error)]
        if problems:errors.append({'image':row.image,'errors':problems})
    return {'valid':bool(rows) and not errors,'images_checked':len(rows),'errors':errors,
            'checks':['canonical raster-to-label','inverse-warped final raster-to-label','type layout/order','quad/bbox/footprint','image/canonical/mask hashes','unknown context'],
            'meaning':'Template contract/readability checks on authored synthetic font; not a real OCR or model accuracy benchmark.'}
