"""Image/split/label/geometry publication checks and visual contact sheets."""
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.phase_f_setup import DATA,REPORT
from tools.phase_e import read,write,load_image
from src.geometry_f import validate_quad,bbox_from_quad
from dataset.schema import sha256


def validate():
    m=read(DATA/'manifest.json');seen_images={};identity_splits={};errors=[]
    hashes={};split_stats={};classes=('type1','type1a','type1b')
    for r in m['records']:
        path=DATA/'detect/images'/r['split']/r['image'];frame=load_image(path)
        if frame.shape[:2]!=(r['height'],r['width']):errors.append('image dimensions '+r['image'])
        if sha256(path)!=r['sha256']:errors.append('image hash '+r['image'])
        previous=seen_images.get(r['sha256'])
        if previous is not None and previous!=r['split']:errors.append('Image leaks across splits')
        seen_images[r['sha256']]=r['split']
        for a in r['annotations']:
            validate_quad(a['quad'],frame.shape)
            if not np.allclose(bbox_from_quad(a['quad']),a['bbox'],atol=1e-4):errors.append('bbox envelope')
            text=a['plate_num'];previous=identity_splits.get(text)
            if previous is not None and previous!=r['split']:errors.append('Plate identity leaks '+text)
            identity_splits[text]=r['split']
        for task,count in [('detect',5),('obb',9),('pose',17)]:
            label=DATA/task/'labels'/r['split']/Path(r['image']).with_suffix('.txt')
            lines=[line.split() for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
            if len(lines)!=len(r['annotations']):errors.append('target count '+str(label))
            for line,a in zip(lines,r['annotations']):
                vals=list(map(float,line))
                if len(vals)!=count or int(vals[0])!=classes.index(a['plate_type']):errors.append('label shape/class')
                if task in ('detect','pose'):
                    box=np.array(a['bbox']);expected=np.r_[(box[:2]+box[2:])/2,box[2:]-box[:2]]
                    if not np.allclose(np.array(vals[1:5])*[r['width'],r['height'],r['width'],r['height']],expected,atol=1e-4):
                        errors.append('detector bbox label differs from true quad envelope')
                coordinates=vals[1:] if task!='pose' else vals[1:5]+[v for i,v in enumerate(vals[5:]) if i%3!=2]
                if not all(0<=v<=1 for v in coordinates):errors.append('normalized coordinate bounds')
                if task=='obb' and not np.allclose(np.array(vals[1:]).reshape(4,2)*[r['width'],r['height']],a['obb'],atol=1e-4):errors.append('OBB label differs')
                if task=='pose' and not np.allclose(np.array(vals[5:]).reshape(4,3)[:,:2]*[r['width'],r['height']],a['quad'],atol=1e-4):errors.append('quad label differs')
            hashes[str(label)]=sha256(label)
        split_stats.setdefault(r['split'],Counter())['scenes']+=1
        split_stats[r['split']]['real_replicates' if not r['is_synthetic'] else 'synthetic_scenes']+=1
        if not r['annotations']:split_stats[r['split']]['negatives']+=1
        for a in r['annotations']:split_stats[r['split']][a['plate_type']]+=1
    result={'status':'PASS' if not errors else 'FAIL','visual_review':'PENDING','errors':errors,
        'images_checked':len(m['records']),'label_files_checked':len(hashes),'label_hashes':hashes,
        'manifest_sha256':sha256(DATA/'manifest.json'),'statistics':{k:dict(v) for k,v in split_stats.items()},
        'checks':['image hashes/readability/dimensions','scene hash disjoint splits','plate identity disjoint splits',
            'every positive has actual convex ordered quad','bbox equals quad envelope','pose equals real corners',
            'OBB equals minimum-area rectangle','normalized coordinate limits','negative targets empty',
            'canonical raster checks performed during generation'],
        'generalization':'GENERALIZATION NOT YET VERIFIED'}
    write(REPORT/'dataset_validation.json',result)
    if errors:raise ValueError(errors[:10])
    print({k:v for k,v in result.items() if k!='label_hashes'},flush=True)
    return m


def preview(m):
    rows=[]
    for kind in ('type1','type1a','type1b'):
        candidates=[r for r in m['records'] if r['is_synthetic'] and r['split']=='val' and any(a['plate_type']==kind for a in r['annotations'])]
        # Include smallest, median, largest observed target sizes.
        candidates=sorted(candidates,key=lambda r:r['annotations'][0]['bbox'][2]-r['annotations'][0]['bbox'][0])
        rows.extend([candidates[i] for i in (0,len(candidates)//4,len(candidates)//2,-1)])
    for kind in ('vehicle_without_plate','billboard','road_sign','shop_sign','screen','printed_plate','text','background'):
        rows.append(next(r for r in m['records'] if r.get('negative_kind')==kind))
    rows.extend([r for r in m['records'] if not r['is_synthetic'] and r['split']=='val'])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    paths=[]
    for page in range((len(rows)+7)//8):
        canvas=Image.new('RGB',(1200,1280),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(rows[page*8:(page+1)*8]):
            x=i%2*600;y=i//2*320
            frame=load_image(DATA/'detect/images'/r['split']/r['image']);original=frame.copy()
            for a in r['annotations']:
                cv2.polylines(frame,[np.int32(a['quad'])],True,(0,255,0),2)
            label=' / '.join(a['plate_type']+' '+a['plate_num'] for a in r['annotations']) or 'NEGATIVE '+r.get('negative_kind','unknown')
            draw.text((x+5,y+3),r['image']+' '+r['split'],fill='black',font=font)
            draw.text((x+5,y+22),label,fill='black',font=font)
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((440,265));canvas.paste(im,(x+5,y+47))
            if r['annotations']:
                a=r['annotations'][0];x1,y1,x2,y2=np.array(a['bbox'],int)
                crop=original[max(0,y1):y2+1,max(0,x1):x2+1]
                im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));im.thumbnail((140,210))
                if im.width<140:im=im.resize((140,max(1,round(im.height*140/im.width))))
                canvas.paste(im,(x+452,y+65))
        path=REPORT/'dataset_review'/f'page_{page+1:02}.jpg';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path);paths.append(str(path))
    write(REPORT/'dataset_review/manifest.json',{'paths':paths,'selected_samples':rows,'status':'PENDING',
        'scope':'Representative procedural scenes including tiny/blurred positives, all8 negative categories, allreal validation scenes; not a claim of photographic realism'})
    print('Review sheets:',paths,flush=True)


if __name__=='__main__':preview(validate())
