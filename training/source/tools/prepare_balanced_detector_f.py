"""Publish v2 without altering v1. Additional manual quads are train-only."""
from collections import Counter
import copy
import csv
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import yaml
from tools.phase_e import ROOT,IMAGES,read,write,load_image
from dataset.schema import sha256
from src.geometry_f import canonical_rectify,bbox_from_quad,order_obb,validate_quad
from PIL import Image,ImageDraw,ImageFont

REPORT=ROOT/'reports/phase_f'
V1=ROOT.parent/'generated_data/detector_f_v1'
V2=ROOT.parent/'generated_data/detector_f_v2'
MANUAL={
 'img_018.jpg':('B256AY550',[[747,1084],[892,1084],[889,1169],[743,1165]]),
 'img_026.jpg':('E180YP125',[[441,566],[553,562],[554,618],[442,622]]),
 'img_032.jpg':('M941MT14',[[1225,1048],[1358,1038],[1365,1125],[1231,1137]]),
 'img_033.jpg':('M493PH125',[[973,466],[1035,467],[1036,504],[974,504]]),
 'img_087.jpg':('T728HC14',[[708,791],[893,795],[891,889],[708,885]]),
 'img_092.jpg':('Y575CA125',[[635,972],[838,978],[837,1095],[632,1084]]),
 'img_097.jpg':('X958KP163',[[902,1056],[1007,1058],[1007,1117],[903,1114]])}

def main():
    assert not V2.exists(),'Versioned publication must never silently overwrite data'
    m=read(V1/'manifest.json')
    for task in ('detect','obb','pose'):
        for split in ('train','val','test'):
            for kind in ('images','labels'):
                dst=V2/task/kind/split;dst.mkdir(parents=True,exist_ok=True)
                for p in (V1/task/kind/split).iterdir():
                    if p.suffix in ('.jpg','.txt'):os.link(p,dst/p.name)
        data=yaml.safe_load((V1/f'{task}.yaml').read_text(encoding='utf-8'))
        data['path']=str(V2/task).replace('\\','/')
        (V2/f'{task}.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    known=read(ROOT/'reports/phase_e/oracle_expansion/manifest.json')['records']
    with (IMAGES.parent/'debug_labels.csv').open(encoding='utf-8-sig') as f:gt=list(csv.DictReader(f,delimiter=';'))
    annotations=[];crops=[]
    for name,(visible,q) in MANUAL.items():
        source=load_image(IMAGES/name);h,w=source.shape[:2];validate_quad(q,source.shape)
        records=[(r['ground_truth'],r['plate_type'],r['manual_quad'],'human_phase_E') for r in known if r['image']==name]
        records.append((visible,'type1a',q,'human_phase_F_pixel_grid_review'))
        assert len(records)==sum(g['image']==name for g in gt),'Only exhaustive scenes allowed'
        aa=[];lines={t:[] for t in ('detect','obb','pose')}
        for text,kind,quad,provenance in records:
            quad=np.float32(quad);box=np.array(bbox_from_quad(quad));obb=order_obb(cv2.boxPoints(cv2.minAreaRect(quad)))
            a={'plate_type':kind,'plate_num':text,'quad':quad.tolist(),'bbox':box.tolist(),'obb':obb.tolist(),'quad_source':provenance}
            if name=='img_032.jpg' and kind=='type1a':
                a['organizer_literal_plate_num']='M491MT14';a['annotation_issue']='Visible raster reads M941MT14; official scorer/labels retained unchanged, detector training uses observed identity only'
            aa.append(a);c=('type1','type1a','type1b').index(kind)
            xywh=np.r_[(box[:2]+box[2:])/2,box[2:]-box[:2]]/[w,h,w,h]
            lines['detect'].append(' '.join(map(str,[c,*xywh])))
            lines['obb'].append(' '.join(map(str,[c,*(obb/[w,h]).ravel()])))
            lines['pose'].append(' '.join(map(str,[c,*xywh,*[v for p in quad/[w,h] for v in (*p,2)]])))
            crop,_=canonical_rectify(source,quad,kind,'manual_train_annotation')
            crops.append((name+' '+kind+' '+text,crop))
        annotations.append({'original_image':name,'source_sha256':sha256(IMAGES/name),'annotations':aa})
        for repeat in range(6):
            output=f'real_{Path(name).stem}_{repeat}.jpg'
            for task in lines:
                os.link(IMAGES/name,V2/task/'images/train'/output)
                (V2/task/'labels/train'/Path(output).with_suffix('.txt')).write_text('\n'.join(lines[task])+'\n',encoding='utf-8')
            m['records'].append({'image':output,'original_image':name,'split':'train','width':w,'height':h,
                'is_synthetic':False,'domain':'organizer_public_DEVELOPMENT','annotations':copy.deepcopy(aa),'sha256':sha256(IMAGES/name),'replicate':repeat})
    m['dataset_version']='detector_f_v2';m['parent_manifest_sha256']=sha256(V1/'manifest.json')
    m['real_import']['original_train_images']=sorted(m['real_import']['original_train_images']+list(MANUAL))
    m['real_import'].update(unique_frames=28,train_frames=20)
    m['composition'].update(real_unique_scenes=28,real_scenes=28,real_training_replicates=120)
    m['amendment']='Seven exhaustively completed mixed-type TRAIN scenes; same eight real val scenes and same synthetic splits; no new real holdout'
    write(V2/'manifest.json',m)
    write(REPORT/'real_annotation_v2/manual_annotations.json',{'status':'PENDING_VISUAL_RECTIFIED_REVIEW','records':annotations,
        'provenance':'Physical plate borders manually selected from original source with absolute coordinate grid; detector proposals used only to locate display; no model-generated quad ground truth',
        'scope':'All additional scenes are training only; public official GT unchanged; no OCR training'})
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    canvas=Image.new('RGB',(1200,((len(crops)+2)//3)*230),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,crop) in enumerate(crops):
        x=i%3*400;y=i//3*230;draw.text((x+5,y+5),label,fill='black',font=font)
        im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));im.thumbnail((380,185));canvas.paste(im,(x+5,y+35))
    canvas.save(REPORT/'real_annotation_v2/rectified_review.jpg')
    # Architecture class count is fixed BEFORE weight transfer to retain compatible class branches.
    import ultralytics
    cfgroot=Path(ultralytics.__file__).parent/'cfg/models/11'
    for task,name in [('detect','F02_balanced'),('detect','F02_clean_v2'),('obb','F03_obb_v2'),('pose','F04_pose_v2')]:
        suffix='' if task=='detect' else '-'+task
        architecture=yaml.safe_load((cfgroot/f'yolo11{suffix}.yaml').read_text(encoding='utf-8'))
        architecture.update(nc=3,scale='n')
        if task=='pose':architecture['kpt_shape']=[4,3]
        archpath=ROOT/'configs/phase_f'/f'yolo11n_f{suffix}.yaml'
        archpath.write_text(yaml.safe_dump(architecture,sort_keys=False),encoding='utf-8')
        config=read(ROOT/'configs/phase_f/F02_finetune.json')
        config.update(name=name,task=task,architecture=str(archpath),data=str(V2/f'{task}.yaml'),dataset_root=str(V2),
            initialization='nc=3 before transfer; preserve matching plate-class feature branches; replace final class projection/task-only heads')
        if name=='F02_clean_v2':config['initial_weights']=str(ROOT/'models/phase_f/pretrained/yolo11n_coco.pt')
        config['train']['name']=name
        write(ROOT/'configs/phase_f'/f'{name}.json',config)
    print('Published',len(m['records']),'scenes; added',dict(Counter(a['plate_type'] for r in annotations for a in r['annotations'])),flush=True)

if __name__=='__main__':main()
