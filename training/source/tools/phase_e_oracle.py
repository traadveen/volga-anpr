"""Expanded manual spatial annotations, frozen before their OCR evaluation."""
from datetime import datetime,timezone
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from tools.phase_e import ROOT,REPORT,IMAGES,BEST,read,write,load_image,summary
from tools.phase_e_diagnostics import FONT,SMALL
from src.image_anpr import FrozenCRNNOCR,save_image,current_postprocess
from src.ocr_preprocess import prepare_ocr_input
from dataset.schema import sha256
from tools.benchmark_multiline import distance

# TL,TR,BR,BL in original pixels, manually read from coordinate previews.
# No text, model confidence or predicted characters set these coordinates.
QUADS={
 'img_010_1':[[24,542],[99,541],[100,560],[24,561]],
 'img_018_0':[[99,843],[299,839],[299,881],[100,886]],
 'img_018_1':[[1539,731],[1696,729],[1695,763],[1538,768]],
 'img_026_0':[[245,392],[352,391],[352,417],[245,417]],
 'img_032_0':[[95,765],[246,749],[246,787],[96,800]],
 'img_033_0':[[158,453],[242,453],[242,476],[159,476]],
 'img_087_1':[[1418,481],[1518,481],[1518,507],[1418,507]],
 'img_092_0':[[88,445],[180,445],[180,469],[88,469]],
 'img_097_1':[[1878,1067],[1980,1068],[1980,1093],[1878,1091]],
 'img_011_0':[[1091,925],[1181,925],[1180,981],[1090,981]],
 'img_025_0':[[555,864],[628,863],[628,906],[555,906]],
 'img_031_0':[[1097,1046],[1197,1045],[1197,1109],[1097,1110]],
 'img_037_0':[[570,895],[641,896],[641,938],[571,939]],
 'img_046_0':[[536,815],[610,814],[612,861],[536,861]],
 'img_051_0':[[609,809],[682,808],[682,849],[609,849]],
 'img_005_0':[[325,540],[441,548],[431,634],[314,623]],
 'img_007_0':[[319,811],[368,811],[368,839],[319,838]],
 'img_010_0':[[1098,739],[1170,737],[1176,795],[1102,798]],
 'img_012_0':[[360,666],[424,701],[426,758],[363,728]],
 'img_016_0':[[1088,667],[1192,656],[1205,737],[1099,748]],
 'img_019_0':[[868,715],[896,712],[895,747],[867,750]],
 'img_029_0':[[908,676],[1007,671],[1002,744],[904,750]],
 'img_030_0':[[975,850],[1036,847],[1038,891],[975,894]],
}


def prepare_annotations():
    folder=REPORT/'oracle_expansion';source=read(folder/'candidates.json')
    records=[]
    for r in source['records']:
        key=f"{Path(r['image']).stem}_{r['index']}";quad=QUADS[key]
        records.append({'id':key,'image':r['image'],'ground_truth':r['ground_truth'],
            'plate_type':r['plate_type'],'manual_quad':quad,'detector_bbox':r['bbox']})
    for r in source['original_eight']:
        refs=read(REPORT/'detections.json')['records']
        ref=next(v for v in refs if v['image']==r['image'] and v['ground_truth']==r['ground_truth'])
        records.append({'id':r['id'],'image':r['image'],'ground_truth':r['ground_truth'],
            'plate_type':r['plate_type'],'manual_quad':r['manual_quad'],'detector_bbox':ref['bbox']})
    for r in records:
        source_path=IMAGES/r['image'];r['source_sha256']=sha256(source_path)
        quad=np.float32(r['manual_quad']);assert cv2.isContourConvex(quad)
        w,h=(290,170) if r['plate_type']=='type1a' else (520,112)
        target=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        matrix=cv2.getPerspectiveTransform(quad,target)
        crop=cv2.warpPerspective(load_image(source_path),matrix,(w,h))
        prepared=prepare_ocr_input(crop,r['plate_type'])
        path=folder/f"{r['id']}_rectified.png";save_image(path,crop)
        inp=folder/f"{r['id']}_input.png"
        save_image(inp,np.clip((prepared.normalized+1)*127.5,0,255).astype(np.uint8))
        r.update({'rectified_crop':str(path),'crop_sha256':sha256(path),'OCR_input':str(inp),
            'homography':matrix.tolist(),'preprocessing':prepared.diagnostics})
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'records':records,
        'evaluation_only':True,'no_training':True,'review_status':'PENDING',
        'annotation_basis':'Human plate border coordinates from original pixels; 23 additions plus the eight Phase D quads; subset selected by visibility before expanded oracle OCR evaluation',
        'canonical_sizes':{'type1a':[290,170],'type1':[520,112]},
        'limitation':'Convenience subset with visible detectable plates, not random/representative. Manual coordinates are approximate and lack independent annotation agreement.'}
    write(folder/'manifest.json',manifest)
    for page in range((len(records)+7)//8):
        canvas=Image.new('RGB',(1120,1000),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(records[page*8:(page+1)*8]):
            x=i%2*560;y=i//2*250
            draw.text((x+5,y+5),f"{r['id']} {r['ground_truth']} {r['plate_type']}",fill='black',font=FONT)
            im=Image.open(r['rectified_crop']);im.thumbnail((530,170));canvas.paste(im,(x+5,y+35))
            im=Image.open(r['OCR_input']);im=im.resize((320,64));canvas.paste(im,(x+5,y+177))
        canvas.save(folder/f'verify_{page+1:02}.png')
    print(f'Prepared {len(records)} manual crops, review before evaluation')


def metrics(rows):
    return {'count':len(rows),'correct':sum(r['ground_truth']==r['prediction'] for r in rows),
            'full_plate_accuracy':sum(r['ground_truth']==r['prediction'] for r in rows)/len(rows),
            'cer':sum(distance(r['ground_truth'],r['prediction']) for r in rows)/sum(len(r['ground_truth']) for r in rows)}


def evaluate():
    manifest=read(REPORT/'oracle_expansion/manifest.json')
    assert manifest['review_status']=='PASS'
    torch.set_num_threads(4);cv2.setNumThreads(1)
    ocr=FrozenCRNNOCR(BEST,'cuda')
    rows=[]
    for r in manifest['records']:
        assert sha256(IMAGES/r['image'])==r['source_sha256']
        assert sha256(Path(r['rectified_crop']))==r['crop_sha256']
        x1,y1,x2,y2=map(int,r['detector_bbox']);frame=load_image(IMAGES/r['image'])
        auto=frame[y1:y2,x1:x2]
        for mode,image in (('auto_known_type',auto),('oracle_known_type',load_image(Path(r['rectified_crop'])))):
            # Known type for both isolates crop geometry from type routing.
            ocr.synchronize();start=time.perf_counter()
            h,w=image.shape[:2];scale=max(480/w,160/h)
            if scale>1:image=cv2.resize(image,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_CUBIC)
            p=prepare_ocr_input(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),r['plate_type'])
            raw,confidence,probabilities=ocr.recognize(p.normalized)
            ocr.synchronize();elapsed=(time.perf_counter()-start)*1000
            rows.append({'id':r['id'],'image':r['image'],'ground_truth':r['ground_truth'],
                'plate_type':r['plate_type'],'mode':mode,'prediction':raw,'confidence':confidence,
                'postprocessed':current_postprocess(raw,probabilities),'preprocessing':p.diagnostics,
                'prepared_crop_OCR_ms':elapsed})
    result={'count':len(manifest['records']),'manifest_sha256':sha256(REPORT/'oracle_expansion/manifest.json'),
        'scope':'OCR-only manually localized subset; same known GT type and common preprocessing for auto/oracle; never eligible for production selection',
        'cases':rows,'metrics':{}}
    for mode in ('auto_known_type','oracle_known_type'):
        result['metrics'][mode]={}
        for kind in ('all','type1','type1a'):
            group=[r for r in rows if r['mode']==mode and (kind=='all' or r['plate_type']==kind)]
            result['metrics'][mode][kind]=metrics(group)
    result['latency']={mode:summary([r['prepared_crop_OCR_ms'] for r in rows if r['mode']==mode])
                       for mode in ('auto_known_type','oracle_known_type')}
    result['latency_scope']='Conditional supplied crop upscale+common preprocessing+GPU OCR+greedy; no detector, file IO or manual annotation cost; first call cold'
    write(REPORT/'oracle_expanded_comparison.json',result)
    # IoU measured only for actual manually annotated plates, using all83 boxes.
    refs=read(REPORT/'detections.json')['records'];spatial=[]
    for r in manifest['records']:
        q=np.array(r['manual_quad']);gt=np.r_[q.min(axis=0),q.max(axis=0)]
        candidates=[v for v in refs if v['image']==r['image']]
        def iou(box):
            box=np.array(box);inter=np.maximum(0,np.minimum(gt[2:],box[2:])-np.maximum(gt[:2],box[:2])).prod()
            return float(inter/((gt[2:]-gt[:2]).prod()+(box[2:]-box[:2]).prod()-inter))
        overlap=max((iou(v['bbox']) for v in candidates),default=0)
        spatial.append({'id':r['id'],'best_detector_IoU':overlap,'GT_bbox':gt.tolist(),
            'GT_quad':r['manual_quad'],'note':'IoU of enclosing axis-aligned GT bbox; detector has no quadrilateral output'})
    write(REPORT/'spatial_oracle_analysis.json',{'count':len(spatial),'records':spatial,
        'recall_at_IoU':{str(t):sum(r['best_detector_IoU']>=t for r in spatial)/len(spatial) for t in (.3,.5,.75)},
        'scope':'Selected visibly detectable convenience subset. Not global detector recall; no undetected examples annotated here.'})
    print(json.dumps(result['metrics'],indent=2))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='evaluate':evaluate()
    else:prepare_annotations()
