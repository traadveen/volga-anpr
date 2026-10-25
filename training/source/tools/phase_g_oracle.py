"""Diagnostic supplied quad/manual split; never an inference candidate."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from tools.phase_g_common import ROOT,REPORT,IMAGES,freeze,read,write
from tools.phase_g_audit import gray_input,FONT,paste
from tools.phase_e import load_image,summary
from src.geometry_f import canonical_rectify,bbox_iou,bbox_from_quad
from src.image_anpr import save_image,FrozenCRNNOCR,current_postprocess
from src.multiline import stitch_rows,MultilineConfig
from src.ocr_preprocess import prepare_ocr_input
from ocr.crnn import _normalize_image
from tools.evaluation import levenshtein

def references():
    records={}
    for r in read(ROOT/'reports/phase_e/oracle_expansion/manifest.json')['records']:
        records[(r['image'],r['ground_truth'])]={'image':r['image'],'ground_truth':r['ground_truth'],'plate_type':r['plate_type'],'quad':r['manual_quad']}
    for r in read(ROOT/'reports/phase_f/real_annotation_v2/manual_annotations.json')['records']:
        for a in r['annotations']:
            key=(r['original_image'],a.get('organizer_literal_plate_num',a['plate_num']))
            records[key]={'image':key[0],'organizer_ground_truth':key[1],'ground_truth':a['plate_num'],'plate_type':a['plate_type'],'quad':a['quad']}
    return list(records.values())

def manual_stitch(gray,cut_ratio):
    h,w=gray.shape;cut=round(h*cut_ratio)
    # Full row width and conservative vertical trim keep every character;
    # oracle cut is reviewed separately. Same F height/gap/padding renderer.
    margin=max(1,round(h*.025));rows=(gray[margin:cut,:],gray[cut:h-margin,:])
    image,transforms=stitch_rows(rows,MultilineConfig())
    return image,{'cut_y':cut,'cut_ratio':cut_ratio,'row_boxes':[[0,margin,w,cut],[0,cut,w,h-margin]],'stitch_transforms':[t.tolist() for t in transforms]}

def prepare():
    freeze();records=references()
    for i,r in enumerate(records):
        r['id']=f'{i:02}_{Path(r["image"]).stem}';frame=load_image(IMAGES/r['image'])
        crop,_=canonical_rectify(frame,r['quad'],r['plate_type'],'diagnostic_manual_E_F_quad')
        folder=REPORT/'oracle'/r['id'];folder.mkdir(parents=True,exist_ok=True)
        save_image(folder/'quad_crop.png',crop);gray=gray_input(crop)
        r.update(crop=str(folder/'quad_crop.png'),review_status='PENDING',source_degraded=r['image']=='img_018.jpg' and r['plate_type']=='type1a')
        if r['plate_type']=='type1a':
            r['proposed_manual_cut_ratio']=.5;stitched,diagnostics=manual_stitch(gray,.5)
            marked=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);cv2.line(marked,(0,round(gray.shape[0]*.5)),(gray.shape[1]-1,round(gray.shape[0]*.5)),(0,0,255),1)
            save_image(folder/'cut_review.png',marked);save_image(folder/'manual_stitch.png',stitched)
            save_image(folder/'manual_normalized.png',np.clip((_normalize_image(stitched,32,160)+1)*127.5,0,255).astype(np.uint8))
            r['manual_split_diagnostics']=diagnostics
    for page in range((len(records)+7)//8):
        c=Image.new('RGB',(1600,1400),'white');dr=ImageDraw.Draw(c)
        for j,r in enumerate(records[page*8:(page+1)*8]):
            x=j%2*800;y=j//2*350;dr.text((x+5,y+5),r['id']+' '+r['ground_truth']+' '+r['plate_type'],font=FONT,fill='black')
            folder=Path(r['crop']).parent;paste(c,folder/'cut_review.png' if r['plate_type']=='type1a' else r['crop'],(x+5,y+40,360,230))
            if r['plate_type']=='type1a':paste(c,folder/'manual_stitch.png',(x+375,y+40,420,210));paste(c,folder/'manual_normalized.png',(x+375,y+260,420,84))
        c.save(REPORT/'oracle'/f'review_{page+1:02}.jpg',quality=95)
    write(REPORT/'oracle/manifest.json',{'records':records,'review_status':'PENDING','scope':'Existing approximate manually localized 38 targets, convenience exposed sample; manual split proposals need direct visual confirmation before evaluation; corrected identity stored separately from official literal.'})
    print('Prepared',len(records),'oracle targets; review cut and normalized inputs',flush=True)

def evaluate():
    manifest=read(REPORT/'oracle/manifest.json');assert manifest['review_status']=='PASS'
    torch.set_num_threads(4);cv2.setNumThreads(1);ocr=FrozenCRNNOCR(freeze()['baseline']['ocr'],'cuda')
    traces={r['image']:r for r in read(REPORT/'audit/artifact_index.json')['records']};rows=[]
    for r in manifest['records']:
        assert r['review_status']=='PASS';frame=load_image(IMAGES/r['image']);ds=traces[r['image']]['detections']
        overlap,d=max(((bbox_iou(d['bbox'],bbox_from_quad(r['quad'])),d) for d in ds),key=lambda z:z[0],default=(0,None))
        for mode in ('AUTO_OBB_AUTO_SPLIT','GT_QUAD_AUTO_SPLIT','GT_QUAD_MANUAL_SPLIT'):
            if mode=='AUTO_OBB_AUTO_SPLIT' and overlap<.5:
                rows.append(dict(r,mode=mode,detected=False,raw='',post='',ms=None,good_input_eligible=False));continue
            if mode=='AUTO_OBB_AUTO_SPLIT':
                crop,_=canonical_rectify(frame,d['observed_quad'],d['plate_type'],d['quad_source'])
            else:crop=load_image(Path(r['crop']))
            gray=gray_input(crop);start=time.perf_counter()
            if mode=='GT_QUAD_MANUAL_SPLIT' and r['plate_type']=='type1a':
                image,diagnostics=manual_stitch(gray,r['manual_cut_ratio']);normalized=_normalize_image(image,32,160)
            else:
                p=prepare_ocr_input(gray,r['plate_type']);normalized=p.normalized;diagnostics=p.diagnostics
            raw,confidence,probs=ocr.recognize(normalized);ocr.synchronize();ms=(time.perf_counter()-start)*1000
            rows.append(dict(r,mode=mode,detected=overlap>=.5,raw=raw,post=current_postprocess(raw,probs),ocr_confidence=confidence,ms=ms,
                diagnostics=diagnostics,good_input_eligible=not r['source_degraded']))
    metrics={}
    for mode in ('AUTO_OBB_AUTO_SPLIT','GT_QUAD_AUTO_SPLIT','GT_QUAD_MANUAL_SPLIT'):
        metrics[mode]={}
        for kind in ('ALL','type1','type1a'):
            group=[r for r in rows if r['mode']==mode and (kind=='ALL' or r['plate_type']==kind)]
            metrics[mode][kind]={'count':len(group),'raw_correct':sum(r['raw']==r['ground_truth'] for r in group),
                'post_correct':sum(r['post']==r['ground_truth'] for r in group),
                'raw_CER':sum(levenshtein(r['ground_truth'],r['raw']) for r in group)/sum(len(r['ground_truth']) for r in group)}
    good=[r for r in rows if r['mode']=='GT_QUAD_MANUAL_SPLIT' and r['good_input_eligible']]
    write(REPORT/'oracle/comparison.json',{'metrics':metrics,'rows':rows,'manual_good_input':{'count':len(good),'raw_correct':sum(r['raw']==r['ground_truth'] for r in good),'post_correct':sum(r['post']==r['ground_truth'] for r in good)},
        'scope':'All38 comparison includes2 auto detector misses; geometry/split ceiling is conditional convenience development. Manual split supplied only here; not eligible for BEST-G. Timing excludes detector and warp.',
        'latency':{mode:summary([r['ms'] for r in rows if r['mode']==mode and r['ms'] is not None]) for mode in metrics}})
    print(metrics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate']);args=p.parse_args()
    (prepare if args.action=='prepare' else evaluate)()
